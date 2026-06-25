"""Builds train_and_test.ipynb. Run from repo root."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); cells = []
def md(s):   cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ---------------------------------------------------------------------------
# 1. Title
# ---------------------------------------------------------------------------
md("# Stage 3 — Train & Test (per dataset × arch)")

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
code(r"""import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # CPU fallback for ops MPS lacks
import sys, json
sys.path.insert(0, os.path.abspath('../src'))
import numpy as np, torch, csi
from pathlib import Path

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
# Prefer GPU: CUDA, else Apple-Silicon Metal (MPS), else CPU.
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'      # M-series GPU — ~4x faster for the conv CsiNet
else:
    DEVICE = 'cpu'
print('device:', DEVICE)""")

# ---------------------------------------------------------------------------
# 3. Config constants + helper functions (ALL in one cell)
# ---------------------------------------------------------------------------
code(r'''# ── constants ───────────────────────────────────────────────────────────────
# Two model families (CsiNet legacy vs TransNet) at matched M, on the canonical
# CDL-C-vs-synthetic pair used by the comparison's two-panel figure.
LABELS   = ['cdlc_3p5ghz', 'synthetic_beam']
ARCHS    = ['csinet16', 'csinet32', 'csinet64',
            'transnet16', 'transnet32', 'transnet64']
EPOCHS   = 30
AI_BITS  = [2, 3, 4]
N_DELAY  = 32

RAW_ROOT     = csi.RAW_ROOT      # models/raw/<arch>/
TRAINED_ROOT = csi.TRAINED_ROOT  # models/trained/<label>/<arch>/

# ── helpers ─────────────────────────────────────────────────────────────────
def load_raw(arch):
    # Load an untrained (raw) model from models/raw/<arch>/
    arch_dir  = RAW_ROOT / arch
    arch_json = json.loads((arch_dir / 'arch.json').read_text())
    net       = getattr(csi, arch_json['class'])(**arch_json['kwargs'])
    net.load_state_dict(torch.load(arch_dir / 'model.pt', map_location='cpu'))
    n_code    = arch_json['kwargs']['n_code']
    return net, n_code, arch_json


def train_one(label, arch):
    # Train arch on the given label dataset, save model + metrics, return metrics dict.
    # — load data —
    data  = csi.load_dataset(csi.DATA_ROOT / label)
    Htr   = data['H_train']
    Hte   = data['H_test']
    n_tx  = int(data['cfg'].gnb_tx)
    W_true = np.asarray(data['reports']['W_true'])   # (Nte, n_tx) complex64

    # — angular-delay transform + standardise —
    Xtr  = csi.complex_to_real_imag(csi.to_angular_delay(Htr, N_DELAY))
    Xte  = csi.complex_to_real_imag(csi.to_angular_delay(Hte, N_DELAY))
    sc   = csi.Standardizer().fit(Xtr)
    Xn   = sc.transform(Xtr)
    Xtn  = sc.transform(Xte)

    # — reproducible init, load raw weights, train —
    torch.manual_seed(0)
    net, M, arch_json = load_raw(arch)
    net, hist = csi.train_autoencoder(net, Xn, Xtn,
                                      epochs=EPOCHS, device=DEVICE, verbose=False)

    # — full-precision reconstruction metrics —
    net.eval()
    import torch as T
    with T.no_grad():
        y   = net(T.tensor(Xtn, device=DEVICE)).cpu().numpy()
        ztr = net.encode(T.tensor(Xn,  device=DEVICE)).cpu().numpy()
        zte = net.encode(T.tensor(Xtn, device=DEVICE)).cpu().numpy()

    Hh        = csi.from_angular_delay(csi.real_imag_to_complex(sc.inverse(y)),
                                       Hte.shape[1])
    nmse      = float(csi.nmse_db(Hte, Hh))
    sgcs_full = float(csi.sgcs(W_true, csi.dominant_eigenvector(Hh)))

    # — bit sweep —
    lq        = csi.LatentQuantizer().fit(ztr)
    bit_sweep = []
    for b in AI_BITS:
        zq = lq.transform(zte, b)
        with T.no_grad():
            yq = net.decode(T.tensor(zq, device=DEVICE)).cpu().numpy()
        Hq = csi.from_angular_delay(csi.real_imag_to_complex(sc.inverse(yq)),
                                    Hte.shape[1])
        bit_sweep.append(dict(
            b    = int(b),
            bits = int(M * b),
            sgcs = float(csi.sgcs(W_true, csi.dominant_eigenvector(Hq)))
        ))

    # — save model + metrics —
    outdir = TRAINED_ROOT / label / arch
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), outdir / 'model.pt')

    metrics = dict(
        channel_label = label,
        arch          = arch,
        arch_class    = arch_json['class'],
        params        = int(arch_json.get('params', 0)),
        flops         = int(arch_json.get('flops', 0)),
        n_code        = int(M),
        epochs        = EPOCHS,
        device        = DEVICE,
        history       = [float(h) for h in hist],
        nmse_db       = nmse,
        sgcs_full     = sgcs_full,
        bit_sweep     = bit_sweep,
        standardizer  = dict(mu=float(sc.mu), sd=float(sc.sd)),
        n_delay       = N_DELAY,
        n_tx          = n_tx,
        n_sub         = int(Hte.shape[1]),
    )
    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f'{label}/{arch}: NMSE={nmse:.2f}dB  sgcs_full={sgcs_full:.4f}  '
          f'bits@b{AI_BITS[-1]}={bit_sweep[-1]}')
    return metrics

print('config + helpers ready')''')

# ---------------------------------------------------------------------------
# 4. ONE code cell per (label, arch) pair  — 6 cells total
# ---------------------------------------------------------------------------
LABELS = ['cdlc_3p5ghz', 'synthetic_beam']
ARCHS  = ['csinet16', 'csinet32', 'csinet64',
          'transnet16', 'transnet32', 'transnet64']

for label in LABELS:
    for arch in ARCHS:
        code(f"train_one({label!r}, {arch!r})")

# ---------------------------------------------------------------------------
# 5. Final markdown + verification cell
# ---------------------------------------------------------------------------
md(r"""## Verification

Assert every `metrics.json` exists for all 6 (label × arch) pairs.""")

code(r"""# Verify all 6 trained artefacts exist
LABELS = ['cdlc_3p5ghz', 'synthetic_beam']
ARCHS  = ['csinet16', 'csinet32', 'csinet64',
          'transnet16', 'transnet32', 'transnet64']

missing = []
for label in LABELS:
    for arch in ARCHS:
        p = csi.TRAINED_ROOT / label / arch / 'metrics.json'
        if not p.exists():
            missing.append(str(p))

assert not missing, f"Missing metrics files:\n" + "\n".join(missing)
print('TRAIN OK')""")

# ---------------------------------------------------------------------------
# Finalise and write
# ---------------------------------------------------------------------------
nb['cells'] = cells
nb.metadata['kernelspec'] = {
    'name':         'python3',
    'display_name': 'Python 3',
    'language':     'python',
}

out = 'notebooks/train_and_test.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)
print('wrote', out, 'with', len(cells), 'cells')
