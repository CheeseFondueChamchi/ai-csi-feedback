"""Builds notebooks/model_zoo.ipynb. Run from repo root."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); cells = []
def md(s):   cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ---------------------------------------------------------------------------
# Cell 1 — title / explanation
# ---------------------------------------------------------------------------
md(r"""# Stage 2 — Model Zoo (raw, untrained two-sided models)

Each architecture in the zoo is **instantiated untrained** (random weights,
fixed seed) and immediately saved to `models/raw/<arch>/` as two files:

| file | contents |
|------|----------|
| `model.pt` | `state_dict` — raw (untrained) weights via `torch.save` |
| `arch.json` | constructor recipe `{class, kwargs, variant}` |

Downstream notebooks (**train_and_test**, **comparison**) load
`arch.json`, reconstruct the model with `build_from_arch`, then call
`load_state_dict` to restore weights — they never hard-code constructor args.
This single source of truth means adding a new variant here automatically
propagates to training and comparison.
""")

# ---------------------------------------------------------------------------
# Cell 2 — setup imports
# ---------------------------------------------------------------------------
code(r"""import sys, os, json
sys.path.insert(0, os.path.abspath('../src'))
import torch
import csi
from pathlib import Path

print('torch', torch.__version__)
print('RAW_ROOT', csi.RAW_ROOT)""")

# ---------------------------------------------------------------------------
# Cell 3 — zoo definition
# ---------------------------------------------------------------------------
code(r"""# ── Model Zoo ────────────────────────────────────────────────────────────
# Each entry: (variant_name, class_name, constructor kwargs)
# n_code = M = latent / feedback dimension. Two model families at matched M:
#   CsiNet   — the original CNN autoencoder (legacy baseline)
#   TransNet — transformer (full-attention) codec (the replacement model)
ZOO = [
    ('csinet16',    'CsiNet',   dict(n_delay=32, n_tx=32, n_code=16, final_activation='linear')),
    ('csinet32',    'CsiNet',   dict(n_delay=32, n_tx=32, n_code=32, final_activation='linear')),
    ('csinet64',    'CsiNet',   dict(n_delay=32, n_tx=32, n_code=64, final_activation='linear')),
    ('transnet16',  'TransNet', dict(n_delay=32, n_tx=32, n_code=16, final_activation='linear')),
    ('transnet32',  'TransNet', dict(n_delay=32, n_tx=32, n_code=32, final_activation='linear')),
    ('transnet64',  'TransNet', dict(n_delay=32, n_tx=32, n_code=64, final_activation='linear')),
]

print('Zoo entries:', [name for name, _, _ in ZOO])""")

# ---------------------------------------------------------------------------
# Cell 4 — build_from_arch helper
# ---------------------------------------------------------------------------
code(r"""# ── Shared loader helper ─────────────────────────────────────────────────
# train_and_test and comparison notebooks use the IDENTICAL rule:
#   model = build_from_arch(json.loads((d / 'arch.json').read_text()))
# Define it once here; copy-paste (or import) into downstream notebooks.

def build_from_arch(arch: dict):
    # arch = {'class': 'CsiNet', 'kwargs': {...}, 'variant': '...'}
    cls = getattr(csi, arch['class'])
    return cls(**arch['kwargs'])

print('build_from_arch defined — train/comparison use the identical rule')""")

# ---------------------------------------------------------------------------
# Cell 5 — build loop: instantiate, save model.pt + arch.json
# ---------------------------------------------------------------------------
code(r"""# ── Build & save raw (untrained) models + record complexity ──────────────
for name, cls, kw in ZOO:
    torch.manual_seed(0)                          # reproducible random init
    net = getattr(csi, cls)(**kw)

    d = Path(csi.RAW_ROOT) / name
    d.mkdir(parents=True, exist_ok=True)

    # weights
    torch.save(net.state_dict(), d / 'model.pt')

    # complexity (params exact, FLOPs estimated) — saved into arch.json
    cx = csi.model_complexity(net, input_shape=(1, 2, kw['n_delay'], kw['n_tx']))

    # architecture recipe (+ complexity for the comparison report)
    arch = {'class': cls, 'kwargs': kw, 'variant': name,
            'params': cx['params'], 'flops': cx['flops']}
    (d / 'arch.json').write_text(json.dumps(arch, indent=2))

    print(f'saved raw  {name:<12} {cls:<9} params={cx["params"]:>10,}  '
          f'flops={cx["flops"]/1e6:7.2f} MFLOP  ->  {d}')""")

# ---------------------------------------------------------------------------
# Cell 6 — verification: reload and round-trip check
# ---------------------------------------------------------------------------
code(r"""# ── Verification: reload arch.json, rebuild, load weights ────────────────
for name, _, _ in ZOO:
    d = Path(csi.RAW_ROOT) / name

    # reconstruct from recipe (same rule downstream notebooks use)
    arch = json.loads((d / 'arch.json').read_text())
    net = build_from_arch(arch)

    # load saved weights
    m = net.load_state_dict(torch.load(d / 'model.pt', weights_only=True))
    assert not m.missing_keys,    f'{name}: missing keys {m.missing_keys}'
    assert not m.unexpected_keys, f'{name}: unexpected keys {m.unexpected_keys}'

print('ZOO OK — all variants round-trip cleanly')""")

# ---------------------------------------------------------------------------
# Cell 7 — schema note
# ---------------------------------------------------------------------------
md(r"""## arch.json contract

```json
{
  "class":   "CsiNet",
  "kwargs":  { "n_delay": 32, "n_tx": 32, "n_code": 64, "final_activation": "linear" },
  "variant": "csinet64"
}
```

**Fields:**

| field | meaning |
|-------|---------|
| `class` | Attribute name on the `csi` module — passed to `getattr(csi, arch['class'])` |
| `kwargs` | Constructor keyword arguments, unpacked verbatim |
| `variant` | Human-readable name; also used as the subdirectory under `models/raw/` |

Both **train_and_test** and **comparison** load this schema via
`build_from_arch(arch)` — no hard-coded constructor calls outside this zoo.
""")

# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------
nb['cells'] = cells
nb.metadata['kernelspec'] = {
    'name': 'python3',
    'display_name': 'Python 3',
    'language': 'python',
}

out = 'notebooks/model_zoo.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)
print('wrote', out, 'with', len(cells), 'cells')
