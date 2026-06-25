"""Builds notebooks/gen_channel_data.ipynb. Run from repo root."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

def md(s):   cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))


# ── Cell 1: title / overview ────────────────────────────────────────────────
md(r"""# Stage 1 — Channel Data Generation (TR 38.843 configs)

Generates **train/test CSI datasets** and **realistic PMI CSI-report data** for each
channel configuration, **verifies** each generated channel against the 3GPP TR 38.901
CDL tables, and saves everything to `data/<channel_label>/` via the `csi` IO contract.

Per config the following files are written:

| File | Contents |
|------|----------|
| `train.npz` | `H` complex64 `(n_train, n_sub, n_tx)` |
| `test.npz`  | `H` complex64 `(n_test,  n_sub, n_tx)` |
| `reports.npz` | dominant eigenvectors + Type I/II PMI codebook metrics (under **noisy CSI estimation**) |
| `config.json` | `ChannelConfig` provenance record |
| `meta.json`   | free-form provenance dict |

Channel models generated: **CDL-A** (NLOS, rich), **CDL-C** (NLOS), **CDL-E** (LOS) via
Sionna TR 38.901, plus a **synthetic** beam-like reference. Each Sionna channel is checked
against TR 38.901 §7.7.1 (per-cluster delay/power/angles) before it is trusted.
""")


# ── Cell 2: imports / setup ─────────────────────────────────────────────────
code(r"""import os
# Single-threaded BLAS so the process pool below scales cleanly across cores.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
_SRC = os.path.abspath('../src')
sys.path.insert(0, _SRC)
# Spawned workers inherit PYTHONPATH (but NOT sys.path), so put src there too
# — this is what lets fresh processes `import csi`.
os.environ['PYTHONPATH'] = _SRC + os.pathsep + os.environ.get('PYTHONPATH', '')

import numpy as np
import csi
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ── multiprocessing helper for the per-sample codebook loops ────────────────
N_JOBS = max(1, (os.cpu_count() or 4) - 1)
_SPAWN = _mp.get_context('spawn')         # robust under nbclient on macOS
_EXEC = None

def _pool(n):
    global _EXEC
    if _EXEC is None:
        _EXEC = ProcessPoolExecutor(max_workers=n, mp_context=_SPAWN)
    return _EXEC

def pmap(func, X, n_jobs=N_JOBS, **kw):
    '''Apply a csi codebook *func* over rows of X across processes -> (W_hat, bits).
    func(X_chunk, **kw) must return (array, bits); bits is identical per chunk.'''
    if n_jobs <= 1 or len(X) < 2 * n_jobs:
        return func(X, **kw)
    chunks = np.array_split(X, n_jobs)
    res = list(_pool(n_jobs).map(partial(func, **kw), chunks))
    return np.concatenate([r[0] for r in res], axis=0), res[0][1]

import atexit
atexit.register(lambda: _EXEC.shutdown(wait=False) if _EXEC is not None else None)

print('csi loaded from:', _SRC, '| multiprocessing N_JOBS =', N_JOBS)
""")


# ── Cell 3: configuration overview ──────────────────────────────────────────
md(r"""## Channel configurations

Each dataset is fully specified by a `csi.ChannelConfig`. Sionna configs map to a
TR 38.901 CDL profile via `cfg.sionna_kwargs()`; the synthetic config uses the
pure-NumPy `csi.generate_csi_dataset`.

New TR-38.843 knobs used here:
- **`snr_db` / `pathloss_db`** — the CSI is estimated under AWGN at the *effective*
  SNR `snr_db - pathloss_db` (`csi.add_awgn`); PMI is reported from the **noisy**
  estimate but scored against the **true** precoder, so estimation error is visible.
- **`ue_speed` / `num_time_steps`** — Doppler mobility (demonstrated at the end).
""")


# ── Cell 4: define CONFIGS list ──────────────────────────────────────────────
code(r"""# Number of delay taps kept in the angular-delay representation.
N_DELAY = 32

# ---------------------------------------------------------------------------
# TR 38.843 channel configurations (CDL-A/C/E + synthetic).
# Shared grid: 3.5 GHz / 30 kHz / 51 RB = 612 subcarriers / 32 TX / dual-pol.
# 25000 train + 5000 test, UE speed 0.5 m/s, CSI estimated at 20 dB SNR.
# delay_spread is per-profile (in SECONDS): 100 ns = 100e-9, 300 ns = 300e-9.
# ---------------------------------------------------------------------------
def _cdl(model, label, delay_spread, ue_speed=0.5):
    return csi.ChannelConfig(
        channel_model=model, data_source='sionna',
        carrier_frequency=3.5e9, bandwidth=20e6, scs=30e3, rb=51, nfu=612,
        gnb_tx=32, ue_rx=1, max_rank=1, dual_pol=True,
        delay_spread=delay_spread, ue_speed=ue_speed,
        snr_db=20.0, pathloss_db=0.0,
        n_orient=16, n_train=25000, n_test=5000, seed=0,
        channel_label=label,
    )

CONFIGS = [
    _cdl('CDL-A', 'cdla_3p5ghz', 100e-9),   # NLOS, rich multipath
    _cdl('CDL-C', 'cdlc_3p5ghz', 300e-9),   # NLOS, moderate
    _cdl('CDL-E', 'cdle_3p5ghz', 100e-9),   # LOS, sparse
    # Synthetic beam-like — pure-NumPy DFT-sparse model
    csi.ChannelConfig(
        channel_model='synthetic', data_source='synthetic',
        carrier_frequency=3.5e9, bandwidth=20e6, scs=30e3, rb=51, nfu=612,
        gnb_tx=32, ue_rx=1, max_rank=1, snr_db=20.0,
        n_train=25000, n_test=5000, seed=1, channel_label='synthetic_beam',
    ),
]

for cfg in CONFIGS:
    print(f'  {cfg.to_dirname():16s}  model={cfg.channel_model:9s} '
          f'n_sub={cfg.n_sub()} n_tx={cfg.gnb_tx} '
          f'DS={cfg.delay_spread*1e9:.0f}ns SNR={cfg.effective_snr_db():.0f}dB '
          f'n_train={cfg.n_train} n_test={cfg.n_test}')
""")


# ── Cell 5: build_reports function ──────────────────────────────────────────
code(r"""def build_reports(H_test, n_tx, n_delay, snr_db, seed=0, dual_pol=False):
    # Per-test-sample PMI metrics under NOISY CSI estimation.
    #
    # Realistic chain: the UE estimates a noisy channel Hhat = H + AWGN at
    # `snr_db`, derives its precoder from Hhat, and reports a PMI. We score every
    # reported precoder against the TRUE eigenvector of the clean H, so the SGCS
    # reflects estimation error AND codebook quantization (not quantization alone).
    rng = np.random.default_rng(1234 + seed)
    reports = {}

    # ── ground-truth precoder from the CLEAN channel ─────────────────────
    W_true = csi.dominant_eigenvector(H_test).astype(np.complex64)
    reports['W_true'] = W_true

    # ── noisy estimate the UE actually sees ──────────────────────────────
    H_est = csi.add_awgn(H_test, snr_db, rng)
    W_est = csi.dominant_eigenvector(H_est).astype(np.complex64)
    # estimation-only reference (no codebook): how good is the raw noisy precoder
    reports['sgcs_estimation'] = np.float64(csi.sgcs(W_true, W_est))
    reports['snr_db'] = np.float64(snr_db)

    # ── delay-truncation reference (near-lossless, on clean H) ───────────
    H_ad  = csi.to_angular_delay(H_test, n_delay)
    H_tr  = csi.from_angular_delay(H_ad, H_test.shape[1])
    reports['sgcs_trunc'] = np.float64(csi.sgcs(W_true, csi.dominant_eigenvector(H_tr)))
    reports['n_delay']    = np.int64(n_delay)

    # ── PMI reported from the NOISY precoder W_est ───────────────────────
    # Legacy codebooks: Type I (single beam) + Type II (L-beam linear combo).
    type1_W, type1_bits = csi.type1_pmi(W_est, n_tx)
    reports['type1_W'] = type1_W.astype(np.complex64)
    reports['type1_bits'] = int(type1_bits)
    for L in [2, 3, 4, 6]:
        W_hat, bits = pmap(csi.type2_pmi, W_est, n_tx=n_tx, L=L)
        reports[f'type2_L{L}_W'] = W_hat.astype(np.complex64)
        reports[f'type2_L{L}_bits'] = int(bits)
    # ── aligned wideband summary arrays (Type I/II, scored vs CLEAN W_true) ──
    scheme_names = ['Type I'] + [f'Type II L={L}' for L in [2, 3, 4, 6]]
    scheme_W     = [reports['type1_W']] + [reports[f'type2_L{L}_W'] for L in [2, 3, 4, 6]]
    scheme_bits  = [reports['type1_bits']] + [reports[f'type2_L{L}_bits'] for L in [2, 3, 4, 6]]
    reports['pmi_schemes'] = np.array(scheme_names)
    reports['pmi_family']  = np.array(['Type I'] + ['Type II'] * 4)
    reports['pmi_bits']    = np.array(scheme_bits, dtype=int)
    reports['pmi_sgcs']    = np.array([float(csi.sgcs(W_true, W)) for W in scheme_W],
                                      dtype=np.float64)

    # ── True Rel-16 eType II: spatial-frequency 2D, evaluated PER-SUBBAND ─────
    # Separate track with its OWN metric basis (per-subband precoders, mean SGCS
    # across subbands). Uses the noisy estimate for the report, scored against the
    # clean per-subband precoders. Sweeps spatial L and frequency M.
    N_SB = 13
    W_sb_true = csi.subband_precoders(H_test, N_SB)   # clean ground truth
    W_sb_est  = csi.subband_precoders(H_est,  N_SB)   # noisy estimate the UE sees
    # Rel-16 K0 truncation: only ceil(beta*L*M) strongest coeffs reported (+ bitmap).
    E2D_BETA = 0.5
    reports['n_subband'] = np.int64(N_SB)
    reports['etype2_2d_beta'] = np.float64(E2D_BETA)
    reports['sgcs_subband_estimation'] = np.float64(csi.sgcs_subband(W_sb_true, W_sb_est))
    e2d_names, e2d_bits, e2d_sgcs = [], [], []
    for (L, M) in [(4, 1), (4, 2), (4, 4), (6, 2), (6, 4), (6, 7)]:
        W_hat_sb, bits = pmap(csi.etype2_pmi_2d, W_sb_est, n_tx=n_tx, L=L, M=M,
                              beta=E2D_BETA, dual_pol=dual_pol)
        e2d_names.append(f'eType2D L={L} M={M}')
        e2d_bits.append(int(bits))
        e2d_sgcs.append(float(csi.sgcs_subband(W_sb_true, W_hat_sb)))
    reports['etype2_2d_schemes'] = np.array(e2d_names)
    reports['etype2_2d_bits']    = np.array(e2d_bits, dtype=int)
    reports['etype2_2d_sgcs']    = np.array(e2d_sgcs, dtype=np.float64)
    return reports


print('build_reports() defined (noisy CSI estimation)')
""")


# ── Cell 6: main generation loop ─────────────────────────────────────────────
code(r"""for cfg in CONFIGS:
    label = cfg.to_dirname()
    n_total = cfg.n_train + cfg.n_test
    print(f'\n=== Generating: {label} ({cfg.channel_model}) ===')

    # ── generate raw channel matrix H ────────────────────────────────────
    if cfg.data_source == 'sionna':
        H = csi.generate_sionna_csi_parallel(n_jobs=N_JOBS, **cfg.sionna_kwargs())  # (n_total, n_sub, n_tx)
    else:
        H = csi.generate_csi_dataset(n_samples=n_total, n_tx=cfg.gnb_tx,
                                     n_sub=cfg.n_sub(), rng=np.random.default_rng(cfg.seed))
    # temporal configs return (N, T, n_sub, n_tx) -> flatten time into samples
    if H.ndim == 4:
        H = H.reshape(-1, H.shape[-2], H.shape[-1])
    print(f'  H shape: {H.shape}  dtype: {H.dtype}')

    # ── reproducible shuffle then split ──────────────────────────────────
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(H))
    H_train = H[idx[:cfg.n_train]]
    H_test  = H[idx[cfg.n_train: cfg.n_train + cfg.n_test]]

    # ── PMI reports under noisy estimation at the config's effective SNR ──
    reports = build_reports(H_test, cfg.gnb_tx, N_DELAY,
                            snr_db=cfg.effective_snr_db(), seed=cfg.seed,
                            dual_pol=bool(cfg.dual_pol))

    meta = dict(generator=cfg.data_source, channel_model=cfg.channel_model,
                n_sub=cfg.n_sub(), n_tx=cfg.gnb_tx,
                n_train=cfg.n_train, n_test=cfg.n_test,
                effective_snr_db=cfg.effective_snr_db(),
                shape_train=list(H_train.shape))

    d = csi.dataset_dir(cfg)
    csi.save_dataset(d, H_train, H_test, reports, meta, cfg)
    print(f'  sgcs_trunc={float(reports["sgcs_trunc"]):.4f} '
          f'sgcs_estimation={float(reports["sgcs_estimation"]):.4f} '
          f'TypeII(L=4)={reports["pmi_sgcs"][3]:.4f}')
    print(f'  eType2D[{reports["n_subband"]} subbands]  '
          f'best={reports["etype2_2d_sgcs"].max():.4f} '
          f'(schemes: {list(reports["etype2_2d_schemes"])})')
    print(f'  saved -> {d}')

print('\nAll configs generated.')
""")


# ── Cell 7: TR 38.901 channel verification ───────────────────────────────────
md(r"""## Channel verification against TR 38.901 §7.7.1

For every Sionna (CDL) config we check the generated channel **two ways**:

1. **Config level** — the generator's per-cluster parameters (normalized delay, power,
   AOD/AOA/ZOD/ZOA, cluster spreads, XPR) must equal the TR 38.901 Table 7.7.1-x values
   **exactly** (`csi.verify_cdl_table`).
2. **Data level** — statistics measured from the generated `H`: unit power, the fraction
   of delay energy inside the table's cluster window, and the PDP shape correlation against
   the binned table (`csi.verify_generated`).
""")

code(r"""all_ok = True
for cfg in CONFIGS:
    if cfg.data_source != 'sionna':
        continue
    ds = csi.load_dataset(csi.dataset_dir(cfg))
    H_te = ds['H_test']
    table_rep = csi.verify_cdl_table(cfg.channel_model)
    gen_rep   = csi.verify_generated(H_te, cfg.channel_model,
                                     delay_spread=cfg.delay_spread, scs=cfg.scs)
    print(csi.format_report(table_rep, gen_rep)); print()
    all_ok = all_ok and table_rep['ok'] and gen_rep['ok']

print('TR 38.901 VERIFICATION:', 'ALL PASS ✅' if all_ok else 'SOME FAILED ❌')
assert all_ok, 'channel verification failed — generated data does not match TR 38.901'
""")


# ── Cell 8: dataset reload sanity check ──────────────────────────────────────
code(r"""# Reload each dataset and sanity-check shapes/dtypes/keys.
for cfg in CONFIGS:
    ds = csi.load_dataset(csi.dataset_dir(cfg))
    H_tr, H_te, rpt = ds['H_train'], ds['H_test'], ds['reports']
    assert H_tr.dtype == np.complex64 and H_te.dtype == np.complex64
    assert H_tr.shape == (cfg.n_train, cfg.n_sub(), cfg.gnb_tx)
    assert H_te.shape == (cfg.n_test,  cfg.n_sub(), cfg.gnb_tx)
    assert 'W_true' in rpt and 'pmi_sgcs' in rpt
    print(f'  {cfg.to_dirname():16s} train={H_tr.shape} test={H_te.shape} '
          f'sgcs_trunc={float(rpt["sgcs_trunc"]):.4f}  OK')
print('\nAll datasets verified.')
""")


# ── Cell 9: directory layout note ────────────────────────────────────────────
md(r"""## Per-config directory layout

```
data/<channel_label>/
  train.npz      # H complex64 (n_train, n_sub, n_tx)
  test.npz       # H complex64 (n_test,  n_sub, n_tx)
  reports.npz    # W_true, sgcs_estimation, snr_db, type1/type2 PMI (noisy),
                 # pmi_schemes, pmi_bits, pmi_sgcs, sgcs_trunc, n_delay
  config.json    # ChannelConfig (incl. snr_db, pathloss_db, num_time_steps)
  meta.json      # provenance (channel_model, effective_snr_db, shapes)
```

Downstream notebooks load via `csi.load_dataset(csi.dataset_dir(cfg))`.
""")


# ── finalise and write ───────────────────────────────────────────────────────
nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}

out = 'notebooks/gen_channel_data.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)

print('wrote', out, 'with', len(cells), 'cells')
