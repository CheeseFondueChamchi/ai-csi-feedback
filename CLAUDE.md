# CLAUDE.md — ai-csi-feedback

AI/ML CSI feedback compression study (3GPP TR 38.843). Decoupled 4-stage notebook
pipeline; reusable `csi` package in `src/`.

## Environment & running
- Conda env **`sionna`** (Python 3.10; torch 2.11, tensorflow 2.20, sionna 1.2.1, CPU/MPS).
- Run Python as: `conda run -n sionna python ...`. Import the package with `sys.path.insert(0, 'src'); import csi`.
- No `jupyter` CLI. Notebooks are **emitted** by `notebooks/build_<x>.py` (nbformat) and
  **executed** via `nbclient.NotebookClient` (cwd = `notebooks/`). To change a notebook,
  edit its `build_*.py` and re-run it; never hand-edit the `.ipynb`.
- Training auto-selects device CUDA → **MPS** (Apple Silicon, ~4.3× for CsiNet) → CPU,
  with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Pipeline (each stage reads/writes only on-disk artifacts)
1. `gen_channel_data` — `ChannelConfig` → Sionna CDL channels + verify + reports → `data/<label>/`
2. `model_zoo` — raw CsiNet/TransNet (+ params/FLOPs) → `models/raw/<arch>/`
3. `train_and_test` — train + eval (per label×arch) → `models/trained/<label>/<arch>/`
4. `comparison` — loads artifacts only → SGCS-vs-bits figures

## Package map (`src/csi/`)
`config` (ChannelConfig + dataset IO) · `sionna_data` (CDL gen `generate_sionna_csi[_parallel|_mixed]`) ·
`verify` (TR 38.901 tables + checks) · `noise` (AWGN) · `baselines` (PMI: eType II 2D `etype2_pmi_2d`,
per-subband helpers) · `reports` (`build_reports` — shared by gen + refresh) · `models` (CsiNet, TransNet,
`model_complexity`) · `metrics` (`sgcs`, `sgcs_subband`, `dominant_eigenvector`) · `transform`, `train`, `quantize`.

## Conventions / gotchas (read before editing)
- **NEVER write to or delete a real `data/<label>/` directory in a test or smoke run.**
  Use a throwaway `channel_label` or an in-memory array. (A past `shutil.rmtree` on a prod
  label reusing the same name deleted real datasets.)
- `data/` and `models/` are **git-ignored** (multi-GB, regenerable). Don't commit them.
- **eType II** is the true Rel-16 2D codebook (`W1·C·Wfᴴ`, dual-pol, K0+bitmap), scored
  **per-subband** (`sgcs_subband`) — a different, stricter metric than wideband SGCS. Keep
  the two metric bases separate. Legacy wideband Type I/II are **excluded** from the comparison.
- eType II beam selection must use an **orthogonal DFT subset** (cols spaced by oversampling),
  not the L globally-strongest oversampled beams (those overlap → captured power >1).
- Report logic lives in **`csi.build_reports`** (the gen notebook delegates to it). Edit there,
  not in the notebook, so gen and the refresh tool stay in lockstep.
- Multiprocessing (gen + codebook loops) is **spawn**-based and sets `PYTHONPATH` so workers
  can `import csi` (spawned procs inherit PYTHONPATH, not `sys.path`).
- `delay_spread` is in **seconds** (300 ns = `300e-9`), not ns. `lsp_variation` (default off)
  adds mild per-sample channel variation (TR 38.901 §7.5).
- Authoritative 3GPP specs are in `/Users/hjl/Documents/Github/gpp_DB/specs/` (TR 38.901,
  TS 38.214, TR 38.843). TR 38.843 eType II rank-1 baseline SGCS ≈ 0.65–0.83 (median 0.705)
  at 59–80 bits — use this, not paper proxies.

## Git
- Public repo. Commit/push only when asked. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
