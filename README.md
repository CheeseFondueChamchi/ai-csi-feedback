# AI/ML-based CSI Compression (3GPP TR 38.843 study)

A runnable, **modular** study of **AI/ML CSI feedback compression** for the NR
air interface, following the 3GPP Release-18 study (**TR 38.843**). It pairs a
hands-on **4-stage notebook pipeline** with an Obsidian study vault (theory,
derivations, and a checklist of claims to verify against the primary 3GPP text).

## Layout
```
csi_report/
├── notebooks/
│   ├── gen_channel_data.ipynb           # Stage 1: reads ChannelConfig → generates CSI datasets + PMI reports
│   ├── model_zoo.ipynb                  # Stage 2: defines AI architectures (csinet16/32/64), saves raw models
│   ├── train_and_test.ipynb             # Stage 3: loads dataset + raw model → trains/evaluates → saves trained model + metrics
│   ├── comparison.ipynb                 # Stage 4: loads artifacts → renders SGCS-vs-bits comparison (synthetic vs CDL-C)
│   ├── build_gen_channel_data.py        # emits gen_channel_data.ipynb
│   ├── build_model_zoo.py               # emits model_zoo.ipynb
│   ├── build_train_and_test.py          # emits train_and_test.ipynb
│   └── build_comparison.py              # emits comparison.ipynb
├── src/csi/                             # the modular toolkit — each file = one job
│   ├── config.py      # ChannelConfig dataclass + save_dataset/load_dataset/dataset_dir IO contract
│   ├── data.py        # channel SOURCE   -> generate_csi_dataset()        [synthetic, fast]
│   ├── sionna_data.py # channel SOURCE   -> generate_sionna_csi()         [real TR 38.901 CDL via Sionna]
│   ├── transform.py   # angular-delay 2D-DFT  + real/complex helpers
│   ├── models.py      # two-sided codec  -> CsiNet (encode@UE / decode@gNB) [swap architecture]
│   ├── metrics.py     # NMSE, SGCS, GCS, cosine-rho, dominant_eigenvector
│   ├── train.py       # Standardizer + train_autoencoder() (model-agnostic)
│   ├── quantize.py    # LatentQuantizer — real bit cost for the AI latent (fair comparison)
│   ├── baselines.py   # CURRENT system: PMI codebooks (Type I / Type II)      [the baseline]
│   └── __init__.py    # re-exports the public API + a module map
├── data/                                # artifact tree: data/<channel_label>/{train.npz, test.npz, reports.npz, config.json, meta.json}
├── models/
│   ├── raw/           # untrained models: <arch>/{model.pt, arch.json}
│   └── trained/       # trained models: <channel_label>/<arch>/{model.pt, metrics.json}
├── obsidian_vault/                      # open this folder in Obsidian
│   ├── 00 - MOC.md                      # start here (map of content)
│   ├── 01 - Concepts/  02 - Math/  03 - To Verify/
```

## Design philosophy: easy to read, easy to swap
Each module has **one responsibility** and a small, stable public interface, so
any stage can be replaced without touching the others:

| want to change… | edit only |
|---|---|
| the on-disk artifact layout + ChannelConfig serialisation | `csi/config.py` |
| the channel data (e.g. plug in Sionna TR 38.901) | `csi/data.py` |
| the sparsifying transform | `csi/transform.py` |
| the neural codec (CRNet, CLNet, Transformer…) | `csi/models.py` — keep `encode/decode/forward` |
| a metric | `csi/metrics.py` — `(truth, pred) -> float` |
| the training recipe (loss, optimiser, schedule) | `csi/train.py` |
| the current PMI baseline (Type I/II codebook) | `csi/baselines.py` |

```python
import csi
H   = csi.generate_csi_dataset(6000, n_environments=12)     # (N, n_sub, n_tx)
Xad = csi.complex_to_real_imag(csi.to_angular_delay(H, 32))
std = csi.Standardizer().fit(Xad)
net = csi.CsiNet(32, 32, n_code=128)
net, hist = csi.train_autoencoder(net, std.transform(Xad), std.transform(Xad), epochs=80)
```

## The 4-stage pipeline
The analysis is now **decoupled into four execution stages** that communicate via
on-disk artifacts. Each notebook reads from and writes to the `data/`, `models/raw/`,
and `models/trained/` directories:

**Stage 1: `gen_channel_data.ipynb`** — Channel data generation
- Reads a `ChannelConfig` (carrier frequency, bandwidth, SCS, antenna counts, channel model, etc.)
- Generates per-config train/test CSI datasets via Sionna TR 38.901 (CDL-C, real 3GPP channels) and synthetic (pure-NumPy, fast)
- Computes realistic PMI (Type I/II) CSI-report data as a baseline
- Saves to `data/<channel_label>/{train.npz, test.npz, reports.npz, config.json, meta.json}`

**Stage 2: `model_zoo.ipynb`** — Model definition
- Defines AI two-sided architectures (CsiNet16/32/64 variants)
- Saves **untrained** models to `models/raw/<arch>/{model.pt, arch.json}`

**Stage 3: `train_and_test.ipynb`** — Training and evaluation
- Loads a dataset (from Stage 1) + untrained model (from Stage 2)
- Trains the two-sided autoencoder (encoder@UE / decoder@gNB)
- Evaluates on test set: **NMSE**, **SGCS** (dominant eigenvector precoder accuracy), and quantized-latent bit sweep
- Saves trained model + metrics to `models/trained/<channel_label>/<arch>/{model.pt, metrics.json}`

**Stage 4: `comparison.ipynb`** — Results visualization
- Loads datasets + PMI reports + trained metrics from artifact directories (fully decoupled, no retraining)
- Renders clean comparison plots: **SGCS-vs-feedback-bits** showing PMI codebook vs AI codec (two panels: synthetic vs CDL-C)

Each stage is **emitted by a builder** (e.g., `python notebooks/build_gen_channel_data.py` emits
the Stage 1 notebook) and executed via nbclient.

> **Key subtlety** (`obsidian_vault/02 - Math/Sparsity vs Learnable Manifold.md`):
> per-sample sparsity is *not* enough for compression — the dataset must lie near
> a low-dimensional manifold. The `n_environments` parameter controls this.

## Run it
```bash
# uses the `sionna` conda env (torch, tensorflow, sionna, numpy, matplotlib...)

# Execute each stage in order (each builder emits the notebook, then run via nbclient):
python notebooks/build_gen_channel_data.py && nbclient notebooks/gen_channel_data.ipynb       # Stage 1: generate datasets
python notebooks/build_model_zoo.py        && nbclient notebooks/model_zoo.ipynb              # Stage 2: create raw models
python notebooks/build_train_and_test.py   && nbclient notebooks/train_and_test.ipynb        # Stage 3: train and evaluate
python notebooks/build_comparison.py       && nbclient notebooks/comparison.ipynb             # Stage 4: visualize results

# Or view the final comparison interactively:
jupyter lab notebooks/comparison.ipynb
```

## Study notes (Obsidian)
Open `obsidian_vault/` as a vault in [Obsidian](https://obsidian.md). Notes use
wikilinks, LaTeX math, and callouts. Start at **`00 - MOC.md`**. Anything marked
**⚠️ TO VERIFY** lives in `03 - To Verify/` — confirm those against the actual
3GPP documents before citing them.

> Note: no Obsidian MCP server was configured in this environment, so the notes
> were written directly as an Obsidian-compatible markdown vault.
