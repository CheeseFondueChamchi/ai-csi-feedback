# AI/ML CSI Feedback Compression (3GPP TR 38.843)

Do AI autoencoders beat the 5G NR codebooks at compressing CSI feedback?
This repo compares **CsiNet / TransNet** against **PMI codebooks** (Type I/II and a
true **Rel-16 eType II**) on standards-compliant channels, scoring **SGCS vs feedback bits**.

## Pipeline

```
        ChannelConfig  (carrier · SCS · CDL profile · SNR · UE speed)
              |
              v
   +---------------------------------------------------------+
   |  STAGE 1 : gen_channel_data                             |
   |    Sionna CDL-A/C/E  ->  verify vs TR 38.901            |
   |                      ->  Type I/II + eType II 2D PMI    |
   +---------------------------------------------------------+
              |  writes
              v
          [ data/ ] -------------------------------+
              |                                     |
              v                                     |
   +-----------------------------+                  |
   |  STAGE 2 : model_zoo        |                  |
   |    CsiNet + TransNet (raw)  | --> [ models/raw/ ]
   +-----------------------------+                  |
              |                                     |
              v                                     |
   +---------------------------------------------+  |
   |  STAGE 3 : train_and_test                   |  |
   |    data/ + raw  ->  train + eval            |  |
   |                 ->  [ models/trained/ ]     |  |
   +---------------------------------------------+  |
              |                                     |
              v                                     v
   +---------------------------------------------------------+
   |  STAGE 4 : comparison                                   |
   |    data/ + trained  ->  SGCS-vs-bits figure            |
   |    PMI   vs   AI (CsiNet/TransNet)   vs   eType II 2D   |
   +---------------------------------------------------------+
```

Each stage only reads/writes files in `data/` and `models/`, so they run independently.

## Two things done carefully

**Channel generation** — real TR 38.901 CDL channels, then *checked against the standard*:
- CDL-A/C/E (+ a fast synthetic channel), driven by one `ChannelConfig`
- `csi.verify` confirms per-cluster delays/powers/angles match TR 38.901 §7.7.1 exactly
- noisy CSI estimation (SNR/pathloss), Doppler mobility, dual-pol; ~3× faster via multiprocessing

**eType II PMI** — the *true* Rel-16 codebook, not a wideband stand-in:  `W = W1 · C · Wfᴴ`
- **W1** = L spatial beams (L ≤ 6), **Wf** = M frequency beams, **C** = per-polarization coefficients
- spatial **and** frequency compression, dual-pol, K0 coefficient truncation
- scored per-subband (`sgcs_subband`) — the strict baseline the AI must beat

## Run

```bash
# conda env `sionna`; build each notebook then execute it (nbclient or Jupyter), in order:
python notebooks/build_gen_channel_data.py   #  → run gen_channel_data.ipynb
python notebooks/build_model_zoo.py          #  → run model_zoo.ipynb
python notebooks/build_train_and_test.py     #  → run train_and_test.ipynb   (uses GPU/MPS if present)
python notebooks/build_comparison.py         #  → run comparison.ipynb       (the final figure)
```

`data/` and `models/` are regenerable and git-ignored.

## Code (`src/csi/`)

| file | role |
|---|---|
| `config.py` | `ChannelConfig` + dataset save/load |
| `sionna_data.py` | TR 38.901 CDL generation (+ parallel) |
| `verify.py` | TR 38.901 reference tables + checks |
| `baselines.py` | PMI: Type I/II + **eType II 2D** |
| `models.py` | CsiNet, TransNet, complexity (FLOPs/params) |
| `metrics.py`, `transform.py`, `train.py`, `quantize.py`, `noise.py` | metrics, angular-delay DFT, training, latent quantizer, AWGN |

Background theory notes: `obsidian_vault/`.
