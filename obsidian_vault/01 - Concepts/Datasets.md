---
title: Datasets — COST 2100, TR 38.901, Sionna/QuaDRiGa
tags: [csi, dataset, channel-model, cost2100, sionna]
created: 2026-06-24
---

# Datasets for CSI Compression

> [!warning] There is **no single official downloadable 3GPP dataset**
> 3GPP does not ship a CSI dataset file. Instead, **TR 38.843** fixes the
> **evaluation assumptions** (channel model, antenna config, bandwidth, SNR…)
> and each company **generates its own** TR 38.901-compliant data. For
> reproducibility, academia mostly uses **COST 2100**.

## 1. COST 2100 — the academic standard (CsiNet)
The dataset released with CsiNet (`sydney222/Python_CsiNet`):
- **Indoor** picocell, **5.3 GHz**; **Outdoor** rural, **300 MHz**.
- gNB **ULA, $N_t=32$**; **$N_c=1024$** subcarriers → angular-delay → truncated to **$N_d=32$** delay taps → samples are $32\times32$ complex.
- Split: **train 100 000 / val 30 000 / test 20 000** per scenario *(verified)*.
- Compression ratio table (CsiNet, $2N_cN_t=2048$): $\gamma=1/4\to M{=}512$, $1/16\to128$, $1/32\to64$, $1/64\to32$.
- Stored as real/imag → `(N, 2, 32, 32)` — exactly the tensor shape our notebook uses.

## 2. 3GPP evaluation methodology — TR 38.901 generated
3GPP companies generate channels from **TR 38.901** (*"Channel model for 0.5–100
GHz"*):
- **Cluster Delay Line (CDL)** A–E and **TDL** A–E — link-level tapped models.
- **System-level** stochastic models: **UMa, UMi, RMa, InH (Indoor Hotspot)**.
- Agreed eval knobs (verified via literature; still cross-check TR 38.843 §6.2):
  - **gNB 32 ports** = **8H × 4V dual-polarisation UPA**; **UE 4 Rx**.
  - carrier **~2.0 GHz FDD DL** (UMa dense-urban), **15 kHz SCS**, ~20 MHz BW.
  - **13 subbands**; primarily **rank-1** (higher ranks also studied).
  - **Baseline = Rel-16 eType II** codebook, **~300 bits** for this config.
  - scenarios: UMa NLOS (CDL-A/-C), UMi NLOS, InH.

## 3. Generators you can actually run
| Tool | What | In this repo |
|---|---|---|
| **NVIDIA Sionna** | TR 38.901 CDL/TDL/UMa/UMi/InH, differentiable | **runnable**: `csi.generate_sionna_csi` (`src/csi/sionna_data.py`) + `notebooks/gen_channel_data.ipynb` |
| **QuaDRiGa** (Fraunhofer HHI) | 3GPP-calibrated geometry-based SCM | MATLAB; common in 3GPP contributions |
| **our `csi.generate_csi_dataset`** | fast synthetic clustered-ray, angular-delay sparse | main notebook §1 — pedagogical, not standards-grade |

> [!tip] Swap-in path (verified working)
> `csi.generate_sionna_csi(...)` returns $H$ of shape `(N, n_sub, n_tx)` — the
> *same contract* as the synthetic generator — so **the entire pipeline re-runs
> unchanged** on TR 38.901 CDL channels. Observed on **CDL-C** (NLOS, 32-ULA):
> ~95 % energy in 32 delay taps, truncation SGCS ≈ 0.995, but **Type II PMI only
> ≈ 0.69 at 84 bits** (vs ≈ 0.88 on beam-like synthetic data) — the rich NLOS
> channel is where the fixed DFT codebook leaves the most headroom for AI/ML.

## Dataset = the crux of [[Two-Sided Model and Training Collaboration|Type 3 training]]
Because the gNB and UE vendors differ, the **dataset exchange format** (what
channels, what target representation — raw $H$ vs [[Eigenvector Precoder and Covariance|eigenvector]]) is itself a standardization topic. Distribution shift between the
training set and the deployed cell is the main **generalization** risk.

## Related
- [[Angular-Delay Transform]]
- [[CsiNet Autoencoder Architecture]]
- [[Open Questions and Claims to Verify]]
