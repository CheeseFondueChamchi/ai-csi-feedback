---
title: Current NR CSI Feedback — CSI-RS, PMI/RI/CQI, Type I/II Codebooks
tags: [3gpp, csi, pmi, codebook, baseline, nr]
created: 2026-06-24
---

# Current NR CSI Feedback (the PMI codebook system)

> [!abstract] This is the **baseline** AI/ML must beat
> Before any AI/ML, NR already has a complete CSI feedback loop. The UE does
> **not** send the channel — it sends a **PMI** (Precoding Matrix Indicator)
> chosen from a standardized **codebook**, plus **RI** and **CQI**. The AI/ML
> two-sided model proposes to *replace the PMI codebook* with a learned codec.
> In TR 38.843 the explicit baseline is the **Rel-16 (enhanced) Type II** codebook.

## The end-to-end loop (TS 38.214 / 38.211 / 38.212)
```
gNB --- CSI-RS (pilots) ---> UE
                              │  1. estimate per-subband channel H_k
                              │  2. R_k = H_k^H H_k ; eigen-decompose
                              │  3. choose RANK (RI) and best codebook entry (PMI)
                              │  4. compute CQI for the chosen precoder+rank
UE  --- CSI report (UCI) ---> gNB     (on PUCCH, or PUSCH if larger)
                              gNB looks up PMI -> precoder W, schedules with CQI
```
The reported quantities:
- **RI** (Rank Indicator) — number of layers (eigenvectors) to transmit.
- **PMI** (Precoding Matrix Indicator) — index/indices into the codebook giving
  the precoder $W$ (the quantized [[Eigenvector Precoder and Covariance|eigenvector(s)]]).
- **CQI** (Channel Quality Indicator) — MCS the UE can sustain with that $W$.

> [!note] "Channel estimation system" vs "CSI feedback"
> The UE *estimates* the channel from CSI-RS; what it *transmits back* is the
> **compressed precoder decision (PMI)**, not the channel. AI/ML CSI compression
> targets this **feedback/quantization** step, not the estimation step.

## The codebooks (the "compression" used today)
> [!example] Type I (coarse, cheap)
> A single (oversampled) **DFT beam** per layer: $W \approx b_{\text{PMI}}$.
> Payload ≈ $\lceil\log_2(N_t\,O)\rceil$ bits (e.g. **7 bits** for $N_t{=}32$,
> oversampling $O{=}4$). Coarse — limited by beam-grid resolution.
> → `csi.type1_pmi`.

> [!example] Type II (fine, expensive) — the 3GPP AI/ML baseline
> A **linear combination of $L$ strongest beams** with per-subband quantized
> **amplitude + phase** coefficients:
> $$ \hat W = \sum_{l=1}^{L} c_l\, b_l, \qquad c_l = (\text{amp})_l e^{j(\text{phase})_l}. $$
> Payload = $L\lceil\log_2(N_tO)\rceil$ (beam indices) $+\,L\,(b_{\text{amp}}+b_{\text{phase}})$
> (coeffs), per subband — typically **~300 bits** for 32-port / 13-subband.
> Rel-17 **eType II** adds port selection / overhead reduction.
> → `csi.type2_pmi`.

## How it stacks up (this project's synthetic data)
| scheme | payload | SGCS |
|---|---|---|
| Type I (1 beam) | ~7 b | ≈ 0.76 |
| Type II (L=4) | ~56 b | ≈ 0.83 |
| Type II (L=6) | ~84 b | ≈ 0.88 |
| learned CsiNet | see notebook §6 | ≈ 0.87 (M=128) |
| truncation ref | 2048 reals | ≈ 0.998 |

> [!warning] Why the codebook is hard to beat here (and the honest caveat)
> Our synthetic channel is a sum of **DFT-like steering vectors**, so the
> **DFT-beam Type II codebook is near-ideal for it** — it compresses to ~84 bits
> at SGCS 0.88. A naively (scalar-)quantized learned codec does *not* obviously
> win. This mirrors the **Rel-18 conclusion that two-sided compression gains
> were "not sufficient"** ([[3GPP AI-ML CSI Overview]]). The learned codec's edge
> grows on **richer, non-ULA, dual-pol, scattering** channels where the fixed
> DFT basis is sub-optimal — and hinges on **efficient latent quantization**
> ([[Quantization and Feedback Overhead]]), not the 8-bit scalar stand-in.

## Related
- [[Eigenvector Precoder and Covariance]] · [[Two-Sided Model and Training Collaboration]]
- [[Metrics - NMSE and SGCS]] · [[Quantization and Feedback Overhead]]
- [[CsiNet Autoencoder Architecture]] (the AI/ML replacement)
