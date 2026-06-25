---
title: Quantization & Feedback Overhead
tags: [math, quantization, rate, overhead]
created: 2026-06-24
---

# Quantization & Feedback Overhead

> [!abstract] The point
> The codeword $z\in\mathbb{R}^M$ must cross the air interface as **bits**.
> Compression is only meaningful once you count those bits and compare against
> the codebook baseline at **equal payload**.

## Compression ratio (dimension-level)
$$\gamma = \frac{M}{2\,N_d\,N_t}\quad(\text{kept reals / original reals}).$$
Smaller $\gamma$ = harder problem. (Notebook uses $N_d=N_t=32$, so the input is
$2\cdot32\cdot32=2048$ reals; $M=64\Rightarrow\gamma=0.03125$.)

## Feedback payload (bit-level)
With $b$-bit scalar quantization of each codeword entry:
$$\text{bits} = M\cdot b.\qquad(\texttt{csi.feedback\_bits})$$
This is the honest axis for rate–distortion comparison vs Type II.

> [!important] Quantize the latent for a *fair* comparison
> Reporting full-precision SGCS at a nominal $M\cdot 8$ bits is **not** an
> achievable operating point. `csi.LatentQuantizer` (per-dim uniform scalar,
> fit on TRAIN) gives the *real* point: quantize the latent, decode, then score.
> Empirically SGCS **saturates around $b\approx4$ bits/dim** — so the true AI
> report length is $M\cdot 4$, not $M\cdot 8$. Below that it degrades fast
> ($b{=}1$ loses a lot). The full PMI-codebook-vs-AI rate–distortion sweep is in
> `notebooks/comparison.ipynb` (see [[Current NR CSI Feedback (PMI)]]).

## Quantizing the latent — the real engineering problem
- **Scalar quantization (SQ):** independent $b$-bit per entry. Simple; standard-friendly.
- **Vector quantization (VQ):** a learned codebook of latent vectors; better rate
  but adds a shared codebook to standardize → ties into [[Two-Sided Model and Training Collaboration|training collaboration]].
- **Training through the quantizer:** quantization is non-differentiable.
  Surrogates: **straight-through estimator (STE)**, additive **uniform noise**
  (à la soft quantization), or a **Gumbel-softmax** over codebook entries.

> [!note] Two-stage training pattern
> 1. Train the autoencoder with a continuous (or noise-perturbed) latent.
> 2. Insert the real quantizer and **fine-tune** (or train a small entropy model)
>    to recover the SGCS lost to quantization.

## Where the bits go in NR signalling
The codeword is carried in the uplink CSI report (cf. CSI part 1 / part 2
structure of the codebook reports). ⚠️ confirm exactly how an AI/ML report maps
onto UCI / PUSCH-CSI containers in the Rel-19 spec →
[[Open Questions and Claims to Verify]].

## Rate–distortion intuition
Plot **SGCS (or NMSE) vs bits** (notebook §6). The AI/ML claim is that the
learned curve sits **above/left** of the Type II curve — same accuracy for fewer
bits, or more accuracy for the same bits.

## Related
- [[Metrics - NMSE and SGCS]]
- [[CsiNet Autoencoder Architecture]]
- [[Two-Sided Model and Training Collaboration]]
