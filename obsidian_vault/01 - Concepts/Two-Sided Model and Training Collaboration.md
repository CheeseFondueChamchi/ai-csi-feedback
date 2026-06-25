---
title: Two-Sided Model & Training Collaboration (Type 1/2/3)
tags: [3gpp, csi, two-sided, training]
created: 2026-06-24
---

# Two-Sided Model & Training Collaboration

> [!abstract] Definition
> A **two-sided model** is a single ML pipeline whose **encoder runs at the UE**
> and whose **decoder runs at the gNB**. Only the **codeword** (the latent)
> crosses the air interface. The two halves must be *jointly consistent* — a
> decoder only understands codewords from the encoder it was trained with.

```
   UE  ───────────────[ uplink: codeword (M·b bits) ]───────────────▶  gNB
  H ──▶ [ Encoder θ_e ] ──▶ z (M-dim) ──▶ quantize ──▶  ...  ──▶ [ Decoder θ_d ] ──▶ Ĥ / ŵ
```

The fundamental challenge: the UE and the gNB are usually built by **different
vendors**, who will not share their proprietary model internals. 3GPP therefore
defined three **training collaboration types**.

## Training collaboration types (TR 38.843)

> [!note] Type 1 — Joint training at a single side
> One entity trains **both** encoder and decoder jointly, then delivers the
> other half to the counterpart (e.g., as model or as a reference encoder/decoder).
> Best performance, weakest on vendor independence.

> [!note] Type 2 — Joint training across UE-side and network-side
> Encoder and decoder are trained **jointly but in different entities**, with
> **real-time exchange of forward activations and backward gradients** during
> training. Strong performance but heavy coordination → **deprioritised in
> Rel-19** as impractical for commercial networks.

> [!note] Type 3 — Separate training at UE-side and network-side ⭐ Rel-19 focus
> Encoder (UE vendor) and decoder (NW vendor) trained **independently**, sharing
> only structure and/or a dataset. Two directions studied:
> - **Direction I — standardized reference model(s):** 3GPP fixes encoder and/or
>   decoder parameters → interoperability by construction.
> - **Direction II — dataset / parameter sharing:** one side delivers a training
>   dataset $\{(\text{target},\text{feedback})\}$ (or encoder params) so the
>   other trains offline to match.
> Most vendor-realistic; the open standardization questions (interface, data
> format, versioning, SGCS-based monitoring/fallback) live here.

*(Type 1/2/3 + Type 3 directions verified via literature; confirm wording in the primary text — [[Open Questions and Claims to Verify]].)*

## Why this matters for design
- **Generalization**: the decoder must work across many UE encoders → motivates
  *one decoder, many encoders* schemes and **nominal/reference models**.
- **Dataset exchange**: Type 3 turns the problem into "agree on a dataset
  format + a target representation" — links to [[Datasets]].
- **Quantization** of the latent must be standardized or signalled →
  [[Quantization and Feedback Overhead]].

## Related
- [[3GPP AI-ML CSI Overview]]
- [[CsiNet Autoencoder Architecture]]
- [[Quantization and Feedback Overhead]]
