---
title: Per-sample Sparsity ≠ a Learnable Dataset Manifold
tags: [math, insight, compression, manifold, pitfall]
created: 2026-06-24
---

# Per-sample Sparsity ≠ a Learnable Dataset Manifold

> [!danger] The subtlety that breaks naive CSI-compression demos
> "The channel is sparse in the [[Angular-Delay Transform|angular-delay domain]],
> therefore it is compressible" is only **half true**. A *single* channel is
> sparse, but whether a **low-rate codec can be trained** depends on the
> geometry of the **whole dataset**, not of one sample.

## The trap (discovered empirically in this project)
We first generated each sample with **fully random continuous** cluster angles
over $\pm 60^\circ$. Each sample was sparse — yet **PCA to 256 of 2048 real
dims captured only ≈49 % energy** and reached only **SGCS ≈ 0.56**, and a CsiNet
autoencoder did no better. Why?

- A peak at a **continuous** angle is **not** sparse in a *fixed* DFT basis — it
  leaks across many angular bins (off-grid leakage).
- Across the dataset, the peak can be **anywhere**, so the union of all samples
  **spans the entire space**. The dataset's linear dimension ≈ full $2 N_d N_t$.
- A codeword of $M \ll 2N_dN_t$ reals (linear-ish bottleneck) then cannot
  represent an arbitrary sample → both PCA and the near-linear CsiNet stall.

> [!important] The distinction
> - **Per-sample sparsity:** each $H_{\mathrm{ad}}$ has few large entries — but
>   their *locations* vary sample to sample.
> - **Dataset manifold dimension:** how many DOF the *collection* of samples
>   actually occupies. Compression rate is bounded by **this**, not by per-sample
>   sparsity.
> Exploiting per-sample sparsity with *varying support* requires **adaptive /
> nonlinear** coding (which is exactly why learned codecs can beat fixed
> transforms — but only when there is shared structure to learn).

## Why real CSI *is* compressible (and COST 2100 / 3GPP works)
A real cell has a **limited, stable set of scattering geometries**. Channels
cluster near a **low-dimensional manifold**, so a codec trained on many
snapshots learns that manifold and compresses hard. We reproduce this with the
`n_environments` knob in `csi.data.generate_csi_dataset`:

| `n_environments` | dataset manifold | PCA SGCS @ M=256 |
|---|---|---|
| 8 | low-dim, easy | ≈ 0.99 |
| 12–20 | moderate | ≈ 0.9 |
| 50 | high-dim | ≈ 0.76 |
| ≈ n_samples | "everything random" (the trap) | ≈ 0.56 |

> [!tip] Practical rules
> - Always sanity-check with a **linear PCA baseline** (notebook §6). If PCA is
>   already bad, your *data* (not your model) is the problem.
> - Train/test must **share the same environments** (generate one pool, then
>   split) — otherwise you are testing generalisation to unseen geometries,
>   which is a *different*, harder question (the 3GPP **generalization** issue;
>   see [[Datasets]] and [[Two-Sided Model and Training Collaboration]]).
> - The autoencoder's edge over PCA comes from **nonlinearity** on a manifold
>   with curvature — strongest at *moderate* `n_environments`.

## Related
- [[Angular-Delay Transform]] · [[CsiNet Autoencoder Architecture]] · [[Metrics - NMSE and SGCS]]
- [[Open Questions and Claims to Verify]] (item F: limits of the synthetic data)
