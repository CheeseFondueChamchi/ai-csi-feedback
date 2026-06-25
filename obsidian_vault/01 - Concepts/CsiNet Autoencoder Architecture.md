---
title: CsiNet & the Autoencoder Family for CSI Feedback
tags: [csi, autoencoder, csinet, architecture, deep-learning]
created: 2026-06-24
---

# CsiNet & the Autoencoder Family

> [!info] Seminal paper
> C.-K. Wen, W.-T. Shih, S. Jin, **"Deep Learning for Massive MIMO CSI
> Feedback"**, *IEEE Wireless Comm. Letters*, 2018. arXiv:1712.08919.
> Reference code: `sydney222/Python_CsiNet`. ⚠️ confirm split sizes in [[Datasets]].

## The pipeline (what our notebook implements)
1. Channel $H\in\mathbb{C}^{N_c\times N_t}$ → [[Angular-Delay Transform|angular-delay 2D DFT]] → truncate to the first $N_d$ delay taps → $H_{\mathrm{ad}}\in\mathbb{C}^{N_d\times N_t}$.
2. Split into **2 real channels** (real, imag) → tensor $(2, N_d, N_t)$; min-max normalize to $[0,1]$.
3. **Encoder (UE):** one $3\times3$ conv (2→2) + BN + LeakyReLU → flatten → **dense → codeword $z\in\mathbb{R}^{M}$**.
4. **Decoder (gNB):** dense → reshape → **2× RefineNet residual blocks** → conv → sigmoid → $\hat H_{\mathrm{ad}}$.
5. Inverse transform + de-truncate → $\hat H$.

> [!tip] RefineNet block
> Residual block of three convs (2→8→16→2 channels, $3\times3$, BN, LeakyReLU)
> with a skip connection: $x \mapsto \mathrm{LeakyReLU}(x + f(x))$. Refines the
> coarse reconstruction without changing tensor shape.

## Compression ratio & overhead
$$\gamma = \frac{M}{2\,N_d\,N_t}, \qquad \text{feedback bits} = M\cdot b.$$
See [[Quantization and Feedback Overhead]].

## Descendant architectures (know the lineage)
| Model | Idea | Note |
|---|---|---|
| **CsiNet** (2018) | CNN autoencoder + RefineNet | arXiv:1712.08919 |
| **CsiNet+** / **CsiNet-LSTM** | bigger kernels / temporal recurrence | time correlation |
| **CRNet** (2020) | multi-resolution CRBlock (3×3, 1×9, 9×1 paths) | arXiv:2006.10097 |
| **CLNet** (2021) | complex-valued input + channel attention, lightweight | arXiv:2102.07507 |
| **EVCsiNet** (2021) | feeds back the **eigenvector** directly | IEEE WCL Dec 2021; aligns with [[Eigenvector Precoder and Covariance|3GPP precoder feedback]] |
| **Transformer / Swin** | self-attention encoder/decoder | arXiv:2401.06435 |

*(arXiv IDs verified via literature pass; confirm headline NMSE/SGCS claims yourself — [[Open Questions and Claims to Verify]].)*

## Design knobs that matter
- **Input representation**: full channel vs **eigenvector** (3GPP-aligned) vs angular-delay truncated.
- **Latent quantization**: scalar vs vector; differentiable surrogate during training → [[Quantization and Feedback Overhead]].
- **Loss**: MSE on $H_{ad}$ vs an **[[Metrics - NMSE and SGCS|SGCS]]-aligned** loss (cosine/GCS loss matches the KPI better than plain MSE).

## Related
- [[Two-Sided Model and Training Collaboration]]
- [[Angular-Delay Transform]]
- [[Metrics - NMSE and SGCS]]
