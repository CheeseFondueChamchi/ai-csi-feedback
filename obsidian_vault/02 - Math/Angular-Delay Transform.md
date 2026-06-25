---
title: Angular-Delay Domain Transform (2D DFT)
tags: [math, dsp, sparsity, dft]
created: 2026-06-24
---

# Angular–Delay Domain Transform

> [!abstract] Why
> The raw channel $H\in\mathbb{C}^{N_c\times N_t}$ (subcarriers × Tx antennas)
> is **dense**, but in the **angular–delay** domain it is **sparse** — a few
> clusters of arrivals at a few delays. Sparsity = compressibility. This is the
> mathematical reason CSI feedback can be compressed at all.

## The transform
$$\boxed{\;H_{\mathrm{ad}} \;=\; F_{\mathrm{delay}}^{H}\, H \, F_{\mathrm{angle}}\;}$$
- $F_{\mathrm{angle}}\in\mathbb{C}^{N_t\times N_t}$: DFT across the **antenna** axis → **angular** (beam) domain. Justified because a **ULA** steering vector $a(\theta)[m]=\tfrac{1}{\sqrt{N_t}}e^{-j2\pi d\, m\sin\theta}$ is a DFT column → each physical angle maps to ~one angular bin.
- $F_{\mathrm{delay}}\in\mathbb{C}^{N_c\times N_c}$: (I)DFT across the **subcarrier** axis → **delay** domain. A path at delay $\tau$ has frequency response $e^{-j2\pi f\tau}$ → maps to ~one delay tap.

> [!note] Implementation (matches `csi_lib.to_angular_delay`)
> `H_delay = ifft(H, axis=subcarrier)` then `H_ad = fft(H_delay, axis=antenna)`.
> The exact DFT/IDFT convention only changes a constant scale — irrelevant for
> [[Metrics - NMSE and SGCS|SGCS]] which is scale-invariant.

## Delay truncation (the classical compressor)
Keep only the first $N_d \ll N_c$ delay taps:
$$\tilde H_{\mathrm{ad}} = H_{\mathrm{ad}}[0:N_d,\,:]\,, \qquad \hat H = \text{zero-pad} \to \text{inverse 2D DFT}.$$
The energy retained is
$$\eta(N_d) = \frac{\sum_{k<N_d}\sum_n |H_{\mathrm{ad}}[k,n]|^2}{\sum_{k}\sum_n |H_{\mathrm{ad}}[k,n]|^2}.$$
In the notebook (§2) $\eta$ saturates by $N_d\approx 16$–$32$ for a 16-tap delay
spread — truncation alone is already a strong baseline the autoencoder must beat.

> [!warning] When the DFT basis is *not* optimal
> DFT is only the right sparsifying basis for an **ideal ULA** with half-wave
> spacing. For **dual-polarized / planar (UPA)** arrays, mutual coupling, or
> calibration error, the true sparsifying basis differs — a learned encoder can
> adapt where a fixed DFT cannot. This is part of the AI/ML gain argument.
> ⚠️ confirm the array geometries used in 3GPP evals → [[Open Questions and Claims to Verify]].

## Related
- [[CsiNet Autoencoder Architecture]]
- [[Eigenvector Precoder and Covariance]]
- [[Metrics - NMSE and SGCS]]
