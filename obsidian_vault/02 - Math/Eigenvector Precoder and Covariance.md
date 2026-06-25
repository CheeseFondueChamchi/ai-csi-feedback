---
title: Eigenvector Precoder & Channel Covariance
tags: [math, mimo, precoding, eigenvector, linear-algebra]
created: 2026-06-24
---

# Eigenvector Precoder & Channel Covariance

> [!abstract] The key 3GPP framing
> In the 3GPP CSI-compression sub-use-case the UE usually feeds back **not the
> raw channel** but the **dominant eigenvector(s)** of the channel — the same
> quantity the **Rel-16 Type II codebook** reports. The gNB uses it as the
> downlink **precoder**.

## Covariance and the optimal rank-1 precoder
Stack the per-subcarrier channel rows $h_f\in\mathbb{C}^{N_t}$ and form the
spatial covariance (over a subband / the band):
$$R \;=\; \sum_{f} h_f^{H} h_f \;=\; H^{H}H \;\in\;\mathbb{C}^{N_t\times N_t},\qquad R = R^H \succeq 0.$$
Eigendecomposition $R = U\Lambda U^H$ with $\lambda_1\ge\lambda_2\ge\dots$. The
**rank-1 capacity-optimal precoder** maximizes received power:
$$w \;=\; \arg\max_{\|v\|=1} v^{H} R\, v \;=\; u_1 \quad(\text{top eigenvector}).$$
> [!note] Matches `csi_lib.dominant_eigenvector`
> Uses `eigh` (Hermitian) and takes the eigenvector of the **largest**
> eigenvalue. For rank-$r$ transmission you keep $u_1,\dots,u_r$.

## Why feed back $w$ instead of $H$?
- $w$ is what the gNB actually needs for beamforming → smaller, task-aligned target.
- $w$ is defined only up to a **global phase** $e^{j\phi}$ and **scale** — so the
  feedback metric must be invariant to both → motivates **[[Metrics - NMSE and SGCS|SGCS]]**.
- Per-subband eigenvectors are smooth across frequency → still compressible.

## Phase/scale ambiguity (don't get burned)
$w$ and $e^{j\phi}w$ are the **same beam**. Any reconstruction loss on $w$ must
quotient this out (cosine / GCS), otherwise the network wastes capacity encoding
an irrelevant phase. This is exactly why the KPI is
$$\frac{|w^H\hat w|^2}{\|w\|^2\|\hat w\|^2}\quad\text{(invariant to phase \& scale).}$$

## Higher rank → generalized cosine similarity
For rank $r$ with subspaces $\mathrm{span}(W), \mathrm{span}(\hat W)$, the
generalized version averages the squared cosines of the **principal angles**
between the two subspaces (chordal-distance flavour). ⚠️ confirm the exact 3GPP
rank-$>1$ GCS definition → [[Open Questions and Claims to Verify]].

## Related
- [[Metrics - NMSE and SGCS]]
- [[Angular-Delay Transform]]
- [[CsiNet Autoencoder Architecture]]
