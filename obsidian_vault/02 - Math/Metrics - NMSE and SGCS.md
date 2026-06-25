---
title: Evaluation Metrics — NMSE, GCS, SGCS
tags: [math, metrics, kpi, sgcs, nmse]
created: 2026-06-24
---

# Evaluation Metrics — NMSE, GCS, SGCS

> [!abstract] The hierarchy
> 3GPP separates **intermediate KPIs** (cheap, model-level) from **eventual
> KPIs** (system-level throughput). For CSI compression the headline
> intermediate KPI is **SGCS**; **NMSE** is the common academic companion;
> the eventual KPI is **throughput gain vs the Rel-16 Type II codebook**.

## SGCS — Squared Generalized Cosine Similarity ⭐ primary
$$\boxed{\;\mathrm{SGCS} \;=\; \mathbb{E}\!\left[\frac{\bigl|\,w^{H}\hat{w}\,\bigr|^{2}}{\lVert w\rVert^{2}\,\lVert \hat{w}\rVert^{2}}\right]\in[0,1]\;}$$
- $w$ = true dominant [[Eigenvector Precoder and Covariance|eigenvector]], $\hat w$ = reconstructed.
- **Invariant to global phase and scale** → measures only *beam direction* accuracy.
- $1.0$ = perfect; typical good operating points are $\approx 0.9$–$0.99$.
- Matches `csi.sgcs`. The non-squared version $\mathrm{GCS}=\sqrt{\mathrm{SGCS}}$ (`csi.gcs`).
- Averaged **per subband then over samples**: $\bar\rho^2=\frac1{KN}\sum_i\sum_k \rho_{k}^{2(i)}$.
- **Confirmed primary intermediate KPI** in TR 38.843 (formula e.g. arXiv:2409.13494 Eq. 3). Reported gains: **+4.6–7 % (S-F)**, **+8.8–16.5 % (S-T-F)** vs Rel-16 eType II.

> [!tip] As a loss
> Training with $\mathcal{L}=1-\mathrm{SGCS}$ (or $-\log$ SGCS) aligns the
> objective with the KPI better than plain MSE on $H_{ad}$, because MSE wastes
> effort on the irrelevant phase/scale.

## NMSE — Normalized Mean Square Error
$$\boxed{\;\mathrm{NMSE} \;=\; 10\log_{10}\!\left(\frac{\mathbb{E}\,\lVert H-\hat H\rVert_F^{2}}{\mathbb{E}\,\lVert H\rVert_F^{2}}\right)\ \text{dB}\;}$$
- Reported in **dB** (more negative = better). Matches `csi_lib.nmse_db`.
- Sensitive to phase/scale — good for raw-channel reconstruction, less aligned
  with the precoding task than SGCS.

## ρ — CsiNet cosine correlation
Per-subcarrier complex correlation, averaged:
$$\rho = \mathbb{E}_{n}\!\left[\frac{|h_n^{H}\hat h_n|}{\|h_n\|\,\|\hat h_n\|}\right].$$
Matches `csi_lib.cosine_rho`. This is the metric in the original CsiNet paper.

## Eventual KPI — system-level throughput (UPT)
- Run the reconstructed precoder $\hat w$ through a system simulator and measure
  **mean User Plane Throughput (UPT)**; report **% gain over Rel-16 eType II**
  at matched feedback overhead.
- Reported data point (TR 38.843): AI/ML at **~26 bits** ≈ eType II at **~155
  bits** → **~83 % overhead reduction** at equal mean UPT.
- This is the number that justifies normative work, but it is expensive →
  SGCS is used as the day-to-day proxy.

> [!warning] Mind the matched-overhead comparison
> A learned codec at SGCS 0.95 is only "better" if its **[[Quantization and Feedback Overhead|feedback bits]]** are ≤ the codebook it beats. Always compare on the **rate–distortion** plane, not SGCS alone. (Notebook §6.)

## Related
- [[Eigenvector Precoder and Covariance]]
- [[Quantization and Feedback Overhead]]
- [[3GPP AI-ML CSI Overview]]
