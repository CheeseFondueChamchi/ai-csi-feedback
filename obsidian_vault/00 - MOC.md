---
title: CSI Compression (AI/ML) — Map of Content
tags: [moc, 3gpp, csi, ai-ml, home]
created: 2026-06-24
---

# 🛰️ AI/ML CSI Compression — Map of Content

> [!abstract] What this vault is
> Study notes for an **AI/ML-based CSI (Channel State Information) compression**
> model following the **3GPP** study on AI/ML for the NR air interface
> ([[3GPP AI-ML CSI Overview|TR 38.843]]). The runnable companion is the
> 4-stage notebook pipeline in `../notebooks/` (see "Runnable pipeline" below).

## 🚀 Runnable pipeline
The companion repository (`../notebooks/`) implements a **decoupled 4-stage simulation pipeline**:
1. **Gen channel data** — reads `ChannelConfig`, generates train/test CSI datasets (Sionna CDL-C + synthetic) + PMI reports → `data/<channel_label>/`
2. **Model zoo** — defines AI architectures (CSiNet variants) → `models/raw/<arch>/`
3. **Train and test** — loads dataset + raw model, trains/evaluates (NMSE, SGCS, bit sweep) → `models/trained/<channel_label>/<arch>/`
4. **Comparison** — fully decoupled visualization (PMI vs AI on SGCS-vs-bits, two panels)

Each stage reads/writes artifacts; run via `python notebooks/build_<x>.py && nbclient notebooks/<x>.ipynb`. The shared `src/csi/config.py` provides the `ChannelConfig` dataclass and IO contract (`save_dataset`/`load_dataset`).

## 🗺️ Reading order

1. [[3GPP AI-ML CSI Overview]] — standardization context, use cases, timeline
2. [[Current NR CSI Feedback (PMI)]] — the *baseline* system AI/ML must beat (PMI/RI/CQI, Type I/II)
3. [[Two-Sided Model and Training Collaboration]] — the encoder@UE / decoder@gNB paradigm + Type 1/2/3
3. [[CsiNet Autoencoder Architecture]] — the canonical model and its descendants
4. [[Datasets]] — COST 2100, TR 38.901 generation, Sionna/QuaDRiGa
5. Math core:
   - [[Angular-Delay Transform]]
   - [[Eigenvector Precoder and Covariance]]
   - [[Metrics - NMSE and SGCS]]
   - [[Quantization and Feedback Overhead]]
   - [[Sparsity vs Learnable Manifold]] — ⭐ subtle, read this
6. ⚠️ [[Open Questions and Claims to Verify]] — **check these yourself**

## 🎯 The one-paragraph mental model

The UE measures a high-dimensional downlink channel $H$. Feeding it back raw is
too expensive, so we **compress** it. Classical NR uses the Rel-16 **Type II
codebook**; the AI/ML approach replaces that with a **[[Two-Sided Model and Training Collaboration|two-sided autoencoder]]**:
a neural **encoder at the UE** maps $H$ (or its dominant
[[Eigenvector Precoder and Covariance|eigenvector]] $w$) to a short codeword,
and a neural **decoder at the gNB** reconstructs it. The whole thing is judged
by **[[Metrics - NMSE and SGCS|SGCS]]** (direction accuracy of the precoder) at
a given **[[Quantization and Feedback Overhead|feedback bit budget]]**.

## 🔑 Key quantities at a glance

| Symbol | Meaning | Note |
|---|---|---|
| $H \in \mathbb{C}^{N_c\times N_t}$ | channel, subcarriers × Tx antennas | [[Angular-Delay Transform]] |
| $w$ | dominant eigenvector / rank-1 precoder | [[Eigenvector Precoder and Covariance]] |
| $M$ | codeword length | [[Quantization and Feedback Overhead]] |
| $\gamma = M/(2N_dN_t)$ | compression ratio | smaller = harder |
| SGCS | $\mathbb{E}\,|w^H\hat w|^2/(\|w\|^2\|\hat w\|^2)$ | primary KPI |
| NMSE | $\mathbb{E}\|H-\hat H\|^2/\mathbb{E}\|H\|^2$ | secondary KPI |

## ✅ Self-check questions
- Why is the channel sparse in the [[Angular-Delay Transform|angular-delay domain]]?
- Why does [[Metrics - NMSE and SGCS|SGCS]] use a *normalized* inner product (what is it invariant to)?
- What breaks if the UE and gNB are made by different vendors? → [[Two-Sided Model and Training Collaboration]]
