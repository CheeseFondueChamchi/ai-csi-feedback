---
title: 3GPP AI/ML for NR Air Interface — CSI Overview
tags: [3gpp, csi, ai-ml, standardization]
created: 2026-06-24
---

# 3GPP AI/ML for the NR Air Interface — CSI Compression Context

> [!info] Primary document
> **3GPP TR 38.843** — *"Study on Artificial Intelligence (AI)/Machine Learning
> (ML) for NR air interface"* (Release 18 **Study Item** `FS_NR_AIML_air`,
> RAN1). Version **v18.0.0, Dec 2023** (~187 pp).
> *(Verified via literature pass — still spot-check the primary text, see [[Open Questions and Claims to Verify]].)*

## The three study use cases (Rel-18 SI)
1. **CSI feedback enhancement** ← *our focus*
   - sub-use-case **CSI compression** (spatial-frequency, *two-sided* model)
   - sub-use-case **CSI prediction** (temporal, can be one-sided)
2. **Beam management** (spatial/temporal beam prediction)
3. **Positioning accuracy enhancement** (direct / AI-assisted)

## Why CSI feedback is the hard, interesting one
- It is the only use case requiring a **[[Two-Sided Model and Training Collaboration|two-sided model]]** (an ML model split across the UE and the gNB), which raises the thorny **inter-vendor training collaboration** problem.
- The baseline it must beat is the **Rel-16 Type II codebook** (already a strong, eigenvector-based linear compressor).

## Timeline (study → normative)
- **Rel-18**: study item → TR 38.843, evaluation methodology + observations.
- **Rel-19**: work item — **WID `RP-234039`** (Dec 2023). Normative support for
  the **one-sided** use cases — **CSI prediction, beam management, positioning**
  (impacts TS 38.214 procedures, TS 38.331 RRC; LCM in TS 28.105).
- ⚠️ **Important nuance**: Rel-18 concluded the **two-sided CSI *compression***
  gains were *not sufficient* vs complexity/overhead to justify normative work,
  so it stayed a **study** item in Rel-19 (inter-vendor Type 3 collaboration,
  data collection, interoperability). Rel-19 functional freeze ≈ **June 2025**.

> [!example] Reported gains in TR 38.843 (company evals)
> - Spatial-frequency (S-F) compression: **+4.6 – 7 % SGCS** vs Rel-16 eType II.
> - Spatio-temporal-frequency (S-T-F): **+8.8 – 16.5 % SGCS**.
> - System-level: AI/ML at **~26 bits** ≈ Rel-16 eType II at **~155 bits**
>   → **~83 % feedback-overhead reduction** at equal throughput.
> See [[Metrics - NMSE and SGCS]].

## Life-cycle management (LCM) themes the SI introduced
- Model **training**, **inference**, **monitoring**, **activation/deactivation**, **switching**, **fallback**.
- **Performance monitoring** metrics so the network can detect when the model degrades (distribution shift) and fall back to the codebook.

## Related
- [[Two-Sided Model and Training Collaboration]]
- [[CsiNet Autoencoder Architecture]]
- [[Metrics - NMSE and SGCS]]
- [[Datasets]]
