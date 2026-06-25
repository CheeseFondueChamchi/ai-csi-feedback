---
title: ⚠️ Open Questions & Claims to Verify
tags: [to-verify, checklist, 3gpp, research]
created: 2026-06-24
---

# ⚠️ Open Questions & Claims to Verify

> [!danger] Read me first
> These notes were written partly from memory + a fast literature pass. Each
> item below is a claim I want **you** to confirm against the **primary 3GPP
> text** or the original papers before you rely on it. Tick the box when checked.

## A · 3GPP document facts
- [ ] **TR number & version**: CSI study is in **TR 38.843** — confirm the exact version (e.g. v18.0.0) and approval date. → [[3GPP AI-ML CSI Overview]]
- [ ] **Study Item RP-number** that launched the Rel-18 AI/ML SI (and the WG, RAN1 lead).
- [ ] **Rel-19 normative**: which AI/ML use cases became normative, the **WI number**, and which TS got changed (TS **38.214** procedures? **38.331** RRC? **38.212** UCI coding?).
- [ ] The exact 3GPP wording of **training collaboration Type 1 / 2 / 3** and sub-variants (NW-first vs UE-first for Type 3). → [[Two-Sided Model and Training Collaboration]]
- [ ] The agreed **LCM** procedures (monitoring, switching, fallback) terminology.

## B · Metric definitions (get the formula exactly right)
- [ ] **SGCS** exact 3GPP definition for **rank 1** vs **rank > 1** (principal-angle / chordal form). → [[Metrics - NMSE and SGCS]] · [[Eigenvector Precoder and Covariance]]
- [ ] Whether SGCS is averaged **per subband then over samples**, or jointly.
- [ ] The precise **eventual KPI** definition: throughput gain vs **Rel-16 Type II** (and is the baseline eType II / Rel-17?), and the % gains 3GPP actually observed.
- [ ] Is **NMSE** or **SGCS** the agreed *primary* intermediate KPI? (I claim SGCS — confirm.)

## C · Channel / dataset assumptions
- [ ] The **agreed antenna configuration** (port count, UPA layout, polarization) in TR 38.843 eval assumptions. → [[Datasets]]
- [ ] Carrier frequency, SCS, bandwidth, **number of subbands**, UE speed set, SNR distribution used in evaluations.
- [ ] Which **TR 38.901** scenarios (CDL vs UMa/UMi/InH) were mandatory vs optional for CSI evals.
- [ ] **COST 2100** split sizes (I wrote train 100k / val 30k / test 20k — verify) and the indoor/outdoor carrier (5.3 GHz / 300 MHz). → [[Datasets]]

## D · Model / architecture claims
- [ ] CsiNet citation details: **arXiv:1712.08919**, IEEE WCL 2018, authors Wen–Shih–Jin. → [[CsiNet Autoencoder Architecture]]
- [ ] Descendant models table (CRNet, CLNet, EVCsiNet, TransNet): authors, years, and their headline NMSE/SGCS claims.
- [ ] Whether 3GPP evaluated a **reference/nominal model** and what architecture it was.

## E · Mathematics to derive yourself (don't just trust the note)
- [ ] Derive that the **top eigenvector of $R=H^HH$** maximizes $v^HRv$ s.t. $\|v\|=1$ (Rayleigh quotient / Courant–Fischer). → [[Eigenvector Precoder and Covariance]]
- [ ] Show why a **ULA steering vector is a DFT column** ⇒ angular sparsity. → [[Angular-Delay Transform]]
- [ ] Confirm **SGCS ∈ [0,1]** and its invariance to $\hat w \mapsto e^{j\phi}c\,\hat w$ (Cauchy–Schwarz).
- [ ] Work out the **rate** of an $M$-dim codeword at $b$ bits and compare to a Type II codebook payload at the same SGCS. → [[Quantization and Feedback Overhead]]
- [ ] Justify the **straight-through estimator** gradient for the quantizer (bias/variance).

## F · Things my synthetic data does NOT capture (limits of the notebook)
- [ ] No **dual-polarization / UPA** spatial structure (DFT-sparsity assumption is cleaner than reality).
- [ ] No **channel estimation error / noise** at the UE before compression.
- [ ] No **temporal correlation** → cannot study CSI *prediction* sub-use-case.
- [ ] Single Rx antenna → no true MIMO rank-$>1$ precoding.
- [ ] → For any quantitative claim, re-run on **Sionna TR 38.901** data (notebook §7).

## Source log
Literature pass (2026-06-24) already corroborated the items below — but these
came from secondary sources (surveys, arXiv, tech blogs), so the **primary TR/TS
text is still the final word**. Treat ✓ as "likely, double-check", not "done".

| Claim | Source checked | Verdict |
|---|---|---|
| CSI study is **TR 38.843 v18.0.0 (Dec 2023)** | arXiv:2308.05315; tech-invite | ✓ likely |
| Rel-19 WID **RP-234039**; two-sided compression stays *study* | arXiv:2312.15174 | ✓ likely |
| **SGCS** is the primary intermediate KPI; formula | arXiv:2409.13494 Eq.3 | ✓ likely |
| **COST 2100** split 100k/30k/20k; 32×32 | sydney222/Python_CsiNet | ✓ likely |
| Eval config 32-port (8H×4V dual-pol), 4 Rx, 13 subbands, ~2 GHz | arXiv:2206.15132 | ✓ likely |
| eType II baseline ≈ 300 bits | survey | ✓ likely |
| Gains: +4.6–7 % (S-F), +8.8–16.5 % (S-T-F) SGCS | survey of submissions | ✓ check magnitudes |
| UPT: ~26 b ≈ 155 b → ~83 % overhead cut | survey | ✓ check |
| CRNet 2006.10097 / CLNet 2102.07507 / Swin 2401.06435 | arXiv | ✓ likely |

## Related
- [[00 - MOC]]
