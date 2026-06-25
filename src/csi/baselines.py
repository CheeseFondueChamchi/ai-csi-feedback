"""
csi.baselines — The *current* NR CSI feedback: PMI codebooks (no ML).
=====================================================================

WHAT THIS MODULE DOES
    Implements today's standardised CSI feedback so the AI/ML codec has an
    honest, same-axes baseline. In NR the UE measures CSI-RS, computes the
    dominant eigenvector (precoder) per subband, and reports a **PMI**
    (Precoding Matrix Indicator) chosen from a **codebook** — not the raw
    channel. The gNB looks the PMI up and beamforms.

    * Type I  : pick the single best DFT beam (cheap, coarse).  -> rank-1 PMI
    * Type II : linear combination of L strongest beams with quantised
                amplitude/phase coefficients (expensive, accurate). This is the
                Rel-16 baseline AI/ML is compared against in TR 38.843.

PUBLIC API (the stable "contract")
    dft_codebook(n_tx, oversampling)            -> B            (n_tx, K) beams
    type1_pmi(W_true, n_tx, ...)                -> (W_hat, bits)
    type2_pmi(W_true, n_tx, L, ...)             -> (W_hat, bits)

    W_true / W_hat are (N, n_tx) dominant eigenvectors; score them with
    ``csi.sgcs(W_true, W_hat)`` exactly like the learned codec — a fair,
    bits-vs-SGCS comparison.

HOW TO SWAP THIS MODULE
    Replace with a Rel-17 enhanced Type II, a port-selection codebook, or the
    real 38.214 codebook tables; keep the (W_true, ...) -> (W_hat, bits) shape.
"""
from __future__ import annotations
import numpy as np


def dft_codebook(n_tx: int, oversampling: int = 4) -> np.ndarray:
    """Oversampled DFT beam codebook for a ULA: columns are unit-norm beams.

    Returns B of shape (n_tx, n_tx * oversampling). This is the spatial-beam
    basis underlying the NR Type I / Type II codebooks (a ULA Type I PMI is
    essentially "which oversampled DFT beam").
    """
    K = n_tx * oversampling
    n = np.arange(n_tx)[:, None]
    k = np.arange(K)[None, :]
    return np.exp(1j * 2 * np.pi * n * k / K) / np.sqrt(n_tx)   # (n_tx, K)


def type1_pmi(W_true: np.ndarray, n_tx: int, oversampling: int = 4):
    """Type-I-style PMI: report the single best DFT beam (rank-1, wideband).

    Returns
    -------
    W_hat : (N, n_tx) the selected beam per sample (the reported precoder).
    bits  : feedback payload = ceil(log2(K)) (one beam index).
    """
    B = dft_codebook(n_tx, oversampling)              # (n_tx, K)
    with np.errstate(all="ignore"):                   # silence spurious macOS complex-matmul warnings
        corr = np.abs(W_true.conj() @ B) ** 2         # (N, K) beamforming power
    idx = corr.argmax(axis=1)
    W_hat = B[:, idx].T                               # (N, n_tx)
    bits = int(np.ceil(np.log2(B.shape[1])))
    return W_hat, bits


def type2_pmi(W_true: np.ndarray, n_tx: int, L: int = 4, oversampling: int = 4,
              amp_bits: int = 3, phase_bits: int = 4):
    """Type-II-style PMI: quantised linear combination of L strongest beams.

    Projects the eigenvector onto its L best DFT beams and scalar-quantises the
    L complex combining coefficients (amplitude + phase). This is a compact
    stand-in for the Rel-16 Type II codebook (the 3GPP baseline).

    Returns
    -------
    W_hat : (N, n_tx) reconstructed (unit-norm) precoder per sample.
    bits  : L*ceil(log2(K)) beam indices + L*(amp_bits+phase_bits) coefficients.
    """
    B = dft_codebook(n_tx, oversampling)              # (n_tx, K)
    with np.errstate(all="ignore"):                   # silence spurious macOS complex-matmul warnings
        corr = np.abs(W_true.conj() @ B)              # (N, K)
    n_amp, n_ph = 2 ** amp_bits, 2 ** phase_bits
    W_hat = np.zeros_like(W_true)
    for i in range(len(W_true)):
        sel = np.argsort(corr[i])[-L:]                # L strongest beams
        Bl = B[:, sel]                                # (n_tx, L)
        c = Bl.conj().T @ W_true[i]                   # (L,) combining coeffs
        # quantise amplitude (relative to the strongest) and phase
        amp = np.abs(c); amp_n = amp / (amp.max() + 1e-12)
        amp_q = np.round(amp_n * (n_amp - 1)) / (n_amp - 1)
        ph_q = np.round(np.angle(c) / (2 * np.pi) * n_ph) / n_ph * 2 * np.pi
        c_q = amp_q * np.exp(1j * ph_q)
        w = Bl @ c_q
        W_hat[i] = w / (np.linalg.norm(w) + 1e-12)
    bits = int(L * np.ceil(np.log2(B.shape[1])) + L * (amp_bits + phase_bits))
    return W_hat, bits


# Rel-16 Type II amplitude code점 집합 (TS 38.214 Table 5.2.2.2.5-2/3 의 wideband 진폭).
# 3비트(8레벨)일 때 진폭 후보는 sqrt(0.5)^k 형태의 등비수열로 정의된다:
#   {1, 1/sqrt(2), 1/2, 1/(2*sqrt(2)), 1/4, 1/(4*sqrt(2)), 1/8, 0}
# 마지막 레벨이 0(=빔 사실상 비활성)인 것이 Rel-16 진폭 집합의 특징이다.
_REL16_AMP_3BIT = np.array(
    [1.0,
     np.sqrt(0.5),
     0.5,
     np.sqrt(0.5) ** 3,
     0.25,
     np.sqrt(0.5) ** 5,
     0.125,
     0.0],
    dtype=float,
)


# ===========================================================================
# True Rel-16 Enhanced Type II — spatial-FREQUENCY 2D compression (L x M)
# ===========================================================================
def subband_precoders(H: np.ndarray, n_sb: int) -> np.ndarray:
    """Per-subband dominant precoders, preserving frequency selectivity.

    Splits the ``n_sub`` subcarriers into ``n_sb`` contiguous subbands and takes
    the dominant eigenvector of each subband's spatial covariance. Unlike the
    wideband ``dominant_eigenvector(H)`` (one vector per sample), this keeps the
    frequency axis that enhanced Type II compresses.

    Parameters
    ----------
    H : complex (N, n_sub, n_tx)

    Returns
    -------
    W_sb : complex64 (N, n_sb, n_tx) — unit-norm precoder per (sample, subband).
    """
    from .metrics import dominant_eigenvector

    N, n_sub, n_tx = H.shape
    edges = np.linspace(0, n_sub, n_sb + 1).astype(int)
    W_sb = np.zeros((N, n_sb, n_tx), dtype=np.complex64)
    for s in range(n_sb):
        a, b = int(edges[s]), int(edges[s + 1])
        W_sb[:, s, :] = dominant_eigenvector(H[:, a:b, :])
    return W_sb


def sgcs_subband(W_true_sb: np.ndarray, W_hat_sb: np.ndarray) -> float:
    """Mean SGCS across subbands (the per-subband precoder-feedback metric)."""
    from .metrics import sgcs
    n_sb = W_true_sb.shape[1]
    return float(np.mean([sgcs(W_true_sb[:, s, :], W_hat_sb[:, s, :])
                          for s in range(n_sb)]))


def etype2_pmi_2d(W_sb: np.ndarray, n_tx: int, L: int = 4, M: int = 2,
                  oversampling: int = 4, amp_bits: int = 3, phase_bits: int = 3,
                  beta: float = 0.5, dual_pol: bool = False):
    """True Rel-16 enhanced Type II: spatial-frequency 2D compression.

    Reports the per-subband precoder matrix as W1 · C · Wf^H — L spatial DFT
    beams (W1) AND M frequency-domain DFT basis vectors (Wf), with a quantised
    coefficient matrix C (TS 38.214 §5.2.2.2.5). The wideband single-eigenvector
    codebook cannot represent this.

    Dual polarization (``dual_pol=True``)
    -------------------------------------
    The n_tx ports are split into two polarization groups of n_tx/2 (3GPP port
    layout: [pol-0 ports, pol-1 ports]). The L spatial DFT beams W1 are **shared
    across both polarizations** (computed over the per-pol spatial dimension), the
    M frequency basis vectors are shared too, but each polarization gets its **own**
    L×M coefficient matrix — so the coefficient grid is 2L×M, exactly the Rel-16
    dual-pol structure. With ``dual_pol=False`` it falls back to single-pol (L×M).

    K0 truncation
    -------------
    Only ``K0 = ceil(beta * P * L * M)`` strongest coefficients are reported (P =
    1 or 2 pols), with a bitmap of their positions — the Rel-16 CSI-Part-2
    mechanism. ``beta=1.0`` recovers the all-coefficient upper bound.

    Parameters
    ----------
    W_sb : complex (N, n_sb, n_tx) — per-subband precoders (see subband_precoders).
    L, M : spatial beams / frequency-domain basis vectors.
    beta : fraction of the P·L·M coefficients kept (non-zero cap K0).
    dual_pol : split ports into two polarizations with shared beams, per-pol coeffs.

    Returns
    -------
    W_hat_sb : complex (N, n_sb, n_tx) reconstructed per-subband precoders.
    bits     : total per-report feedback payload.
    """
    # ──────────────────────────────────────────────────────────────────────
    # 진짜 eType II: 공간(W1)·주파수(Wf) 2D 압축 + 이중 편파(dual-pol).
    #   각 편파별 precoder P_p (n_sp×n_sb) ≈ W1 (n_sp×L) · C_p (L×M) · Wf^H
    #   - W1 : 두 편파가 *공유*하는 L 개 공간 DFT 빔 (per-pol 차원 n_sp=n_tx/P 위에서 선택)
    #   - Wf : 두 편파가 공유하는 M 개 주파수 DFT 기저
    #   - C_p: 편파마다 *독립*인 L×M 계수 → 전체 P·L·M (단일편파는 P=1)
    # ──────────────────────────────────────────────────────────────────────
    N, n_sb, _ = W_sb.shape
    n_pol = 2 if dual_pol else 1
    n_sp = n_tx // n_pol                               # 편파당 공간 소자 수
    B = dft_codebook(n_sp, oversampling)              # (n_sp, K) per-pol 공간 빔
    K = B.shape[1]
    M = min(M, n_sb)
    n_coeff = n_pol * L * M
    K0 = min(n_coeff, max(1, int(np.ceil(beta * n_coeff))))

    nn = np.arange(n_sb)
    Wf_full = np.exp(-2j * np.pi * np.outer(nn, nn) / n_sb) / np.sqrt(n_sb)  # (n_sb, n_sb)

    n_ph = 2 ** phase_bits
    amp_levels = _REL16_AMP_3BIT if amp_bits == 3 else np.linspace(0.0, 1.0, 2 ** amp_bits)

    W_hat = np.zeros_like(W_sb)
    for i in range(N):
        P = W_sb[i].T                                  # (n_tx, n_sb)
        Pp = [P[p * n_sp:(p + 1) * n_sp] for p in range(n_pol)]   # 편파별 (n_sp, n_sb)
        # 1) 공유 공간 빔 선택: 두 편파+서브밴드 합산 전력 상위 L 개.
        with np.errstate(all="ignore"):
            beam_pow = sum(np.sum(np.abs(B.conj().T @ Pp[p]) ** 2, axis=1)
                           for p in range(n_pol))      # (K,)
        s_sel = np.argsort(beam_pow)[-L:]
        W1 = B[:, s_sel]                               # (n_sp, L)
        # 2) 편파별 빔공간 채널 A_p = W1^H P_p (L × n_sb).
        with np.errstate(all="ignore"):
            A = [W1.conj().T @ Pp[p] for p in range(n_pol)]
        # 3) 공유 주파수 기저 선택: 두 편파 합산 전력 상위 M 개.
        Af = [A[p] @ Wf_full for p in range(n_pol)]    # 편파별 (L, n_sb)
        f_pow = sum(np.sum(np.abs(Af[p]) ** 2, axis=0) for p in range(n_pol))  # (n_sb,)
        f_sel = np.argsort(f_pow)[-M:]
        Wf_sel = Wf_full[:, f_sel]                     # (n_sb, M)
        C = np.stack([Af[p][:, f_sel] for p in range(n_pol)])    # (n_pol, L, M)
        # 4) K0 절단 + 양자화 (전체 P·L·M 계수에 대해, 공통 reference).
        Cf = C.reshape(-1)
        amp = np.abs(Cf)
        keep = np.argsort(amp)[-K0:]
        ref = int(keep[int(np.argmax(amp[keep]))])
        amp_ref = amp[ref] + 1e-12
        ph_ref = np.angle(Cf[ref])
        Cq = np.zeros_like(Cf)
        for j in keep:
            j = int(j)
            if j == ref:
                Cq[j] = 1.0
                continue
            a_q = amp_levels[np.argmin(np.abs(amp_levels - amp[j] / amp_ref))]
            ph = np.angle(Cf[j]) - ph_ref
            ph_q = np.round(ph / (2 * np.pi) * n_ph) / n_ph * 2 * np.pi
            Cq[j] = a_q * np.exp(1j * ph_q)
        Cq = Cq.reshape(n_pol, L, M)
        # 5) 편파별 재구성 후 스택, 서브밴드별 unit-norm 정규화.
        P_rec = np.zeros((n_tx, n_sb), dtype=complex)
        with np.errstate(all="ignore"):
            for p in range(n_pol):
                P_rec[p * n_sp:(p + 1) * n_sp] = W1 @ (Cq[p] @ Wf_sel.conj().T)
        norms = np.linalg.norm(P_rec, axis=0, keepdims=True) + 1e-12
        W_hat[i] = (P_rec / norms).T                   # (n_sb, n_tx)

    # 6) 비트 회계 — 빔/주파수 기저 선택 + 비트맵(P·L·M) + 강계수 지시 + 계수 비트.
    from math import comb, log2, ceil
    beam_bits = int(ceil(log2(comb(K, L)))) if comb(K, L) > 1 else 0
    freq_bits = int(ceil(log2(comb(n_sb, M)))) if M < n_sb else 0
    bitmap_bits = n_coeff
    strongest_bits = int(ceil(log2(K0))) if K0 > 1 else 0
    coeff_bits = (K0 - 1) * (amp_bits + phase_bits)
    bits = int(beam_bits + freq_bits + bitmap_bits + strongest_bits + coeff_bits)
    return W_hat, bits
