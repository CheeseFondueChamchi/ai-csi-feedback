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
                TS 38.214 §5.2.2.2.1.
    * Type II : linear combination of L strongest beams with quantised
                amplitude/phase coefficients (expensive, accurate).
                TS 38.214 §5.2.2.2.3.
    * eType II: Rel-16 enhanced Type II — adds a *frequency-domain* DFT basis
                (M vectors) on top of the L spatial beams, i.e. the 2D
                W1·C·Wf^H compression. TS 38.214 §5.2.2.2.5. This is the
                non-AI baseline the two-sided AI/ML CSI-compression model of
                TR 38.843 (UE encoder / gNB decoder) is benchmarked against,
                using SGCS as the intermediate KPI plus a bits / complexity
                budget.

WHY THIS IS THE TR 38.843 BASELINE
    TR 38.843 (Rel-18 "Study on AI/ML for NR air interface") evaluates the CSI
    *compression* sub-use-case with a two-sided model: a UE-side encoder emits a
    bit payload, a gNB-side decoder reconstructs the precoder. SGCS (squared
    generalised cosine similarity, see ``csi.sgcs``) is the agreed intermediate
    KPI, and results must be reported *against a non-AI baseline at matched
    feedback overhead* (typically Rel-16 eType II) together with model
    complexity (FLOPs / parameters). This module supplies that baseline so the
    learned codec is judged on the same bits-vs-SGCS axes.

PUBLIC API (the stable "contract")
    dft_codebook(n_tx, oversampling)            -> B            (n_tx, K) beams
    type1_pmi(W_true, n_tx, ...)                -> (W_hat, bits)
    type2_pmi(W_true, n_tx, L, ...)             -> (W_hat, bits)
    subband_precoders(H, n_sb)                  -> W_sb (N, n_sb, n_tx)
    sgcs_subband(W_true_sb, W_hat_sb)           -> float
    etype2_pmi_2d(W_sb, n_tx, L, M, ...)        -> (W_hat_sb, bits)

    W_true / W_hat are (N, n_tx) dominant eigenvectors; score them with
    ``csi.sgcs(W_true, W_hat)`` exactly like the learned codec — a fair,
    bits-vs-SGCS comparison. The eType II path works on per-subband precoders
    (N, n_sb, n_tx) and is scored with ``sgcs_subband``.

REALISTIC PARAMETER EXAMPLE (FR1 n78, 100 MHz, 32-port gNB)
    Scenario: 3.5 GHz carrier, 30 kHz SCS, 100 MHz -> 273 RB; CSI is reported
    over a 13-subband configuration; gNB has a 32-port dual-pol panel
    (8x2 cross-pol => 16 ports/pol); UE 4 Rx; nominal 100 ns delay spread,
    3 km/h (0.83 m/s) low mobility, CSI-RS every 5 slots. A typical Rel-16
    eType II operating point is L=4 spatial beams, M=2 FD basis vectors,
    beta=1/2 (so K0 = ceil(0.5 * 2L*M) non-zero coefficients per layer),
    3-bit amplitude + 3-bit (8PSK) phase, landing around SGCS ~0.6-0.9:

        H   = ...                                   # (N, 273*12 -> grouped, 32)
        W_sb = subband_precoders(H, n_sb=13)        # (N, 13, 32)
        W_hat, bits = etype2_pmi_2d(W_sb, n_tx=32, L=4, M=2,
                                    beta=0.5, dual_pol=True,
                                    amp_bits=3, phase_bits=3)
        sgcs_subband(W_sb, W_hat)                    # compare vs the AI codec

HOW TO SWAP THIS MODULE
    Replace with a Rel-17 further-enhanced Type II, a port-selection codebook,
    or the real 38.214 codebook tables; keep the (W_true, ...) -> (W_hat, bits)
    (and (W_sb, ...) -> (W_hat_sb, bits)) shapes.
"""
from __future__ import annotations
import numpy as np


def dft_codebook(n_tx: int, oversampling: int = 4) -> np.ndarray:
    """Oversampled DFT beam codebook for a ULA: columns are unit-norm beams.

    Returns B of shape (n_tx, n_tx * oversampling). This is the spatial-beam
    basis underlying the NR Type I / Type II codebooks (a ULA Type I PMI is
    essentially "which oversampled DFT beam"). In TS 38.214 §5.2.2.2 the spatial
    beams are oversampled 2D DFT vectors with oversampling factors O1, O2 (per
    panel dimension); here we model a single ULA dimension with one factor.

    Math
    ----
    Column k is the steering vector b_k[n] = exp(j 2π n k / K) / sqrt(n_tx),
    n = 0..n_tx-1, for K = n_tx * oversampling candidate pointing angles. The
    1/sqrt(n_tx) scaling makes each column unit-norm (||b_k|| = 1).

    Parameters
    ----------
    n_tx : number of (per-polarization) antenna ports along the ULA dimension.
    oversampling : DFT oversampling factor O (TS 38.214 uses O1=O2=4 typically),
        giving K = n_tx * oversampling beam candidates.

    Returns
    -------
    B : complex (n_tx, K) — each column a unit-norm DFT beam.

    Example
    -------
    A 16-port-per-pol ULA with O=4 gives a 64-beam grid::

        B = dft_codebook(16, oversampling=4)   # B.shape == (16, 64)
    """
    K = n_tx * oversampling
    n = np.arange(n_tx)[:, None]                      # (n_tx, 1) antenna index
    k = np.arange(K)[None, :]                         # (1, K) beam index
    # exp(j 2π n k / K): DFT phase ramp per antenna; /sqrt(n_tx) -> unit-norm cols.
    return np.exp(1j * 2 * np.pi * n * k / K) / np.sqrt(n_tx)   # (n_tx, K)


def type1_pmi(W_true: np.ndarray, n_tx: int, oversampling: int = 4):
    """Type-I-style PMI: report the single best DFT beam (rank-1, wideband).

    TS 38.214 §5.2.2.2.1 (Type I single-panel codebook): a rank-1 PMI selects
    one oversampled DFT beam from the grid. We model the codebook-selection step
    by maximising the beamforming power |w^H b_k|^2 over the beam candidates b_k.

    Parameters
    ----------
    W_true : complex (N, n_tx) — true dominant eigenvector / precoder per sample.
    n_tx   : number of Tx ports along the ULA dimension.
    oversampling : DFT oversampling factor (beam grid size K = n_tx*oversampling).

    Returns
    -------
    W_hat : (N, n_tx) the selected beam per sample (the reported precoder).
    bits  : feedback payload = ceil(log2(K)) (one beam index, no coefficients).
    """
    B = dft_codebook(n_tx, oversampling)              # (n_tx, K) beam grid
    with np.errstate(all="ignore"):                   # silence spurious macOS complex-matmul warnings
        # |w^H b_k|^2 = received beamforming power of beam k for each sample.
        corr = np.abs(W_true.conj() @ B) ** 2         # (N, K) beamforming power
    idx = corr.argmax(axis=1)                         # (N,) best beam per sample
    W_hat = B[:, idx].T                               # (N, n_tx) reported precoder
    # One beam index out of K candidates -> ceil(log2 K) bits of PMI payload.
    bits = int(np.ceil(np.log2(B.shape[1])))
    return W_hat, bits


def type2_pmi(W_true: np.ndarray, n_tx: int, L: int = 4, oversampling: int = 4,
              amp_bits: int = 3, phase_bits: int = 4):
    """Type-II-style PMI: quantised linear combination of L strongest beams.

    TS 38.214 §5.2.2.2.3 (Type II codebook): the precoder is a linear
    combination of L orthogonal DFT beams with per-beam quantised amplitude and
    phase. Here we project the eigenvector onto its L best DFT beams and
    scalar-quantise the L complex combining coefficients. This is a compact
    *wideband* stand-in for the Rel-16 Type II codebook (one combination for the
    whole band; the per-subband frequency structure is added by ``etype2_pmi_2d``).

    Quantisation model
    ------------------
    * amplitude: normalised to the strongest beam, uniformly quantised to
      ``2**amp_bits`` levels (the real codebook uses non-uniform sqrt(0.5)^k
      levels — see ``_REL16_AMP_3BIT`` and ``etype2_pmi_2d``).
    * phase: uniformly quantised to ``2**phase_bits`` points on the unit circle
      (e.g. phase_bits=2 -> QPSK, 3 -> 8PSK, as in 38.214).

    Parameters
    ----------
    W_true : complex (N, n_tx) — true precoder per sample.
    n_tx, oversampling : ULA size / DFT beam grid (K = n_tx*oversampling).
    L : number of combined beams (38.214 allows L in {2,4,6}).
    amp_bits, phase_bits : amplitude / phase quantiser resolutions.

    Returns
    -------
    W_hat : (N, n_tx) reconstructed (unit-norm) precoder per sample.
    bits  : L*ceil(log2(K)) beam indices + L*(amp_bits+phase_bits) coefficients.
            (Simplification: the strongest coefficient's normalised amplitude=1 /
            phase reference is not deducted here; see ``etype2_pmi_2d`` for the
            full Rel-16 reference-coefficient accounting.)
    """
    B = dft_codebook(n_tx, oversampling)              # (n_tx, K) beam grid
    with np.errstate(all="ignore"):                   # silence spurious macOS complex-matmul warnings
        corr = np.abs(W_true.conj() @ B)              # (N, K) per-beam magnitude
    n_amp, n_ph = 2 ** amp_bits, 2 ** phase_bits      # quantiser level counts
    W_hat = np.zeros_like(W_true)
    for i in range(len(W_true)):
        sel = np.argsort(corr[i])[-L:]                # L strongest beams (indices)
        Bl = B[:, sel]                                # (n_tx, L) selected beams
        c = Bl.conj().T @ W_true[i]                   # (L,) combining coeffs = B_l^H w
        # quantise amplitude (relative to the strongest beam) and phase.
        amp = np.abs(c); amp_n = amp / (amp.max() + 1e-12)   # normalise to strongest
        amp_q = np.round(amp_n * (n_amp - 1)) / (n_amp - 1)  # uniform amplitude grid
        # angle in (-π,π]; map to one of n_ph points on the unit circle.
        ph_q = np.round(np.angle(c) / (2 * np.pi) * n_ph) / n_ph * 2 * np.pi
        c_q = amp_q * np.exp(1j * ph_q)               # (L,) quantised coefficients
        w = Bl @ c_q                                  # (n_tx,) reconstructed precoder
        W_hat[i] = w / (np.linalg.norm(w) + 1e-12)    # unit-norm (SGCS is scale-free)
    # Bit budget: L beam indices (ceil(log2 K) each) + L amp/phase coefficient words.
    bits = int(L * np.ceil(np.log2(B.shape[1])) + L * (amp_bits + phase_bits))
    return W_hat, bits


# Rel-16 Type II amplitude code점 집합 (TS 38.214 Table 5.2.2.2.5-2/3 의 wideband 진폭).
# 3비트(8레벨)일 때 진폭 후보는 sqrt(0.5)^k 형태의 등비수열로 정의된다:
#   {1, 1/sqrt(2), 1/2, 1/(2*sqrt(2)), 1/4, 1/(4*sqrt(2)), 1/8, 0}
# 마지막 레벨이 0(=빔 사실상 비활성)인 것이 Rel-16 진폭 집합의 특징이다.
# (Rel-16 eType II amplitude alphabet, TS 38.214 §5.2.2.2.5: a geometric ladder
#  p_k = sqrt(0.5)**k with a terminal 0 level meaning "coefficient switched off".
#  Quantising amplitude *relative to the strongest (reference) coefficient* is the
#  exact 38.214 behaviour; etype2_pmi_2d snaps to the nearest of these levels.)
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

    A "subband" here is a contiguous group of subcarriers, mirroring the NR
    CSI subband concept (TS 38.214 §5.2.1.4: a subband is a set of contiguous
    PRBs whose size depends on BWP bandwidth). Frequency selectivity across
    subbands is exactly what the FD basis Wf in eType II exploits.

    Parameters
    ----------
    H : complex (N, n_sub, n_tx) — per-subcarrier channel (Rx-combined / SIMO).
    n_sb : number of subbands to split the n_sub subcarriers into.

    Returns
    -------
    W_sb : complex64 (N, n_sb, n_tx) — unit-norm precoder per (sample, subband).

    Example
    -------
    100 MHz / 30 kHz SCS (273 RB -> 3276 subcarriers) reported over 13 subbands::

        W_sb = subband_precoders(H, n_sb=13)   # (N, 13, n_tx)
    """
    from .metrics import dominant_eigenvector

    N, n_sub, n_tx = H.shape
    # Contiguous, near-equal subband edges over the subcarrier axis.
    edges = np.linspace(0, n_sub, n_sb + 1).astype(int)
    W_sb = np.zeros((N, n_sb, n_tx), dtype=np.complex64)
    for s in range(n_sb):
        a, b = int(edges[s]), int(edges[s + 1])
        # Dominant eigenvector of that subband's spatial covariance H^H H.
        W_sb[:, s, :] = dominant_eigenvector(H[:, a:b, :])
    return W_sb


def sgcs_subband(W_true_sb: np.ndarray, W_hat_sb: np.ndarray) -> float:
    """Mean SGCS across subbands (the per-subband precoder-feedback metric).

    SGCS (TR 38.843 intermediate KPI) is computed per subband and averaged, so
    the eType II baseline and the AI codec are scored on the same frequency axis.
    Inputs are (N, n_sb, n_tx); returns a scalar in [0, 1].
    """
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
    n_pol = 2 if dual_pol else 1                       # P = number of polarizations (1 or 2)
    n_sp = n_tx // n_pol                               # 편파당 공간 소자 수 (ports per pol)
    B = dft_codebook(n_sp, oversampling)              # (n_sp, K) per-pol 공간 빔 (W1 candidates)
    K = B.shape[1]                                     # spatial beam grid size
    M = min(M, n_sb)                                   # cannot pick more FD vectors than subbands
    # Total coefficient grid is the 2L*M (dual-pol) / L*M (single-pol) of §5.2.2.2.5.
    n_coeff = n_pol * L * M                             # P·L·M coefficients
    # K0 = ceil(beta * P·L·M): the non-zero-coefficient cap (38.214 K0 / beta).
    K0 = min(n_coeff, max(1, int(np.ceil(beta * n_coeff))))

    # Frequency-domain DFT basis Wf (orthonormal, n_sb x n_sb). Columns are the
    # candidate FD basis vectors over the N3=n_sb subbands (38.214 §5.2.2.2.5).
    nn = np.arange(n_sb)
    Wf_full = np.exp(-2j * np.pi * np.outer(nn, nn) / n_sb) / np.sqrt(n_sb)  # (n_sb, n_sb)

    n_ph = 2 ** phase_bits                             # phase points (3 bits -> 8PSK)
    # amplitude alphabet: exact Rel-16 sqrt(0.5)^k ladder at 3 bits, else uniform.
    amp_levels = _REL16_AMP_3BIT if amp_bits == 3 else np.linspace(0.0, 1.0, 2 ** amp_bits)

    W_hat = np.zeros_like(W_sb)
    for i in range(N):
        # P is the per-subband precoder matrix laid out as (ports, subbands).
        P = W_sb[i].T                                  # (n_tx, n_sb)
        # Split ports into polarization groups [pol-0 | pol-1] (3GPP port layout).
        Pp = [P[p * n_sp:(p + 1) * n_sp] for p in range(n_pol)]   # 편파별 (n_sp, n_sb)
        # 1) Shared spatial beams W1 (shared across both pols, TS 38.214 §5.2.2.2.5).
        #    Rel-16 requires the L beams to come from ONE *orthogonal* DFT subset, not
        #    the L globally-strongest oversampled beams: adjacent oversampled columns
        #    overlap (|<b,b'>| ~ 0.9), which double-counts power (captured fraction can
        #    exceed 1) and collapses the W1^H projection. The orthogonal subset for
        #    oversampling rotation q is columns {q, q+O, q+2O, ...}; pick the rotation
        #    whose top-L beams capture the most power.
        beam_pow = np.zeros(K)
        with np.errstate(all="ignore"):
            for p in range(n_pol):
                beam_pow += np.sum(np.abs(B.conj().T @ Pp[p]) ** 2, axis=1)  # (K,) per-beam power
        s_sel, best_pw = np.arange(L), -1.0   # placeholder; loop always runs (O>=1)
        for q in range(oversampling):
            cols = np.arange(q, K, oversampling)       # one orthogonal DFT subset (n_sp beams)
            sub = cols[np.argsort(beam_pow[cols])[-L:]]   # top-L within this rotation
            pw = float(beam_pow[sub].sum())
            if pw > best_pw:
                best_pw, s_sel = pw, sub
        W1 = B[:, s_sel]                               # (n_sp, L) orthogonal shared basis
        # 2) Per-pol beam-space channel A_p = W1^H P_p (project onto the L beams).
        with np.errstate(all="ignore"):
            A = [W1.conj().T @ Pp[p] for p in range(n_pol)]   # each (L, n_sb)
        # 3) Shared FD basis Wf: project A onto the orthonormal DFT basis and pick
        #    the M columns carrying the most power summed over both pols (shared M).
        Af = [A[p] @ Wf_full for p in range(n_pol)]    # 편파별 (L, n_sb) FD coeffs
        f_pow = sum(np.sum(np.abs(Af[p]) ** 2, axis=0) for p in range(n_pol))  # (n_sb,)
        f_sel = np.argsort(f_pow)[-M:]                 # indices of M FD basis vectors
        Wf_sel = Wf_full[:, f_sel]                     # (n_sb, M) shared FD basis
        # C[p] is the L×M coefficient matrix of pol p -> full grid is (n_pol, L, M).
        C = np.stack([Af[p][:, f_sel] for p in range(n_pol)])    # (n_pol, L, M)
        # 4) K0 truncation + quantisation over the full P·L·M grid, with a single
        #    strongest "reference" coefficient (amp=1, phase=0), §5.2.2.2.5.
        Cf = C.reshape(-1)                             # flatten the P·L·M grid
        amp = np.abs(Cf)
        keep = np.argsort(amp)[-K0:]                   # keep K0 strongest (bitmap = on)
        ref = int(keep[int(np.argmax(amp[keep]))])     # strongest coeff = reference
        amp_ref = amp[ref] + 1e-12                     # amplitudes are relative to this
        ph_ref = np.angle(Cf[ref])                     # phases are relative to this
        Cq = np.zeros_like(Cf)                          # un-kept coeffs stay 0 (bitmap off)
        for j in keep:
            j = int(j)
            if j == ref:
                Cq[j] = 1.0                            # reference: amp=1, phase=0
                continue
            # snap relative amplitude to the Rel-16 alphabet (sqrt(0.5)^k ladder).
            a_q = amp_levels[np.argmin(np.abs(amp_levels - amp[j] / amp_ref))]
            # relative phase quantised to one of n_ph points (QPSK/8PSK).
            ph = np.angle(Cf[j]) - ph_ref
            ph_q = np.round(ph / (2 * np.pi) * n_ph) / n_ph * 2 * np.pi
            Cq[j] = a_q * np.exp(1j * ph_q)
        Cq = Cq.reshape(n_pol, L, M)
        # 5) Reconstruct each pol as W1 · Cq_p · Wf_sel^H, then unit-norm per subband.
        P_rec = np.zeros((n_tx, n_sb), dtype=complex)
        with np.errstate(all="ignore"):
            for p in range(n_pol):
                # Cq[p] @ Wf_sel^H expands L×M coeffs back to L×n_sb; W1 @ ... -> ports.
                P_rec[p * n_sp:(p + 1) * n_sp] = W1 @ (Cq[p] @ Wf_sel.conj().T)
        norms = np.linalg.norm(P_rec, axis=0, keepdims=True) + 1e-12
        W_hat[i] = (P_rec / norms).T                   # (n_sb, n_tx) reconstructed

    # 6) Bit accounting (TS 38.214 §5.2.2.2.5 CSI Part 2 structure):
    #      beam_bits   : which L of K spatial beams      -> ceil(log2 C(K,L))
    #      freq_bits   : which M of n_sb FD basis vectors -> ceil(log2 C(n_sb,M))
    #      bitmap_bits : P·L·M-bit bitmap of which coeffs are non-zero (K0 of them)
    #      strongest   : index of the reference (strongest) coefficient
    #      coeff_bits  : (K0-1) amplitude+phase words (reference costs no amp/phase)
    from math import comb, log2, ceil
    # beam cost: choose L from one orthogonal subset of n_orth = K/O beams, plus
    # log2(O) bits to signal which oversampling rotation (q1/q2) the subset uses.
    n_orth = K // oversampling
    beam_bits = ((int(ceil(log2(comb(n_orth, L)))) if comb(n_orth, L) > 1 else 0)
                 + (int(ceil(log2(oversampling))) if oversampling > 1 else 0))
    freq_bits = int(ceil(log2(comb(n_sb, M)))) if M < n_sb else 0
    bitmap_bits = n_coeff                              # one bit per coefficient in the grid
    strongest_bits = int(ceil(log2(K0))) if K0 > 1 else 0
    coeff_bits = (K0 - 1) * (amp_bits + phase_bits)    # reference excluded
    bits = int(beam_bits + freq_bits + bitmap_bits + strongest_bits + coeff_bits)
    return W_hat, bits
