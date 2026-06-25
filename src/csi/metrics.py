"""
csi.metrics — Evaluation metrics & precoder extraction (pure NumPy).
====================================================================

WHAT THIS MODULE DOES
    The numbers 3GPP TR 38.843 uses to score CSI compression:
      * dominant_eigenvector : the rank-1 precoder fed back (cf. Type II)
      * sgcs / gcs           : (Squared) Generalized Cosine Similarity  <- primary KPI
      * nmse_db              : Normalized MSE in dB
      * cosine_rho           : CsiNet per-subcarrier correlation

WHY SGCS IS THE KPI (3GPP TR 38.843, Rel-18 "AI/ML for NR air interface")
    TR 38.843 studies a *two-sided* model for the CSI-compression sub-use-case:
    a UE-side neural ENCODER compresses the measured downlink channel into a
    small feedback payload, and a gNB-side DECODER reconstructs it. The study
    item's agreed *intermediate* KPI for this sub-use-case is the
    Squared Generalized Cosine Similarity (SGCS) between the eigenvector
    (precoder) derived from the ground-truth channel and the eigenvector
    derived from the reconstructed channel. Evaluation methodology in TR 38.843
    mandates comparison against a *non-AI baseline* (the TS 38.214 Type II /
    eType II codebook PMI — see csi.baselines) plus complexity reporting
    (FLOPs / parameter count — see csi.models.model_complexity).

    SGCS measures *precoder subspace alignment*, NOT raw channel reconstruction
    error. What matters for throughput is whether the beamforming direction the
    gNB applies (the dominant right-singular vector of H, i.e. the rank-1
    precoder) points the same way as the true one. Because beamforming gain is
    insensitive to a global phase/scale on the precoder, the metric is made
    phase- and scale-invariant (see sgcs()). NMSE (channel-domain MSE) and
    cosine_rho (per-subcarrier correlation, the original CsiNet metric) are
    reported as *secondary* diagnostics; they penalise errors that do not
    affect the precoder direction and so correlate only loosely with SGCS.

    Typical eType II operating points (TS 38.214 §5.2.2.2.5, L spatial beams in
    {2,4}, FD basis count M, amplitude/phase quantization) land at SGCS ~0.6-0.9
    for L in {2,4} and beta (NZC ratio) in {1/4, 1/2, 3/4}; an AI codec is
    judged "useful" when it beats that baseline at equal or lower feedback bits.

ARRAY-SHAPE CONTRACT
    Channel H        : (N, n_sub, n_tx) complex  — N samples, n_sub subcarriers
                       (or FD units), n_tx gNB Tx antenna ports.
    Precoder w       : (N, n_tx)        complex  — one rank-1 precoder per sample
                       (or per (sample, subband) slice when called by
                       csi.baselines.sgcs_subband).
    All metric functions take (truth, prediction) and return a Python float so
    the notebook can treat them as interchangeable scorers.

REALISTIC PARAMETER EXAMPLE (FR1 n78 macro cell)
    Carrier 3.5 GHz, SCS 30 kHz, 100 MHz -> 273 RB -> 273*12 = 3276 subcarriers;
    gNB 32 ports (8x2 cross-pol panel), UE 4 Rx; CDL-C (TR 38.901 §7.7.1-3),
    delay spread 100 ns (nominal), UE 30 km/h (8.3 m/s). One scoring call:

        H_true = csi.generate_csi_dataset(2000)        # (2000, 3276, 32)
        H_hat  = decode(encode(H_true))                # reconstructed channel
        w_t    = csi.dominant_eigenvector(H_true)      # (2000, 32)
        w_p    = csi.dominant_eigenvector(H_hat)       # (2000, 32)
        kpi    = csi.sgcs(w_t, w_p)                     # -> ~0.6..0.9

HOW TO SWAP THIS MODULE
    Add new metrics with the same (truth, prediction) -> float signature; the
    notebook treats them as interchangeable scorers.
"""
from __future__ import annotations
import numpy as np


def nmse_db(H_true: np.ndarray, H_pred: np.ndarray) -> float:
    """Normalized Mean-Squared Error in dB (secondary, channel-domain KPI).

        NMSE_dB = 10 * log10( mean_n[ ||H_n - Hhat_n||^2 / ||H_n||^2 ] )

    This is the per-sample-normalised variant: the error energy of each sample
    is divided by that sample's own channel energy *before* averaging, so a few
    high-power samples cannot dominate the mean. Lower (more negative) is
    better; 0 dB means the reconstruction error has the same energy as the
    channel itself. Reported as a diagnostic alongside the SGCS KPI of
    TR 38.843 — it penalises reconstruction errors that may not change the
    precoder direction, so it does not always track SGCS.

    Parameters
    ----------
    H_true, H_pred : np.ndarray, complex, shape (N, ...) with matching shapes.
        Norm/sum is taken over every axis except axis 0 (the sample axis), so
        this works for (N, n_sub, n_tx) channels or any (N, ...) tensor.

    Returns
    -------
    float : NMSE in dB.
    """
    # All non-sample axes are reduced; axis 0 stays as the per-sample axis.
    axes = tuple(range(1, H_true.ndim))
    # ||H - Hhat||^2 per sample (|.|^2 handles complex magnitude correctly).
    num = np.sum(np.abs(H_true - H_pred) ** 2, axis=axes)
    # ||H||^2 per sample; +1e-12 guards against division by zero (all-zero sample).
    den = np.sum(np.abs(H_true) ** 2, axis=axes) + 1e-12
    return float(10 * np.log10(np.mean(num / den)))


def cosine_rho(H_true: np.ndarray, H_pred: np.ndarray) -> float:
    """CsiNet correlation rho: mean per-subcarrier |h^H hhat| / (||h|| ||hhat||).

    This is the correlation metric from the original CsiNet line of work: for
    each subcarrier it computes the normalised inner product between the true
    and reconstructed Tx-antenna response vectors, then averages over samples
    and subcarriers. It is bounded in [0, 1] (magnitude makes it phase-blind per
    subcarrier) and reported as a secondary diagnostic, distinct from the
    TR 38.843 SGCS KPI which works on the *eigenvector precoder*, not raw
    per-subcarrier channels.

    Parameters
    ----------
    H_true, H_pred : np.ndarray, complex, shape (N, n_sub, n_tx).
        The inner product / norm is taken over the last axis (n_tx), so it is
        evaluated independently per (sample, subcarrier).

    Returns
    -------
    float : mean correlation in [0, 1] (averaged over samples and subcarriers).
    """
    # |h^H hhat| per (sample, subcarrier): conjugate true, dot over n_tx, abs.
    num = np.abs(np.sum(np.conj(H_true) * H_pred, axis=-1))
    # ||h|| ||hhat|| per (sample, subcarrier); norm over the n_tx axis.
    # np.linalg.norm on complex input returns the real 2-norm. +1e-12 guards /0.
    den = np.linalg.norm(H_true, axis=-1) * np.linalg.norm(H_pred, axis=-1) + 1e-12
    return float(np.mean(num / den))


def dominant_eigenvector(H: np.ndarray) -> np.ndarray:
    """Rank-1 precoder: the top eigenvector of the spatial covariance R = H^H H.

    For each sample we form the n_tx x n_tx spatial covariance R = H^H H (summed
    over subcarriers) and take the eigenvector of its largest eigenvalue. That
    eigenvector is the dominant right-singular vector of H, i.e. the rank-1
    beamforming/precoding direction the gNB would apply. This is the channel
    quantity that the TR 38.843 SGCS KPI is computed on, and it is the same
    rank-1 precoder concept underlying the TS 38.214 Type II / eType II codebook
    PMI (§5.2.2.2.3 / §5.2.2.2.5) used as the non-AI baseline in csi.baselines.

    Parameters
    ----------
    H : np.ndarray, complex, shape (N, n_sub, n_tx).
        n_sub is the number of subcarriers / FD units to average the covariance
        over (use a single-subband slice for per-subband precoders — see
        csi.baselines.subband_precoders).

    Returns
    -------
    np.ndarray : complex128, shape (N, n_tx). Each row is a unit-norm precoder
        (eigh returns normalised eigenvectors). The global phase is arbitrary,
        which is fine because SGCS is phase-invariant.
    """
    # Promote to float64-backed complex for a stable, well-conditioned eigh.
    H = H.astype(np.complex128)
    out = np.zeros((H.shape[0], H.shape[-1]), dtype=np.complex128)
    with np.errstate(all="ignore"):
        for i in range(H.shape[0]):
            R = H[i].conj().T @ H[i]            # (n_tx, n_tx), Hermitian PSD
            # eigh is for Hermitian matrices and returns eigenvalues in ASCENDING
            # order, so the last column V[:, -1] is the dominant eigenvector.
            _, V = np.linalg.eigh(R)
            out[i] = V[:, -1]                   # eigenvector of the largest eigenvalue
    return out


def sgcs(w_true: np.ndarray, w_pred: np.ndarray) -> float:
    """Squared Generalized Cosine Similarity — the primary 3GPP TR 38.843 KPI.

        SGCS = E[ |w^H w_hat|^2 / (||w||^2 ||w_hat||^2) ]   in [0, 1]

    This is the agreed intermediate KPI for the CSI-compression sub-use-case of
    the two-sided model in TR 38.843: it scores how well the *precoder subspace*
    (the dominant-eigenvector beamforming direction) survives compression and
    reconstruction. The squared normalised inner product is BOTH phase-invariant
    (|.|^2 removes any e^{j*theta} on either vector) AND scale-invariant (the
    ||.||^2 denominators cancel any amplitude), so only the *direction* of the
    precoder matters — exactly the quantity that determines beamforming gain.

    SGCS = 1 means the predicted precoder is identical (up to phase/scale) to the
    true one; SGCS = 0 means they are orthogonal. Practical eType II / AI codec
    operating points sit around 0.6-0.9 (TS 38.214 §5.2.2.2.5).

    Parameters
    ----------
    w_true, w_pred : np.ndarray, complex, shape (N, n_tx).
        Rank-1 precoders, e.g. from dominant_eigenvector(). The inner product /
        norms are taken over the last axis (n_tx) and the result is averaged
        over the N samples (the E[.]).

    Returns
    -------
    float : SGCS in [0, 1]. (gcs() returns its square root.)
    """
    # |w^H w_hat|^2 per sample: conjugate truth, dot over n_tx, magnitude squared.
    num = np.abs(np.sum(np.conj(w_true) * w_pred, axis=-1)) ** 2
    # ||w||^2 ||w_hat||^2 per sample; +1e-12 guards against a zero precoder.
    den = (np.sum(np.abs(w_true) ** 2, axis=-1)
           * np.sum(np.abs(w_pred) ** 2, axis=-1)) + 1e-12
    return float(np.mean(num / den))


def gcs(w_true: np.ndarray, w_pred: np.ndarray) -> float:
    """Generalized Cosine Similarity (non-squared) = sqrt(SGCS), in [0, 1].

    Some TR 38.843 evaluation tables report the non-squared GCS instead of SGCS;
    it is monotonically related (GCS = sqrt(SGCS)) and given here for
    convenience. Inputs/shapes are identical to sgcs().
    """
    return float(np.sqrt(sgcs(w_true, w_pred)))
