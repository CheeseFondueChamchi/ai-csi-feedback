"""
csi.metrics — Evaluation metrics & precoder extraction (pure NumPy).
====================================================================

WHAT THIS MODULE DOES
    The numbers 3GPP TR 38.843 uses to score CSI compression:
      * dominant_eigenvector : the rank-1 precoder fed back (cf. Type II)
      * sgcs / gcs           : (Squared) Generalized Cosine Similarity  <- primary KPI
      * nmse_db              : Normalized MSE in dB
      * cosine_rho           : CsiNet per-subcarrier correlation

PUBLIC API (the stable "contract")
    dominant_eigenvector(H)        -> w          (N, n_tx)
    sgcs(w_true, w_pred)           -> float in [0,1]
    gcs(w_true, w_pred)            -> float in [0,1]   (= sqrt of SGCS)
    nmse_db(H_true, H_pred)        -> float (dB)
    cosine_rho(H_true, H_pred)     -> float in [0,1]

HOW TO SWAP THIS MODULE
    Add new metrics with the same (truth, prediction) -> float signature; the
    notebook treats them as interchangeable scorers.
"""
from __future__ import annotations
import numpy as np


def nmse_db(H_true: np.ndarray, H_pred: np.ndarray) -> float:
    """Normalized MSE in dB:  10 log10( E||H-Hhat||^2 / E||H||^2 )."""
    axes = tuple(range(1, H_true.ndim))
    num = np.sum(np.abs(H_true - H_pred) ** 2, axis=axes)
    den = np.sum(np.abs(H_true) ** 2, axis=axes) + 1e-12
    return float(10 * np.log10(np.mean(num / den)))


def cosine_rho(H_true: np.ndarray, H_pred: np.ndarray) -> float:
    """CsiNet correlation: mean per-subcarrier |h^H hhat| / (|h||hhat|).

    H arrays have shape (N, n_sub, n_tx); averaged over samples and subcarriers.
    """
    num = np.abs(np.sum(np.conj(H_true) * H_pred, axis=-1))
    den = np.linalg.norm(H_true, axis=-1) * np.linalg.norm(H_pred, axis=-1) + 1e-12
    return float(np.mean(num / den))


def dominant_eigenvector(H: np.ndarray) -> np.ndarray:
    """Top eigenvector of the spatial covariance R = H^H H (the rank-1 precoder).

    H has shape (N, n_sub, n_tx); returns (N, n_tx) complex.
    """
    H = H.astype(np.complex128)
    out = np.zeros((H.shape[0], H.shape[-1]), dtype=np.complex128)
    with np.errstate(all="ignore"):
        for i in range(H.shape[0]):
            R = H[i].conj().T @ H[i]            # (n_tx, n_tx), Hermitian PSD
            _, V = np.linalg.eigh(R)
            out[i] = V[:, -1]                   # eigenvector of the largest eigenvalue
    return out


def sgcs(w_true: np.ndarray, w_pred: np.ndarray) -> float:
    """Squared Generalized Cosine Similarity (primary 3GPP intermediate KPI).

        SGCS = E[ |w^H w_hat|^2 / (||w||^2 ||w_hat||^2) ] in [0, 1]

    Phase- and scale-invariant: only the *direction* of the precoder matters.
    Inputs are (N, n_tx) complex eigenvectors.
    """
    num = np.abs(np.sum(np.conj(w_true) * w_pred, axis=-1)) ** 2
    den = (np.sum(np.abs(w_true) ** 2, axis=-1)
           * np.sum(np.abs(w_pred) ** 2, axis=-1)) + 1e-12
    return float(np.mean(num / den))


def gcs(w_true: np.ndarray, w_pred: np.ndarray) -> float:
    """Generalized Cosine Similarity (non-squared) = sqrt(SGCS)."""
    return float(np.sqrt(sgcs(w_true, w_pred)))
