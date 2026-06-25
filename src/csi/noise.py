"""
csi.noise — Additive white Gaussian noise for CSI-estimation modeling.
======================================================================

WHY THIS MODULE EXISTS
    A real UE never sees the clean channel H; it estimates a noisy version
    Ĥ = H + n from the CSI-RS pilots. CSI feedback (PMI or AI) is computed on Ĥ,
    but the precoder is judged against the TRUE channel H. Lower SNR (or higher
    pathloss, which lowers the effective SNR) therefore degrades the reported
    CSI — the realistic effect this module injects.

SNR CONVENTION
    SNR is defined per the whole array as
        SNR = mean(|H|^2) / sigma^2,
    so for the unit-average-power channels produced by the generators
    (mean|H|^2 = 1) the noise variance is simply sigma^2 = 10**(-SNR_dB/10),
    split equally across the real and imaginary parts.

USAGE
    Hhat = csi.add_awgn(H, snr_db=10.0, rng=np.random.default_rng(0))
    # compute PMI / AI feedback on Hhat, score SGCS against eigvec(H).
"""
from __future__ import annotations

import numpy as np


def add_awgn(H: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Return Ĥ = H + n with complex AWGN at the given SNR (dB).

    Parameters
    ----------
    H : complex array of any shape — the (clean) channel.
    snr_db : float — SNR = mean(|H|^2) / noise_var, in dB. Use a large value
        (e.g. 50) for an effectively noiseless estimate.
    rng : np.random.Generator, optional — for reproducibility.
    """
    H = np.asarray(H)
    if rng is None:
        rng = np.random.default_rng()
    sig_power = float(np.mean(np.abs(H) ** 2))
    noise_var = sig_power / (10.0 ** (snr_db / 10.0))
    n = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    n *= np.sqrt(noise_var / 2.0)
    return (H + n).astype(H.dtype)
