"""
csi.noise — Additive white Gaussian noise for CSI-estimation modeling.
======================================================================

WHY THIS MODULE EXISTS
    A real UE never sees the clean channel H; it estimates a noisy version
    Ĥ = H + n from the CSI-RS pilots. CSI feedback (PMI or AI) is computed on Ĥ,
    but the precoder is judged against the TRUE channel H. Lower SNR (or higher
    pathloss, which lowers the effective SNR) therefore degrades the reported
    CSI — the realistic effect this module injects.

    In the two-sided AI/ML framework of 3GPP TR 38.843 (Rel-18 AI/ML for the NR
    air interface, CSI-compression sub-use-case), the UE-side *encoder* compresses
    its estimate Ĥ into a latent message and the gNB-side *decoder* reconstructs
    Ĥ_rec. The end KPI is the Squared Generalized Cosine Similarity (SGCS, see
    csi.metrics) between the precoder derived from Ĥ_rec and the eigen-precoder of
    the clean H, evaluated against a non-AI baseline (eType II PMI, TS 38.214
    §5.2.2.2.5) with model-complexity reporting. Pilot/estimation noise modeled
    here is what separates an idealized study (train/score on clean H) from a
    realistic one (train/score on Ĥ), so this module is the single knob that turns
    "genie CSI" into "estimated CSI".

SNR CONVENTION
    SNR is defined *per the whole array* (array-wide / per-snapshot average over
    all Rx-Tx-subcarrier entries) as

        SNR = mean(|H|^2) / sigma^2,                                   (linear)
        SNR_dB = 10 * log10( mean(|H|^2) / sigma^2 ),                  (dB)

    where sigma^2 = E[|n|^2] is the *total* complex-noise variance per entry
    (real part + imaginary part). Inverting,

        sigma^2 = mean(|H|^2) / 10**(SNR_dB/10).

    Because the channel generators (csi.data.generate_csi_dataset and
    csi.sionna_data.generate_sionna_csi) normalize each snapshot to unit average
    power, mean(|H|^2) ≈ 1 and the noise variance collapses to

        sigma^2 = 10**(-SNR_dB/10),

    split equally across the real and imaginary parts: each part is i.i.d.
    N(0, sigma^2 / 2) so that E[|n|^2] = sigma^2 (circularly-symmetric complex
    Gaussian, CN(0, sigma^2)). This module measures mean(|H|^2) from the actual
    array rather than assuming 1.0, so it stays correct for un-normalized H too.

HOW PATHLOSS FOLDS INTO EFFECTIVE SNR
    Real link SNR is set by transmit power, pathloss, and the receiver noise
    floor. With a unit-average-power *small-scale* channel H_ss (what the
    generators emit), the physical channel is H = sqrt(g) * H_ss where g is the
    large-scale gain = 10**(-PL_dB/10) (pathloss PL_dB > 0 reduces g). The
    received SNR then scales as

        SNR_eff_dB = SNR_tx_dB - PL_dB,

    i.e. pathloss is a pure dB *offset* on the SNR. Because this function defines
    noise relative to the *measured* mean(|H|^2), you can model pathloss in either
    of two equivalent ways:
      (a) Keep H unit-power and simply lower snr_db by PL_dB (preferred — one knob,
          numerically identical effect on Ĥ since both signal and noise scale
          together in the ratio); or
      (b) Pre-scale H by sqrt(g) (so mean(|H|^2)=g) and pass the *transmit* SNR;
          mean(|H|^2) then carries the pathloss and sigma^2 grows accordingly.
    Both yield the same effective SNR = mean(|H|^2)/sigma^2 at the estimator, so a
    cell-edge UE at, say, SNR_tx=20 dB with PL margin 15 dB is modeled by
    snr_db≈5.

REALISTIC PARAMETER EXAMPLES (3GPP-style)
    FR1 mid-band, good coverage (3.5 GHz n78, 100 MHz, SCS 30 kHz -> 273 RB ->
    3276 subcarriers, gNB 32 ports = 8x2 dual-pol panel, UE 4 Rx, CDL-C 300 ns
    delay spread, 3 km/h):
        Hhat = csi.add_awgn(H, snr_db=20.0, rng=np.random.default_rng(0))
        # ~20 dB is a typical "good CSI" operating point; SGCS lands ~0.85-0.9.
    FR1 cell edge (same config, +15 dB pathloss margin):
        Hhat = csi.add_awgn(H, snr_db=5.0, rng=np.random.default_rng(0))
        # degraded estimate; eType II (L=4, beta=1/2) baseline SGCS drops ~0.6-0.7.
    FR2 (28 GHz, 100 MHz, SCS 120 kHz -> 66 RB, 64 ports): use snr_db in 0-15 dB.
    Effectively-noiseless "genie" estimate for upper-bound studies: snr_db=50.

USAGE
    Hhat = csi.add_awgn(H, snr_db=10.0, rng=np.random.default_rng(0))
    # compute PMI / AI feedback on Hhat, score SGCS against eigvec(H).
"""
from __future__ import annotations

import numpy as np


def add_awgn(H: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Return Ĥ = H + n with circularly-symmetric complex AWGN at the given SNR.

    Adds i.i.d. CN(0, sigma^2) noise to every entry of H, where sigma^2 is set so
    that the array-wide SNR = mean(|H|^2) / sigma^2 equals ``snr_db`` (see the
    module docstring for the convention and the pathloss-as-dB-offset relation).
    Models pilot/CSI-RS estimation error for the noisy-CSI study of
    3GPP TR 38.843 (two-sided UE-encoder / gNB-decoder CSI compression).

    Parameters
    ----------
    H : complex ndarray of any shape — the (clean) channel. Typical shape from the
        generators is (n_samples, n_sub, n_tx), e.g. (2000, 3276, 32) for a
        100 MHz / 273 RB / 32-port config. A real-valued H is promoted to complex
        so the imaginary noise component is preserved.
    snr_db : float — array-wide SNR = mean(|H|^2) / noise_var, in dB. Lower it by
        the pathloss margin PL_dB to model a worse-coverage UE. Use a large value
        (e.g. 50) for an effectively noiseless estimate.
    rng : np.random.Generator, optional — for reproducibility. A fresh default_rng
        is created if None (results then vary run to run).

    Returns
    -------
    Ĥ : complex ndarray, same shape as H. dtype matches H for complex inputs
        (e.g. complex64 in, complex64 out); real inputs are returned as the
        matching complex type (float32 -> complex64, float64 -> complex128).

    Notes
    -----
    * sigma^2 = mean(|H|^2) / 10**(snr_db/10); the per-component std is
      sqrt(sigma^2 / 2) so real and imag each carry sigma^2/2 and E[|n|^2]=sigma^2.
    * mean(|H|^2) is measured from H, so the SNR convention holds for both the
      unit-power channels (mean|H|^2≈1) and any pre-scaled / un-normalized array.
    """
    H = np.asarray(H)
    # Ensure a complex working/return dtype: a real-valued H would otherwise cause
    # the complex noise to be truncated to its real part on the final cast, halving
    # the noise power and breaking the SNR convention. Map real floats to the
    # matching complex width (float32->complex64, float64/other->complex128) and
    # leave complex inputs (e.g. complex64 from the generators) untouched.
    if not np.iscomplexobj(H):
        out_dtype = np.complex64 if H.dtype == np.float32 else np.complex128
    else:
        out_dtype = H.dtype
    if rng is None:
        rng = np.random.default_rng()
    # Array-wide average channel power, mean(|H|^2); ≈1 for unit-power snapshots.
    sig_power = float(np.mean(np.abs(H) ** 2))
    # Total complex-noise variance sigma^2 from the SNR = sig_power / sigma^2 def.
    noise_var = sig_power / (10.0 ** (snr_db / 10.0))
    # Circularly-symmetric complex Gaussian: real & imag are i.i.d. unit-variance,
    # then scaled by sqrt(sigma^2 / 2) so each carries sigma^2/2 and |n| carries sigma^2.
    n = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    n *= np.sqrt(noise_var / 2.0)
    return (H + n).astype(out_dtype)
