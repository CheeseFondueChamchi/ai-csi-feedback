"""
csi.transform — Angular-delay domain transform (pure NumPy, no ML deps).
=========================================================================

WHAT THIS MODULE DOES
    Moves the channel between the **spatial-frequency** domain (what the UE
    measures) and the **angular-delay** domain (where the channel is sparse and
    therefore compressible), plus tiny real<->complex helpers for feeding CNNs.

PUBLIC API (the stable "contract" other modules rely on)
    to_angular_delay(H, n_delay)   -> H_ad      # forward 2D-DFT + truncation
    from_angular_delay(H_ad, n_sub)-> H          # inverse (zero-pads)
    complex_to_real_imag(H_ad)     -> x          # (N,D,T) -> (N,2,D,T)
    real_imag_to_complex(x)        -> H_ad        # inverse of the above

CONVENTIONS
    H      : complex array (..., n_sub, n_tx)   subcarriers x Tx-antennas
    H_ad   : complex array (..., n_delay, n_tx) delay-taps x angular-bins

HOW TO SWAP THIS MODULE
    The DFT basis is only optimal for an ideal half-wavelength ULA. To use a
    learned/other sparsifying basis, replace these two functions but keep the
    same shapes and the round-trip property `from_(to_(H)) ~= H`.
"""
from __future__ import annotations
import numpy as np


def to_angular_delay(H: np.ndarray, n_delay: int) -> np.ndarray:
    """2D DFT to the angular-delay domain, then keep the first ``n_delay`` taps.

    H_ad = F_delay^H @ H @ F_angle, truncated to the leading delay taps where
    the channel energy concentrates.
    """
    H_delay = np.fft.ifft(H, axis=-2)      # subcarrier -> delay
    H_ad = np.fft.fft(H_delay, axis=-1)    # antenna    -> angular
    return H_ad[..., :n_delay, :]


def from_angular_delay(H_ad: np.ndarray, n_sub: int) -> np.ndarray:
    """Inverse of :func:`to_angular_delay` (zero-pads the truncated delay taps)."""
    n_delay = H_ad.shape[-2]
    shape = list(H_ad.shape)
    shape[-2] = n_sub
    H_delay = np.zeros(shape, dtype=np.complex128)
    H_delay[..., :n_delay, :] = H_ad
    H = np.fft.ifft(H_delay, axis=-1)      # angular -> antenna
    H = np.fft.fft(H, axis=-2)             # delay   -> subcarrier
    return H


def complex_to_real_imag(H_ad: np.ndarray) -> np.ndarray:
    """Split complex ``(N, D, T)`` into a real 2-channel tensor ``(N, 2, D, T)``."""
    return np.stack([H_ad.real, H_ad.imag], axis=1).astype(np.float32)


def real_imag_to_complex(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`complex_to_real_imag`: ``(N, 2, D, T)`` -> complex ``(N, D, T)``."""
    return x[:, 0] + 1j * x[:, 1]
