"""
csi.quantize — Latent (codeword) quantization for a *fair* bit comparison.
==========================================================================

WHAT THIS MODULE DOES
    The autoencoder's codeword is a vector of floats; to send it over the air it
    must become **bits**. To compare the AI codec against the PMI codebook on the
    same SGCS-vs-bits axis, the AI report length must be a REAL operating point:
    quantize the latent, then measure SGCS *after* decode.

    `LatentQuantizer` does **per-dimension uniform scalar quantization**: it fits
    each latent dimension's [min, max] on the TRAIN set, then maps test latents
    to `bits` bits/dimension.

        report length (bits) = n_code * bits

PUBLIC API (the stable "contract")
    LatentQuantizer().fit(z_train).transform(z, bits) -> z_hat (same shape)

NOTE
    This is *post-training* quantization (no fine-tuning), so it is an honest but
    pessimistic operating point — straight-through / VQ fine-tuning would recover
    some SGCS. See `obsidian_vault/02 - Math/Quantization and Feedback Overhead`.
"""
from __future__ import annotations
import numpy as np


class LatentQuantizer:
    """Per-dimension uniform scalar quantizer fitted on the training latent."""

    def __init__(self):
        self.lo = None
        self.hi = None

    def fit(self, z: np.ndarray) -> "LatentQuantizer":
        """z: (N, n_code) training codewords."""
        self.lo = z.min(axis=0)
        self.hi = z.max(axis=0)
        return self

    def transform(self, z: np.ndarray, bits: int) -> np.ndarray:
        """Quantize each dimension of z to `bits` bits (uniform over [lo, hi])."""
        if self.lo is None or self.hi is None:
            raise RuntimeError("LatentQuantizer.transform called before fit()")
        levels = 2 ** bits
        span = (self.hi - self.lo).copy()
        span[span < 1e-12] = 1e-12
        zc = np.clip(z, self.lo, self.hi)
        q = np.round((zc - self.lo) / span * (levels - 1))
        return (self.lo + q / (levels - 1) * span).astype(np.float32)
