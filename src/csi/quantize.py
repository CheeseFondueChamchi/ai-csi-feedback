"""
csi.quantize — Latent (codeword) quantization for a *fair* bit comparison.
==========================================================================

WHAT THIS MODULE DOES
    The autoencoder's codeword (the "latent" z) is a vector of floats; to send it
    over the air as CSI feedback it must become **bits**. 3GPP TR 38.843 (Rel-18
    AI/ML for the NR air interface) evaluates the CSI *compression* sub-use-case
    with a **two-sided model** — a UE-side encoder produces z, a gNB-side decoder
    reconstructs the channel — and reports **SGCS** (squared generalized cosine
    similarity) as the KPI, *against a non-AI baseline at matched overhead* plus a
    complexity report. A float latent has no well-defined bit cost, so to place the
    AI codec on the SAME SGCS-vs-bits axis as the PMI codebook baselines
    (TS 38.214 §5.2.2.2 Type I / Type II / eType II) we must turn z into a REAL
    operating point: quantize the latent, decode, then measure SGCS *after* decode.

    `LatentQuantizer` does **per-dimension uniform scalar quantization**: it learns
    each latent dimension's dynamic range [min, max] on the TRAIN set, then maps
    test latents to `bits` bits/dimension. With ``n_code`` latent dimensions:

        report length (bits) = n_code * bits

    This mirrors how the eType II PMI baseline spends bits on *coefficients*
    (TS 38.214 §5.2.2.2.5: 2L beams x M FD basis vectors, K0 non-zero-coefficient
    cap + bitmap, per-coefficient amplitude + QPSK/8PSK phase): both schemes end up
    as "N scalars, each quantized to a few bits", so the bit axis is comparable.

    Quantization scheme (per dimension d, independent of all other dimensions):
        1.  fit():  lo[d] = min over train, hi[d] = max over train.
        2.  clip z[d] to [lo[d], hi[d]]            (handles test-set outliers).
        3.  normalize to u = (z - lo) / (hi - lo)  in [0, 1].
        4.  L = 2**bits reconstruction levels, evenly spaced lo..hi inclusive.
        5.  q = round(u * (L - 1))                 integer code in {0, ..., L-1}.
        6.  reconstruct  z_hat = lo + q/(L-1) * (hi - lo).
    Step size is  (hi - lo) / (L - 1)  per dimension; the endpoints lo and hi are
    both representable (a "mid-tread" / endpoint-inclusive uniform quantizer).

PUBLIC API (the stable "contract")
    LatentQuantizer().fit(z_train).transform(z, bits) -> z_hat   (same shape as z)

EXAMPLE (realistic operating point)
    A TR 38.843 CSI-compression run at 3.5 GHz (FR1 n78), 30 kHz SCS, 100 MHz =
    273 RB, gNB 32 ports (8x2 dual-pol panel), CDL-C @ 300 ns delay spread,
    UE @ 3 km/h. The angular-delay channel is fed to a CsiNet with n_code = 128:

        net = csi.CsiNet(n_delay=32, n_tx=32, n_code=128)
        lq  = csi.LatentQuantizer().fit(z_train)     # z_train: (N, 128)
        z_q = lq.transform(z_test, bits=4)           # 4 bits/dim
        # report length = 128 * 4 = 512 bits  -> compare SGCS vs eType II at ~512b
        # eType II here: L=4 beams, beta=1/2 -> SGCS operating point ~0.6-0.9.

    A coarser point (bits=2) costs 128*2 = 256 bits; a finer point (bits=6) costs
    768 bits. Sweeping `bits` traces the AI codec's SGCS-vs-overhead curve.

NOTE
    This is *post-training* quantization (no fine-tuning), so it is an honest but
    pessimistic operating point — straight-through estimator / VQ-VAE fine-tuning
    would recover some SGCS by training the network to be robust to the rounding.
    See `obsidian_vault/02 - Math/Quantization and Feedback Overhead`.
"""
from __future__ import annotations
import numpy as np


class LatentQuantizer:
    """Per-dimension uniform scalar quantizer fitted on the training latent.

    State (set by ``fit``):
        lo : (n_code,) float — per-dimension minimum over the train set.
        hi : (n_code,) float — per-dimension maximum over the train set.
    Both are ``None`` until ``fit`` is called; ``transform`` raises if called first.
    """

    def __init__(self):
        # Per-dimension dynamic range; learned from the training latent in fit().
        self.lo = None  # (n_code,) lower bound per latent dimension
        self.hi = None  # (n_code,) upper bound per latent dimension

    def fit(self, z: np.ndarray) -> "LatentQuantizer":
        """Learn the per-dimension quantization range from training codewords.

        z : (N, n_code) array of training codewords (one row per sample). We take
            the min/max along axis 0 (over samples) so each of the ``n_code`` latent
            dimensions gets its OWN [lo, hi] interval — different dimensions of an
            autoencoder latent have very different scales, so a shared global range
            would waste levels on the wider dimensions. Returns self for chaining.
        """
        self.lo = z.min(axis=0)  # (n_code,) tightest lower bound seen in training
        self.hi = z.max(axis=0)  # (n_code,) tightest upper bound seen in training
        return self

    def transform(self, z: np.ndarray, bits: int) -> np.ndarray:
        """Quantize each dimension of z to `bits` bits (uniform over [lo, hi]).

        z    : (N, n_code) test codewords to quantize (any N; n_code must match fit).
        bits : bits per latent dimension -> 2**bits reconstruction levels. The total
               feedback cost of one report is ``n_code * bits`` (see module docstring).
        Returns z_hat with the SAME shape/semantics as z (float32), where every
        value has been snapped to the nearest of the per-dimension levels.
        """
        if self.lo is None or self.hi is None:
            raise RuntimeError("LatentQuantizer.transform called before fit()")
        # Need at least 2 levels to define a step size; with <2 levels the
        # reconstruction "(levels - 1)" denominator below would be 0 (div-by-zero)
        # and a single level cannot carry information. bits=1 (2 levels) is the
        # coarsest meaningful setting; the report's AI bit sweep uses bits in 2..4+.
        if bits < 1:
            raise ValueError(f"bits must be >= 1 (got {bits})")
        levels = 2 ** bits                  # number of reconstruction levels L
        # Guard against dead (constant) latent dimensions: if a dimension never
        # varied in training, hi == lo and the span is 0; floor it to a tiny
        # epsilon so the normalize/round step does not divide by zero. (Such a
        # dimension reconstructs to lo for any input, which is correct — it carries
        # no information — but we must avoid 0/0.)
        span = (self.hi - self.lo).copy()   # (n_code,) per-dim range, may be 0
        span[span < 1e-12] = 1e-12
        # Clip test values into the trained range so out-of-range test-set outliers
        # map to the nearest endpoint level rather than overflowing past lo/hi.
        zc = np.clip(z, self.lo, self.hi)
        # Normalize to [0,1], scale to [0, L-1], round to the nearest integer code.
        q = np.round((zc - self.lo) / span * (levels - 1))   # integer codes 0..L-1
        # Dequantize: map the integer code back to a real value on the lo..hi grid.
        return (self.lo + q / (levels - 1) * span).astype(np.float32)
