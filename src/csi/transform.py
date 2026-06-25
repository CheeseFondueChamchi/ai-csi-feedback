"""
csi.transform — Angular-delay domain transform (pure NumPy, no ML deps).
=========================================================================

WHAT THIS MODULE DOES
    Moves the channel between the **spatial-frequency** domain (what the UE
    measures) and the **angular-delay** domain (where the channel is sparse and
    therefore compressible), plus tiny real<->complex helpers for feeding CNNs.

WHY THE ANGULAR-DELAY DOMAIN (the whole point of this file)
    A downlink CSI matrix H has shape (n_sub, n_tx): one complex coefficient per
    (subcarrier, Tx-antenna-port). Physically the channel is a sum of only a few
    multipath clusters (see TR 38.901 §7.7.1 CDL tables: CDL-A has 23 clusters,
    CDL-C 24, CDL-E 14 — small relative to n_sub*n_tx coefficients). Each cluster
    arrives at:
        * a specific *delay*   tau_l  -> a linear phase ramp ACROSS SUBCARRIERS
          (a tone e^{-j 2*pi*f*tau_l} in the frequency direction), and
        * a specific *angle*   AOD_l  -> a linear phase ramp ACROSS ANTENNAS
          (the ULA array-steering phase e^{-j*pi*sin(AOD_l)*n} for half-lambda
          spacing; cf. csi.data.ula_steering).
    A 1-D DFT turns a single tone into a single bin (a delta). So:
        * IDFT along the subcarrier axis collapses each path's frequency ramp to
          a localized *delay tap*, and
        * DFT along the antenna axis collapses each path's spatial ramp to a
          localized *angular bin*.
    The result H_ad concentrates almost all energy into a handful of
    (delay, angle) bins near the origin -> the matrix is approximately sparse,
    which is exactly what a neural encoder (UE side) or a Type II codebook can
    exploit. This is the structural prior that makes CSI feedback compression
    work at all (TR 38.843, CSI compression sub-use-case, two-sided model:
    UE-side encoder + gNB-side decoder; SGCS is the KPI, see csi.metrics).

    The 2-D DFT basis is *exactly* the basis TS 38.214 §5.2.2.2 eType II uses:
    L spatial DFT beams (angular axis) x M frequency-domain DFT basis vectors
    (delay axis), with 2L coefficient groups for the two polarizations. Here we
    use the dense, untruncated DFT and instead keep only the leading delay taps
    (n_delay), which is the AI/ML analogue of eType II's K0 non-zero-coefficient
    cap + bitmap (TS 38.214 §5.2.2.2.5).

SPARSITY vs DELAY SPREAD (how many taps to keep)
    The number of significant leading delay taps scales with delay-spread x
    bandwidth. With subcarrier spacing the delay resolution is 1/(n_sub*df) and
    the unambiguous delay window is 1/df. For a 100 MHz FR1 carrier at 30 kHz
    SCS, 273 RB -> n_sub = 273*12 = 3276 subcarriers; df = 30 kHz gives a delay
    bin of ~1/(3276*30kHz) ~= 10 ns, so a 300 ns "long" delay spread (TR 38.901
    §7.7.3) occupies on the order of ~30 leading taps. In practice n_delay in
    {16, 32, 48} captures the bulk of the energy; the rest is discarded as the
    compression's first (lossy) step.

PUBLIC API (the stable "contract" other modules rely on)
    to_angular_delay(H, n_delay)   -> H_ad      # forward 2D-DFT + truncation
    from_angular_delay(H_ad, n_sub)-> H          # inverse (zero-pads)
    complex_to_real_imag(H_ad)     -> x          # (N,D,T) -> (N,2,D,T)
    real_imag_to_complex(x)        -> H_ad        # inverse of the above

CONVENTIONS
    H      : complex array (..., n_sub, n_tx)   subcarriers x Tx-antennas
    H_ad   : complex array (..., n_delay, n_tx) delay-taps x angular-bins
             (last axis length is n_tx: a full antenna DFT produces exactly n_tx
             angular bins, one per Tx port; nothing is dropped on the angular
             axis here, only on the delay axis.)

REALISTIC EXAMPLE CONFIG (FR1 n78, the workhorse mid-band scenario)
    Carrier 3.5 GHz, SCS 30 kHz, 100 MHz / 273 RB, gNB 32 ports (8x2 dual-pol
    panel), UE speed 3 km/h, CDL-C with 300 ns delay spread:
        H   = csi.generate_csi_dataset(2000)        # (N, n_sub, n_tx)
        Xad = csi.complex_to_real_imag(
                  csi.to_angular_delay(H, n_delay=32))   # keep 32 delay taps
    The eType II baseline for this point would use L=4 spatial beams,
    beta=1/2 coefficient ratio, targeting an SGCS operating point ~0.6-0.9
    (TS 38.214 §5.2.2.2.5; TR 38.843 eval methodology vs the non-AI baseline
    plus encoder/decoder complexity reporting).

HOW TO SWAP THIS MODULE
    The DFT basis is only optimal for an ideal half-wavelength ULA. To use a
    learned/other sparsifying basis, replace these two functions but keep the
    same shapes and the round-trip property `from_(to_(H)) ~= H`.
"""
from __future__ import annotations
import numpy as np


def to_angular_delay(H: np.ndarray, n_delay: int) -> np.ndarray:
    """Forward 2-D DFT into the angular-delay domain, then keep ``n_delay`` taps.

    Pipeline (each path's phase ramp -> a localized bin; see module docstring):
        1. IDFT over the subcarrier axis (-2): frequency tone -> delay tap.
        2. DFT  over the antenna   axis (-1): ULA steering phase -> angular bin.
    Symbolically  H_ad = F_delay^H @ H @ F_angle  (F_delay^H is the IDFT, the
    Hermitian/conjugate of the DFT; F_angle is the DFT), followed by truncation
    to the leading ``n_delay`` taps, where the channel energy concentrates.

    NOTE on DFT direction: which of fft/ifft is "forward" is a pure convention;
    the only requirement is that :func:`from_angular_delay` applies the exact
    inverse pair so the round trip is identity (np.fft scales ifft by 1/N).

    Args:
        H:       complex array (..., n_sub, n_tx). Spatial-frequency CSI:
                 subcarriers x Tx antenna ports. The leading ``...`` axes
                 (e.g. a batch dimension N) are untouched.
        n_delay: number of leading delay taps to keep (int, 0 < n_delay <= n_sub).
                 This truncation is the first, lossy compression step (analogous
                 to the eType II K0 non-zero-coefficient cap, TS 38.214
                 §5.2.2.2.5). Typical values: 16 / 32 / 48.

    Returns:
        H_ad: complex array (..., n_delay, n_tx) in the angular-delay domain.
              dtype follows NumPy's FFT promotion (complex128 for real/complex128
              input, complex64 preserved for complex64 input).
    """
    H_delay = np.fft.ifft(H, axis=-2)      # subcarrier -> delay  (IDFT)
    H_ad = np.fft.fft(H_delay, axis=-1)    # antenna    -> angular (DFT)
    return H_ad[..., :n_delay, :]


def from_angular_delay(H_ad: np.ndarray, n_sub: int) -> np.ndarray:
    """Inverse of :func:`to_angular_delay` (zero-pads the truncated delay taps).

    Steps mirror the forward transform in reverse order, each operation being the
    exact inverse of its counterpart so that ``from_angular_delay(to_angular_delay
    (H, n_delay), n_sub) ~= H`` (exactly, up to float round-off, when
    ``n_delay == n_sub``; otherwise H is the band-limited reconstruction that
    drops the discarded trailing delay taps):
        0. Zero-pad the delay axis from n_delay back up to n_sub. Because
           :func:`to_angular_delay` keeps the *leading* taps, the discarded taps
           sit at indices [n_delay:], so zeros are appended there — this is the
           correct placement (the dropped high-delay taps are assumed ~0).
        1. IDFT over the antenna axis (-1): angular bin -> ULA steering phase.
           (inverse of the forward DFT on that axis)
        2. DFT  over the delay   axis (-2): delay tap   -> frequency tone.
           (inverse of the forward IDFT on that axis)

    Args:
        H_ad:  complex array (..., n_delay, n_tx) angular-delay CSI.
        n_sub: target number of subcarriers to reconstruct (>= n_delay).

    Returns:
        H: complex array (..., n_sub, n_tx) spatial-frequency CSI (complex128).
    """
    n_delay = H_ad.shape[-2]
    shape = list(H_ad.shape)
    shape[-2] = n_sub                       # restore full subcarrier count
    H_delay = np.zeros(shape, dtype=np.complex128)
    H_delay[..., :n_delay, :] = H_ad        # leading taps kept, rest = 0
    H = np.fft.ifft(H_delay, axis=-1)      # angular -> antenna    (inverse DFT)
    H = np.fft.fft(H, axis=-2)             # delay   -> subcarrier (inverse IDFT)
    return H


def complex_to_real_imag(H_ad: np.ndarray) -> np.ndarray:
    """Split complex ``(N, D, T)`` into a real 2-channel tensor ``(N, 2, D, T)``.

    Neural CSI codecs (e.g. CsiNet, TR 38.843 two-sided model) operate on real
    tensors, so the complex angular-delay map is packed as two image-like
    channels: channel 0 = real part, channel 1 = imaginary part. Cast to float32
    to match typical deep-learning pipelines (halves memory; negligible accuracy
    cost for the SGCS operating points of interest).

    Args:
        H_ad: complex array (N, n_delay, n_tx).
    Returns:
        x: float32 array (N, 2, n_delay, n_tx), NCHW with C=2 (real, imag).
    """
    return np.stack([H_ad.real, H_ad.imag], axis=1).astype(np.float32)


def real_imag_to_complex(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`complex_to_real_imag`: ``(N, 2, D, T)`` -> complex ``(N, D, T)``.

    Recombines the two real channels (0=real, 1=imag) into a single complex
    array. Output dtype is complex64 when ``x`` is float32 (NumPy promotion),
    matching the precision chosen in :func:`complex_to_real_imag`.
    """
    return x[:, 0] + 1j * x[:, 1]
