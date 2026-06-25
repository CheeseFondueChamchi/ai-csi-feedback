"""
csi.data — CSI dataset source (pure NumPy, no ML deps).
=======================================================

WHAT THIS MODULE DOES
    Produces synthetic downlink channel matrices that look like 3GPP massive
    MIMO CSI: a few angular **clusters**, each with a compact **delay**, over a
    uniform linear array (ULA). This gives the angular-delay sparsity that CSI
    compression exploits.

PUBLIC API (the stable "contract")
    generate_csi_dataset(...) -> H   # complex array (n_samples, n_sub, n_tx)
    ula_steering(...)         -> a   # ULA steering vectors (helper)

HOW TO SWAP THIS MODULE  <-- this is the #1 thing people replace
    Any function that returns an array of shape (n_samples, n_sub, n_tx) of
    dtype complex can drop in here. For standards-grade channels use NVIDIA
    Sionna (TR 38.901 CDL/UMa/UMi) or QuaDRiGa, or load the COST 2100 dataset,
    then hand the resulting H to the rest of the pipeline unchanged.
"""
from __future__ import annotations
import numpy as np


def ula_steering(n_ant: int, angles_rad: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Uniform-linear-array steering vectors.

    Returns a complex array of shape (n_ant, L); column l is
    a(theta_l) = exp(-j 2pi d n sin(theta_l)) / sqrt(n_ant).
    """
    n = np.arange(n_ant)[:, None]
    phase = 2 * np.pi * spacing * n * np.sin(angles_rad)[None, :]
    return np.exp(-1j * phase) / np.sqrt(n_ant)


def generate_csi_dataset(
    n_samples: int = 4000,
    n_tx: int = 32,
    n_sub: int = 256,
    n_clusters: int = 3,
    rays_per_cluster: int = 6,
    delay_spread_taps: float = 12.0,
    n_environments: int = 12,
    angle_jitter_deg: float = 2.0,
    delay_jitter_taps: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic massive-MIMO CSI dataset with a *learnable manifold*.

    Channel model (one sample):
        H(f, m) = sum_l alpha_l * a_tx(theta_l)[m] * exp(-j 2pi f tau_l)

    KEY DESIGN POINT — ``n_environments``
        A real cell has a *limited, stable* set of dominant scattering
        geometries, not a fresh random one per snapshot. We model this with
        ``n_environments`` fixed templates of cluster (angle, delay); each
        sample picks one template, perturbs it slightly (``angle_jitter_deg`` /
        ``delay_jitter_taps``), and draws fresh random ray gains.

        This matters a lot for compression: if every sample had fully random
        continuous angles, the dataset would span the whole angular space and
        NO low-rate codec (learned or not) could compress it — per-sample
        sparsity is not the same as a low-dimensional dataset manifold. With a
        finite set of environments the dataset lives near a low-dim manifold
        that the autoencoder can learn. Smaller ``n_environments`` = easier;
        larger = harder. Set it huge (≈ n_samples) to recover the degenerate
        "everything random" case.

    Returns
    -------
    H : complex64 array, shape (n_samples, n_sub, n_tx).
    """
    rng = rng or np.random.default_rng(0)
    # Fixed environment templates (the shared structure across samples).
    env_angles = rng.uniform(-np.pi / 3, np.pi / 3, size=(n_environments, n_clusters))
    env_delays = rng.uniform(0, delay_spread_taps, size=(n_environments, n_clusters))

    H = np.zeros((n_samples, n_sub, n_tx), dtype=np.complex64)
    f = np.arange(n_sub)[:, None]

    # macOS Accelerate can emit spurious divide/overflow warnings on complex
    # matmul; results are correct, so silence them for this block only.
    with np.errstate(all="ignore"):
        for s in range(n_samples):
            env = rng.integers(n_environments)
            acc = np.zeros((n_sub, n_tx), dtype=np.complex128)
            for c in range(n_clusters):
                ang = env_angles[env, c] + np.deg2rad(angle_jitter_deg) * rng.standard_normal(rays_per_cluster)
                dly = env_delays[env, c] + delay_jitter_taps * rng.uniform(0, 1, size=rays_per_cluster)
                gains = rng.standard_normal(rays_per_cluster) + 1j * rng.standard_normal(rays_per_cluster)
                gains *= np.exp(-env_delays[env, c] / delay_spread_taps) / np.sqrt(2 * rays_per_cluster)
                a = ula_steering(n_tx, ang)
                phase = np.exp(-1j * 2 * np.pi * f * (dly[None, :] / n_sub))
                acc += (phase * gains[None, :]) @ a.conj().T
            acc /= np.sqrt(np.mean(np.abs(acc) ** 2) + 1e-12)   # unit average power
            H[s] = acc.astype(np.complex64)
    return H
