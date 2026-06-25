"""
csi.data — CSI dataset source (pure NumPy, no ML deps).
=======================================================

WHAT THIS MODULE DOES
    Produces synthetic downlink channel matrices that look like 3GPP massive
    MIMO CSI: a few angular **clusters**, each with a compact **delay**, over a
    uniform linear array (ULA). This gives the angular-delay sparsity that CSI
    compression exploits.

WHY A SYNTHETIC GENERATOR (vs. the Sionna path)
    This module is a *toy* small-scale model: a deterministic, dependency-free
    sum of a handful of complex sinusoids in angle and delay. It exists so the
    whole CSI-report pipeline (angular-delay transform -> autoencoder / PMI
    codebook -> SGCS scoring) can be developed, unit-tested and CI'd in
    milliseconds, with no TensorFlow/Sionna install and no GPU.

    It is the deliberately simplified counterpart of ``csi.sionna_data``, which
    drives NVIDIA Sionna's clustered-delay-line (CDL) model — the standards
    channel of TR 38.901 §7.7.1 that companies actually use for 3GPP AI/ML CSI
    evaluations. The two share an identical output contract
    ((n_samples, n_sub, n_tx) complex64, unit average power per snapshot) so
    they are drop-in interchangeable for the rest of the pipeline.

    HOW THIS TOY DIFFERS FROM THE TR 38.901 CDL PATH
      * Geometry — here each "cluster" is a raw (angle, delay) pair we draw
        ourselves; CDL uses the *standardised* per-cluster normalized
        delay / power[dB] / AOD / AOA / ZOD / ZOA tables of TR 38.901 §7.7.1
        (e.g. CDL-A Table 7.7.1-1, CDL-C Table 7.7.1-3, CDL-E Table 7.7.1-5)
        and the delay-spread scaling of §7.7.3. We do NOT reproduce those
        tables — angles/delays here are uniform-random, not the fixed CDL rays.
      * Array response — a 1-D ULA steering law (``ula_steering``) vs. Sionna's
        full 3-D dual-pol panel response with 3GPP antenna patterns.
      * Polarization — single-pol only. The eType II 2L-coefficient dual-pol
        structure (TS 38.214 §5.2.2.2.5) is not modelled here; use the Sionna
        path with ``dual_pol=True`` for that.
      * Doppler / mobility — none (each sample is an independent snapshot);
        Sionna can evolve the channel in time under a UE speed.
    Net effect: this generator yields a beam-sparse, low-rank-friendly channel
    that the angular-delay DFT basis (the spatial DFT beams + FD DFT basis of
    TS 38.214 §5.2.2.2) compresses cleanly — ideal for sanity-checking the
    autoencoder before moving to the harder CDL data.

ROLE IN THE 3GPP TR 38.843 STUDY
    TR 38.843 (Rel-18 "Study on AI/ML for NR air interface") studies the CSI
    *compression* sub-use-case as a two-sided model: a UE-side encoder produces
    a compact payload from H (or its precoder), a gNB-side decoder reconstructs
    it, and quality is scored by SGCS (the squared generalized cosine
    similarity, see ``csi.metrics``) against a non-AI baseline (e.g. eType II)
    with model-complexity reporting. The arrays produced here are the ``H`` that
    feeds that loop; the actual SGCS/baseline machinery lives in ``csi.metrics``
    and ``csi.baselines``.

PUBLIC API (the stable "contract")
    generate_csi_dataset(...) -> H   # complex array (n_samples, n_sub, n_tx)
    ula_steering(...)         -> a   # ULA steering vectors (helper)

HOW TO SWAP THIS MODULE  <-- this is the #1 thing people replace
    Any function that returns an array of shape (n_samples, n_sub, n_tx) of
    dtype complex can drop in here. For standards-grade channels use NVIDIA
    Sionna (TR 38.901 CDL/UMa/UMi, via ``csi.sionna_data``) or QuaDRiGa, or load
    the COST 2100 dataset, then hand the resulting H to the rest of the pipeline
    unchanged.

REALISTIC PARAMETER EXAMPLE (FR1 n78, 100 MHz, 32-port gNB)
    A scenario roughly matching a 3.5 GHz (FR1 band n78) macro cell with a
    100 MHz carrier at 30 kHz SCS (273 RB -> 273*12 = 3276 subcarriers, here
    decimated to one value per PRB-ish grid of 256 bins) and an 8x2 dual-pol
    panel reported as 32 CSI-RS ports:

        H = generate_csi_dataset(
                n_samples=4000,        # snapshots in the dataset
                n_tx=32,               # gNB CSI-RS ports (8x2 dual-pol panel)
                n_sub=256,             # frequency bins across the band
                n_clusters=3,          # ~3 dominant scattering clusters (NLOS-ish)
                delay_spread_taps=12.0,# ~100 ns nominal DS at this bin spacing
                n_environments=12,     # finite set of stable cell geometries
            )
        # -> H.shape == (4000, 256, 32), complex64, unit avg power per snapshot

    Other realistic operating points the rest of the codebase targets:
      * Carrier / SCS: 3.5 GHz or 2.6 GHz @ 15/30 kHz (FR1); 28 GHz @ 120 kHz (FR2).
      * Bandwidth / RB: 20 MHz (51/106 RB) or 100 MHz (273 RB); subcarriers = RB*12.
      * gNB ports: 32 (8x2 dual-pol) or 64; UE 1-4 Rx (here a single rank-1 stream).
      * Delay spread (TR 38.901): 30 ns short / 100 ns nominal / 300 ns long /
        1000 ns very long — pick ``delay_spread_taps`` to match the bin spacing.
      * eType II reference codebook (TS 38.214 §5.2.2.2.5): L in {2,4} (up to 6)
        spatial beams, beta in {1/4, 1/2, 3/4}; SGCS operating points ~0.6-0.9.
"""
from __future__ import annotations
import numpy as np


def ula_steering(n_ant: int, angles_rad: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    """Uniform-linear-array (ULA) array-response / steering vectors.

    For a ULA of ``n_ant`` isotropic elements, the element-``n`` response to a
    plane wave arriving/departing at angle ``theta`` (measured from broadside)
    is the progressive phase ``exp(-j 2pi (d/lambda) n sin theta)``. Here
    ``spacing`` is the element spacing ``d`` in wavelengths, so the default
    ``0.5`` is the canonical half-wavelength array; at half-wavelength spacing
    these steering vectors over a regular angle grid become exactly the columns
    of a spatial DFT matrix — i.e. the *spatial DFT beams* that the NR Type II /
    eType II codebooks select from (TS 38.214 §5.2.2.2.1 / §5.2.2.2.3 /
    §5.2.2.2.5). This is the single-polarization, azimuth-only stand-in for
    Sionna's full 3-D dual-pol panel response in the TR 38.901 path.

    Parameters
    ----------
    n_ant : int
        Number of array elements (here the gNB Tx ports, e.g. 32 or 64).
    angles_rad : np.ndarray
        1-D array of L angles in radians (broadside = 0). Typically the
        per-ray/per-cluster departure angles within (-pi/2, pi/2).
    spacing : float, default 0.5
        Inter-element spacing in wavelengths (0.5 = half-wavelength ULA).

    Returns
    -------
    a : complex array, shape (n_ant, L).
        Column ``l`` is the unit-norm steering vector
        ``a(theta_l)[n] = exp(-j 2pi d n sin(theta_l)) / sqrt(n_ant)``.
        The ``1/sqrt(n_ant)`` factor makes each column have unit L2 norm so the
        array gain does not change the channel's overall power normalization.
    """
    # Element index n = 0..n_ant-1 as a column so it broadcasts against the
    # row of L angles -> phase has shape (n_ant, L).
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

    Channel model (one sample), per frequency bin ``k`` and Tx port ``m``:
        H(k, m) = sum_c sum_r alpha_{c,r}
                    * a_tx(theta_{c,r})[m]              # ULA spatial response
                    * exp(-j 2pi k tau_{c,r} / n_sub)   # delay -> freq phase ramp
    i.e. a sum over ``n_clusters`` clusters of ``rays_per_cluster`` rays, each
    ray being one complex sinusoid in the spatial domain (its departure angle)
    multiplied by one complex sinusoid in the frequency domain (its delay). The
    delay-to-frequency phase ``exp(-j 2pi k tau / n_sub)`` is exactly the DFT
    kernel, so a ray at integer tap ``tau`` is a pure tone on the FD DFT basis —
    the same basis the NR FD compression of TS 38.214 §5.2.2.2.5 uses. This is a
    hand-rolled small-scale model, NOT the TR 38.901 §7.7.1 CDL tables (see the
    module docstring for the contrast with ``csi.sionna_data``).

    Units / conventions
        * ``theta`` in radians, broadside = 0, drawn within +/- pi/3.
        * ``tau`` ("delay_spread_taps", "delay_jitter_taps") in *DFT taps*, i.e.
          delay measured in units of the OFDM sampling period (one delay tap =
          one frequency-domain DFT bin period over the ``n_sub`` grid). 12 taps
          over a 256-bin grid is a short-to-nominal delay spread (~100 ns once
          mapped to a real bandwidth); raise it for the 300/1000 ns long-DS
          cases of TR 38.901.
        * Power: each ray gain is CN(0,1)-like, scaled per cluster by an
          exponential delay-power profile (longer-delay clusters are weaker),
          then the whole sample is renormalized to unit average power.

    Parameters
    ----------
    n_samples : int      Number of channel snapshots (rows of H).
    n_tx : int           gNB Tx / CSI-RS ports (e.g. 32 = 8x2 dual-pol, or 64).
    n_sub : int          Frequency bins (subcarriers / PRBs) per snapshot.
    n_clusters : int     Dominant scattering clusters per environment.
    rays_per_cluster : int   Sub-rays per cluster (intra-cluster angular spread).
    delay_spread_taps : float  Max cluster delay (in DFT taps); sets DS.
    n_environments : int     Size of the fixed pool of cell geometries (see below).
    angle_jitter_deg : float   Per-ray angular jitter (deg, Gaussian) around the
                               cluster center -> intra-cluster angle spread.
    delay_jitter_taps : float  Per-ray extra delay (taps, uniform [0,1)*this).
    rng : np.random.Generator | None   Seeded for reproducibility (default 0).

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
    # Fixed environment templates: the *shared* angular-delay structure across
    # samples. Each environment is a set of n_clusters (center angle, center
    # delay) pairs. Drawing these ONCE (not per sample) is what gives the
    # dataset a low-dimensional manifold — see the docstring above. Angles live
    # in +/- 60 deg (a sector-ish field of view); delays in [0, delay_spread].
    env_angles = rng.uniform(-np.pi / 3, np.pi / 3, size=(n_environments, n_clusters))
    env_delays = rng.uniform(0, delay_spread_taps, size=(n_environments, n_clusters))

    H = np.zeros((n_samples, n_sub, n_tx), dtype=np.complex64)
    f = np.arange(n_sub)[:, None]   # frequency-bin index k as a column (n_sub, 1)

    # macOS Accelerate can emit spurious divide/overflow warnings on complex
    # matmul; results are correct, so silence them for this block only.
    with np.errstate(all="ignore"):
        for s in range(n_samples):
            env = rng.integers(n_environments)             # pick one cell geometry
            # Accumulate in complex128 for numerical headroom; cast at the end.
            acc = np.zeros((n_sub, n_tx), dtype=np.complex128)
            for c in range(n_clusters):
                # Per-ray departure angles: cluster center + Gaussian jitter
                # (intra-cluster angular spread). shape (rays_per_cluster,).
                ang = env_angles[env, c] + np.deg2rad(angle_jitter_deg) * rng.standard_normal(rays_per_cluster)
                # Per-ray delays: cluster center + small uniform excess delay.
                dly = env_delays[env, c] + delay_jitter_taps * rng.uniform(0, 1, size=rays_per_cluster)
                # Complex ray gains ~ CN(0,1): Rayleigh-fading sub-rays.
                gains = rng.standard_normal(rays_per_cluster) + 1j * rng.standard_normal(rays_per_cluster)
                # Exponential delay-power profile (later clusters are weaker),
                # and 1/sqrt(2*rays) so the per-cluster power is ~consistent
                # regardless of rays_per_cluster. This is the toy analogue of
                # the standardized per-cluster power[dB] column in the TR 38.901
                # §7.7.1 CDL tables (here exponential rather than tabulated).
                gains *= np.exp(-env_delays[env, c] / delay_spread_taps) / np.sqrt(2 * rays_per_cluster)
                # Spatial response of the rays at the ULA, shape (n_tx, rays).
                a = ula_steering(n_tx, ang)
                # Delay -> frequency phase ramp exp(-j 2pi k tau / n_sub),
                # shape (n_sub, rays). tau in taps, k the FD DFT bin index.
                phase = np.exp(-1j * 2 * np.pi * f * (dly[None, :] / n_sub))
                # Sum the rays: (n_sub, rays) @ (rays, n_tx) -> (n_sub, n_tx).
                # a.conj().T fixes the sign convention so a delay/angle tone maps
                # to a single bin on the angular-delay DFT grid (clean sparsity).
                acc += (phase * gains[None, :]) @ a.conj().T
            # Renormalize each snapshot to unit average power so the dataset has
            # a consistent scale (matches the Sionna path's normalization and
            # keeps SGCS/NMSE comparable across samples).
            acc /= np.sqrt(np.mean(np.abs(acc) ** 2) + 1e-12)   # unit average power
            H[s] = acc.astype(np.complex64)
    return H
