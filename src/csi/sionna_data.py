"""
csi.sionna_data — Standards-compliant CSI from NVIDIA Sionna (3GPP TR 38.901).
=============================================================================

WHAT THIS MODULE DOES
    A drop-in replacement for ``csi.data.generate_csi_dataset`` that produces
    *real* TR 38.901 channels using NVIDIA **Sionna**'s CDL link-level model —
    the channel companies actually use for 3GPP CSI evaluations. Output shape
    and dtype match the synthetic generator, so the whole CSI-report pipeline
    (angular-delay transform, PMI codebooks, autoencoder, SGCS) runs unchanged.

WHY THIS MATTERS FOR TR 38.843
    3GPP TR 38.843 (Rel-18 "Study on AI/ML for NR air interface") studies a
    *two-sided* model for the CSI-compression sub-use-case: a UE-side **encoder**
    compresses the estimated downlink channel / eigenvector into a small payload,
    and a gNB-side **decoder** reconstructs it. The agreed KPI is **SGCS** (squared
    generalized cosine similarity) of the reconstructed precoder against the
    ground-truth eigenvector, evaluated against a *non-AI baseline* (the TS 38.214
    Type II / eType II PMI codebook) with model **complexity** (FLOPs / params)
    reported alongside. All of that needs realistic channel realizations; this
    module supplies them from the TR 38.901 CDL clustered-delay-line models
    (see csi.metrics.sgcs, csi.baselines for the eType II baseline, and
    csi.models.model_complexity for the complexity reporting).

PUBLIC API (same contract as csi.data)
    generate_sionna_csi(...) -> H   # complex64 (n_samples, n_sub, n_tx)

SCENARIO (defaults)
    * Downlink, single-antenna UE, an N_t-element **ULA** gNB (single pol so the
      DFT codebook / angular-delay basis are well matched — set dual_pol=True for
      a realistic 8H×... dual-pol panel instead).
    * CDL profile (default "C" = NLOS, rich scattering), TR 38.901 §7.7.1.
      The CDL profiles encode per-cluster *normalized delay*, *power [dB]*, and
      angles AOD/AOA/ZOD/ZOA from the tables of §7.7.1 (CDL-A = Table 7.7.1-1,
      CDL-C = Table 7.7.1-3, CDL-E = Table 7.7.1-5); CDL-A/B/C are NLOS, CDL-D/E
      are LOS. The (unitless) tabulated delays are scaled by ``delay_spread`` per
      §7.7.3 (tau_n = tau_n,normalized * DS). Sionna stores these tables and does
      the §7.7.1 ray-to-array steering / §7.7.3 scaling internally.
    * UE-angle diversity: CDL has *fixed* cluster angles, so we synthesize a
      cell of UEs by drawing ``n_orient`` random BS-array orientations (azimuth)
      — analogous to the ``n_environments`` knob of the synthetic generator.

REALISTIC PARAMETER EXAMPLE (FR1 n78 macro, the common 3GPP CSI study point)
    >>> H = generate_sionna_csi(
    ...     n_samples=4000,            # dataset size for autoencoder training
    ...     n_tx=32, dual_pol=True,    # 32-port gNB: 8(H) x 2(V) x dual-pol panel
    ...     n_sub=624,                 # 52 RB * 12 subcarriers (sub-band-grouped later)
    ...     model="C",                 # CDL-C, NLOS rich scattering (TR 38.901 7.7.1-3)
    ...     carrier_frequency=3.5e9,   # FR1 band n78
    ...     subcarrier_spacing=30e3,   # 30 kHz numerology (FR1)
    ...     delay_spread=100e-9,       # 100 ns "nominal" delay spread (TR 38.901 7.7.3)
    ...     max_speed=8.3,             # 30 km/h UE mobility (8.3 m/s) -> Doppler
    ... )                              # -> (4000, 624, 32) complex64
    # FR2 mmWave point instead: carrier_frequency=28e9, subcarrier_spacing=120e3,
    # delay_spread=30e-9 (short DS), model="A".

HOW TO SWAP
    Change ``model`` to "A".."E" (or use UMa/UMi system-level for dropped UEs).
    Returns the same (n_samples, n_sub, n_tx) array the rest of csi expects.
"""
from __future__ import annotations
import numpy as np


def generate_sionna_csi(
    n_samples: int = 2000,
    n_tx: int = 32,
    n_sub: int = 256,
    model: str = "C",
    carrier_frequency: float = 3.5e9,
    delay_spread: float = 100e-9,
    subcarrier_spacing: float = 30e3,
    n_orient: int = 16,
    dual_pol: bool = False,
    max_speed: float = 0.0,
    num_time_steps: int = 1,
    time_sampling_frequency: float | None = None,
    seed: int | None = 0,
    batch: int = 250,
) -> np.ndarray:
    """Generate a TR 38.901 CDL CSI dataset via Sionna.

    Parameters
    ----------
    n_samples : int
        Number of channel realizations (rows of the returned dataset).
    n_tx : int
        Number of gNB transmit *ports*. For ``dual_pol=True`` these are split as
        ``n_tx//2`` co-located cross-polarized element pairs (so n_tx must be
        even). Typical 3GPP CSI study values: 32 (8x2 dual-pol panel) or 64.
    n_sub : int
        Number of OFDM subcarriers in the channel frequency response. In NR this
        is ``RB * 12`` (e.g. 51/106/273 RB -> 612/1272/3276 subcarriers for
        20/40/100 MHz at 30 kHz SCS); studies often use a smaller grid.
    model : str
        CDL profile letter "A".."E" (TR 38.901 §7.7.1). A/B/C = NLOS, D/E = LOS.
    carrier_frequency : float
        Carrier f_c in Hz. FR1: 3.5e9 (n78) / 2.6e9; FR2: 28e9.
    delay_spread : float
        RMS delay spread (seconds) used to scale the normalized CDL delays
        (TR 38.901 §7.7.3). 30 ns short / 100 ns nominal / 300 ns long / 1000 ns
        very long.
    subcarrier_spacing : float
        SCS in Hz: 15e3/30e3 (FR1), 120e3 (FR2). Sets the subcarrier grid via
        ``subcarrier_frequencies`` and the default temporal sampling rate.
    n_orient : int
        Number of random BS-array azimuth orientations used to fan a single CDL
        profile (fixed cluster geometry) into a *cell* of distinct UE angular
        situations. See module docstring.
    dual_pol : bool
        If True, build a cross-polarized (V/H) panel; else a single-pol ULA.
    max_speed : float
        Maximum UE speed (m/s) -> max Doppler shift f_D = max_speed * f_c / c.
        3 km/h = 0.83, 30 km/h = 8.3, 120 km/h = 33 m/s. 0.0 = static channel.
    num_time_steps : int
        Number of consecutive time samples (snapshots) per realization.
    time_sampling_frequency : float | None
        Temporal sampling rate (Hz) when num_time_steps > 1. Defaults to the SCS.
    seed : int | None
        Seed for the NumPy RNG (orientations) and TF RNG (per-cluster phases).
    batch : int
        Sionna call batch size (memory/throughput knob; does not affect output).

    Mobility / temporal sampling
    ----------------------------
    With ``num_time_steps > 1`` each realization is sampled at ``num_time_steps``
    consecutive instants spaced by ``1 / time_sampling_frequency`` seconds, so
    the channel (and its angles/phases) **evolves over time** under the Doppler
    set by ``max_speed`` (max Doppler f_D = max_speed * f_c / c). Set
    ``time_sampling_frequency`` to the CSI-report rate, e.g.
    1 / (csi_rs_periodicity_slots * slot_duration); at 30 kHz SCS a slot is
    0.5 ms, so a 5-slot CSI-RS period -> 1/2.5 ms = 400 Hz. With a single time
    step the behaviour is unchanged (a static snapshot per realization).

    Returns
    -------
    H : complex64 array, unit average power per snapshot.
        * ``num_time_steps == 1`` -> shape (n_samples, n_sub, n_tx)
        * ``num_time_steps  > 1`` -> shape (n_samples, num_time_steps, n_sub, n_tx)

    Notes
    -----
    The downstream pipeline (csi.transform angular-delay 2D-DFT, then either the
    autoencoder or the TS 38.214 §5.2.2.2 PMI baseline) consumes the
    (n_samples, n_sub, n_tx) layout directly. For eType II (§5.2.2.2.5) the
    n_tx ports map to 2L spatial DFT beams (dual-pol) and n_sub to M FD-DFT basis
    vectors; this generator just provides the raw H those steps operate on.
    """
    # TensorFlow + Sionna are imported lazily *inside* the call so that merely
    # importing csi (and the pure-NumPy pipeline) never pulls in TF. This is also
    # what makes the spawn-based parallel path safe: each fresh worker process
    # imports TF for the first time here, after the process has started.
    import tensorflow as tf
    from sionna.phy.channel.tr38901 import CDL, AntennaArray
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

    # Two independent RNGs: NumPy draws the BS orientations (UE-angle diversity),
    # TF seeds the per-cluster random initial phases inside Sionna's CDL so the
    # whole dataset is reproducible for a given ``seed``.
    rng = np.random.default_rng(seed)
    if seed is not None:
        tf.random.set_seed(seed)

    # ---- antenna arrays (TR 38.901 §7.3 panel model) ----
    # gNB array geometry. AntennaArray lays out a (num_rows x num_cols) grid of
    # elements with half-wavelength spacing; the *port* count is rows*cols for a
    # single polarization, or 2*rows*cols for dual ("VH" = +90/0 deg slant V & H).
    if dual_pol:
        # Cross-polarized panel: n_tx/2 spatial element positions, each carrying a
        # V and an H port -> n_tx total ports. This mirrors the TS 38.214 dual-pol
        # codebook layout, where the 2L beams of eType II (§5.2.2.2.5) span the two
        # polarizations. A 32-port "8x2 dual-pol" panel = num_cols=16 here.
        assert n_tx % 2 == 0, "dual-pol needs an even port count"
        bs_kwargs = dict(num_rows=1, num_cols=n_tx // 2,
                         polarization="dual", polarization_type="VH")
    else:
        # Single-pol ULA of n_tx elements along the array axis. This is the layout
        # the angular-delay 2D-DFT basis / oversampled DFT codebook is matched to,
        # so it gives the cleanest sparsity for the autoencoder and Type I/II PMI.
        bs_kwargs = dict(num_rows=1, num_cols=n_tx,
                         polarization="single", polarization_type="V")
    # Single omni UE antenna (downlink Rx) -> n_rx_ant = 1 in the CIR shape below.
    ut = AntennaArray(num_rows=1, num_cols=1, polarization="single",
                      polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=carrier_frequency)

    # Subcarrier offsets (Hz) of the OFDM grid: k * subcarrier_spacing, used to
    # turn the time-domain CIR (taps a, delays tau) into a frequency response.
    freqs = subcarrier_frequencies(n_sub, subcarrier_spacing)
    T = int(num_time_steps)
    # Temporal sampling rate for multi-snapshot (Doppler) sampling; defaults to SCS.
    fs = subcarrier_spacing if time_sampling_frequency is None else float(time_sampling_frequency)
    if T > 1:
        H = np.zeros((n_samples, T, n_sub, n_tx), dtype=np.complex64)
    else:
        H = np.zeros((n_samples, n_sub, n_tx), dtype=np.complex64)

    # CDL cluster angles are *fixed* by the TR 38.901 §7.7.1 tables, so a single
    # CDL profile alone yields one angular geometry. To synthesize a cell of UEs
    # we draw ``n_orient`` random BS-array azimuth orientations and split the
    # samples evenly across them; rotating the array re-points all clusters and
    # acts like placing the UE at a different bearing relative to the panel.
    per = int(np.ceil(n_samples / n_orient))
    filled = 0
    for _ in range(n_orient):
        if filled >= n_samples:
            break
        az = float(rng.uniform(-np.pi, np.pi))      # random BS azimuth in [-pi, pi)
        bs = AntennaArray(antenna_pattern="38.901",
                          carrier_frequency=carrier_frequency, **bs_kwargs)
        # CDL channel: downlink (gNB->UE). ``bs_orientation=[azimuth, down-tilt,
        # slant]`` rotates the panel; ``max_speed`` sets the max Doppler (the
        # per-realization velocity is drawn in [min_speed, max_speed] internally).
        cdl = CDL(model, delay_spread, carrier_frequency, ut, bs,
                  direction="downlink", max_speed=max_speed,
                  bs_orientation=[az, 0.0, 0.0])
        need = min(per, n_samples - filled)         # samples for this orientation
        got = 0
        while got < need:
            b = min(batch, need - got)              # this Sionna call's batch size
            # CDL(b, T, fs) -> time-domain CIR:
            #   a   : complex path gains, shape
            #         [b, n_rx=1, n_rx_ant=1, n_tx=1, n_tx_ant, n_paths, T]
            #   tau : path delays (s), shape [b, n_rx=1, n_tx=1, n_paths]
            a, tau = cdl(b, num_time_steps=T, sampling_frequency=fs)
            # OFDM channel frequency response H(f) = sum_p a_p * exp(-j 2pi f tau_p),
            # evaluated at every subcarrier. normalize=True scales so the average
            # energy over the resource grid is 1 (TR 38.901-style channel power
            # normalization); we additionally renormalize per snapshot below.
            h = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
            # h shape: [b, n_rx=1, n_rx_ant=1, n_tx=1, n_tx_ant, T, n_sub]
            hh = np.asarray(h)[:, 0, 0, 0, :, :, :]      # (b, n_tx, T, n_sub)
            hh = hh.transpose(0, 2, 3, 1)                # (b, T, n_sub, n_tx)
            # Per-(sample, time) power normalization: divide each snapshot by the
            # RMS of |H| over (subcarriers, ports) so every snapshot has unit
            # average power. This removes large-scale fading / path-loss scaling so
            # the autoencoder and SGCS see only the (scale-invariant) channel shape;
            # the +1e-12 guards against divide-by-zero on a degenerate snapshot.
            p = np.sqrt(np.mean(np.abs(hh) ** 2, axis=(2, 3), keepdims=True)) + 1e-12
            hh = (hh / p).astype(np.complex64)
            if T > 1:
                H[filled:filled + b] = hh
            else:
                H[filled:filled + b] = hh[:, 0]          # drop the time axis -> (b, n_sub, n_tx)
            got += b
            filled += b
    return H


# ===========================================================================
# Parallel channel generation (multiprocessing)
# ===========================================================================
def _gen_worker(kw: dict) -> np.ndarray:
    """Spawned-process worker: generate one sample-chunk with limited threads.

    Each worker runs a *fresh* TensorFlow/Sionna in its own process (TF cannot be
    forked safely after init), generating ``n_samples`` realizations with its own
    seed so the chunks are independent. Thread count is capped so N workers share
    the cores instead of each grabbing all of them.

    ``kw`` is the kwargs dict for ``generate_sionna_csi`` plus a private
    ``_n_threads`` entry (popped here) controlling the per-worker thread budget.
    The thread-env vars must be set *before* TF is imported to take effect, which
    is exactly why TF is imported lazily (inside generate_sionna_csi) and not at
    module top level.
    """
    import os
    n_threads = int(kw.pop("_n_threads", 2))
    # Cap BLAS/OpenMP/TF threading via env vars (setdefault: don't clobber an
    # explicit outer setting). With N workers each pinned to a few threads we
    # avoid oversubscription (N * all_cores) that would thrash the scheduler.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTEROP_THREADS",
              "TF_NUM_INTRAOP_THREADS"):
        os.environ.setdefault(v, str(n_threads))
    try:
        # Belt-and-suspenders: also set TF's op-parallelism explicitly (these
        # calls must run before any TF graph executes, hence first-touch here).
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    return generate_sionna_csi(**kw)


def generate_sionna_csi_parallel(n_jobs: int = 4, **kwargs) -> np.ndarray:
    """Parallel `generate_sionna_csi`: split the samples across `n_jobs` processes.

    Splits ``n_samples`` into ``n_jobs`` chunks, each generated in a separate
    spawned process (fresh TF) with a distinct seed, then concatenates. Because
    CDL cluster angles are fixed and UE diversity comes from random BS
    orientations, distinct per-worker seeds simply add orientation diversity —
    the union is a valid dataset. Falls back to serial for small jobs.

    Returns the same array shape as ``generate_sionna_csi``.
    """
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    n_samples = int(kwargs["n_samples"])
    # Not worth the process-spawn + TF-import overhead for tiny jobs; also guards
    # the case where there are fewer than ~2 samples per worker.
    if n_jobs <= 1 or n_samples < 2 * n_jobs:
        return generate_sionna_csi(**kwargs)

    # ``seed or 0`` collapses both 0 and None to 0 as the base seed.
    base_seed = int(kwargs.get("seed", 0) or 0)
    # Divide cores among workers (>=1 each) to avoid thread oversubscription.
    n_threads = max(1, (os.cpu_count() or 4) // n_jobs)
    # Balanced split of n_samples across workers (remainder spread over the first
    # few), so the chunks concatenate back to exactly n_samples rows.
    sizes = [n_samples // n_jobs + (1 if i < n_samples % n_jobs else 0)
             for i in range(n_jobs)]
    tasks = []
    for i, sz in enumerate(sizes):
        kw = dict(kwargs)
        kw["n_samples"] = sz
        # Distinct, well-separated seeds per worker -> independent orientation
        # draws / cluster phases, so the union is a valid larger dataset.
        kw["seed"] = base_seed + 1000 * i + 1     # distinct realizations per worker
        kw["_n_threads"] = n_threads
        tasks.append(kw)

    # "spawn" (not "fork") start method: a forked child inherits a half-initialized
    # TF/CUDA runtime from the parent and deadlocks/crashes, whereas spawn boots a
    # clean interpreter that imports TF fresh. Spawned workers do NOT inherit the
    # parent's sys.path modifications, so make the (often non-installed, src-layout)
    # csi package importable by prepending its parent dir to PYTHONPATH, which the
    # spawned children *do* inherit via the environment.
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", "")

    ctx = mp.get_context("spawn")                  # TF-safe (no fork after init)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        # ex.map preserves task order, so concatenation order is deterministic.
        chunks = list(ex.map(_gen_worker, tasks))
    return np.concatenate(chunks, axis=0)
