"""
csi.sionna_data — Standards-compliant CSI from NVIDIA Sionna (3GPP TR 38.901).
=============================================================================

WHAT THIS MODULE DOES
    A drop-in replacement for ``csi.data.generate_csi_dataset`` that produces
    *real* TR 38.901 channels using NVIDIA **Sionna**'s CDL link-level model —
    the channel companies actually use for 3GPP CSI evaluations. Output shape
    and dtype match the synthetic generator, so the whole CSI-report pipeline
    (angular-delay transform, PMI codebooks, autoencoder, SGCS) runs unchanged.

PUBLIC API (same contract as csi.data)
    generate_sionna_csi(...) -> H   # complex64 (n_samples, n_sub, n_tx)

SCENARIO (defaults)
    * Downlink, single-antenna UE, an N_t-element **ULA** gNB (single pol so the
      DFT codebook / angular-delay basis are well matched — set dual_pol=True for
      a realistic 8H×... dual-pol panel instead).
    * CDL profile (default "C" = NLOS, rich scattering), TR 38.901.
    * UE-angle diversity: CDL has *fixed* cluster angles, so we synthesize a
      cell of UEs by drawing ``n_orient`` random BS-array orientations (azimuth)
      — analogous to the ``n_environments`` knob of the synthetic generator.

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

    Mobility / temporal sampling
    ----------------------------
    With ``num_time_steps > 1`` each realization is sampled at ``num_time_steps``
    consecutive instants spaced by ``1 / time_sampling_frequency`` seconds, so
    the channel (and its angles/phases) **evolves over time** under the Doppler
    set by ``max_speed``. Set ``time_sampling_frequency`` to the CSI-report rate
    (e.g. 1 / (csi_rs_periodicity * slot_duration)). With a single time step the
    behaviour is unchanged.

    Returns
    -------
    H : complex64 array, unit average power per snapshot.
        * ``num_time_steps == 1`` -> shape (n_samples, n_sub, n_tx)
        * ``num_time_steps  > 1`` -> shape (n_samples, num_time_steps, n_sub, n_tx)
    """
    import tensorflow as tf
    from sionna.phy.channel.tr38901 import CDL, AntennaArray
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

    rng = np.random.default_rng(seed)
    if seed is not None:
        tf.random.set_seed(seed)

    # ---- arrays ----
    if dual_pol:
        assert n_tx % 2 == 0, "dual-pol needs an even port count"
        bs_kwargs = dict(num_rows=1, num_cols=n_tx // 2,
                         polarization="dual", polarization_type="VH")
    else:
        bs_kwargs = dict(num_rows=1, num_cols=n_tx,
                         polarization="single", polarization_type="V")
    ut = AntennaArray(num_rows=1, num_cols=1, polarization="single",
                      polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=carrier_frequency)

    freqs = subcarrier_frequencies(n_sub, subcarrier_spacing)
    T = int(num_time_steps)
    fs = subcarrier_spacing if time_sampling_frequency is None else float(time_sampling_frequency)
    if T > 1:
        H = np.zeros((n_samples, T, n_sub, n_tx), dtype=np.complex64)
    else:
        H = np.zeros((n_samples, n_sub, n_tx), dtype=np.complex64)

    # split the samples across n_orient random BS azimuth orientations (UE diversity)
    per = int(np.ceil(n_samples / n_orient))
    filled = 0
    for _ in range(n_orient):
        if filled >= n_samples:
            break
        az = float(rng.uniform(-np.pi, np.pi))
        bs = AntennaArray(antenna_pattern="38.901",
                          carrier_frequency=carrier_frequency, **bs_kwargs)
        cdl = CDL(model, delay_spread, carrier_frequency, ut, bs,
                  direction="downlink", max_speed=max_speed,
                  bs_orientation=[az, 0.0, 0.0])
        need = min(per, n_samples - filled)
        got = 0
        while got < need:
            b = min(batch, need - got)
            a, tau = cdl(b, num_time_steps=T, sampling_frequency=fs)
            h = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
            # [b, n_rx=1, n_rx_ant=1, n_tx=1, n_tx_ant, T, n_sub]
            hh = np.asarray(h)[:, 0, 0, 0, :, :, :]      # (b, n_tx, T, n_sub)
            hh = hh.transpose(0, 2, 3, 1)                # (b, T, n_sub, n_tx)
            # unit average power per (sample, time) snapshot
            p = np.sqrt(np.mean(np.abs(hh) ** 2, axis=(2, 3), keepdims=True)) + 1e-12
            hh = (hh / p).astype(np.complex64)
            if T > 1:
                H[filled:filled + b] = hh
            else:
                H[filled:filled + b] = hh[:, 0]          # (b, n_sub, n_tx)
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
    """
    import os
    n_threads = int(kw.pop("_n_threads", 2))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTEROP_THREADS",
              "TF_NUM_INTRAOP_THREADS"):
        os.environ.setdefault(v, str(n_threads))
    try:
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
    if n_jobs <= 1 or n_samples < 2 * n_jobs:
        return generate_sionna_csi(**kwargs)

    base_seed = int(kwargs.get("seed", 0) or 0)
    n_threads = max(1, (os.cpu_count() or 4) // n_jobs)
    sizes = [n_samples // n_jobs + (1 if i < n_samples % n_jobs else 0)
             for i in range(n_jobs)]
    tasks = []
    for i, sz in enumerate(sizes):
        kw = dict(kwargs)
        kw["n_samples"] = sz
        kw["seed"] = base_seed + 1000 * i + 1     # distinct realizations per worker
        kw["_n_threads"] = n_threads
        tasks.append(kw)

    # Spawned workers start fresh; ensure the (non-installed) csi package is
    # importable by putting its parent dir on PYTHONPATH (inherited by children).
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", "")

    ctx = mp.get_context("spawn")                  # TF-safe (no fork after init)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        chunks = list(ex.map(_gen_worker, tasks))
    return np.concatenate(chunks, axis=0)
