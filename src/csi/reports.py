"""
csi.reports — Build the per-dataset CSI-feedback report (PMI + eType II 2D).
============================================================================

WHAT THIS MODULE DOES
    Computes, for one test set, the codebook-feedback metrics stored in
    ``reports.npz`` and consumed by the comparison stage. Factored out of the
    gen notebook so the SAME logic builds reports during generation AND when
    refreshing reports after a codebook change (no duplicated/divergent code).

THE REPORTED CHAIN (realistic, noisy CSI estimation — TR 38.843 spirit)
    The UE estimates a noisy channel Ĥ = H + AWGN at ``snr_db`` (csi.add_awgn),
    derives its precoder from Ĥ, and reports a PMI. Every reported precoder is
    scored with SGCS (TR 38.843 intermediate KPI) against the TRUE eigenvector of
    the CLEAN H, so the score reflects estimation error AND codebook quantization.

TWO TRACKS (different metric bases — kept separate on purpose)
    * Wideband Type I / Type II (TS 38.214 §5.2.2.2.1/.3): one dominant
      eigenvector per sample; SGCS of that wideband precoder.
    * True Rel-16 eType II 2D (§5.2.2.2.5): per-subband precoders compressed in
      space (L beams) AND frequency (M DFT basis), dual-pol; scored with mean
      SGCS across subbands (csi.sgcs_subband) — a stricter, per-subband metric.
"""
from __future__ import annotations

import numpy as np

from .metrics import dominant_eigenvector, sgcs
from .noise import add_awgn
from .transform import to_angular_delay, from_angular_delay
from .baselines import (
    type1_pmi, type2_pmi, etype2_pmi_2d, subband_precoders, sgcs_subband,
)

# eType II 2D sweep (spatial L, frequency M) and the K0-truncation fraction.
E2D_SWEEP = [(4, 1), (4, 2), (4, 4), (6, 2), (6, 4), (6, 7)]
E2D_BETA = 0.5
N_SB = 13               # NR-style subband count for the per-subband eType II track


def _serial_map(func, X, **kw):
    """Default codebook evaluator (serial). Mirrors the notebook's pmap()."""
    return func(X, **kw)


def build_reports(H_test, n_tx, n_delay, snr_db, seed=0, dual_pol=False,
                  codebook_map=None) -> dict:
    """Compute the reports dict for one test set.

    Parameters
    ----------
    H_test : complex (n_test, n_sub, n_tx) — clean test channel.
    n_tx, n_delay : ports, kept delay taps for the truncation reference.
    snr_db : CSI-estimation SNR for the AWGN noisy estimate.
    dual_pol : pass-through to the eType II 2D codebook (per-pol beams).
    codebook_map : optional ``f(func, X, **kw) -> (W_hat, bits)`` for parallel
        codebook evaluation (the notebook passes its spawn-pool ``pmap``);
        defaults to serial.

    Returns
    -------
    reports : dict of named arrays/scalars for np.savez_compressed.
    """
    cb = codebook_map if codebook_map is not None else _serial_map
    rng = np.random.default_rng(1234 + seed)
    reports: dict = {}

    # ground-truth precoder from the CLEAN channel
    W_true = dominant_eigenvector(H_test).astype(np.complex64)
    reports['W_true'] = W_true

    # noisy estimate the UE actually sees
    H_est = add_awgn(H_test, snr_db, rng)
    W_est = dominant_eigenvector(H_est).astype(np.complex64)
    reports['sgcs_estimation'] = np.float64(sgcs(W_true, W_est))   # estimation-only ref
    reports['snr_db'] = np.float64(snr_db)

    # delay-truncation reference (near-lossless, on clean H)
    H_tr = from_angular_delay(to_angular_delay(H_test, n_delay), H_test.shape[1])
    reports['sgcs_trunc'] = np.float64(sgcs(W_true, dominant_eigenvector(H_tr)))
    reports['n_delay'] = np.int64(n_delay)

    # ── wideband Type I / Type II (reported from the NOISY precoder) ──────────
    type1_W, type1_bits = type1_pmi(W_est, n_tx)
    reports['type1_W'] = type1_W.astype(np.complex64)
    reports['type1_bits'] = int(type1_bits)
    for L in (2, 3, 4, 6):
        W_hat, bits = cb(type2_pmi, W_est, n_tx=n_tx, L=L)
        reports[f'type2_L{L}_W'] = W_hat.astype(np.complex64)
        reports[f'type2_L{L}_bits'] = int(bits)
    scheme_W = [reports['type1_W']] + [reports[f'type2_L{L}_W'] for L in (2, 3, 4, 6)]
    scheme_bits = [reports['type1_bits']] + [reports[f'type2_L{L}_bits'] for L in (2, 3, 4, 6)]
    reports['pmi_schemes'] = np.array(['Type I'] + [f'Type II L={L}' for L in (2, 3, 4, 6)])
    reports['pmi_family'] = np.array(['Type I'] + ['Type II'] * 4)
    reports['pmi_bits'] = np.array(scheme_bits, dtype=int)
    reports['pmi_sgcs'] = np.array([float(sgcs(W_true, W)) for W in scheme_W], dtype=np.float64)

    # ── true Rel-16 eType II 2D, evaluated PER-SUBBAND ───────────────────────
    W_sb_true = subband_precoders(H_test, N_SB)    # clean ground truth
    W_sb_est = subband_precoders(H_est, N_SB)      # noisy estimate
    reports['n_subband'] = np.int64(N_SB)
    reports['etype2_2d_beta'] = np.float64(E2D_BETA)
    reports['sgcs_subband_estimation'] = np.float64(sgcs_subband(W_sb_true, W_sb_est))
    e2d_names, e2d_bits, e2d_sgcs = [], [], []
    for (L, M) in E2D_SWEEP:
        W_hat_sb, bits = cb(etype2_pmi_2d, W_sb_est, n_tx=n_tx, L=L, M=M,
                            beta=E2D_BETA, dual_pol=dual_pol)
        e2d_names.append(f'eType2D L={L} M={M}')
        e2d_bits.append(int(bits))
        e2d_sgcs.append(float(sgcs_subband(W_sb_true, W_hat_sb)))
    reports['etype2_2d_schemes'] = np.array(e2d_names)
    reports['etype2_2d_bits'] = np.array(e2d_bits, dtype=int)
    reports['etype2_2d_sgcs'] = np.array(e2d_sgcs, dtype=np.float64)
    return reports
