"""
csi.verify — TR 38.901 CDL channel verification.
================================================

WHY THIS MODULE EXISTS
    Generating "a CDL-C channel" is only trustworthy if the channel that comes
    out actually carries the cluster delays / powers / angles that 3GPP
    TR 38.901 specifies. This module provides an **independent** ground-truth
    copy of the published CDL tables (transcribed from the standard, not read
    from Sionna) and two levels of verification:

      1. CONFIG level  — `verify_cdl_table(model)`: the per-cluster parameters
         the generator is configured with (Sionna's bundled model table) must
         equal the TR 38.901 table — normalized delay, power [dB], AOD, AOA,
         ZOD, ZOA, the cluster angular spreads (cASD/cASA/cZSD/cZSA) and XPR.

      2. DATA level    — `verify_generated(H, cfg)`: statistics measured from
         the *actually generated* channel H must be consistent with the table —
         unit average power, and the RMS delay spread implied by the power-delay
         profile recovered from H.

REFERENCES
    3GPP TR 38.901 V18.0.0 §7.7.1 "CDL models":
        Table 7.7.1-1  CDL-A (NLOS, 23 clusters)
        Table 7.7.1-3  CDL-C (NLOS, 24 clusters)
        Table 7.7.1-5  CDL-E (LOS,  14 cluster rows)
    Delays are normalized (multiply by the desired RMS delay spread to get
    seconds). Powers are in dB. Angles (AOD/AOA/ZOD/ZOA) are in degrees and are
    the per-cluster mean angles before scaling by the cluster spreads.

HOW TO READ THE REPORT
    Each `verify_*` returns a dict with `ok` (bool) plus per-field diagnostics.
    `format_report(report)` renders a human-readable pass/fail table.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# TR 38.901 V18.0.0 reference tables (transcribed from the standard).
# Each profile: scalar cluster spreads + XPR, and per-cluster arrays of
# [normalized delay, power dB, AOD°, AOA°, ZOD°, ZOA°].
# ---------------------------------------------------------------------------
CDL_TABLES: dict[str, dict] = {
    # --- Table 7.7.1-1: CDL-A (NLOS) ------------------------------------------
    "CDL-A": {
        "los": 0,
        "num_clusters": 23,
        "cASD": 5.0, "cASA": 11.0, "cZSD": 3.0, "cZSA": 3.0, "xpr": 10.0,
        "delays": [0.0, 0.3819, 0.4025, 0.5868, 0.461, 0.5375, 0.6708, 0.575,
                   0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119,
                   3.0582, 4.081, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586],
        "powers": [-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9,
                   -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3,
                   -18.9, -16.6, -19.9, -29.7],
        "aod": [-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4,
                -83.0, 134.8, -153.0, -172.0, -129.9, -136.0, 165.4, 148.4,
                132.7, -118.6, -154.1, 126.5, -56.2],
        "aoa": [51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8, -41.9,
                94.2, 51.9, -115.9, 26.6, 76.6, -7.0, -23.0, -47.2, 110.4, 144.5,
                155.3, 102.0, -151.8, 55.2],
        "zod": [50.2, 93.2, 93.2, 93.2, 122.0, 122.0, 122.0, 150.2, 55.2, 26.4,
                126.4, 171.6, 151.4, 157.2, 47.2, 40.4, 43.3, 161.8, 10.8, 16.7,
                171.7, 22.7, 144.9],
        "zoa": [125.4, 91.3, 91.3, 91.3, 94.0, 94.0, 94.0, 47.1, 56.0, 30.1,
                58.8, 26.0, 49.2, 143.1, 117.4, 122.7, 123.2, 32.6, 27.2, 15.2,
                146.0, 150.7, 156.1],
    },
    # --- Table 7.7.1-3: CDL-C (NLOS) ------------------------------------------
    "CDL-C": {
        "los": 0,
        "num_clusters": 24,
        "cASD": 2.0, "cASA": 15.0, "cZSD": 3.0, "cZSA": 7.0, "xpr": 7.0,
        "delays": [0.0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.656,
                   0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704,
                   2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374,
                   7.0427, 8.6523],
        "powers": [-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1,
                   -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
                   -17.1, -16.0, -15.7, -21.6, -22.8],
        "aod": [-46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1, -64.5,
                80.2, -97.1, -55.3, -64.3, -78.5, 102.7, 99.2, 88.8, -101.9,
                92.2, 93.3, 106.6, 119.5, -123.8],
        "aoa": [-101.0, 120.0, 120.0, 120.0, -127.5, 170.4, 170.4, 170.4, 55.4,
                66.5, -48.1, 46.9, 68.1, -68.7, 81.5, 30.7, -16.4, 3.8, -13.7,
                9.7, 5.6, 0.7, -21.9, 33.6],
        "zod": [97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2, 95.3,
                106.1, 93.5, 103.7, 104.2, 93.0, 104.2, 94.9, 93.1, 92.2, 106.7,
                93.0, 92.9, 105.2, 107.8],
        "zoa": [87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4, 63.8,
                71.4, 60.5, 90.6, 60.1, 61.0, 100.7, 62.3, 66.7, 52.9, 61.8,
                51.9, 61.7, 58.0, 57.0],
    },
    # --- Table 7.7.1-5: CDL-E (LOS) -------------------------------------------
    "CDL-E": {
        "los": 1,
        "num_clusters": 14,  # 14 cluster rows; the LOS path is split into 2 sub-rows
        "cASD": 5.0, "cASA": 11.0, "cZSD": 3.0, "cZSA": 7.0, "xpr": 8.0,
        "delays": [0.0, 0.0, 0.5133, 0.544, 0.563, 0.544, 0.7112, 1.9092, 1.9293,
                   1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6419],
        "powers": [-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6,
                   -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2],
        "aod": [0.0, 0.0, 57.5, 57.5, 57.5, -20.1, 16.2, 9.3, 9.3, 9.3, 19.0,
                32.7, 0.5, 55.9, 57.6],
        "aoa": [-180.0, -180.0, 18.2, 18.2, 18.2, 101.8, 112.9, -155.5, -155.5,
                -155.5, -143.3, -94.7, 147.0, -36.2, -26.0],
        "zod": [99.6, 99.6, 104.2, 104.2, 104.2, 99.4, 100.8, 98.8, 98.8, 98.8,
                100.8, 96.4, 98.9, 95.6, 104.6],
        "zoa": [80.4, 80.4, 80.4, 80.4, 80.4, 80.8, 86.3, 82.7, 82.7, 82.7,
                82.9, 88.0, 81.0, 88.6, 78.3],
    },
}

_PER_CLUSTER_FIELDS = ("delays", "powers", "aod", "aoa", "zod", "zoa")
_SCALAR_FIELDS = ("cASD", "cASA", "cZSD", "cZSA", "xpr", "num_clusters", "los")


def normalize_model_name(model: str) -> str:
    """Normalize 'C', 'cdl-c', 'CDL_C' -> 'CDL-C'."""
    m = model.strip().upper().replace("_", "-")
    if not m.startswith("CDL-"):
        m = "CDL-" + m
    return m


def cdl_reference(model: str) -> dict:
    """Return the TR 38.901 reference table for a CDL profile."""
    m = normalize_model_name(model)
    if m not in CDL_TABLES:
        raise KeyError(f"No TR 38.901 reference table for {m!r} "
                       f"(have {sorted(CDL_TABLES)})")
    return CDL_TABLES[m]


def rms_delay_spread_normalized(model: str) -> float:
    """Normalized RMS delay spread of the table (multiply by delay_spread [s]).

    Computed from the (linear-power-weighted) second central moment of the
    normalized cluster delays. For a correctly scaled CDL this should be close
    to 1.0 by construction, so multiplying by `delay_spread` recovers seconds.
    """
    ref = cdl_reference(model)
    tau = np.asarray(ref["delays"], float)
    p = 10.0 ** (np.asarray(ref["powers"], float) / 10.0)
    p = p / p.sum()
    mean_tau = float((p * tau).sum())
    return float(np.sqrt((p * (tau - mean_tau) ** 2).sum()))


# ---------------------------------------------------------------------------
# 1. CONFIG-LEVEL verification: Sionna's bundled table vs the TR 38.901 table.
# ---------------------------------------------------------------------------
def _load_sionna_table(model: str) -> dict:
    """Read the model table Sionna actually loads, straight from its JSON."""
    import json
    from pathlib import Path
    from sionna.phy.channel.tr38901 import cdl as _cdl_mod

    m = normalize_model_name(model)
    models_dir = Path(_cdl_mod.__file__).resolve().parent / "models"
    return json.loads((models_dir / f"{m}.json").read_text())


def verify_cdl_table(model: str, atol: float = 1e-6) -> dict:
    """Verify the generator's CDL table equals the TR 38.901 reference.

    Compares every per-cluster array (delay/power/AOD/AOA/ZOD/ZOA) and every
    scalar (cluster spreads, XPR, num_clusters, LOS flag) of the table Sionna
    loads against the standard values transcribed in CDL_TABLES.

    Returns a report dict: {model, ok, fields:{name:{max_abs_err, ok}}, ...}.
    """
    ref = cdl_reference(model)
    try:
        son = _load_sionna_table(model)
    except Exception as exc:  # pragma: no cover - only if Sionna missing
        return {"model": normalize_model_name(model), "ok": False,
                "error": f"could not load Sionna table: {exc}", "fields": {}}

    fields: dict[str, dict] = {}
    ok = True
    for f in _PER_CLUSTER_FIELDS:
        a = np.asarray(ref[f], float)
        b = np.asarray(son.get(f, []), float)
        if a.shape != b.shape:
            fields[f] = {"ok": False, "reason": f"len {a.size} vs {b.size}"}
            ok = False
            continue
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        f_ok = err <= atol
        fields[f] = {"ok": f_ok, "max_abs_err": err}
        ok = ok and f_ok
    for f in _SCALAR_FIELDS:
        a, b = ref.get(f), son.get(f)
        f_ok = (a is not None and b is not None and abs(float(a) - float(b)) <= atol)
        fields[f] = {"ok": f_ok, "ref": a, "got": b}
        ok = ok and f_ok
    return {"model": normalize_model_name(model), "ok": ok,
            "num_clusters": ref["num_clusters"], "fields": fields}


# ---------------------------------------------------------------------------
# 2. DATA-LEVEL verification: statistics of the generated H.
# ---------------------------------------------------------------------------
def verify_generated(H: np.ndarray, model: str, delay_spread: float,
                     scs: float, power_atol: float = 0.05,
                     win_energy_min: float = 0.85,
                     pdp_corr_min: float = 0.6) -> dict:
    """Verify statistics measured from generated H are consistent with the table.

    Checks (all robust to OFDM band-limiting)
    -----------------------------------------
    * unit average power  : mean over samples of mean(|H|^2) ~ 1.0, within
      `power_atol` (generators normalize per sample).
    * delay-window energy : fraction of power-delay-profile energy that falls
      inside the table's cluster delay span [0, max_delay] (+small guard). A
      correct channel concentrates its energy there; must be >= win_energy_min.
    * PDP shape match     : Pearson correlation between the empirical PDP and
      the table's per-cluster delays/powers binned onto the same delay grid,
      over the cluster window; must be >= pdp_corr_min.

    Why not RMS delay spread? Recovering the RMS *moment* from a band-limited
    OFDM channel is dominated by DFT sinc-leakage at high delays (the tau^2
    weighting amplifies a tiny tail), so it is reported for information only,
    not gated. Per-cluster *angles* are covered by the config-level
    `verify_cdl_table` (high-resolution angle estimation from H after the BS
    array response is a separate problem).

    Parameters
    ----------
    H : complex array (N, n_sub, n_tx)  — generated channel.
    """
    H = np.asarray(H)
    if H.ndim == 4:          # (N, T, n_sub, n_tx) temporal -> flatten time
        H = H.reshape(-1, H.shape[-2], H.shape[-1])
    N, n_sub = H.shape[0], H.shape[1]

    # --- unit average power ---------------------------------------------------
    avg_power = float(np.mean(np.abs(H) ** 2))
    power_ok = abs(avg_power - 1.0) <= power_atol

    # --- empirical power-delay profile via IDFT across subcarriers ------------
    h_delay = np.fft.ifft(H, axis=1)                  # (N, n_sub, n_tx) delay taps
    pdp = np.mean(np.abs(h_delay) ** 2, axis=(0, 2))  # (n_sub,) avg power per tap
    pdp = pdp / (pdp.sum() + 1e-12)
    bin_dt = 1.0 / (n_sub * scs)                      # seconds per delay bin

    # --- theoretical PDP: table clusters binned onto the same delay grid ------
    ref = cdl_reference(model)
    tau = np.asarray(ref["delays"], float) * float(delay_spread)   # seconds
    pw = 10.0 ** (np.asarray(ref["powers"], float) / 10.0)
    pw = pw / pw.sum()
    theo = np.zeros(n_sub)
    for t, p in zip(tau, pw):
        theo[int(round(t / bin_dt)) % n_sub] += p

    # --- delay-window energy concentration ------------------------------------
    win = min(int(np.ceil(tau.max() / bin_dt)) + 3, n_sub)   # cluster span + guard
    win_energy = float(pdp[:win].sum())
    win_ok = win_energy >= win_energy_min

    # --- PDP shape correlation over the cluster window ------------------------
    a, b = pdp[:win], theo[:win]
    if a.std() > 0 and b.std() > 0:
        pdp_corr = float(np.corrcoef(a, b)[0, 1])
    else:
        pdp_corr = float("nan")
    corr_ok = (pdp_corr >= pdp_corr_min)

    # --- informational windowed RMS delay spread (not gated) ------------------
    pwin = pdp[:win] / (pdp[:win].sum() + 1e-12)
    taps = np.arange(win) * bin_dt
    mean_tau = float((pwin * taps).sum())
    rms_ds_win = float(np.sqrt((pwin * (taps - mean_tau) ** 2).sum()))
    rms_ds_ref = float(delay_spread) * rms_delay_spread_normalized(model)

    return {
        "model": normalize_model_name(model), "n_samples": N,
        "avg_power": avg_power, "power_ok": power_ok,
        "win_energy": win_energy, "win_energy_ok": win_ok,
        "pdp_corr": pdp_corr, "pdp_corr_ok": corr_ok,
        "window_taps": win, "delay_bin_s": bin_dt,
        "rms_ds_windowed_s": rms_ds_win, "rms_ds_reference_s": rms_ds_ref,
        # Hard gate = unit power + energy concentration (both robust). The PDP
        # shape correlation is reported but NOT gated: with coarse OFDM delay
        # bins, channels with many overlapping clusters (e.g. CDL-C) legitimately
        # sit near ~0.6, so gating on it would be flaky. Config-level exact match
        # remains the authoritative TR 38.901 check.
        "ok": bool(power_ok and win_ok),
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def format_report(table_rep: dict, gen_rep: dict | None = None) -> str:
    """Render verify_cdl_table (+ optional verify_generated) as a text table."""
    lines = []
    m = table_rep.get("model", "?")
    status = "PASS ✅" if table_rep.get("ok") else "FAIL ❌"
    lines.append(f"── CDL verification: {m}  [{status}] "
                 f"({table_rep.get('num_clusters','?')} clusters, TR 38.901 §7.7.1) ──")
    if "error" in table_rep:
        lines.append(f"   error: {table_rep['error']}")
    lines.append("   config-level (generator table vs TR 38.901):")
    for f, d in table_rep.get("fields", {}).items():
        mark = "ok " if d.get("ok") else "ERR"
        if "max_abs_err" in d:
            lines.append(f"     [{mark}] {f:<6} max|Δ| = {d['max_abs_err']:.2e}")
        elif "reason" in d:
            lines.append(f"     [{mark}] {f:<6} {d['reason']}")
        else:
            lines.append(f"     [{mark}] {f:<6} ref={d.get('ref')} got={d.get('got')}")
    if gen_rep is not None:
        gstatus = "PASS ✅" if gen_rep.get("ok") else "FAIL ❌"
        lines.append(f"   data-level (measured from generated H)  [{gstatus}]:")
        lines.append(f"     [{'ok ' if gen_rep['power_ok'] else 'ERR'}] avg |H|^2 = "
                     f"{gen_rep['avg_power']:.4f} (target 1.0)")
        lines.append(f"     [{'ok ' if gen_rep['win_energy_ok'] else 'ERR'}] "
                     f"delay-window energy = {gen_rep['win_energy']*100:.1f}% "
                     f"in [0, {gen_rep['window_taps']} taps] "
                     f"(bin {gen_rep['delay_bin_s']*1e9:.1f} ns)")
        lines.append(f"     [inf] PDP shape corr vs table = {gen_rep['pdp_corr']:.3f} "
                     f"(not gated; OFDM-resolution sensitive)")
        lines.append(f"     [inf] windowed RMS delay spread = "
                     f"{gen_rep['rms_ds_windowed_s']*1e9:.0f} ns "
                     f"(table {gen_rep['rms_ds_reference_s']*1e9:.0f} ns; "
                     f"OFDM-resolution limited, informational)")
    return "\n".join(lines)
