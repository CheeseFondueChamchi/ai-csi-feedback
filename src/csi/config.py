"""
csi.config — Shared IO contract for the CSI-simulation pipeline.
=================================================================

On-disk layout per dataset::

    data/<channel_label>/
        train.npz       — complex64 array H, shape (n_train, n_sub, n_tx)
        test.npz        — complex64 array H, shape (n_test,  n_sub, n_tx)
        reports.npz     — arbitrary named arrays (codebook metrics, etc.)
        config.json     — ChannelConfig serialised with to_json()
        meta.json       — free-form provenance dict (timestamps, git hash, …)

All four pipeline notebooks import ChannelConfig, save_dataset, load_dataset,
and dataset_dir from this module so that path logic lives in exactly one place.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT    = Path(__file__).resolve().parents[2]      # .../csi_report
DATA_ROOT    = REPO_ROOT / "data"
RAW_ROOT     = REPO_ROOT / "models" / "raw"
TRAINED_ROOT = REPO_ROOT / "models" / "trained"


@dataclass
class ChannelConfig:
    """Full specification of a simulated channel dataset.

    Instances are serialisable (to_json / from_json) and compare equal by
    value, so they can be used as a reproducibility record alongside the
    numpy arrays they describe.
    """

    # --- RF carrier -----------------------------------------------------------
    carrier_frequency: float = 3.5e9   # Hz
    bandwidth: float         = 20e6    # Hz, informational
    scs: float               = 30e3    # Hz subcarrier spacing -> Sionna subcarrier_spacing
    rb: int                  = 52      # resource blocks, 12 subcarriers each
    frequency_unit: str      = "Hz"    # label only

    # --- Frequency grid -------------------------------------------------------
    nfu: int = 624  # number of frequency units = simulated subcarriers; default = rb*12

    # --- Antenna configuration ------------------------------------------------
    gnb_tx: int  = 32  # -> n_tx
    ue_rx: int   = 1   # single UE antenna
    max_rank: int = 1

    # --- Channel model --------------------------------------------------------
    channel_model: str  = "CDL-C"   # parsed to Sionna model by stripping "CDL-"
    ue_speed: float     = 1       # m/s -> CDL max_speed (Doppler)
    delay_spread: float = 300e-9    # s

    # --- Link budget (CSI estimation SNR) -------------------------------------
    snr_db: float       = 20.0      # operating SNR for noisy CSI estimation
    pathloss_db: float  = 0.0       # per-scenario extra loss; lowers effective SNR

    # --- Temporal sampling (mobility) -----------------------------------------
    num_time_steps: int = 1         # snapshots per realization; >1 -> time axis

    # --- Dataset identity -----------------------------------------------------
    channel_label: str = ""  # dir slug; if empty, derived in to_dirname()

    # --- Scheduling / reference signals ---------------------------------------
    csi_rs_periodicity: int = 20  # slots; informational

    # --- Data source ----------------------------------------------------------
    data_source: str = "sionna"  # "sionna" or "synthetic"

    # --- Split sizes ----------------------------------------------------------
    n_train: int   = 25000
    n_test: int    = 5000
    n_orient: int  = 16
    dual_pol: bool = True
    seed: int      = 0

    # --------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.rb * 12 != self.nfu:
            warnings.warn(
                f"rb*12 ({self.rb * 12}) != nfu ({self.nfu}); using nfu"
            )
        if self.n_sub() * self.scs > self.bandwidth:
            warnings.warn(
                f"occupied BW ({self.n_sub() * self.scs / 1e6:.3g} MHz) "
                f"exceeds bandwidth ({self.bandwidth / 1e6:.3g} MHz)"
            )

    def n_sub(self) -> int:
        """Return the number of simulated subcarriers."""
        return int(self.nfu)

    def effective_snr_db(self) -> float:
        """Operating SNR after subtracting the per-scenario pathloss."""
        return float(self.snr_db - self.pathloss_db)

    def slot_duration_s(self) -> float:
        """NR slot duration: 1 ms / 2**mu, mu = log2(scs / 15 kHz)."""
        mu = int(round(np.log2(self.scs / 15e3)))
        return 1e-3 / (2 ** mu)

    def csi_report_rate_hz(self) -> float:
        """CSI-report rate = 1 / (csi_rs_periodicity slots * slot duration)."""
        return 1.0 / (self.csi_rs_periodicity * self.slot_duration_s())

    def to_dirname(self) -> str:
        """Return a filesystem-safe slug identifying this configuration."""
        if self.channel_label:
            return self.channel_label
        raw = (
            f"{self.channel_model}_{self.carrier_frequency / 1e9:g}ghz"
            f"_{self.scs / 1e3:g}khz_{self.gnb_tx}tx"
        )
        slug = re.sub(r"[^a-z0-9]+", "_", raw.lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ChannelConfig":
        """Deserialise from a JSON string produced by to_json()."""
        return cls(**json.loads(s))

    def save_json(self, d) -> None:
        """Write config.json into directory *d*."""
        path = Path(d) / "config.json"
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path) -> "ChannelConfig":
        """Load a ChannelConfig from *path*.

        *path* may be a directory (reads dir/config.json) or a direct file.
        """
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        return cls.from_json(p.read_text(encoding="utf-8"))

    def sionna_kwargs(self) -> dict:
        """Return keyword arguments suitable for generate_sionna_csi().

        # NOTE: ue_speed is forwarded as the CDL max_speed (UE mobility / Doppler).
        # max_rank / csi_rs_periodicity are still recorded for provenance only.
        """
        model = (
            self.channel_model
            .replace("CDL-", "")
            .replace("cdl-", "")
        ) or "C"
        return dict(
            n_samples=self.n_train + self.n_test,
            n_tx=self.gnb_tx,
            n_sub=self.n_sub(),
            model=model,
            carrier_frequency=self.carrier_frequency,
            delay_spread=self.delay_spread,
            subcarrier_spacing=self.scs,
            n_orient=self.n_orient,
            dual_pol=self.dual_pol,
            max_speed=self.ue_speed,
            num_time_steps=self.num_time_steps,
            time_sampling_frequency=self.csi_report_rate_hz(),
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Module-level IO helpers
# ---------------------------------------------------------------------------

def dataset_dir(cfg: ChannelConfig) -> Path:
    """Return the canonical data directory for *cfg*."""
    return DATA_ROOT / cfg.to_dirname()


def save_dataset(
    d,
    H_train,
    H_test,
    reports: dict,
    meta: dict,
    cfg: ChannelConfig = None,
) -> None:
    """Persist a complete dataset to directory *d*.

    Creates *d* if it does not exist.  All arrays are stored as complex64.
    """
    out = Path(d)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "train.npz",   H=H_train.astype(np.complex64))
    np.savez_compressed(out / "test.npz",    H=H_test.astype(np.complex64))
    np.savez_compressed(out / "reports.npz", **reports)

    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    if cfg is not None:
        cfg.save_json(out)


def load_dataset(d) -> dict:
    """Load a dataset written by save_dataset().

    Returns a dict with keys:

    * ``H_train``  — complex64 ndarray
    * ``H_test``   — complex64 ndarray
    * ``reports``  — dict of arrays (from reports.npz)
    * ``cfg``      — ChannelConfig (from config.json)
    * ``meta``     — dict (from meta.json, or {} if absent)
    """
    out = Path(d)

    H_train = dict(np.load(out / "train.npz",   allow_pickle=True))["H"].astype(np.complex64)
    H_test  = dict(np.load(out / "test.npz",    allow_pickle=True))["H"].astype(np.complex64)
    reports = dict(np.load(out / "reports.npz", allow_pickle=True))

    cfg = ChannelConfig.load_json(out)

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    return {
        "H_train": H_train,
        "H_test":  H_test,
        "reports": reports,
        "cfg":     cfg,
        "meta":    meta,
    }
