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

Standards context
-----------------
This codebase implements the **CSI compression** sub-use-case of 3GPP
TR 38.843 (Rel-18 "Study on AI/ML for NR air interface"). The two-sided model
there places a CSI **encoder at the UE** and a **decoder at the gNB**; the
quantised latent in between is the "CSI feedback". The agreed KPI is the
**SGCS** (Squared Generalised Cosine Similarity) of the reconstructed
eigenvector vs the ideal one, reported against a **non-AI eType-II baseline**
together with model **complexity** (FLOPs / params) — see csi.metrics,
csi.baselines and csi.models for those pieces. ``ChannelConfig`` only fixes the
*channel and link assumptions* of one such evaluation: it describes the
TR 38.901 CDL channel (csi.sionna_data) and the NR resource grid that the
encoder/decoder operate on.

Realistic example configuration
-------------------------------
A typical FR1 dense-urban operating point (3GPP n78, 100 MHz, 32-port dual-pol
gNB panel, pedestrian UE), as used for the AI/ML-vs-eType-II SGCS comparison::

    cfg = ChannelConfig(
        carrier_frequency=3.5e9,   # n78 (FR1), 3.5 GHz
        bandwidth=100e6,           # 100 MHz channel
        scs=30e3,                  # 30 kHz SCS  -> numerology mu = 1
        rb=273, nfu=273 * 12,      # 273 PRB -> 3276 subcarriers
        gnb_tx=32,                 # 8x2 dual-pol panel (16 cols x 2 pol)
        ue_rx=1, max_rank=1,       # rank-1 CSI report
        channel_model="CDL-C",     # NLOS, rich scattering (TR 38.901 7.7.1-3)
        ue_speed=0.83,             # 3 km/h pedestrian -> CDL max_speed (Doppler)
        delay_spread=100e-9,       # 100 ns nominal (TR 38.901 7.7.3)
        snr_db=20.0,               # CSI-estimation SNR
        csi_rs_periodicity=20,     # CSI-RS every 20 slots
    )
    # cfg.to_dirname() -> "cdl_c_3_5ghz_30khz_32tx"
    # cfg.slot_duration_s() -> 5e-4  (0.5 ms at mu=1)
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

    Field reference (unit -> 3GPP quantity -> realistic range)
    ----------------------------------------------------------
    Every field below carries an inline note giving its **unit**, the **3GPP
    quantity** it maps to, and a **realistic value range**. See the class-level
    example in the module docstring for a complete, concrete config.
    """

    # --- RF carrier -----------------------------------------------------------
    # carrier_frequency : Hz. NR carrier centre frequency (TS 38.104 channel
    #   raster). FR1 bands: 2.6 GHz (n7/n41), 3.5 GHz (n78) typical; FR2:
    #   28 GHz (n257/n261). Forwarded to Sionna CDL (TR 38.901). Range ~0.6-6 GHz
    #   (FR1) or ~24-52 GHz (FR2).
    carrier_frequency: float = 3.5e9   # Hz; default 3.5 GHz = FR1 n78
    # bandwidth : Hz. Nominal channel bandwidth (TS 38.104). Informational here:
    #   the actual occupied grid is n_sub*scs (checked in __post_init__).
    #   Typical: 20 MHz / 100 MHz (FR1), 100-400 MHz (FR2).
    bandwidth: float         = 20e6    # Hz, informational
    # scs : Hz. Subcarrier spacing = NR numerology, scs = 15 kHz * 2**mu
    #   (TS 38.211 4.2). FR1: 15/30 kHz; FR2: 60/120 kHz. -> Sionna
    #   subcarrier_spacing. Drives slot_duration_s().
    scs: float               = 30e3    # Hz subcarrier spacing -> Sionna subcarrier_spacing
    # rb : count. Number of NR Resource Blocks (12 subcarriers each, TS 38.211
    #   4.4.4). Common BWP sizes: 51 (20 MHz @30 kHz), 106 (40 MHz @30 kHz),
    #   273 (100 MHz @30 kHz). n_sub = rb*12 unless nfu overrides it.
    rb: int                  = 52      # resource blocks, 12 subcarriers each
    frequency_unit: str      = "Hz"    # label only

    # --- Frequency grid -------------------------------------------------------
    # nfu : count. Number of "frequency units" = simulated subcarriers fed to the
    #   2D-DFT angular-delay transform (csi.transform) and the encoder. Should
    #   equal rb*12 (warned in __post_init__ otherwise). e.g. 273*12 = 3276.
    nfu: int = 624  # number of frequency units = simulated subcarriers; default = rb*12

    # --- Antenna configuration ------------------------------------------------
    # gnb_tx : count. Number of gNB transmit ports = CSI dimension N_t -> n_tx in
    #   Sionna. A 32-port "8x2 dual-pol" panel (16 cols x 2 pol) or 64-port panel
    #   is typical for the TR 38.843 CSI-compression study. With dual_pol=True
    #   this must be even (16 cols x 2 pol). Range {8,16,32,64}.
    gnb_tx: int  = 32  # -> n_tx
    # ue_rx : count. UE receive antennas (rank determines layers). 1-4 Rx typical;
    #   the pipeline default models a single-Rx, rank-1 CSI report.
    ue_rx: int   = 1   # single UE antenna
    # max_rank : count. Max transmission rank = number of MIMO layers in the CSI
    #   report (TS 38.214 5.2.2.2: rank indicator RI). 1-4 typical; 1 here.
    max_rank: int = 1

    # --- Channel model --------------------------------------------------------
    # channel_model : str. TR 38.901 7.7.1 CDL profile. "CDL-A/B/C" are NLOS
    #   (7.7.1-1 / -2 / -3), "CDL-D/E" are LOS (7.7.1-4 / -5). Parsed to the bare
    #   Sionna letter by stripping the "CDL-" prefix in sionna_kwargs().
    channel_model: str  = "CDL-C"   # parsed to Sionna model by stripping "CDL-"
    # ue_speed : m/s. UE velocity -> CDL max_speed, sets the Doppler shift
    #   f_D = v*f_c/c (TR 38.901 mobility). Convert km/h -> m/s by /3.6:
    #   3 km/h=0.83, 30 km/h=8.3, 120 km/h=33.3 m/s. Range 0-50 m/s.
    ue_speed: float     = 1       # m/s -> CDL max_speed (Doppler)
    # delay_spread : SECONDS (NOT ns). RMS delay spread DS that scales the CDL
    #   normalised per-cluster delays (TR 38.901 7.7.3 delay-spread scaling).
    #   Typical: 30e-9 short, 100e-9 nominal, 300e-9 long, 1000e-9 very long.
    delay_spread: float = 300e-9    # s

    # --- Link budget (CSI estimation SNR) -------------------------------------
    # snr_db : dB. Operating SNR used to add AWGN to the estimated CSI
    #   (csi.noise.add_awgn), modelling imperfect CSI-RS channel estimation.
    #   Range ~0-30 dB; 20 dB is a clean high-SNR operating point.
    snr_db: float       = 20.0      # operating SNR for noisy CSI estimation
    # pathloss_db : dB. Extra per-scenario loss subtracted from snr_db to get the
    #   effective SNR (effective_snr_db()). 0 = none. Range 0-30 dB.
    pathloss_db: float  = 0.0       # per-scenario extra loss; lowers effective SNR

    # --- Temporal sampling (mobility) -----------------------------------------
    # num_time_steps : count. Channel snapshots per realization. 1 -> static
    #   per-sample CSI; >1 adds a time axis sampled at csi_report_rate_hz(),
    #   letting the channel evolve under the ue_speed Doppler.
    num_time_steps: int = 1         # snapshots per realization; >1 -> time axis
    # lsp_variation : turn on MILD per-sample large-scale-parameter variation
    #   (TR 38.901 7.5): log-normal delay spread + small tilt + cluster jitter.
    #   False -> the strict, deterministic CDL table (default).
    lsp_variation: bool = False

    # --- Dataset identity -----------------------------------------------------
    # channel_label : str. Explicit on-disk directory slug. If empty, to_dirname()
    #   derives one from model/freq/scs/ports. NOTE: the derived slug does NOT
    #   encode delay_spread / ue_speed / snr_db, so two configs differing only in
    #   those map to the SAME directory; set channel_label to disambiguate.
    channel_label: str = ""  # dir slug; if empty, derived in to_dirname()

    # --- Scheduling / reference signals ---------------------------------------
    # csi_rs_periodicity : slots. CSI-RS reporting period (TS 38.214 5.2.1.4).
    #   Sets the temporal sampling rate via csi_report_rate_hz(). 5/10/20 slots
    #   typical; 20 = relatively infrequent feedback.
    csi_rs_periodicity: int = 20  # slots

    # --- Data source ----------------------------------------------------------
    # data_source : str. "sionna" (TR 38.901 CDL via csi.sionna_data),
    #   "synthetic" (csi.data toy generator), or "mixed" (a blend of several CDL
    #   profiles via csi.generate_sionna_csi_mixed — for cross-CDL generalization,
    #   a TR 38.843 evaluation scenario). Provenance only.
    data_source: str = "sionna"  # "sionna" | "synthetic" | "mixed"
    # mix_models : str. Used only when data_source="mixed": comma-separated CDL
    #   profiles to blend, e.g. "CDL-A,CDL-C,CDL-E". Each contributes an equal
    #   share of n_train+n_test with its profile-typical delay spread; samples are
    #   shuffled so train/test mix the profiles. Empty for non-mixed datasets.
    mix_models: str = ""

    # --- Split sizes ----------------------------------------------------------
    # n_train / n_test : count. Number of channel realizations in each split.
    n_train: int   = 25000
    n_test: int    = 5000
    # n_orient : count. Number of random BS-array azimuth orientations used to
    #   synthesize UE/cell diversity over CDL's fixed cluster angles
    #   (see csi.sionna_data); 16 is a reasonable default.
    n_orient: int  = 16
    # dual_pol : bool. True -> dual-polarized gNB panel (gnb_tx = 2*cols, the
    #   2L dual-pol structure of TS 38.214 5.2.2.2 Type II). Requires even gnb_tx.
    dual_pol: bool = True
    # seed : int. RNG seed for reproducible orientation draws / TF determinism.
    seed: int      = 0

    # --------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Sanity-check the resource-grid fields after construction.

        Two non-fatal consistency warnings:
        * rb*12 should equal nfu (12 subcarriers per NR Resource Block,
          TS 38.211 4.4.4); nfu is authoritative if they disagree.
        * the occupied bandwidth n_sub*scs must fit inside the nominal
          channel ``bandwidth`` (ignores guard bands, so it is a loose upper
          bound — see TS 38.104 for transmission-bandwidth configurations).
        """
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
        """NR slot duration in seconds: 1 ms / 2**mu, mu = log2(scs / 15 kHz).

        NR numerology mu indexes the subcarrier spacing scs = 15 kHz * 2**mu
        (TS 38.211 4.2); each slot has 14 OFDM symbols and lasts 1 ms / 2**mu
        (TS 38.211 4.3.2). Examples: 15 kHz -> 1 ms, 30 kHz (mu=1) -> 0.5 ms,
        120 kHz (mu=3) -> 0.125 ms.
        """
        mu = int(round(np.log2(self.scs / 15e3)))
        return 1e-3 / (2 ** mu)

    def csi_report_rate_hz(self) -> float:
        """CSI-report rate in Hz = 1 / (csi_rs_periodicity slots * slot duration).

        Used as ``time_sampling_frequency`` for the multi-snapshot Sionna path:
        consecutive CSI snapshots are spaced one CSI-RS period apart. e.g.
        20 slots at 30 kHz (0.5 ms/slot) -> 1/(20*5e-4) = 100 Hz.
        """
        return 1.0 / (self.csi_rs_periodicity * self.slot_duration_s())

    def to_dirname(self) -> str:
        """Return a filesystem-safe slug identifying this configuration.

        Uses ``channel_label`` verbatim if set; otherwise derives a slug from
        model / carrier / SCS / port count **plus** delay spread, UE speed and
        SNR, e.g. "cdl_c_3_5ghz_30khz_32tx_ds300ns_v1mps_20db". Including the
        last three keeps sweep configs that differ only in those from colliding
        on the same directory; setting an explicit ``channel_label`` is still
        recommended for human-readable dataset names.
        """
        if self.channel_label:
            return self.channel_label
        raw = (
            f"{self.channel_model}_{self.carrier_frequency / 1e9:g}ghz"
            f"_{self.scs / 1e3:g}khz_{self.gnb_tx}tx"
            f"_ds{self.delay_spread * 1e9:g}ns_v{self.ue_speed:g}mps"
            f"_{self.snr_db:g}db"
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

        Maps this config onto the TR 38.901 CDL generator in csi.sionna_data.
        The leading "CDL-" is stripped so "CDL-C" -> "C" (the bare Sionna model
        letter; falls back to "C" if empty). Every returned key matches a
        generate_sionna_csi() parameter exactly.

        # NOTE: ue_speed is forwarded as the CDL max_speed (UE mobility / Doppler).
        # csi_rs_periodicity enters only via time_sampling_frequency; max_rank /
        # ue_rx / bandwidth are recorded for provenance only (not passed here).
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
            lsp_variation=self.lsp_variation,
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
    cfg: "ChannelConfig | None" = None,
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
