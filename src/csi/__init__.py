"""
csi — A small, modular toolkit for AI/ML CSI compression (3GPP TR 38.843).
==========================================================================

STANDARDS CONTEXT (why this package exists)
    3GPP TR 38.843 (Rel-18, "Study on AI/ML for NR air interface") studies the
    **CSI compression** sub-use-case with a *two-sided model*: a neural
    **encoder runs at the UE** (compresses the estimated downlink channel into
    a small bit payload sent on the uplink) and a matching **decoder runs at
    the gNB** (reconstructs the channel / precoder). The KPI used to score the
    reconstruction is **SGCS** (Squared Generalized Cosine Similarity) on the
    dominant eigenvector(s); the study mandates evaluation **against a non-AI
    baseline** (the legacy Type II / eType II PMI codebooks of TS 38.214) and
    **complexity reporting** (FLOPs / parameter count) for both sides.

    This package mirrors that exact structure:
      * the two-sided model           -> ``csi.models``   (encoder@UE + decoder@gNB)
      * the SGCS KPI                   -> ``csi.metrics``  (``sgcs`` / ``gcs``)
      * the non-AI PMI baseline        -> ``csi.baselines`` (TS 38.214 Type I/II, eType II)
      * complexity reporting           -> ``csi.models.model_complexity``
      * standards-faithful channel data-> ``csi.sionna_data`` + ``csi.verify`` (TR 38.901 CDL)

The pipeline is split into five **swappable** modules. Each has ONE job and a
small, stable public interface, so you can replace any one of them without
touching the others:

    ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
    │  csi.data    │ → │ csi.transform │ → │  csi.models  │ → │ csi.metrics  │
    │ channel H    │   │ angular-delay │   │ encoder@UE / │   │ NMSE, SGCS,  │
    │ (the source) │   │  2D-DFT       │   │ decoder@gNB  │   │ eigenvector  │
    └──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘
                                  ▲  csi.train (standardise + fit any model)

WHICH MODULE DO I EDIT?
    * different channel data  -> csi.data        (e.g. plug in Sionna/QuaDRiGa)
    * different sparsifying basis -> csi.transform
    * different neural codec   -> csi.models      (keep encode/decode/forward)
    * new scoring metric       -> csi.metrics
    * different training recipe -> csi.train
    * the current PMI baseline  -> csi.baselines  (Type I / Type II codebooks)

QUICK START
    import csi
    H   = csi.generate_csi_dataset(2000)            # (N, n_sub, n_tx) complex
    Xad = csi.complex_to_real_imag(csi.to_angular_delay(H, 32))   # (N, 2, 32, 32) float32
    std = csi.Standardizer().fit(Xad)
    net = csi.CsiNet(32, 32, n_code=128)            # n_delay=32, n_tx=32, latent dim=128
    net, hist = csi.train_autoencoder(net, std.transform(Xad), std.transform(Xad), epochs=40)

    Shape walk-through: ``generate_csi_dataset`` returns ``(N, n_sub, n_tx)``;
    ``to_angular_delay(H, 32)`` 2D-DFTs to the angular-delay domain and keeps the
    leading 32 delay taps -> ``(N, 32, n_tx)``; ``complex_to_real_imag`` splits
    real/imag into 2 channels -> ``(N, 2, 32, n_tx)``, which is exactly the
    ``(batch, 2, n_delay, n_tx)`` tensor ``CsiNet`` / ``model_complexity`` expect.

REALISTIC CONFIG EXAMPLE (FR1 mid-band, eType II operating point)
    A typical Rel-18 CSI-compression eval point — 3.5 GHz (NR band n78, FR1),
    30 kHz SCS, 100 MHz / 273 RB, gNB with 32 CSI-RS ports (8x2 dual-pol panel),
    UE at 3 km/h (0.83 m/s) under TR 38.901 CDL-C with ~100 ns nominal delay
    spread. The angular-delay grid below mirrors the 32 antenna ports x 32 kept
    delay taps used above, and ``n_code=128`` is a CSI feedback payload roughly
    comparable to an eType II report with L=4 beams, beta=1/2:

        cfg = csi.ChannelConfig(carrier_frequency=3.5e9, scs=30e3,
                                rb=273, nfu=273 * 12, gnb_tx=32,
                                channel_model="CDL-C", ue_speed=0.83,
                                delay_spread=100e-9)      # see csi.config for all fields
        net = csi.CsiNet(n_delay=32, n_tx=32, n_code=128)
        # Compare against the non-AI baseline (TS 38.214 eType II) via:
        #   csi.etype2_pmi_2d(...) + csi.sgcs_subband(...)   # SGCS KPI per TR 38.843
"""
# --- Data source: synthetic SCM-like channel + ULA array steering vectors ----
from .data import generate_csi_dataset, ula_steering
# --- Sparsifying transform: angular-delay 2D-DFT (TS 38.214 uses spatial +
#     frequency-domain DFT bases; here the antenna axis -> angular DFT beams and
#     the subcarrier axis -> delay taps, matching the Type II beam/FD-basis idea)
from .transform import (
    to_angular_delay, from_angular_delay,
    complex_to_real_imag, real_imag_to_complex,
)
# --- KPIs: NMSE (reconstruction error) and SGCS/GCS, the TR 38.843 CSI-
#     compression scoring metric on the dominant eigenvector(s) ----------------
from .metrics import nmse_db, cosine_rho, dominant_eigenvector, sgcs, gcs
# --- Two-sided model (TR 38.843): encoder@UE / decoder@gNB neural codecs, plus
#     compression-ratio / feedback-bit / FLOP-and-param complexity reporting ---
from .models import (
    CsiNet, TransNet, RefineNet, compression_ratio, feedback_bits, model_complexity,
)
# --- Training recipe: per-feature standardisation + generic autoencoder fit ---
from .train import Standardizer, train_autoencoder
# --- Latent quantisation: maps the float bottleneck to a finite bit budget,
#     modelling the actual uplink CSI report payload (cf. TS 38.214 K0/bitmap +
#     amplitude/phase quantization in eType II) -------------------------------
from .quantize import LatentQuantizer
# --- Non-AI baseline (TS 38.214 §5.2.2.2): Type I PMI (§5.2.2.2.1), Type II
#     (§5.2.2.2.3) and enhanced Type II / eType II (§5.2.2.2.5) PMI codebooks
#     with L spatial DFT beams, M frequency-domain DFT basis vectors, plus
#     per-subband precoders and the SGCS scorer used for the AI-vs-baseline gap.
from .baselines import (
    dft_codebook, type1_pmi, type2_pmi,
    subband_precoders, sgcs_subband, etype2_pmi_2d,
)
# Sionna-based TR 38.901 data source (imports TensorFlow lazily, inside the call)
from .sionna_data import (
    generate_sionna_csi, generate_sionna_csi_parallel, generate_sionna_csi_mixed,
)
# TR 38.901 CDL channel verification: reference per-cluster tables (normalized
# delay / power[dB] / AOD / AOA / ZOD / ZOA from §7.7.1 — CDL-A Table 7.7.1-1,
# CDL-C 7.7.1-3, CDL-E 7.7.1-5) and delay-spread scaling (§7.7.3), with checks
# that the generated channel matches the standard tables.
from .verify import (
    CDL_TABLES, cdl_reference, verify_cdl_table, verify_generated, format_report,
)
# AWGN for noisy CSI-estimation modeling (SNR / pathloss)
from .noise import add_awgn
# Shared IO contract: ChannelConfig, dataset paths, save/load helpers
from .config import (
    ChannelConfig,
    save_dataset,
    load_dataset,
    dataset_dir,
    REPO_ROOT,
    DATA_ROOT,
    RAW_ROOT,
    TRAINED_ROOT,
)

__all__ = [
    "generate_csi_dataset", "ula_steering",
    "to_angular_delay", "from_angular_delay",
    "complex_to_real_imag", "real_imag_to_complex",
    "nmse_db", "cosine_rho", "dominant_eigenvector", "sgcs", "gcs",
    "CsiNet", "TransNet", "RefineNet", "compression_ratio", "feedback_bits",
    "model_complexity",
    "Standardizer", "train_autoencoder", "LatentQuantizer",
    "dft_codebook", "type1_pmi", "type2_pmi",
    "subband_precoders", "sgcs_subband", "etype2_pmi_2d",
    "generate_sionna_csi", "generate_sionna_csi_parallel", "generate_sionna_csi_mixed",
    "CDL_TABLES", "cdl_reference", "verify_cdl_table", "verify_generated",
    "format_report", "add_awgn",
    "ChannelConfig", "save_dataset", "load_dataset", "dataset_dir",
    "REPO_ROOT", "DATA_ROOT", "RAW_ROOT", "TRAINED_ROOT",
]
