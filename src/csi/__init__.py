"""
csi — A small, modular toolkit for AI/ML CSI compression (3GPP TR 38.843).
==========================================================================

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
    H   = csi.generate_csi_dataset(2000)            # (N, n_sub, n_tx)
    Xad = csi.complex_to_real_imag(csi.to_angular_delay(H, 32))
    std = csi.Standardizer().fit(Xad)
    net = csi.CsiNet(32, 32, n_code=128)
    net, hist = csi.train_autoencoder(net, std.transform(Xad), std.transform(Xad), epochs=40)
"""
from .data import generate_csi_dataset, ula_steering
from .transform import (
    to_angular_delay, from_angular_delay,
    complex_to_real_imag, real_imag_to_complex,
)
from .metrics import nmse_db, cosine_rho, dominant_eigenvector, sgcs, gcs
from .models import (
    CsiNet, TransNet, RefineNet, compression_ratio, feedback_bits, model_complexity,
)
from .train import Standardizer, train_autoencoder
from .quantize import LatentQuantizer
from .baselines import (
    dft_codebook, type1_pmi, type2_pmi,
    subband_precoders, sgcs_subband, etype2_pmi_2d,
)
# Sionna-based TR 38.901 data source (imports TensorFlow lazily, inside the call)
from .sionna_data import generate_sionna_csi, generate_sionna_csi_parallel
# TR 38.901 CDL channel verification (reference tables + checks)
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
    "generate_sionna_csi", "generate_sionna_csi_parallel",
    "CDL_TABLES", "cdl_reference", "verify_cdl_table", "verify_generated",
    "format_report", "add_awgn",
    "ChannelConfig", "save_dataset", "load_dataset", "dataset_dir",
    "REPO_ROOT", "DATA_ROOT", "RAW_ROOT", "TRAINED_ROOT",
]
