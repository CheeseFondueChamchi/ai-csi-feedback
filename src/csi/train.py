"""
csi.train — Standardisation + a generic training loop (PyTorch).
================================================================

WHAT THIS MODULE DOES
    Reusable training plumbing that works with ANY model honouring the model
    contract in ``csi.models``:
      * Standardizer : zero-mean / unit-std scaling fitted on TRAIN data, so
                       reconstruction MSE tracks NMSE (avoids the constant-
                       prediction collapse that min-max + sigmoid suffers on
                       heavy-tailed angular-delay data).
      * train_autoencoder : a compact Adam + cosine-LR loop returning the
                       trained model and the per-epoch test-loss history.

WHERE THIS SITS IN THE 3GPP PICTURE
    This is the offline training stage for the CSI-compression sub-use-case of
    TR 38.843 (Rel-18 AI/ML for the NR air interface). The model under training
    is the "two-sided model": a UE-side encoder and a gNB-side decoder that
    communicate only through a short codeword. Per TR 38.843 the headline KPI is
    SGCS (squared generalised cosine similarity) against a non-AI baseline
    (eType II PMI, TS 38.214 §5.2.2.2.5), with model complexity (params/FLOPs)
    reported alongside — see ``csi.metrics`` and ``csi.models.model_complexity``.
    The reconstruction MSE minimised here is a differentiable surrogate; the
    Standardizer (below) is what keeps that surrogate aligned with NMSE/SGCS.

PUBLIC API (the stable "contract")
    Standardizer().fit(X) / .transform(X) / .inverse(X)
    train_autoencoder(model, Xtr, Xte, epochs=..., ...) -> (model, history)

HOW TO SWAP THIS MODULE
    Swap the loss (e.g. an SGCS-aligned loss), optimiser, or schedule here
    without touching the model or metrics modules.

REALISTIC EXAMPLE CONFIG (FR1 n78 macro cell)
    Scenario: 3.5 GHz, 30 kHz SCS, 100 MHz BW -> 273 RB -> 273*12 = 3276
    subcarriers; gNB 32 ports (8x2 dual-pol panel), UE 4 Rx; CDL-C, 100 ns
    delay spread (TR 38.901 §7.7.1 Table 7.7.1-3, §7.7.3 scaling); UE 3 km/h.
    After ``csi.transform.to_angular_delay`` truncation to (n_delay=32, n_tx=32)
    the codec input is (N, 2, 32, 32). A typical run:

        H   = csi.generate_csi_dataset(20000)        # (N, n_sub, n_tx)
        Xad = csi.complex_to_real_imag(csi.to_angular_delay(H, 32))
        std = csi.Standardizer().fit(Xtr_raw)        # fit on TRAIN split only
        net = csi.CsiNet(32, 32, n_code=128)         # gamma = 128/2048 ~ 1/16
        net, hist = csi.train_autoencoder(
            net, std.transform(Xtr_raw), std.transform(Xte_raw),
            epochs=60, batch_size=200, lr=1.5e-3, device="mps")

    n_code=128 over a 2*32*32 = 2048-element input is a compression ratio
    gamma ~ 1/16, an eType II-comparable operating point (target SGCS ~0.6-0.9).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class Standardizer:
    """Zero-mean / unit-std (z-score) scaler fitted on training data only.

    WHY STANDARDISE (the rationale, not just the mechanics)
        The codec input is a real/imag angular-delay image (see
        ``csi.transform``). Angular-delay CSI is heavy-tailed and sparse: a few
        cells (the dominant CDL clusters, TR 38.901 §7.7.1) carry almost all the
        energy while most cells are near zero. Two consequences:
          1. Min-max scaling to [0, 1] + a final Sigmoid (the original CsiNet
             recipe) compresses that long tail, so the cheapest way for the
             network to cut MSE is to predict a near-constant value everywhere —
             a collapse that *looks* like low MSE but destroys the structure
             SGCS cares about.
          2. NMSE (the natural CSI error, ||H - Hhat||^2 / ||H||^2) is scale-
             invariant. If we z-score the data to unit variance, the plain MSE
             the optimiser minimises becomes proportional to NMSE, so driving
             MSE down genuinely drives NMSE down (and, empirically, SGCS up).
        Hence we use a linear zero-mean/unit-std transform and a *linear* final
        activation in the model (``final_activation='linear'``), not Sigmoid.

    LEAKAGE NOTE
        ``fit`` must see TRAIN data only; ``mu``/``sd`` are then frozen and
        reused to ``transform`` the test split, mirroring deployment where the
        UE cannot peek at future/test statistics.

    SCALARS, NOT PER-FEATURE
        ``mu`` and ``sd`` are global scalars (computed over the whole array),
        so the transform is a single affine map shared by every cell. This keeps
        the mapping exactly invertible and preserves the relative geometry of
        the angular-delay image that SGCS scores.
    """

    def __init__(self):
        # Identity transform until ``fit`` is called: x -> (x - 0) / 1 == x.
        self.mu = 0.0   # global mean of the fitted training array (scalar)
        self.sd = 1.0   # global std  of the fitted training array (scalar)

    def fit(self, X: np.ndarray) -> "Standardizer":
        """Estimate global mean/std from TRAIN data; return self for chaining.

        ``X`` : float array, any shape (typically (N, 2, n_delay, n_tx)).
        The ``+ 1e-12`` floor guards against a divide-by-zero if a (near-)
        constant array is passed (std == 0). ``float(...)`` collapses the
        0-d numpy results to Python scalars so ``transform``/``inverse`` stay
        dtype- and device-agnostic.
        """
        self.mu, self.sd = float(X.mean()), float(X.std() + 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted z-score map and cast to float32.

        Returns float32 because that is the dtype PyTorch (and Apple MPS, which
        does not support float64) expects for model inputs downstream.
        """
        return ((X - self.mu) / self.sd).astype(np.float32)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        """Undo the z-score map: recover data in the original CSI units.

        Used to map a reconstruction back to physical scale before computing
        NMSE/SGCS, or before the inverse angular-delay DFT in ``csi.transform``.
        """
        return X * self.sd + self.mu


def train_autoencoder(model, Xtr, Xte, epochs: int = 60, batch_size: int = 200,
                      lr: float = 1.5e-3, device: str = "cpu", verbose: bool = True):
    """Train any encode/decode model to reconstruct its standardised input.

    This is a self-supervised reconstruction loop: the target IS the input
    (autoencoding). It implements the offline training of the TR 38.843
    two-sided model — the encoder (UE) and decoder (gNB) are trained jointly
    end-to-end here; only at inference are they split across the air interface.

    THE RECIPE
        * Loss      : mean-squared error (nn.MSELoss) on the standardised
                      tensor. Because the input is z-scored to unit variance
                      (see Standardizer), this MSE is proportional to NMSE, the
                      scale-invariant CSI error, which in turn tracks the
                      TR 38.843 SGCS KPI. Swap in an SGCS-aligned loss here if
                      you want to optimise the KPI directly.
        * Optimiser : Adam (adaptive per-parameter step sizes; robust default
                      for the BatchNorm + conv/linear mix in CsiNet/TransNet).
        * Schedule  : CosineAnnealingLR over ``epochs`` epochs — the LR decays
                      lr -> ~0 along a cosine from epoch 0 to ``epochs``. Large
                      early steps explore; the smooth late decay anneals into a
                      sharp minimum. ``T_max == epochs`` with one ``sch.step()``
                      per epoch (NOT per batch) so the cosine completes exactly
                      once over the run.

    DEVICE-AGNOSTIC
        Works on CPU, CUDA, or Apple MPS. Inputs are moved to ``device`` once;
        the shuffle permutation is created on the SAME device as the data so we
        never mix a CPU index tensor with an accelerator tensor (which is
        unreliable across PyTorch/MPS/CUDA versions).

    Parameters
    ----------
    model      : an nn.Module honouring the csi.models contract
                 (encode/decode/forward with matching I/O shapes).
    Xtr, Xte   : standardised arrays/tensors (N, 2, n_delay, n_tx), float32.
                 Typically ``Standardizer.transform(...)`` outputs (numpy).
    epochs     : number of full passes over Xtr (also the cosine period T_max).
    batch_size : SGD minibatch size (e.g. 200 channel realisations per step).
    lr         : initial Adam learning rate (cosine-annealed to ~0).
    device     : "cpu" | "cuda" | "mps".
    verbose    : if True, print test MSE every 10 epochs and on the last epoch.

    Returns
    -------
    (model, history) : the trained model and a list of per-epoch test MSE
                       (length == ``epochs``), useful for plotting convergence.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # T_max = epochs => one full cosine half-period over the whole run when we
    # step the scheduler once per epoch below.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    # Move the full train/test arrays onto the compute device once. float32 is
    # preserved from the Standardizer output (MPS has no float64 support).
    xtr = torch.tensor(Xtr, device=device)
    xte = torch.tensor(Xte, device=device)
    history = []
    for ep in range(epochs):
        model.train()
        # Shuffle each epoch (decorrelates minibatch gradients). Build the
        # permutation on the data's own device so indexing stays device-agnostic
        # (CPU index tensors against MPS/CUDA data are version-dependent).
        perm = torch.randperm(xtr.size(0), device=xtr.device)
        for i in range(0, xtr.size(0), batch_size):
            b = xtr[perm[i:i + batch_size]]          # one minibatch (B, 2, Nd, Nt)
            opt.zero_grad()
            # Autoencoding target == input: reconstruct the standardised CSI.
            loss_fn(model(b), b).backward()
            opt.step()
        sch.step()                                    # advance cosine LR once/epoch
        model.eval()
        with torch.no_grad():
            # Whole-test-set forward pass (no grad) -> scalar test MSE.
            te = loss_fn(model(xte), xte).item()
        history.append(te)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3} | test MSE {te:.5f}")
    return model, history
