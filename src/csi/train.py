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

PUBLIC API (the stable "contract")
    Standardizer().fit(X) / .transform(X) / .inverse(X)
    train_autoencoder(model, Xtr, Xte, epochs=..., ...) -> (model, history)

HOW TO SWAP THIS MODULE
    Swap the loss (e.g. an SGCS-aligned loss), optimiser, or schedule here
    without touching the model or metrics modules.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class Standardizer:
    """Zero-mean / unit-std scaler fitted on training data only."""

    def __init__(self):
        self.mu = 0.0
        self.sd = 1.0

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu, self.sd = float(X.mean()), float(X.std() + 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.sd + self.mu


def train_autoencoder(model, Xtr, Xte, epochs: int = 60, batch_size: int = 200,
                      lr: float = 1.5e-3, device: str = "cpu", verbose: bool = True):
    """Train any encode/decode model to reconstruct its standardised input.

    Parameters
    ----------
    model    : an nn.Module honouring the csi.models contract.
    Xtr, Xte : standardised tensors (N, 2, n_delay, n_tx), float32.

    Returns
    -------
    (model, history) : the trained model and a list of per-epoch test MSE.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    xtr = torch.tensor(Xtr, device=device)
    xte = torch.tensor(Xte, device=device)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.size(0))
        for i in range(0, xtr.size(0), batch_size):
            b = xtr[perm[i:i + batch_size]]
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            te = loss_fn(model(xte), xte).item()
        history.append(te)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3} | test MSE {te:.5f}")
    return model, history
