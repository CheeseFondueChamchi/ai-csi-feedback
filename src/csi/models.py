"""
csi.models — The two-sided autoencoder (PyTorch).
=================================================

WHAT THIS MODULE DOES
    Defines the neural CSI codec: a UE-side **encoder** and a gNB-side
    **decoder** that talk only through a short codeword (the 3GPP "two-sided
    model"). Default architecture is CsiNet (Wen et al., 2018), made a touch
    deeper so it converges within a notebook-friendly budget.

THE MODEL CONTRACT  <-- any replacement model must honour this
    A model is an ``nn.Module`` providing:
        encode(x)      : (N, 2, n_delay, n_tx) -> (N, n_code)     [runs at UE]
        decode(code)   : (N, n_code)          -> (N, 2, n_delay, n_tx)  [at gNB]
        forward(x)     : decode(encode(x))
    Keep these three methods and the I/O shapes and the rest of the pipeline
    (training, metrics) works unchanged.

HOW TO SWAP THIS MODULE
    Drop in CRNet / CLNet / a Transformer codec etc. Just expose
    encode/decode/forward with the shapes above. ``compression_ratio`` and
    ``feedback_bits`` below describe the rate of whatever codeword you choose.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RefineNet(nn.Module):
    """RefineNet residual block from CsiNet (3 convs + residual add)."""

    def __init__(self, ch: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, 8, 3, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.3),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        return self.act(x + self.body(x))


class CsiNet(nn.Module):
    """CsiNet autoencoder for CSI feedback (the two-sided model).

    Parameters
    ----------
    n_delay, n_tx : truncated angular-delay dimensions of the input.
    n_code        : codeword length M (the compression bottleneck).
    final_activation : 'linear' (recommended; for standardised zero-mean inputs,
        so MSE tracks NMSE) or 'sigmoid' (original CsiNet, for [0,1] inputs).
    """

    def __init__(self, n_delay: int = 32, n_tx: int = 32, n_code: int = 64,
                 final_activation: str = "linear"):
        super().__init__()
        self.n_delay, self.n_tx, self.n_code = n_delay, n_tx, n_code
        self.flat = 2 * n_delay * n_tx
        self.final_activation = final_activation

        # ---- UE-side encoder ----
        self.enc_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 2, 3, padding=1), nn.BatchNorm2d(2), nn.LeakyReLU(0.3),
        )
        self.enc_fc = nn.Linear(self.flat, n_code)

        # ---- gNB-side decoder ----
        self.dec_fc = nn.Linear(n_code, self.flat)
        self.refine = nn.Sequential(RefineNet(2), RefineNet(2))
        self.out_conv = nn.Conv2d(2, 2, 3, padding=1)
        self.sig = nn.Sigmoid()

    def encode(self, x):
        """UE side: channel tensor -> codeword (the only thing sent uplink)."""
        z = self.enc_conv(x).reshape(x.size(0), -1)
        return self.enc_fc(z)

    def decode(self, code):
        """gNB side: codeword -> reconstructed channel tensor."""
        z = self.dec_fc(code).reshape(-1, 2, self.n_delay, self.n_tx)
        z = self.refine(z)
        out = self.out_conv(z)
        return self.sig(out) if self.final_activation == "sigmoid" else out

    def forward(self, x):
        return self.decode(self.encode(x))


class TransNet(nn.Module):
    """Transformer-based CSI feedback autoencoder (the two-sided model).

    Honours the same MODEL CONTRACT as CsiNet — encode/decode/forward with
    identical I/O shapes — so it is a drop-in replacement in the training and
    metrics pipeline. Full-attention encoder and decoder after Cui, Guo, Wen,
    Jin & Wang, "Transformer-Empowered CSI Feedback for Massive MIMO", 2022.

    The input (N, 2, n_delay, n_tx) is read as a sequence of ``n_delay`` tokens,
    each of dimension ``2 * n_tx`` (real+imag interleaved over the tx axis). A
    learned input embedding lifts tokens to ``d_model``; learned positional
    embeddings are added; two standard Transformer encoder layers attend; then a
    linear bottleneck produces the ``n_code`` codeword. The decoder mirrors this:
    expand the codeword back to a token sequence, attend with two more encoder
    layers, and project each token back to ``2 * n_tx``.

    Parameters
    ----------
    n_delay, n_tx : truncated angular-delay dimensions of the input.
    n_code        : codeword length M (the compression bottleneck).
    final_activation : 'linear' (recommended; for standardised zero-mean inputs,
        so MSE tracks NMSE) or 'sigmoid' (for [0,1] inputs).
    """

    def __init__(self, n_delay: int = 32, n_tx: int = 32, n_code: int = 64,
                 final_activation: str = "linear"):
        super().__init__()
        self.n_delay, self.n_tx, self.n_code = n_delay, n_tx, n_code
        self.final_activation = final_activation

        # ---- transformer hyper-parameters (CPU-friendly) ----
        d_model = 128            # token embedding width (nhead must divide it)
        nhead = 4                # multi-head attention heads
        n_layers = 2             # encoder layers on each side
        dim_ff = 4 * d_model     # feed-forward width
        self.d_model = d_model
        self.token_dim = 2 * n_tx

        def _stack():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=0.0, batch_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        # ---- UE-side encoder ----
        self.enc_embed = nn.Linear(self.token_dim, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, n_delay, d_model))
        self.enc_tf = _stack()
        self.enc_fc = nn.Linear(n_delay * d_model, n_code)

        # ---- gNB-side decoder ----
        self.dec_fc = nn.Linear(n_code, n_delay * d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, n_delay, d_model))
        self.dec_tf = _stack()
        self.dec_proj = nn.Linear(d_model, self.token_dim)
        self.sig = nn.Sigmoid()

    def encode(self, x):
        """UE side: channel tensor -> codeword (the only thing sent uplink)."""
        n = x.size(0)
        # (N, 2, n_delay, n_tx) -> (N, n_delay, 2*n_tx) tokens
        tok = x.permute(0, 2, 1, 3).reshape(n, self.n_delay, self.token_dim)
        z = self.enc_embed(tok) + self.enc_pos
        z = self.enc_tf(z)
        return self.enc_fc(z.reshape(n, -1))

    def decode(self, code):
        """gNB side: codeword -> reconstructed channel tensor."""
        n = code.size(0)
        z = self.dec_fc(code).reshape(n, self.n_delay, self.d_model)
        z = z + self.dec_pos
        z = self.dec_tf(z)
        tok = self.dec_proj(z)                      # (N, n_delay, 2*n_tx)
        out = tok.reshape(n, self.n_delay, 2, self.n_tx).permute(0, 2, 1, 3)
        return self.sig(out) if self.final_activation == "sigmoid" else out

    def forward(self, x):
        return self.decode(self.encode(x))


def model_complexity(model, input_shape=(1, 2, 32, 32)) -> dict:
    """Return {'params': int, 'flops': int} for a model.

    Params are exact. FLOPs are estimated via forward hooks counting
    multiply-accumulates (×2 ops) on Conv2d / Linear / MultiheadAttention; the
    attention term (QKV in-proj + scores + context) is an estimate (out-proj is
    an nn.Linear, counted separately). Enough to compare CsiNet vs a transformer
    codec on the same axes.
    """
    import torch

    flops = [0]
    handles = []

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        n_out = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
        flops[0] += 2 * n_out * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        n = 1
        for s in out.shape[:-1]:
            n *= s
        flops[0] += 2 * n * m.in_features * m.out_features

    def mha_hook(m, inp, out):
        x = inp[0]
        if x.dim() == 3:                      # batch_first (N, L, E)
            N, L, E = x.shape
        else:
            N, L, E = 1, x.shape[0], x.shape[1]
        flops[0] += 2 * N * L * E * E * 3     # Q,K,V in-projection
        flops[0] += 2 * N * L * L * E * 2     # attention scores + context

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(mha_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    return {"params": int(sum(p.numel() for p in model.parameters())),
            "flops": int(flops[0])}


def compression_ratio(n_code: int, n_delay: int, n_tx: int) -> float:
    """gamma = M / (2 * Nd * Nt): fraction of the truncated CSI that is kept."""
    return n_code / (2 * n_delay * n_tx)


def feedback_bits(n_code: int, bits_per_coeff: int = 8) -> int:
    """Uplink feedback payload if each codeword entry is scalar-quantised."""
    return n_code * bits_per_coeff
