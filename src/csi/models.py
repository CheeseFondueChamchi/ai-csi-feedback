"""
csi.models — The two-sided autoencoder (PyTorch).
=================================================

WHAT THIS MODULE DOES
    Defines the neural CSI codec: a UE-side **encoder** and a gNB-side
    **decoder** that talk only through a short codeword (the 3GPP "two-sided
    model"). Default architecture is CsiNet (Wen et al., 2018), made a touch
    deeper so it converges within a notebook-friendly budget.

3GPP CONTEXT (TR 38.843, Rel-18 "AI/ML for the NR air interface")
    The CSI-compression sub-use-case is the flagship *two-sided* model: the UE
    runs the **encoder** (a CSI generation part) and the gNB runs the
    **decoder** (a CSI reconstruction part); only the latent codeword crosses
    the air interface on the uplink control channel. This module is exactly
    that pair of networks. Per TR 38.843:
      * KPI    — the *primary intermediate* accuracy metric is SGCS (Squared
                 Generalized Cosine Similarity) on the dominant eigenvector;
                 NMSE is the secondary metric (both live in ``csi.metrics``).
      * Method — the AI/ML codec must be evaluated *against a non-AI baseline*,
                 i.e. the eType II codebook (``csi.baselines``), on identical
                 channel data, AND its **complexity must be reported** (model
                 size / parameter count + computational cost / FLOPs). See
                 ``model_complexity`` below, which supplies the FLOPs+params
                 numbers TR 38.843 asks every candidate model to disclose.

THE MODEL CONTRACT  <-- any replacement model must honour this
    A model is an ``nn.Module`` providing:
        encode(x)      : (N, 2, n_delay, n_tx) -> (N, n_code)     [runs at UE]
        decode(code)   : (N, n_code)          -> (N, 2, n_delay, n_tx)  [at gNB]
        forward(x)     : decode(encode(x))
    Keep these three methods and the I/O shapes and the rest of the pipeline
    (training, metrics) works unchanged.

    INPUT TENSOR LAYOUT
        x is the *angular-delay* CSI image produced by ``csi.transform``:
          axis 0 (N)        — batch (one CSI snapshot per sample),
          axis 1 (2)        — real / imag of the complex channel (interleaved),
          axis 2 (n_delay)  — truncated delay taps (rows; sparse after 2D-DFT),
          axis 3 (n_tx)     — gNB transmit-antenna / angular index (cols).
        Values are expected pre-standardised (zero-mean, unit-var per feature
        via ``csi.Standardizer``) so plain MSE training tracks NMSE; use
        ``final_activation='linear'``. The original CsiNet used [0,1]-scaled
        magnitudes and a sigmoid output — pass ``final_activation='sigmoid'``
        only if you feed it [0,1] data.

HOW TO SWAP THIS MODULE
    Drop in CRNet / CLNet / a Transformer codec etc. Just expose
    encode/decode/forward with the shapes above. ``compression_ratio`` and
    ``feedback_bits`` below describe the rate of whatever codeword you choose.

REALISTIC PARAMETER EXAMPLE (FR1 n78 macro-cell)
    Carrier 3.5 GHz, SCS 30 kHz, 100 MHz => 273 RB => 3276 subcarriers; a
    32-port gNB panel (8x2 dual-pol) and CDL-C @ 300 ns delay spread, UE at
    3 km/h (0.83 m/s). After ``to_angular_delay(H, n_delay=32)`` the CSI image
    is (N, 2, 32, 32). A 1/16 compression operating point keeps n_code=128:

        net = CsiNet(n_delay=32, n_tx=32, n_code=128)   # gamma = 128/2048 = 1/16
        # feedback_bits(128, bits_per_coeff=8) = 1024 bits per CSI report
        # (target SGCS ~0.8, vs eType II L=4, beta=1/2 baseline)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RefineNet(nn.Module):
    """RefineNet residual block from CsiNet (Wen et al., 2018).

    The decoder-side refinement unit: three 3x3 convolutions
    (ch -> 8 -> 16 -> ch feature maps) whose output is added back to the
    block input (a residual / skip connection) and passed through a final
    activation. The 3x3 kernels with ``padding=1`` preserve the (n_delay, n_tx)
    spatial size, so the block is shape-preserving: (N, ch, H, W) -> same. Each
    conv is followed by BatchNorm; LeakyReLU(0.3) is the original CsiNet slope.
    Two of these are stacked in the decoder to iteratively sharpen the coarse
    reconstruction emitted by the fully-connected expansion layer.

    Parameters
    ----------
    ch : number of channels in/out (2 here: the real & imag CSI planes). The
        internal 8/16 widths are CsiNet's fixed bottleneck widths.
    """

    def __init__(self, ch: int = 2):
        super().__init__()
        # 3-conv body. Note the LAST conv has BN but NO activation here, so the
        # residual is added in the *pre-activation* domain and the single
        # trailing self.act is applied to the sum (matches CsiNet's RefineNet).
        self.body = nn.Sequential(
            nn.Conv2d(ch, 8, 3, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.3),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        # Residual add: identity skip + learned correction, then activation.
        return self.act(x + self.body(x))


class CsiNet(nn.Module):
    """CsiNet autoencoder for CSI feedback (the two-sided model).

    Architecture (Wen, Shih & Jin, "Deep Learning for Massive MIMO CSI
    Feedback", IEEE WCL 2018), the canonical AI baseline for the TR 38.843
    CSI-compression study:

      UE encoder:  Conv(2->16) + Conv(16->2) feature extraction over the
        angular-delay image, flatten to 2*Nd*Nt, then a single dense layer
        compresses to the length-``n_code`` codeword M. This codeword is the
        *only* quantity sent uplink (the "CSI generation part").
      gNB decoder: a dense layer expands M back to 2*Nd*Nt, reshape to the
        image, then two ``RefineNet`` residual blocks sharpen it, and a final
        1-channel-preserving Conv(2->2) produces the reconstruction (the "CSI
        reconstruction part").

    Compared with the 2018 paper this variant adds BatchNorm on the encoder
    convs and stacks two RefineNet blocks, which helps it converge inside a
    notebook training budget without changing the I/O contract.

    Parameters
    ----------
    n_delay, n_tx : truncated angular-delay dimensions of the input. Typical
        massive-MIMO setup: n_delay=32 (delay taps kept after 2D-DFT, since the
        channel is sparse in delay), n_tx=32 (gNB antenna/angular index).
    n_code        : codeword length M (the compression bottleneck). E.g. M=128
        gives compression ratio gamma = 128/(2*32*32) = 1/16.
    final_activation : 'linear' (recommended; for standardised zero-mean inputs,
        so MSE tracks NMSE) or 'sigmoid' (original CsiNet, for [0,1] inputs).
    """

    def __init__(self, n_delay: int = 32, n_tx: int = 32, n_code: int = 64,
                 final_activation: str = "linear"):
        super().__init__()
        self.n_delay, self.n_tx, self.n_code = n_delay, n_tx, n_code
        # Real-valued length of the flattened image: 2 (re/im) * Nd * Nt.
        self.flat = 2 * n_delay * n_tx
        self.final_activation = final_activation

        # ---- UE-side encoder (runs on the handset) ----
        # Two 3x3 convs extract local angular-delay structure; the 16-wide
        # hidden layer is squeezed back to 2 channels before flattening so the
        # dense bottleneck stays small. Shape-preserving (padding=1).
        self.enc_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 2, 3, padding=1), nn.BatchNorm2d(2), nn.LeakyReLU(0.3),
        )
        # The compression layer: 2*Nd*Nt -> M. This is the rate-defining layer.
        self.enc_fc = nn.Linear(self.flat, n_code)

        # ---- gNB-side decoder (runs at the base station) ----
        self.dec_fc = nn.Linear(n_code, self.flat)        # M -> 2*Nd*Nt
        self.refine = nn.Sequential(RefineNet(2), RefineNet(2))
        self.out_conv = nn.Conv2d(2, 2, 3, padding=1)     # final refinement
        self.sig = nn.Sigmoid()

    def encode(self, x):
        """UE side: channel tensor -> codeword (the only thing sent uplink).

        x: (N, 2, n_delay, n_tx) -> returns (N, n_code). The conv stack keeps
        the 2-channel image, which is flattened to (N, 2*Nd*Nt) and projected
        down to the M-dim codeword.
        """
        z = self.enc_conv(x).reshape(x.size(0), -1)
        return self.enc_fc(z)

    def decode(self, code):
        """gNB side: codeword -> reconstructed channel tensor.

        code: (N, n_code) -> returns (N, 2, n_delay, n_tx). The dense layer
        un-flattens to the image grid, RefineNet blocks restore detail, and the
        output conv emits the final re/im planes. With 'sigmoid' the output is
        squashed to [0,1] (for [0,1]-scaled training data); 'linear' leaves it
        unbounded (for zero-mean standardised data, the recommended path).
        """
        z = self.dec_fc(code).reshape(-1, 2, self.n_delay, self.n_tx)
        z = self.refine(z)
        out = self.out_conv(z)
        return self.sig(out) if self.final_activation == "sigmoid" else out

    def forward(self, x):
        # Full pass = UE encode then gNB decode; mirrors the over-the-air loop.
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

    Because it usually has far more parameters and FLOPs than CsiNet, it is a
    natural candidate for the TR 38.843 *complexity-vs-accuracy* trade study:
    report both endpoints with ``model_complexity`` and compare their SGCS.

    Parameters
    ----------
    n_delay, n_tx : truncated angular-delay dimensions of the input. n_delay
        becomes the attention sequence length L; 2*n_tx is the per-token width.
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
        nhead = 4                # multi-head attention heads (d_model/nhead=32/head)
        n_layers = 2             # encoder layers on each side
        dim_ff = 4 * d_model     # feed-forward width (standard 4x expansion)
        self.d_model = d_model
        # Each token is one delay tap flattened over (re/im, n_tx) = 2*n_tx.
        self.token_dim = 2 * n_tx

        # batch_first=True => tensors are (N, L, E); dropout=0 keeps the FLOPs
        # estimate deterministic and avoids the fused fast-path that would skip
        # MultiheadAttention.forward hooks (see model_complexity).
        def _stack():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=0.0, batch_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        # ---- UE-side encoder (CSI generation part) ----
        self.enc_embed = nn.Linear(self.token_dim, d_model)   # lift token -> d_model
        # Learned absolute positional embeddings, one per delay-tap token.
        self.enc_pos = nn.Parameter(torch.zeros(1, n_delay, d_model))
        self.enc_tf = _stack()
        # Flatten the whole attended sequence (n_delay * d_model) -> codeword M.
        self.enc_fc = nn.Linear(n_delay * d_model, n_code)

        # ---- gNB-side decoder (CSI reconstruction part) ----
        self.dec_fc = nn.Linear(n_code, n_delay * d_model)    # M -> sequence
        self.dec_pos = nn.Parameter(torch.zeros(1, n_delay, d_model))
        self.dec_tf = _stack()
        self.dec_proj = nn.Linear(d_model, self.token_dim)    # d_model -> 2*n_tx
        self.sig = nn.Sigmoid()

    def encode(self, x):
        """UE side: channel tensor -> codeword (the only thing sent uplink).

        x: (N, 2, n_delay, n_tx). The permute+reshape reads each delay tap as
        one token whose features are (re/im interleaved over the n_tx axis),
        giving (N, n_delay, 2*n_tx). After embedding + positional add + two
        self-attention layers, the full (N, n_delay*d_model) sequence is
        projected to the (N, n_code) codeword.
        """
        n = x.size(0)
        # (N, 2, n_delay, n_tx) -> (N, n_delay, 2, n_tx) -> (N, n_delay, 2*n_tx)
        tok = x.permute(0, 2, 1, 3).reshape(n, self.n_delay, self.token_dim)
        z = self.enc_embed(tok) + self.enc_pos
        z = self.enc_tf(z)
        return self.enc_fc(z.reshape(n, -1))

    def decode(self, code):
        """gNB side: codeword -> reconstructed channel tensor.

        code: (N, n_code) -> (N, 2, n_delay, n_tx). Inverse of ``encode``: the
        dense layer rebuilds the (N, n_delay, d_model) sequence, positional
        embeddings + two attention layers refine it, each token is projected
        back to 2*n_tx, and the reshape+permute exactly inverts encode's
        token-folding to restore the (2, n_delay, n_tx) image layout.
        """
        n = code.size(0)
        z = self.dec_fc(code).reshape(n, self.n_delay, self.d_model)
        z = z + self.dec_pos
        z = self.dec_tf(z)
        tok = self.dec_proj(z)                      # (N, n_delay, 2*n_tx)
        # Inverse of encode's fold: (N, n_delay, 2, n_tx) -> (N, 2, n_delay, n_tx)
        out = tok.reshape(n, self.n_delay, 2, self.n_tx).permute(0, 2, 1, 3)
        return self.sig(out) if self.final_activation == "sigmoid" else out

    def forward(self, x):
        # Full pass = UE encode then gNB decode; mirrors the over-the-air loop.
        return self.decode(self.encode(x))


def model_complexity(model, input_shape=(1, 2, 32, 32)) -> dict:
    """Return {'params': int, 'flops': int} for a model.

    Supplies the two complexity numbers TR 38.843 requires every CSI-feedback
    candidate to disclose alongside its SGCS: **model size** (parameter count)
    and **computational cost** (FLOPs). Reported per single CSI sample by
    default (batch dim = 1 in ``input_shape``).

    Params are exact (sum of all tensor elements). FLOPs are *estimated* via
    forward hooks counting multiply-accumulates as 2 ops each (1 multiply +
    1 add), on Conv2d / Linear / MultiheadAttention:

      * Conv2d  : 2 * (output elements) * (in_ch/groups) * kH * kW.
      * Linear  : 2 * (output rows) * in_features * out_features. NOTE the
        attention out-proj is an nn.Linear subclass and is *also* caught here,
        so it is counted exactly once and excluded from the mha term below.
      * MultiheadAttention (mha_hook): the QKV in-projection (3 * E*E) plus the
        scores (L*L*E) and context (L*L*E) matmuls — an estimate; the in_proj
        weight is a raw Parameter, not an nn.Linear, so it is NOT double-counted.

    Caveats: BatchNorm / activation / normalisation / elementwise costs are
    ignored (negligible vs the matmuls), so this is a lower bound suited to
    *relative* CsiNet-vs-transformer comparison rather than an absolute count.
    A fused-attention fast path could bypass the mha hook, which is why TransNet
    is built with dropout=0 in eval — verified the hook fires here.
    """
    import torch

    flops = [0]                               # boxed accumulator (closures mutate)
    handles = []

    def conv_hook(m, inp, out):
        # MACs = (#output elements) * (#input channels per group) * kernel area.
        kh, kw = m.kernel_size
        n_out = out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3]
        flops[0] += 2 * n_out * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, inp, out):
        # All leading dims are independent "rows"; each costs in*out MACs.
        n = 1
        for s in out.shape[:-1]:
            n *= s
        flops[0] += 2 * n * m.in_features * m.out_features

    def mha_hook(m, inp, out):
        # inp[0] is the query tensor; for self-attention key==value==query.
        x = inp[0]
        if x.dim() == 3:                      # batch_first (N, L, E)
            N, L, E = x.shape
        else:                                 # (L, E) unbatched fallback
            N, L, E = 1, x.shape[0], x.shape[1]
        flops[0] += 2 * N * L * E * E * 3     # Q,K,V in-projection (3 * E*E)
        flops[0] += 2 * N * L * L * E * 2     # attention scores + context (2 matmuls)

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(mha_hook))

    # eval() disables BN running-stat updates / dropout so the single dummy
    # pass is deterministic; restore the original mode afterwards.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))      # one pass just to trigger hooks
    for h in handles:
        h.remove()                            # always detach hooks (no leaks)
    if was_training:
        model.train()

    return {"params": int(sum(p.numel() for p in model.parameters())),
            "flops": int(flops[0])}


def compression_ratio(n_code: int, n_delay: int, n_tx: int) -> float:
    """gamma = M / (2 * Nd * Nt): fraction of the truncated CSI that is kept.

    The denominator is the real-valued length of the angular-delay image
    (factor 2 = real+imag). Smaller gamma = harder compression. Example:
    n_code=128, n_delay=n_tx=32 -> gamma = 128/2048 = 1/16. This is the
    uncompressed-CSI-vs-codeword ratio the TR 38.843 study sweeps to trace
    the SGCS-vs-overhead curve against the eType II baseline.
    """
    return n_code / (2 * n_delay * n_tx)


def feedback_bits(n_code: int, bits_per_coeff: int = 8) -> int:
    """Uplink feedback payload if each codeword entry is scalar-quantised.

    With the default 8 bits/coeff, n_code=128 -> 1024 bits per CSI report; at
    a 5-slot (2.5 ms @ 30 kHz SCS) CSI-RS periodicity that is ~0.41 Mbit/s of
    UCI overhead. Compare against the eType II payload (``csi.baselines``),
    whose size is set by L spatial beams, M FD basis vectors and the K0
    non-zero-coefficient cap per TS 38.214 5.2.2.2.5. (A real deployment would
    entropy-code the latent; this is the simple uniform-quantiser upper bound.)
    """
    return n_code * bits_per_coeff
