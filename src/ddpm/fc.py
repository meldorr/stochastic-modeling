"""Fully-connected denoisers — faithful ports of diffusion-models-lab's two
``FCDenoiser`` variants (both named ``FCDenoiser`` there, in different commits):

* :class:`FCDenoiser` — the "fc" arch. Constant-width (hidden=1024, depth=8),
  symmetric skip connections. Reference commit ``85b4980`` ("FCN not U-net shaped").
* :class:`FCUNetDenoiser` — the "fcn_unet" arch. Widths ``(2048,1024,512,256,128)``
  mirrored back, U-Net skips. Reference commit ``19e9741`` ("standard scaler, 10k
  epochs, with Unet architecture improved results a lot").

Only the ``nn.Module`` denoisers are ported; the reference's standalone ``DDPM``
wrapper is not needed — our :class:`~src.ddpm.ddpm.LatentDDPM` provides
q_sample/sampling and calls ``denoiser(x_t, t)`` with integer timesteps, matching
both classes' ``forward(x_t, t)`` signature. Time embedding / block style
(Linear→LayerNorm→SiLU→Dropout, time_dim=256) are kept verbatim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int or float tensor -> (B, dim)."""
        half = self.dim // 2
        device = t.device
        t = t.float()

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device).float()
            / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _block(in_d: int, out_d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_d, out_d),
        nn.LayerNorm(out_d),
        nn.SiLU(),
        nn.Dropout(dropout),
    )


class FCDenoiser(nn.Module):
    """Constant-width fully-connected denoiser (reference "fc", commit 85b4980).

    Fully-connected denoiser for x of shape (B, C, T) = (B, 4, 200); predicts
    epsilon with the same shape. Encoder = ``depth`` blocks all of width
    ``hidden``; symmetric skips; decoder = ``depth`` blocks of (hidden+hidden)->
    hidden with the skips consumed in reverse.
    """

    def __init__(
        self,
        c: int = 4,
        t_len: int = 200,
        hidden: int = 1024,
        time_dim: int = 256,
        depth: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c = c
        self.t_len = t_len
        self.in_dim = c * t_len

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [self.in_dim + time_dim] + [hidden] * depth
        self.enc_layers = nn.ModuleList(
            [_block(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)]
        )

        self.mid = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.dec_layers = nn.ModuleList(
            [_block(hidden + hidden, hidden, dropout) for _ in range(depth)]
        )

        self.out = nn.Linear(hidden, self.in_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        x = x_t.reshape(B, -1)  # (B, 800)

        te = self.time_mlp(self.time_emb(t))  # (B, time_dim)
        h = torch.cat([x, te], dim=-1)

        skips = []
        for layer in self.enc_layers:
            h = layer(h)
            skips.append(h)

        h = self.mid(h)

        for layer, skip in zip(self.dec_layers, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)

        eps = self.out(h)
        return eps.view(B, self.c, self.t_len)


class FCUNetDenoiser(nn.Module):
    """Fully-connected U-Net denoiser (reference "fcn_unet", commit 19e9741).

    U-Net widths (2048, 1024, 512, 256, 128) then mirrored back; predicts epsilon
    of shape (B, C, T). At each up step the skip from the matching encoder width is
    concatenated.
    """

    def __init__(
        self,
        c: int = 4,
        t_len: int = 200,
        time_dim: int = 256,
        widths: tuple[int, ...] = (2048, 1024, 512, 256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c = c
        self.t_len = t_len
        self.in_dim = c * t_len
        self.widths = widths

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder: (x||t) -> 2048 -> 1024 -> 512 -> 256 -> 128
        enc_dims = (self.in_dim + time_dim,) + widths
        self.enc_layers = nn.ModuleList(
            [_block(enc_dims[i], enc_dims[i + 1], dropout) for i in range(len(enc_dims) - 1)]
        )

        # Bottleneck (stays at the narrowest width)
        bottleneck = widths[-1]
        self.mid = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.SiLU(),
        )

        # Decoder: 128 -> 256 -> 512 -> 1024 -> 2048, concat matching-width skip
        dec_dims = widths[::-1]
        self.dec_layers = nn.ModuleList(
            [_block(dec_dims[i] + dec_dims[i + 1], dec_dims[i + 1], dropout)
             for i in range(len(dec_dims) - 1)]
        )

        self.out = nn.Linear(widths[0], self.in_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        x = x_t.reshape(B, -1)

        te = self.time_mlp(self.time_emb(t))
        h = torch.cat([x, te], dim=-1)

        skips = []
        for layer in self.enc_layers:
            h = layer(h)
            skips.append(h)  # [2048, 1024, 512, 256, 128]

        h = self.mid(h)

        up_skips = list(reversed(skips[:-1]))  # [256, 512, 1024, 2048]
        for layer, skip in zip(self.dec_layers, up_skips):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)

        eps = self.out(h)
        return eps.view(B, self.c, self.t_len)
