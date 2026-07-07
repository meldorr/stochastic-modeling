"""Small MLP denoiser with sinusoidal time embeddings for the fPCA latent.

The reference `diffusion-models-lab` denoiser is a TCN U-Net over (B, 4, 200)
trajectory images. Here the DDPM lives in a ~15-25 dim fPCA weight space, so a
compact residual MLP is the right tool. The ``SinusoidalTimeEmbedding`` is kept
identical to the reference for schedule/behaviour parity.
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
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Pre-norm residual MLP block with additive time conditioning."""

    def __init__(self, dim: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.time = nn.Linear(time_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.lin1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)                       # inject t as a bias
        h = self.lin2(self.drop(F.silu(self.norm2(h))))
        return x + h


class MLPDenoiser(nn.Module):
    """Predicts epsilon for latent vectors. Input/output shape ``(B, m)``."""

    def __init__(
        self,
        m: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_proj = nn.Linear(m, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, time_dim, dropout) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, m)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(F.silu(self.out_norm(h)))
