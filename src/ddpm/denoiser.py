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
        h = h + self.time(t_emb)  # inject t as a bias
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


# ---------------------------------------------------------------------------
# TCN denoiser
# ---------------------------------------------------------------------------


def _safe_group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest num_groups in {8,4,2,1} that divides channels."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)  # fallback: instance-norm style


class TCNResBlock(nn.Module):
    """Non-causal dilated Conv1d residual block with additive time conditioning.

    Follows the architecture from ``diffusion-models-lab/ddpm.py`` but adapted
    for arbitrary 1-D latent sequences rather than 4-channel trajectory tensors.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        time_dim: int,
        dropout: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2  # symmetric → non-causal
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.norm1 = _safe_group_norm(channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, dilation=dilation
        )
        self.norm2 = _safe_group_norm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]  # (B, C) -> (B, C, 1)
        h = self.drop(F.silu(self.norm2(self.conv2(h))))
        return x + h


class TCNDenoiser(nn.Module):
    """Dilated TCN denoiser for fPCA latent vectors.

    Treats the flat latent ``(B, m)`` as a single-channel 1-D signal ``(B, 1, m)``
    and applies non-causal dilated Conv1d blocks with sinusoidal time conditioning.
    Input/output shape: ``(B, m)``.
    """

    def __init__(
        self,
        m: int,
        channels: int = 64,
        n_blocks: int = 8,
        kernel_size: int = 3,
        dilations: list | None = None,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.m = m
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.stem = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [
                TCNResBlock(
                    channels,
                    kernel_size,
                    dilations[i % len(dilations)],
                    time_dim,
                    dropout,
                )
                for i in range(n_blocks)
            ]
        )
        self.out_norm = _safe_group_norm(channels)
        self.out_proj = nn.Conv1d(channels, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        h = self.stem(x.unsqueeze(1))  # (B, m) -> (B, 1, m) -> (B, C, m)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(F.silu(self.out_norm(h))).squeeze(1)  # (B, m)


# ---------------------------------------------------------------------------
# U-Net MLP denoiser
# ---------------------------------------------------------------------------


class _UNetLevel(nn.Module):
    """One encoder level: n_blocks ResBlocks at dim `dim`, with a Linear
    down-projection to `dim_out` for the next level."""

    def __init__(
        self, dim: int, dim_out: int, n_blocks: int, time_dim: int, dropout: float
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock(dim, time_dim, dropout) for _ in range(n_blocks)]
        )
        self.down = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x, self.down(F.silu(x))  # skip, downsampled


class UNetMLPDenoiser(nn.Module):
    """MLP U-Net denoiser for fPCA latent vectors.

    Encoder widens hidden dim across ``depth`` levels; decoder mirrors it with
    skip-concatenation at each level. Input/output shape: ``(B, m)``.

    Example with hidden_dim=256, depth=3, channel_mult=(1,2,4)::

        enc0: 256 -> skip(256), down -> 512
        enc1: 512 -> skip(512), down -> 1024
        bottleneck: 1024
        dec1: cat(1024, skip512=512) -> Linear(1536->512) -> ResBlocks
        dec0: cat(512,  skip256=256) -> Linear(768->256)  -> ResBlocks
        out:  256 -> m
    """

    def __init__(
        self,
        m: int,
        hidden_dim: int = 256,
        depth: int = 3,
        channel_mult: tuple = (1, 2, 4),
        blocks_per_level: int = 2,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(channel_mult) == depth, "channel_mult must have `depth` entries"
        self.m = m
        dims = [hidden_dim * c for c in channel_mult]  # e.g. [256, 512, 1024]

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_proj = nn.Linear(m, dims[0])

        # encoder
        self.enc_levels = nn.ModuleList()
        for i in range(depth - 1):
            self.enc_levels.append(
                _UNetLevel(dims[i], dims[i + 1], blocks_per_level, time_dim, dropout)
            )

        # bottleneck (no down-projection)
        self.bottleneck = nn.ModuleList(
            [ResBlock(dims[-1], time_dim, dropout) for _ in range(blocks_per_level)]
        )

        # decoder: input is cat(up, skip), so dim = dims[i+1] + dims[i]
        self.dec_projs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):  # depth-2, ..., 0
            self.dec_projs.append(nn.Linear(dims[i + 1] + dims[i], dims[i]))
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(dims[i], time_dim, dropout)
                        for _ in range(blocks_per_level)
                    ]
                )
            )

        self.out_norm = nn.LayerNorm(dims[0])
        self.out_proj = nn.Linear(dims[0], m)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        h = self.in_proj(x)

        # encoder — collect skips
        skips = []
        for level in self.enc_levels:
            skip, h = level(h, t_emb)
            skips.append(skip)

        # bottleneck
        for blk in self.bottleneck:
            h = blk(h, t_emb)

        # decoder
        for proj, blocks, skip in zip(self.dec_projs, self.dec_blocks, reversed(skips)):
            h = F.silu(proj(torch.cat([h, skip], dim=-1)))
            for blk in blocks:
                h = blk(h, t_emb)

        return self.out_proj(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Trajectory-space TCN denoiser (raw-space diffusion, no fPCA)
# ---------------------------------------------------------------------------


class TrajTCNDenoiser(nn.Module):
    """Dilated TCN denoiser over full trajectories ``(B, C, T)``.

    For the E4 raw-space baseline: the DDPM diffuses the standardized trajectory
    tensor directly (C feature channels x T timesteps), no fPCA latent. Reuses
    :class:`TCNResBlock`; the dilation cycle gives a receptive field spanning the
    200-step sequence.
    """

    def __init__(
        self,
        channels_in: int,
        hidden: int = 64,
        n_blocks: int = 10,
        kernel_size: int = 5,
        dilations: list | None = None,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.channels_in = channels_in
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.stem = nn.Conv1d(channels_in, hidden, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [
                TCNResBlock(hidden, kernel_size, dilations[i % len(dilations)], time_dim, dropout)
                for i in range(n_blocks)
            ]
        )
        self.out_norm = _safe_group_norm(hidden)
        self.out_proj = nn.Conv1d(hidden, channels_in, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(F.silu(self.out_norm(h)))
