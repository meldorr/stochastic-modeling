"""TCN U-Net denoiser for raw trajectory diffusion — ported from the reference
implementation in `diffusion-models-lab/ddpm.py` (same blocks, same layout).

Structure (defaults): stem Conv1d -> 4 encoder levels of 3 dilated TCN res-blocks
(channels 64/128/256/512, dilations 1/2/4) with stride-2 downsampling between
levels (200 -> 100 -> 50 -> 25), a 3-block bottleneck, then a mirrored decoder
with ConvTranspose upsampling and skip concatenation. Time conditioning is a
sinusoidal embedding (dim 256) injected as a per-channel bias inside each block.
Predicts epsilon; input/output ``(B, C, T)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoiser import SinusoidalTimeEmbedding, _safe_group_norm


class TemporalBlock1D(nn.Module):
    """Dilated Conv1d (length-preserving, non-causal) + GN + SiLU + dropout."""

    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad)
        self.norm = _safe_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.silu(self.norm(self.conv(x))))


class UNetTCNResBlock(nn.Module):
    """Two TemporalBlocks + time bias + 1x1 skip. (B, Cin, L) -> (B, Cout, L)."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int, time_dim: int, dropout: float):
        super().__init__()
        self.tb1 = TemporalBlock1D(cin, cout, k, dilation, dropout)
        self.tb2 = TemporalBlock1D(cout, cout, k, dilation, dropout)
        self.time_proj = nn.Linear(time_dim, cout)
        self.skip = nn.Conv1d(cin, cout, kernel_size=1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.tb1(x)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]
        h = self.tb2(h)
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)  # L -> L/2

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)  # L -> 2L

    def forward(self, x):
        return self.deconv(x)


class TCNUNetDenoiser(nn.Module):
    """U-Net of dilated TCN residual blocks. Input/output ``(B, c, t_len)``."""

    def __init__(
        self,
        c: int = 4,
        t_len: int = 200,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dilations_per_level: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert t_len % (2 ** (len(channel_mults) - 1)) == 0, (
            f"t_len must be divisible by 2^(#downs); got t_len={t_len}, "
            f"downs={len(channel_mults) - 1}"
        )
        self.c = c
        self.t_len = t_len

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.stem = nn.Conv1d(c, base_channels, kernel_size=3, padding=1)

        chs = [base_channels * m for m in channel_mults]

        self.enc_levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(chs):
            blocks = []
            for d in dilations_per_level:
                blocks.append(UNetTCNResBlock(in_ch, out_ch, kernel_size, d, time_dim, dropout))
                in_ch = out_ch
            self.enc_levels.append(nn.ModuleList(blocks))
            if i != len(chs) - 1:
                self.downs.append(Downsample1D(out_ch))

        mid_ch = chs[-1]
        self.mid = nn.ModuleList(
            [UNetTCNResBlock(mid_ch, mid_ch, kernel_size, d, time_dim, dropout)
             for d in dilations_per_level]
        )

        self.ups = nn.ModuleList()
        self.dec_levels = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            up_ch, out_ch = chs[i + 1], chs[i]
            self.ups.append(Upsample1D(up_ch))
            blocks = []
            in_ch = up_ch + out_ch  # upsampled + skip concat
            for d in dilations_per_level:
                blocks.append(UNetTCNResBlock(in_ch, out_ch, kernel_size, d, time_dim, dropout))
                in_ch = out_ch
            self.dec_levels.append(nn.ModuleList(blocks))

        self.head_norm = _safe_group_norm(base_channels)
        self.head = nn.Conv1d(base_channels, c, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        h = self.stem(x_t)

        skips = []
        for lvl, blocks in enumerate(self.enc_levels):
            for blk in blocks:
                h = blk(h, t_emb)
            skips.append(h)
            if lvl < len(self.downs):
                h = self.downs[lvl](h)

        for blk in self.mid:
            h = blk(h, t_emb)

        _ = skips.pop()  # deepest skip shares the bottleneck resolution
        for up, blocks in zip(self.ups, self.dec_levels):
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:  # guard against off-by-one lengths
                L = min(h.shape[-1], skip.shape[-1])
                h, skip = h[..., :L], skip[..., :L]
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, t_emb)

        return self.head(F.silu(self.head_norm(h)))
