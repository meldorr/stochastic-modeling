"""Raw trajectory denoiser registry — the experiment families:

    fc         plain fully-connected net on the flattened (C*T) trajectory
    fcn_unet   fully-connected U-Net on the flattened trajectory
    tcn_unet   reference TCN U-Net (diffusion-models-lab), operates on (B, C, T)

``fc``/``fcn_unet`` reuse the latent MLP denoisers behind a flatten adapter, so
all three expose the same (B, C, T) -> (B, C, T) epsilon interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .denoiser import MLPDenoiser, UNetMLPDenoiser
from .tcn_unet import TCNUNetDenoiser

RAW_ARCHS = ("fc", "fcn_unet", "tcn_unet")


class FlattenDenoiser(nn.Module):
    """Adapter: run a (B, m)-denoiser on flattened (B, C, T) trajectories."""

    def __init__(self, inner: nn.Module, c: int, t_len: int):
        super().__init__()
        self.inner = inner
        self.c = c
        self.t_len = t_len

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.inner(x.flatten(1), t)
        return out.view(-1, self.c, self.t_len)


def build_raw_denoiser(arch: str, c: int, t_len: int, dropout: float = 0.0) -> nn.Module:
    m = c * t_len
    if arch == "fc":
        return FlattenDenoiser(
            MLPDenoiser(m=m, hidden_dim=1024, n_blocks=4, time_dim=128, dropout=dropout),
            c, t_len,
        )
    if arch == "fcn_unet":
        return FlattenDenoiser(
            UNetMLPDenoiser(m=m, hidden_dim=512, depth=3, channel_mult=(1, 2, 4),
                            blocks_per_level=2, time_dim=128, dropout=dropout),
            c, t_len,
        )
    if arch == "tcn_unet":
        return TCNUNetDenoiser(c=c, t_len=t_len, dropout=dropout)
    raise ValueError(f"unknown raw arch {arch!r}; choose from {RAW_ARCHS}")
