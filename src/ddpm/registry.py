"""Raw trajectory denoiser registry — the experiment families, all faithful
ports of diffusion-models-lab:

    fc         constant-width fully-connected denoiser (commit 85b4980)
    fcn_unet   fully-connected U-Net denoiser (commit 19e9741)
    tcn_unet   bidirectional TCN U-Net (commit 8846ad0)

All three take and return ``(B, C, T)`` and expose the same epsilon interface
``denoiser(x_t, t)`` with integer timesteps, so they plug straight into
:class:`~src.ddpm.ddpm.LatentDDPM`.
"""

from __future__ import annotations

import torch.nn as nn

from .fc import FCDenoiser, FCUNetDenoiser
from .tcn_unet import TCNUNetDenoiser

RAW_ARCHS = ("fc", "fcn_unet", "tcn_unet")


def build_raw_denoiser(arch: str, c: int, t_len: int, dropout: float = 0.0,
                       base_channels: int = 64) -> nn.Module:
    if arch == "fc":
        return FCDenoiser(c=c, t_len=t_len, hidden=1024, time_dim=256, depth=8, dropout=dropout)
    if arch == "fcn_unet":
        return FCUNetDenoiser(c=c, t_len=t_len, time_dim=256,
                              widths=(2048, 1024, 512, 256, 128), dropout=dropout)
    if arch == "tcn_unet":
        return TCNUNetDenoiser(c=c, t_len=t_len, base_channels=base_channels, dropout=dropout)
    raise ValueError(f"unknown raw arch {arch!r}; choose from {RAW_ARCHS}")
