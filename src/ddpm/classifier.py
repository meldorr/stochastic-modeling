"""Noise-conditioned trajectory classifier for classifier-guided diffusion.

``p(y | x_t, t)``: a small dilated-TCN that reads a (possibly noised) standardized
trajectory ``(B, C, T)`` plus the diffusion timestep and outputs class logits. Its
gradient steers a frozen DDPM toward a behavior class (e.g. holdings) via classifier
guidance (Dhariwal & Nichol 2021).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the diffusion timestep (like the denoiser's)."""
    half = max(1, dim // 2)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class HoldingClassifier(nn.Module):
    """Dilated-TCN classifier over a (B, C, T) trajectory, conditioned on timestep t.

    Exponentially growing dilations give a receptive field spanning the whole track,
    so a racetrack/holding loop (a global shape) is visible to the final pooled head.
    """

    def __init__(self, channels: int, t_len: int, n_classes: int = 2,
                 width: int = 96, tdim: int = 128, n_layers: int = 6):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Conv1d(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(width, width, 3, padding=2 ** i, dilation=2 ** i) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n_classes))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.temb(timestep_embedding(t, self.tdim)).unsqueeze(-1)     # (B, width, 1)
        h = self.inp(x)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + F.silu(nrm(blk(h) + te))                                # time-conditioned residual
        return self.head(h.mean(-1))                                        # global avg-pool -> logits


def load_classifier(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("arch", {})
    clf = HoldingClassifier(ck["channels"], ck["t_len"], width=a.get("width", 96),
                            tdim=a.get("tdim", 128), n_layers=a.get("n_layers", 6)).to(device)
    clf.load_state_dict(ck["model_state"])
    clf.eval()
    return clf, ck
