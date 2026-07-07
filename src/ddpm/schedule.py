"""Diffusion noise schedules -> the buffer set a DDPM needs.

Cosine schedule follows Nichol & Dhariwal (2021); linear matches the reference
`diffusion-models-lab/ddpm.py`. Returns plain CPU tensors; the DDPM registers
them as buffers so they move with ``.to(device)``.
"""

from __future__ import annotations

import math

import torch


def _from_alphas_bar(alphas_bar: torch.Tensor) -> dict[str, torch.Tensor]:
    alphas_bar = alphas_bar.clamp(1e-8, 1.0)
    alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])
    betas = (1.0 - alphas_bar / alphas_bar_prev).clamp(0.0, 0.999)
    alphas = 1.0 - betas
    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "alphas_bar_prev": alphas_bar_prev,
        "sqrt_alphas_bar": torch.sqrt(alphas_bar),
        "sqrt_one_minus_alphas_bar": torch.sqrt(1.0 - alphas_bar),
        "posterior_variance": posterior_variance.clamp(min=1e-20),
    }


def cosine_schedule(timesteps: int, s: float = 0.008) -> dict[str, torch.Tensor]:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps + s) / (1.0 + s)) * math.pi / 2) ** 2
    alphas_bar = (f / f[0])[1:].float()
    return _from_alphas_bar(alphas_bar)


def linear_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    return _from_alphas_bar(alphas_bar)


def build_schedule(cfg: dict) -> dict[str, torch.Tensor]:
    if cfg.get("schedule", "cosine") == "linear":
        return linear_schedule(cfg["timesteps"], cfg["beta_start"], cfg["beta_end"])
    return cosine_schedule(cfg["timesteps"], cfg.get("cosine_s", 0.008))
