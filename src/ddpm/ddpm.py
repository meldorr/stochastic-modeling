"""DDPM operating purely in the fPCA latent space.

Epsilon-prediction, identical math to `diffusion-models-lab/ddpm.py`, but every
tensor is ``(B, m)`` instead of ``(B, C, T)`` — the buffer gathers reshape to
``(-1, 1)`` rather than ``(-1, 1, 1)``. The denoiser never sees raw trajectory
points; it only ever touches latent weight vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import build_schedule


class LatentDDPM(nn.Module):
    def __init__(self, denoiser: nn.Module, ddpm_cfg: dict):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(ddpm_cfg["timesteps"])
        for name, buf in build_schedule(ddpm_cfg).items():
            self.register_buffer(name, buf)

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return buf[t].view(-1, 1)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._gather(self.sqrt_alphas_bar, t)
        s2 = self._gather(self.sqrt_one_minus_alphas_bar, t)
        return s1 * x0 + s2 * noise, noise

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """DDPM training loss: MSE between predicted and true noise."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        return F.mse_loss(self.denoiser(x_t, t), noise)

    @torch.no_grad()
    def p_sample(self, x_t, t, deterministic: bool = False):
        """One reverse step x_t -> x_{t-1} (epsilon parameterization)."""
        betas_t = self._gather(self.betas, t)
        alphas_t = self._gather(self.alphas, t)
        alphas_bar_t = self._gather(self.alphas_bar, t)

        eps = self.denoiser(x_t, t)
        mean = (1.0 / torch.sqrt(alphas_t + 1e-8)) * (
            x_t - betas_t / (torch.sqrt(1.0 - alphas_bar_t) + 1e-8) * eps
        )
        if deterministic:
            return mean
        sigma = torch.sqrt(self._gather(self.posterior_variance, t))
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).view(-1, 1).float()
        return mean + nonzero * sigma * noise

    @torch.no_grad()
    def sample(self, n: int, m: int, device=None, deterministic: bool = False,
               clamp: float | None = None):
        """Generate ``n`` latent vectors from pure noise.

        ``clamp`` bounds the running sample each reverse step (in normalized
        latent units, so ~10 = 10 sigma). It never bites a well-trained model
        but prevents an under-trained one from diverging to +/-1e4.
        """
        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(n, m, device=device)
        for ti in reversed(range(self.timesteps)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            x = self.p_sample(x, t, deterministic=deterministic)
            if clamp is not None:
                x = x.clamp(-clamp, clamp)
        return x


class EMA:
    """Exponential moving average of model params for stabler sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return self.shadow
