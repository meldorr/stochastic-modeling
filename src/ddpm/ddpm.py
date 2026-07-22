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

    def _gather(self, buf: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # broadcast over (B, m) latents and (B, C, T) trajectories alike
        return buf[t].view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._gather(self.sqrt_alphas_bar, t, x0)
        s2 = self._gather(self.sqrt_one_minus_alphas_bar, t, x0)
        return s1 * x0 + s2 * noise, noise

    def forward(self, x0: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """DDPM training loss: MSE between predicted and true noise.

        ``mask`` (B, T), from SE(2) free-augmentation, restricts the loss to valid
        (in-TMA) timesteps — zero-padded out-of-box points contribute no gradient.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        pred = self.denoiser(x_t, t)
        if mask is None:
            return F.mse_loss(pred, noise)
        m = mask.unsqueeze(1)                                    # (B,1,T) broadcast over channels
        return ((pred - noise) ** 2 * m).sum() / (m.sum() * pred.shape[1] + 1e-8)

    @torch.no_grad()
    def p_sample(self, x_t, t, deterministic: bool = False):
        """One reverse step x_t -> x_{t-1} (epsilon parameterization)."""
        betas_t = self._gather(self.betas, t, x_t)
        alphas_t = self._gather(self.alphas, t, x_t)
        alphas_bar_t = self._gather(self.alphas_bar, t, x_t)

        eps = self.denoiser(x_t, t)
        mean = (1.0 / torch.sqrt(alphas_t + 1e-8)) * (
            x_t - betas_t / (torch.sqrt(1.0 - alphas_bar_t) + 1e-8) * eps
        )
        if deterministic:
            return mean
        sigma = torch.sqrt(self._gather(self.posterior_variance, t, x_t))
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).view(-1, *([1] * (x_t.dim() - 1))).float()
        return mean + nonzero * sigma * noise

    @torch.no_grad()
    def sample(self, n: int, m: int | None = None, device=None, deterministic: bool = False,
               clamp: float | None = None, shape: tuple | None = None):
        """Generate ``n`` samples from pure noise.

        Latent mode: pass ``m`` -> samples are ``(n, m)``. Trajectory mode: pass
        ``shape=(C, T)`` -> samples are ``(n, C, T)``. ``clamp`` bounds the running
        sample each reverse step (in normalized units, so ~10 = 10 sigma); it never
        bites a well-trained model but stops an under-trained one diverging.
        """
        if device is None:
            device = next(self.parameters()).device
        dims = tuple(shape) if shape is not None else (int(m),)
        x = torch.randn(n, *dims, device=device)
        for ti in reversed(range(self.timesteps)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            x = self.p_sample(x, t, deterministic=deterministic)
            if clamp is not None:
                x = x.clamp(-clamp, clamp)
        return x

    @torch.no_grad()
    def sample_guided(self, n: int, shape: tuple, mask: torch.Tensor, known: torch.Tensor,
                      device=None, resample_u: int = 1, clamp: float | None = None):
        """RePaint-style waypoint inpainting for trajectory samples.

        Forces the masked entries to the given values at every reverse step, so the
        generated ``(n, C, T)`` trajectories pass through the specified waypoints.

        ``mask``/``known`` are ``(C, T)`` (broadcast over the batch) or ``(n, C, T)``,
        both in the model's **standardized** space. ``mask`` is 1 on constrained
        (channel, timestep) entries (e.g. x/y[/alt] at each waypoint index), 0 else.
        ``resample_u`` > 1 enables RePaint's harmonization (renoise a step and redo)
        to smooth the seam between the imposed waypoints and the generated bridge.
        """
        if device is None:
            device = next(self.parameters()).device
        shape = tuple(shape)
        mask = mask.to(device).float()
        known = known.to(device).float()
        if mask.dim() == len(shape):                       # (C,T) -> (n,C,T)
            mask = mask.unsqueeze(0).expand(n, *shape)
            known = known.unsqueeze(0).expand(n, *shape)
        x = torch.randn(n, *shape, device=device)
        for ti in reversed(range(self.timesteps)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            for u in range(max(1, resample_u)):
                x_un = self.p_sample(x, t)                 # model estimate of x_{ti-1}
                if ti - 1 >= 0:                            # known region at noise level ti-1
                    t_prev = torch.full((n,), ti - 1, device=device, dtype=torch.long)
                    known_part, _ = self.q_sample(known, t_prev)
                else:
                    known_part = known                     # final step: clean waypoint values
                x = mask * known_part + (1.0 - mask) * x_un
                if clamp is not None:
                    x = x.clamp(-clamp, clamp)
                if u < resample_u - 1 and ti > 0:          # renoise ti-1 -> ti for another pass
                    beta_t = self._gather(self.betas, t, x)
                    x = torch.sqrt(1.0 - beta_t) * x + torch.sqrt(beta_t) * torch.randn_like(x)
        return x

    def sample_soft_guided(self, n: int, shape: tuple, target: torch.Tensor, mask: torch.Tensor,
                           guidance_scale: float = 1.0, device=None, clamp: float | None = None):
        """Soft (DPS-style) guidance toward a target track — fly *near*, not through.

        At each reverse step it forms the model's clean estimate ``x0_hat``,
        measures the masked distance to ``target``, and nudges ``x_{t-1}`` down that
        gradient. Unlike :meth:`sample_guided` (hard replacement) the waypoints are
        attracted softly, so the trajectory stays on the model's (flyable) manifold
        and bends toward the target instead of snapping onto it.

        ``target``/``mask`` are ``(C, T)`` or ``(n, C, T)`` in standardized space;
        ``mask`` is 1 on guided (channel, timestep) cells (e.g. x/y along the
        desired track). ``guidance_scale`` is the DPS step (normalized by the
        current distance, so ~O(1)); larger = tighter adherence, looser realism.
        Needs autograd, so it is *not* wrapped in ``no_grad``.
        """
        if device is None:
            device = next(self.parameters()).device
        shape = tuple(shape)
        target = target.to(device).float()
        mask = mask.to(device).float()
        if mask.dim() == len(shape):
            target = target.unsqueeze(0).expand(n, *shape)
            mask = mask.unsqueeze(0).expand(n, *shape)

        x = torch.randn(n, *shape, device=device)
        for ti in reversed(range(self.timesteps)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            x = x.detach().requires_grad_(True)
            eps = self.denoiser(x, t)
            abar_t = self._gather(self.alphas_bar, t, x)
            x0_hat = (x - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)

            diff = mask * (x0_hat - target)
            per_sample = diff.flatten(1).pow(2).sum(1)                 # (n,)
            grad, = torch.autograd.grad(per_sample.sum(), x)

            with torch.no_grad():
                betas_t = self._gather(self.betas, t, x)
                alphas_t = self._gather(self.alphas, t, x)
                mean = (1.0 / torch.sqrt(alphas_t + 1e-8)) * (
                    x - betas_t / (torch.sqrt(1.0 - abar_t) + 1e-8) * eps
                )
                if ti > 0:
                    sigma = torch.sqrt(self._gather(self.posterior_variance, t, x))
                    mean = mean + sigma * torch.randn_like(x)
                step = guidance_scale / (torch.sqrt(per_sample) + 1e-8)   # DPS normalization (n,)
                x = mean - step.view(-1, *([1] * (x.dim() - 1))) * grad
                if clamp is not None:
                    x = x.clamp(-clamp, clamp)
        return x.detach()


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
