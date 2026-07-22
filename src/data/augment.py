"""On-the-fly SE(2) augmentation for trajectory batches (rigid = flyable).

Operates in the model's **normalized** space (isotropic x/y, TMA ~ the unit box),
so a rotation + translation of the standardized trajectory is a true rigid motion
and every kinematic quantity (speed, turn rate, curvature) is preserved — the
augmented trajectory is exactly as flyable as the original.

``se2_augment`` rotates each sample about its own centroid by theta ~ U(0, 2pi)
and translates it by (dx, dy) drawn from the *allowable box* — the range that
keeps its bounding box inside ``[-box, box]``. That guarantees the sample stays in
the TMA (no padding, no rejection); the only fallback is a shape larger than the
box on some axis, which is simply centred on that axis. ``timedelta`` and any other
channels are left untouched.
"""

from __future__ import annotations

import math

import torch


def _uniform_in(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Elementwise U(lo, hi); where lo > hi (shape wider than the box) -> midpoint."""
    u = torch.rand_like(lo)
    samp = lo + u * (hi - lo)
    mid = 0.5 * (lo + hi)
    return torch.where(hi >= lo, samp, mid)


def se2_augment(xb: torch.Tensor, x_idx: int, y_idx: int, box: float = 1.0,
                free: bool = False):
    """Rigid SE(2) augmentation of a normalized batch ``(B, C, T)``.

    Rotates the ``x_idx``/``y_idx`` channels about each sample's centroid, then
    translates. Two modes:

    * ``free=False`` (bounded): translate within the in-TMA allowable box so the
      whole trajectory stays inside ``[-box, box]``. Returns ``(out, None)``.
    * ``free=True``: translate so the centroid is **uniform over ``[-box, box]``**
      (respecting the uniform-position assumption), allowing part of the trajectory
      to fall outside the TMA. Out-of-box points are zero-padded and flagged in a
      per-timestep validity ``mask`` ``(B, T)`` (1 = inside, 0 = padded) so the
      training loss can drop their gradients. Returns ``(out, mask)``.
    """
    b = xb.shape[0]
    dev = xb.device
    theta = torch.rand(b, device=dev) * (2 * math.pi)
    cos, sin = torch.cos(theta)[:, None], torch.sin(theta)[:, None]      # (B, 1)

    x, y = xb[:, x_idx], xb[:, y_idx]                                    # (B, T)
    cx, cy = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    x0, y0 = x - cx, y - cy
    xr = x0 * cos - y0 * sin                                            # centroid at origin
    yr = x0 * sin + y0 * cos

    out = xb.clone()
    if not free:
        xr, yr = xr + cx, yr + cy                                      # restore centroid
        dx = _uniform_in(-box - xr.amin(1), box - xr.amax(1))          # keep it all in-box
        dy = _uniform_in(-box - yr.amin(1), box - yr.amax(1))
        out[:, x_idx] = xr + dx[:, None]
        out[:, y_idx] = yr + dy[:, None]
        return out, None

    # free: centroid uniform over [-box, box]; parts may fall out -> pad + mask
    tx = (torch.rand(b, 1, device=dev) * 2.0 - 1.0) * box
    ty = (torch.rand(b, 1, device=dev) * 2.0 - 1.0) * box
    xr, yr = xr + tx, yr + ty
    mask = ((xr.abs() <= box) & (yr.abs() <= box)).float()             # (B, T) 1=inside
    out[:, x_idx] = xr * mask                                          # zero-pad outside
    out[:, y_idx] = yr * mask
    return out, mask
