"""Preview SE(2) augmentation (rotate about centroid + in-bounds translate) on x/y.

    python experiments/augment_se2_preview.py

Each flight is rotated by theta ~ U(0,360) about its own centroid, then translated
by (dx,dy) drawn from the *allowable box* — the range that keeps its bounding box
inside the TMA (no padding, no rejection). timedelta is untouched. Shows the
resulting coverage and reports how much translation room the long approaches
actually have.

Writes results/augment/se2_preview.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

N_FLIGHTS = 1000
K_SAMPLES = 2                      # SE(2) draws per flight
SEED = 0


def rotate_about(xy, center, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    v = xy - center
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1) + center


def se2_augment(traj, tma_lo, tma_hi, rng):
    """Rotate about centroid, then translate within the in-TMA allowable box."""
    r = rotate_about(traj, traj.mean(0), rng.uniform(0, 360))
    lo = tma_lo - r.min(0)                     # shift that puts the min corner on the TMA edge
    hi = tma_hi - r.max(0)
    shift = np.array([rng.uniform(l, h) if l <= h else 0.0 for l, h in zip(lo, hi)])
    return r + shift, np.where(lo <= hi, hi - lo, 0.0)   # translated traj, per-axis room


def main() -> None:
    d = np.load("data/processed.npz", allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    xi, yi = names.index("x"), names.index("y")
    X = d["X"][:, :, [xi, yi]].astype(float)
    tma_lo = X.reshape(-1, 2).min(0)           # TMA = extent of all real traffic
    tma_hi = X.reshape(-1, 2).max(0)
    print(f"[se2] TMA extent: x {tma_lo[0]/1000:.0f}-{tma_hi[0]/1000:.0f} km  "
          f"y {tma_lo[1]/1000:.0f}-{tma_hi[1]/1000:.0f} km  "
          f"(span {(tma_hi[0]-tma_lo[0])/1000:.0f} x {(tma_hi[1]-tma_lo[1])/1000:.0f} km)")

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(X), N_FLIGHTS, replace=False)
    segs, rooms = [], []
    for k in pick:
        for _ in range(K_SAMPLES):
            t, room = se2_augment(X[k], tma_lo, tma_hi, rng)
            segs.append(t); rooms.append(room)
    rooms = np.array(rooms) / 1000.0
    print(f"[se2] {len(segs)} SE(2) samples.  translation room (km): "
          f"x median {np.median(rooms[:,0]):.0f} (p90 {np.percentile(rooms[:,0],90):.0f}), "
          f"y median {np.median(rooms[:,1]):.0f} (p90 {np.percentile(rooms[:,1],90):.0f})")

    out = Path("results/augment"); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(LineCollection(segs, colors="#1f77b4", linewidths=0.3, alpha=0.05))
    ax.add_collection(LineCollection([X[k] for k in pick], colors="#d62728", linewidths=0.4, alpha=0.12))
    ax.plot([tma_lo[0], tma_hi[0], tma_hi[0], tma_lo[0], tma_lo[0]],
            [tma_lo[1], tma_lo[1], tma_hi[1], tma_hi[1], tma_lo[1]], "k--", lw=1, alpha=0.6, label="TMA extent")
    ax.autoscale(); ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    ax.legend([plt.Line2D([0],[0],color="#1f77b4",lw=2), plt.Line2D([0],[0],color="#d62728",lw=2),
               plt.Line2D([0],[0],color="k",ls="--",lw=1)],
              ["SE(2)-augmented", "original flights", "TMA extent"], loc="upper right")
    ax.set_title(f"SE(2) augmentation (rotate about centroid + in-bounds translate) — "
                 f"{N_FLIGHTS} flights x {K_SAMPLES}")
    fig.tight_layout(); fig.savefig(out / "se2_preview.png", dpi=140); plt.close(fig)
    print(f"[se2] wrote {out}/se2_preview.png")


if __name__ == "__main__":
    main()
