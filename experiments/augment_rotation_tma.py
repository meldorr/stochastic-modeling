"""Render the rotation-augmented airspace as a normal TMA traffic plot.

    python experiments/augment_rotation_tma.py

1000 random flights, each rotated about the FAF through 0, 10, ..., 350 deg
(36 orientations) -> 36000 trajectories, drawn in the usual TMA style so we can
see what the augmented airspace looks like as "real" traffic.

Writes results/augment/rotation_tma.png.
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

ANGLES = np.arange(0, 360, 10)
N_FLIGHTS = 1000
SEED = 0


def rotate_about(xy, center, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    v = xy - center
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1) + center


def main() -> None:
    d = np.load("data/processed.npz", allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    xi, yi = names.index("x"), names.index("y")
    X = d["X"][:, :, [xi, yi]].astype(float)
    FAF = X[:, -1, :].mean(0)

    rng = np.random.default_rng(SEED)
    flights = X[rng.choice(len(X), N_FLIGHTS, replace=False)]          # (1000, T, 2)
    segs = []
    for a in ANGLES:
        segs.extend(list(rotate_about(flights, FAF, a)))              # 1000 arrays per angle
    print(f"[aug] {N_FLIGHTS} flights x {len(ANGLES)} rotations (step {ANGLES[1]-ANGLES[0]} deg) "
          f"-> {len(segs)} trajectories")

    out = Path("results/augment"); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    lc = LineCollection(segs, colors="#1f77b4", linewidths=0.3, alpha=0.04)
    ax.add_collection(lc)
    # original (unrotated) flights in red on top -> the real operational corridors
    orig = LineCollection(list(flights), colors="#d62728", linewidths=0.4, alpha=0.12)
    ax.add_collection(orig)
    ax.autoscale(); ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    ax.legend([plt.Line2D([0], [0], color="#1f77b4", lw=2),
               plt.Line2D([0], [0], color="#d62728", lw=2)],
              ["rotated copies (augmented)", f"original {N_FLIGHTS} flights"], loc="upper right")
    ax.set_title(f"Rotation-augmented TMA — {N_FLIGHTS} flights x {len(ANGLES)} rotations "
                 f"({len(segs)} trajectories); red = originals")
    fig.tight_layout(); fig.savefig(out / "rotation_tma.png", dpi=140); plt.close(fig)
    print(f"[aug] wrote {out}/rotation_tma.png")


if __name__ == "__main__":
    main()
