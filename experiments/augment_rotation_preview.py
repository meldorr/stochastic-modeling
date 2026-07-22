"""Preview the rotation augmentation as a single aggregate overlay.

    python experiments/augment_rotation_preview.py

Take 10 random flights, rotate the whole set about the FAF through 0, 2, ..., 358
deg, and overlay every rotated copy on ONE diagram — i.e. exactly the augmented
distribution the model would see if each flight were randomly rotated before being
noised/denoised. Lets us judge visually how uniformly rotation fills the TMA.

Writes results/augment/rotation_overlay.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANGLES = np.arange(0, 360, 10)         # 0, 10, ..., 350
N_FLIGHTS = 10
SEED = 0


def rotate_about(xy, center, deg):
    """Rotate (...,2) points about center by deg. deg may be an array (broadcast)."""
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    v = xy - center
    x = v[..., 0] * c - v[..., 1] * s
    y = v[..., 0] * s + v[..., 1] * c
    return np.stack([x, y], -1) + center


def main() -> None:
    d = np.load("data/processed.npz", allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    xi, yi = names.index("x"), names.index("y")
    X = d["X"][:, :, [xi, yi]].astype(float)
    FAF = X[:, -1, :].mean(0)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(X), N_FLIGHTS, replace=False)
    flights = X[pick]                                     # (10, T, 2)
    print(f"[aug] FAF=({FAF[0]:.0f},{FAF[1]:.0f})  {N_FLIGHTS} random flights x {len(ANGLES)} rotations "
          f"(step {ANGLES[1]-ANGLES[0]} deg) -> {N_FLIGHTS*len(ANGLES)} overlaid trajectories")

    out = Path("results/augment"); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 11))
    # every rotated copy of the whole set, one colour -> shows the coverage cloud
    for a in ANGLES:
        r = rotate_about(flights, FAF, a)                # (10, T, 2)
        for k in range(N_FLIGHTS):
            ax.plot(r[k, :, 0] / 1000, r[k, :, 1] / 1000, color="#1f77b4", lw=0.35, alpha=0.12)
    # the 10 originals (0 deg) on top so the seed flights are visible
    for k in range(N_FLIGHTS):
        ax.plot(flights[k, :, 0] / 1000, flights[k, :, 1] / 1000, color="#d62728", lw=1.6, alpha=0.95)
    ax.scatter(FAF[0] / 1000, FAF[1] / 1000, marker="*", s=340, color="black", zorder=6, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Rotation augmentation — {N_FLIGHTS} flights swept 0-358 deg about the FAF, overlaid\n"
                 "(red = the 10 originals; blue = all rotated copies)")
    fig.tight_layout(); fig.savefig(out / "rotation_overlay.png", dpi=135); plt.close(fig)
    print(f"[aug] wrote {out}/rotation_overlay.png")


if __name__ == "__main__":
    main()
