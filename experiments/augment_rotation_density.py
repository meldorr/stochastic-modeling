"""Rotation-augmentation coverage: many-flight overlay + density heatmap.

    python experiments/augment_rotation_density.py

Checks the uniformity of the rotation-augmented distribution more rigorously than
the 10-flight sketch: a 40-flight swept overlay and a 2D density heatmap of every
rotated point, plus the angular-coverage histogram (should be flat).

Writes results/augment/{rotation_overlay_many.png, rotation_density.png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# one consistent font across every figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 12, "figure.titlesize": 13})

ANGLES = np.arange(0, 360, 10)
N_OVERLAY = 40
N_DENSITY = 400
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
    out = Path("results/augment"); out.mkdir(parents=True, exist_ok=True)

    # --- 1) many-flight overlay ---
    pick = rng.choice(len(X), N_OVERLAY, replace=False)
    fig, ax = plt.subplots(figsize=(11, 11))
    for a in ANGLES:
        r = rotate_about(X[pick], FAF, a)
        for k in range(N_OVERLAY):
            ax.plot(r[k, :, 0] / 1000, r[k, :, 1] / 1000, color="#1f77b4", lw=0.25, alpha=0.05)
    ax.scatter(FAF[0] / 1000, FAF[1] / 1000, marker="*", s=340, color="black", zorder=6, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Rotation augmentation — {N_OVERLAY} flights swept 0-358 deg about the FAF, overlaid")
    fig.tight_layout(); fig.savefig(out / "rotation_overlay_many.png", dpi=135); plt.close(fig)

    # --- 2) density heatmap + angular coverage ---
    pick = rng.choice(len(X), N_DENSITY, replace=False)
    R = 95.0                                             # km half-window about FAF
    bins = 400
    edges = np.linspace(-R, R, bins + 1)
    H = np.zeros((bins, bins))
    ang_pts = []
    for a in ANGLES:
        r = rotate_about(X[pick], FAF, a).reshape(-1, 2)
        dx = (r[:, 0] - FAF[0]) / 1000.0
        dy = (r[:, 1] - FAF[1]) / 1000.0
        h, _, _ = np.histogram2d(dx, dy, bins=[edges, edges])
        H += h
        ang_pts.append(np.degrees(np.arctan2(dx, dy)) % 360)
    ang = np.concatenate(ang_pts)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    ax = axes[0]
    im = ax.imshow(np.log1p(H.T), origin="lower", extent=[-R, R, -R, R], cmap="magma", aspect="equal")
    ax.scatter(0, 0, marker="*", s=200, color="cyan", zorder=6)
    ax.set_xlabel("x - FAF (km)"); ax.set_ylabel("y - FAF (km)")
    ax.set_title(f"Augmented density (log) — {N_DENSITY} flights x {len(ANGLES)} rotations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + count)")

    ax = axes[1]
    ax.hist(ang, bins=72, color="#1f77b4", alpha=0.85)
    ax.axhline(len(ang) / 72, color="red", ls="--", label="perfectly uniform")
    ax.set_xlabel("bearing from FAF (deg)"); ax.set_ylabel("point count")
    ax.set_title("Angular coverage (flat = uniform over all bearings)")
    ax.set_xlim(0, 360); ax.legend()
    fig.tight_layout(); fig.savefig(out / "rotation_density.png", dpi=135); plt.close(fig)

    counts = np.histogram(ang, bins=72)[0]
    cv = float(np.std(counts) / np.mean(counts))
    print(f"[aug] wrote rotation_overlay_many.png + rotation_density.png")
    print(f"[aug] angular-coverage non-uniformity (CV over 72 sectors): {cv:.4f}  (0 = perfectly uniform)")


if __name__ == "__main__":
    main()
