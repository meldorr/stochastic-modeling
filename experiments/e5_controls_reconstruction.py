"""E5 — Position reconstruction: derived controls vs gs/track dead-reckoning.

    python experiments/e5_controls_reconstruction.py

Claim under test: re-integrating *derived* control signals (turn rate, along-track
accel, vertical rate; SavGol w=21 + trapezoid — spec Section 1.4) reconstructs the
flown track far better than dead-reckoning the *recorded* groundspeed/track channels.

Why it wins (and why this is fair): the controls are derived from the positions
themselves, so the (derive -> integrate) round trip is self-consistent; recorded
gs/track are independent measurements that do not exactly match the position
increments, giving dead-reckoning an irreducible error floor (E1). Both routes
below get the same flights, entry states and horizons; dead-reckoning is scored
with both Euler (as used operationally) and trapezoid (same integrator as the
controls route) so the integrator is not the confound.

Writes results/e5_controls_reconstruction/{metrics.json, table.md, overlay.png,
error_growth.png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import DYN, XY, KTS_TO_MS, load_features, out_dir
from src.data.controls import derive_controls, integrate_controls
from src.pipeline.utils import load_config

N_SAMPLE = 2000
HORIZONS = (60, 199)


def deadreckon(track_deg, gs_kts, td, x0, y0, scheme="euler"):
    """Integrate recorded track/groundspeed to positions. (N, T) inputs."""
    dt = np.diff(td, axis=1)
    vx = gs_kts * KTS_TO_MS * np.sin(np.radians(track_deg))
    vy = gs_kts * KTS_TO_MS * np.cos(np.radians(track_deg))
    if scheme == "trapezoid":
        ix = 0.5 * (vx[:, :-1] + vx[:, 1:]) * dt
        iy = 0.5 * (vy[:, :-1] + vy[:, 1:]) * dt
    else:
        ix = vx[:, :-1] * dt
        iy = vy[:, :-1] * dt
    x = np.concatenate([x0[:, None], x0[:, None] + np.cumsum(ix, axis=1)], axis=1)
    y = np.concatenate([y0[:, None], y0[:, None] + np.cumsum(iy, axis=1)], axis=1)
    return np.stack([x, y], axis=-1)


def err_stats(xy_hat, xy_true):
    e = np.hypot(xy_hat[:, :, 0] - xy_true[:, :, 0], xy_hat[:, :, 1] - xy_true[:, :, 1])
    row = {}
    for h in HORIZONS:
        row[f"mean_m@{h}"] = float(e[:, : h + 1].mean())
        row[f"final_m@{h}"] = float(e[:, h].mean())
        row[f"p90_final_m@{h}"] = float(np.percentile(e[:, h], 90))
    return row, e


def main() -> None:
    cfg = load_config("configs/base.yaml")
    cfg["paths"].setdefault("processed", "data/processed.npz")
    out = out_dir(cfg, "e5_controls_reconstruction")

    d_xy = load_features(cfg, XY)
    d_dyn = load_features(cfg, DYN)
    va = d_xy["val_idx"]
    rng = np.random.default_rng(0)
    sel = va[rng.choice(len(va), min(N_SAMPLE, len(va)), replace=False)]

    nx = d_xy["feature_names"]
    x = d_xy["X"][sel][:, :, nx.index("x")].astype(float)
    y = d_xy["X"][sel][:, :, nx.index("y")].astype(float)
    alt = d_xy["X"][sel][:, :, nx.index("altitude")].astype(float)
    td = d_xy["X"][sel][:, :, nx.index("timedelta")].astype(float)
    xy_true = np.stack([x, y], axis=-1)

    nd = d_dyn["feature_names"]
    trk = d_dyn["X"][sel][:, :, nd.index("track")].astype(float)
    gs = d_dyn["X"][sel][:, :, nd.index("groundspeed")].astype(float)

    routes = {}
    routes["deadreckon gs/track (euler)"] = deadreckon(trk, gs, td, x[:, 0], y[:, 0], "euler")
    routes["deadreckon gs/track (trapezoid)"] = deadreckon(trk, gs, td, x[:, 0], y[:, 0], "trapezoid")
    der = derive_controls(x, y, alt, td, cfg["controls"])
    routes["derived controls (SavGol21+trapezoid)"] = integrate_controls(der["entry"], der["controls"])[:, :, :2]

    metrics, curves = {}, {}
    for name, xy_hat in routes.items():
        metrics[name], e = err_stats(xy_hat, xy_true)
        curves[name] = e.mean(0)
        print(f"[e5] {name:40s} " + "  ".join(f"{k}={v:.0f}" for k, v in metrics[name].items()))
    (out / "metrics.json").write_text(json.dumps(
        {"n_flights": int(len(sel)), "routes": metrics}, indent=2))

    hdr = list(next(iter(metrics.values())).keys())
    lines = ["| route | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for name, m in metrics.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.0f}" for k in hdr) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    # --- error growth over time ---
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"deadreckon gs/track (euler)": "#1f77b4",
              "deadreckon gs/track (trapezoid)": "#17becf",
              "derived controls (SavGol21+trapezoid)": "#d62728"}
    for name, c in curves.items():
        ax.plot(c, lw=2, label=name, color=colors.get(name))
    for h in HORIZONS[:-1]:
        ax.axvline(h, color="gray", ls=":", lw=1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean position error (m)")
    ax.set_title(f"E5: position error growth — controls integration vs dead-reckoning ({len(sel)} val flights)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_growth.png", dpi=130)
    plt.close(fig)

    # --- overlay figure: worst dead-reckon flights make the point ---
    e_dr = np.hypot(*(routes["deadreckon gs/track (euler)"] - xy_true).transpose(2, 0, 1))
    order = np.argsort(e_dr.mean(1))
    pick = [order[len(order) // 2], order[-3], order[-2]]     # median + 2 bad cases
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, k in zip(axes, pick):
        ax.plot(xy_true[k, :, 0], xy_true[k, :, 1], color="black", lw=2.2, label="real track")
        ax.plot(*routes["deadreckon gs/track (euler)"][k].T, color="#1f77b4", lw=1.4,
                label="dead-reckon gs/track")
        ax.plot(*routes["derived controls (SavGol21+trapezoid)"][k].T, color="#d62728",
                lw=1.4, ls="--", label="controls re-integration")
        ax.scatter([xy_true[k, 0, 0]], [xy_true[k, 0, 1]], color="green", s=25, zorder=5)
        ax.set_title(f"flight {k}: DR {e_dr[k].mean():.0f} m vs "
                     f"controls {np.hypot(*(routes['derived controls (SavGol21+trapezoid)'][k]-xy_true[k]).T.reshape(2,-1)).mean():.0f} m mean",
                     fontsize=9)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("E5: median + two worst dead-reckoning flights")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=130)
    plt.close(fig)
    print(f"[e5] wrote metrics.json / table.md / error_growth.png / overlay.png -> {out}")


if __name__ == "__main__":
    main()
