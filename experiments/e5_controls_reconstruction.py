"""E5 — FAF-anchored reconstruction: derived controls vs gs/track dead-reckoning.

    python experiments/e5_controls_reconstruction.py

Reconstruction follows the literature (deep-traffic-generation-paper
`_walk_latlon(forward=False)`): anchor at the **FAF** (fixed known destination)
and integrate **backward** in time, so error accumulates at the **entry** point —
which is where we measure drift. Both representations get the identical anchor and
horizon:

    * gs/track  -> `deadreckon_from_faf` (Euler, bearing-flip); factor 1.0 (our UTM
      plane; the reference's 0.99 is worse here) and 0.99 (reference) both shown;
    * controls  -> `integrate_controls_from_faf` (backward trapezoid from FAF state).

Gap-interpolation artifact flights (Section 1.3) are excluded from all routes.
Writes results/e5_controls_reconstruction/{metrics.json, table.md, overlay.png,
error_growth.png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import DYN, XY, load_features, out_dir
from src.data.controls import derive_controls
from src.data.reckoning import deadreckon_from_faf, entry_drift, integrate_controls_from_faf
from src.pipeline.utils import load_config

N_SAMPLE = 2000


def main() -> None:
    cfg = load_config("configs/base.yaml")
    cfg["paths"].setdefault("processed", "data/processed.npz")
    out = out_dir(cfg, "e5_controls_reconstruction")

    d_xy = load_features(cfg, XY)
    d_dyn = load_features(cfg, DYN)
    va = d_xy["val_idx"]
    rng = np.random.default_rng(0)
    sel = va[rng.choice(len(va), min(N_SAMPLE, len(va)), replace=False)]

    nx, nd = d_xy["feature_names"], d_dyn["feature_names"]
    x, y, alt, td = (d_xy["X"][sel][:, :, nx.index(c)].astype(float)
                     for c in ("x", "y", "altitude", "timedelta"))
    trk = d_dyn["X"][sel][:, :, nd.index("track")].astype(float)
    gs = d_dyn["X"][sel][:, :, nd.index("groundspeed")].astype(float)

    der = derive_controls(x, y, alt, td, cfg["controls"])
    ok = der["valid_mask"]
    print(f"[e5] artifact flights excluded: {int((~ok).sum())}/{len(ok)} "
          f"({100 * (~ok).mean():.1f}%)")
    x, y, alt, td, trk, gs = (a[ok] for a in (x, y, alt, td, trk, gs))
    xy_true = np.stack([x, y], axis=-1)
    faf_xy = xy_true[:, -1, :]                       # FAF anchor = true last point
    controls, faf_state = der["controls"][ok], der["faf"][ok]

    routes = {
        "gs/track FAF (factor 1.0)": deadreckon_from_faf(trk, gs, td, faf_xy, gs_factor=1.0),
        "gs/track FAF (factor 0.99, ref)": deadreckon_from_faf(trk, gs, td, faf_xy, gs_factor=0.99),
        "derived controls FAF": integrate_controls_from_faf(faf_state, controls)[:, :, :2],
    }

    metrics, curves = {}, {}
    for name, xy_hat in routes.items():
        metrics[name] = entry_drift(xy_hat, xy_true)
        curves[name] = np.linalg.norm(xy_hat - xy_true, axis=2).mean(0)   # (T,) mean over flights
        m = metrics[name]
        print(f"[e5] {name:32s} entry mean {m['entry_mean_m']:5.0f}m  median {m['entry_median_m']:5.0f}m  "
              f"p90 {m['entry_p90_m']:5.0f}m  path {m['path_mean_m']:5.0f}m")
    metrics["_n_flights"] = int(len(xy_true))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # table
    hdr = ["entry_mean_m", "entry_median_m", "entry_p90_m", "path_mean_m"]
    lines = ["| route | " + " | ".join(h.replace("_m", " (m)") for h in hdr) + " |",
             "|---|" + "---|" * len(hdr)]
    for name, m in metrics.items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | " + " | ".join(f"{m[h]:.0f}" for h in hdr) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    # --- error growth (FAF at right ~0, entry at left = max) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"gs/track FAF (factor 1.0)": "#1f77b4",
              "gs/track FAF (factor 0.99, ref)": "#17becf",
              "derived controls FAF": "#d62728"}
    for name, c in curves.items():
        ax.plot(c, lw=2, label=name, color=colors[name])
    ax.set_xlabel("timestep (0 = entry, 199 = FAF anchor)")
    ax.set_ylabel("mean position error (m)")
    ax.set_title(f"E5: FAF-anchored reconstruction error ({len(xy_true)} val flights)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_growth.png", dpi=130)
    plt.close(fig)

    # --- overlay: worst for gs/track (top) and worst for controls (bottom) ---
    e_dr = np.linalg.norm(routes["gs/track FAF (factor 1.0)"][:, 0] - xy_true[:, 0], axis=1)
    e_ct = np.linalg.norm(routes["derived controls FAF"][:, 0] - xy_true[:, 0], axis=1)
    rows = [("worst gs/track entry drift", np.argsort(e_dr)[-3:]),
            ("worst controls entry drift", np.argsort(e_ct)[-3:])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for r, (row_title, pick) in enumerate(rows):
        for cidx, k in enumerate(pick):
            ax = axes[r, cidx]
            ax.plot(xy_true[k, :, 0], xy_true[k, :, 1], color="black", lw=2.2, label="real")
            ax.plot(*routes["gs/track FAF (factor 1.0)"][k].T, color="#1f77b4", lw=1.3, label="gs/track (FAF)")
            ax.plot(*routes["derived controls FAF"][k].T, color="#d62728", lw=1.3, ls="--", label="controls (FAF)")
            ax.scatter(*faf_xy[k], color="green", s=30, zorder=5, label="FAF (anchor)")
            ax.scatter(*xy_true[k, 0], color="orange", s=30, zorder=5, label="true entry")
            ax.set_title(f"flight {k}: gs/trk {e_dr[k]:.0f} m vs ctrl {e_ct[k]:.0f} m entry drift", fontsize=9)
            ax.set_aspect("equal", "datalim")
            ax.grid(alpha=0.3)
        axes[r, 0].set_ylabel(row_title, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("E5: FAF-anchored reconstruction — green=FAF anchor, orange=true entry")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=130)
    plt.close(fig)
    print(f"[e5] wrote metrics.json / table.md / error_growth.png / overlay.png -> {out}")


if __name__ == "__main__":
    main()
