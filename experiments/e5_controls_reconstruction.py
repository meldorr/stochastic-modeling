"""E5 — FAF-anchored reconstruction: literature gs/track vs ours, plus controls.

    python experiments/e5_controls_reconstruction.py

Reconstruction follows the literature (deep-traffic-generation-paper
`_walk_latlon(forward=False)`): anchor at the **FAF** (fixed known destination)
and integrate **backward** in time, so error accumulates at the **entry** point —
which is where we measure drift. Every route gets the identical anchor and horizon.

Two questions are answered here:

  1. **Tweaks to the literature gs/track walk.** The reference walks RAW ADS-B
     gs/track in WGS84 lat/lon and scales each step by 0.99. We keep the geodesic
     geometry but use factor 1.0 (the 0.99 groundspeed correction under-shoots on
     our data and roughly triples entry drift). A naive UTM-planar walk with raw
     compass track is an ablation — its grid-convergence bearing error (~-0.43 deg
     at LSZH) costs ~320 m, so geodesic is the right choice.

  2. **Raw vs derived gs/track.** Raw ADS-B gs/track are *independent measurements*
     that don't perfectly integrate back to the recorded x/y. Alternatively we can
     *derive* the heading + speed from the x/y path itself (grid-referenced, so a
     self-consistent planar walk with a single integration). We test the smoothed
     derived signals (SavGol w=21, modeling-realistic) and, as a consistency floor,
     the unsmoothed finite-difference version.

Routes on the figure: literature (raw, geodesic, 0.99), ours (raw, geodesic, 1.0),
ours (derived-from-xy, smoothed), and the derived-controls integrator. The full
planar/geodesic x factor 2x2, the derived floor, and controls land in table.md.

Gap-interpolation artifact flights (Section 1.3) are excluded from every route.
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
from src.data.reckoning import (
    deadreckon_from_faf,
    deadreckon_from_faf_geodesic,
    deadreckon_from_faf_velocity,
    entry_drift,
    geodesic_entry_drift,
    integrate_controls_from_faf,
    latlon_to_local_xy,
)
from src.pipeline.utils import load_config

N_SAMPLE = 2000


def main() -> None:
    cfg = load_config("configs/base.yaml")
    cfg["paths"].setdefault("processed", "data/processed.npz")
    out = out_dir(cfg, "e5_controls_reconstruction")

    d_xy = load_features(cfg, XY)
    d_dyn = load_features(cfg, DYN)
    latlon = np.load(cfg["paths"]["processed"], allow_pickle=True)["latlon"]   # (N,T,2) lat,lon
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
    ll = latlon[sel][ok]                             # (M,T,2) true lat/lon
    xy_true = np.stack([x, y], axis=-1)
    faf_xy = xy_true[:, -1, :]                       # planar anchor = true last x/y
    faf_ll = ll[:, -1, :]                            # geodesic anchor = true last lat/lon
    controls, faf_state = der["controls"][ok], der["faf"][ok]

    # ---- reconstructions ----
    # (a) RAW ADS-B gs/track (independent measurement), geodesic walk
    geo_ref = deadreckon_from_faf_geodesic(trk, gs, td, faf_ll, gs_factor=0.99)   # literature
    geo_ours = deadreckon_from_faf_geodesic(trk, gs, td, faf_ll, gs_factor=1.0)   # ours
    # (b) DERIVED gs/track: heading + speed from the x/y path itself (grid-referenced,
    #     consistent-by-construction) -> self-consistent planar walk, single integration
    gs_s, chi_s = der["gs"][ok], der["chi"][ok]                                   # SavGol-smoothed
    deriv_sm = deadreckon_from_faf_velocity(chi_s, gs_s, td, faf_xy)
    dtc = ((td[:, -1] - td[:, 0]) / (td.shape[1] - 1))[:, None]                   # per-flight grid step
    vx, vy = np.gradient(x, axis=1) / dtc, np.gradient(y, axis=1) / dtc
    gs_raw, chi_raw = np.hypot(vx, vy), np.unwrap(np.arctan2(vx, vy), axis=1)
    deriv_raw = deadreckon_from_faf_velocity(chi_raw, gs_raw, td, faf_xy)         # unsmoothed floor
    # (c) derived controls (double integration)
    ctrl_xy = integrate_controls_from_faf(faf_state, controls)[:, :, :2]
    # planar ablation (raw compass track in the UTM grid — grid-convergence error)
    plan_099 = deadreckon_from_faf(trk, gs, td, faf_xy, gs_factor=0.99)
    plan_100 = deadreckon_from_faf(trk, gs, td, faf_xy, gs_factor=1.0)

    # ---- metrics (geodesic routes measured in geodesic metres; xy routes in the plane) ----
    metrics, curves = {}, {}
    def add_geo(name, recon_ll):
        m = geodesic_entry_drift(recon_ll, ll)
        curves[name] = m.pop("curve_m")
        metrics[name] = m
    def add_xy(name, recon_xy):
        metrics[name] = entry_drift(recon_xy, xy_true)
        curves[name] = np.linalg.norm(recon_xy - xy_true, axis=2).mean(0)

    add_geo("gs/track RAW-ADSB geodesic (factor 0.99, literature)", geo_ref)
    add_geo("gs/track RAW-ADSB geodesic (factor 1.0, ours)", geo_ours)
    add_xy("gs/track DERIVED-from-xy, smoothed (ours)", deriv_sm)
    add_xy("derived controls (ours)", ctrl_xy)
    add_xy("gs/track DERIVED-from-xy, unsmoothed (raw finite-diff)", deriv_raw)
    add_xy("[ablation] gs/track planar-UTM raw (factor 0.99)", plan_099)
    add_xy("[ablation] gs/track planar-UTM raw (factor 1.0)", plan_100)

    for name, m in metrics.items():
        print(f"[e5] {name:46s} entry mean {m['entry_mean_m']:5.0f}m  "
              f"median {m['entry_median_m']:5.0f}m  p90 {m['entry_p90_m']:5.0f}m  "
              f"path {m['path_mean_m']:5.0f}m")
    metrics["_n_flights"] = int(len(xy_true))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- table.md (full 2x2 ablation + controls) ----
    hdr = ["entry_mean_m", "entry_median_m", "entry_p90_m", "path_mean_m"]
    labels = {"entry_mean_m": "entry mean (m)", "entry_median_m": "entry median (m)",
              "entry_p90_m": "entry p90 (m)", "path_mean_m": "path mean (m)"}
    lines = ["| route | " + " | ".join(labels[h] for h in hdr) + " |",
             "|---|" + "---|" * len(hdr)]
    for name, m in metrics.items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | " + " | ".join(f"{m[h]:.0f}" for h in hdr) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    # ---- error growth: raw gs/track vs derived gs/track vs controls ----
    main = ["gs/track RAW-ADSB geodesic (factor 0.99, literature)",
            "gs/track RAW-ADSB geodesic (factor 1.0, ours)",
            "gs/track DERIVED-from-xy, smoothed (ours)",
            "derived controls (ours)"]
    colors = {main[0]: "#17becf", main[1]: "#1f77b4", main[2]: "#2ca02c", main[3]: "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in main:
        ax.plot(curves[name], lw=2.2, label=f"{name}  (entry {metrics[name]['entry_mean_m']:.0f} m)",
                color=colors[name])
    ax.set_xlabel("timestep (0 = entry, 199 = FAF anchor)")
    ax.set_ylabel("mean position error (m)")
    ax.set_title(f"E5: FAF-anchored reconstruction error ({len(xy_true)} val flights)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_growth.png", dpi=130)
    plt.close(fig)

    # ---- overlay: worst flights for ours (gs/track) and controls ----
    geo_ours_xy = latlon_to_local_xy(geo_ours, ref_latlon=ll, ref_xy=xy_true)     # project for plotting
    e_dr = np.linalg.norm(geo_ours_xy[:, 0] - xy_true[:, 0], axis=1)
    e_ct = np.linalg.norm(ctrl_xy[:, 0] - xy_true[:, 0], axis=1)
    rows = [("worst gs/track (ours) entry drift", np.argsort(e_dr)[-3:]),
            ("worst controls entry drift", np.argsort(e_ct)[-3:])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for r, (row_title, pick) in enumerate(rows):
        for cidx, k in enumerate(pick):
            ax = axes[r, cidx]
            ax.plot(xy_true[k, :, 0], xy_true[k, :, 1], color="black", lw=2.2, label="real")
            ax.plot(*geo_ours_xy[k].T, color="#1f77b4", lw=1.3, label="gs/track geodesic (ours)")
            ax.plot(*ctrl_xy[k].T, color="#d62728", lw=1.3, ls="--", label="controls (ours)")
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
