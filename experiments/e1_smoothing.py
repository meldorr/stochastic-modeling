"""E1 — Representation study: does smoothing kill the dynamical channels?

    python experiments/e1_smoothing.py

Hypotheses under test (no training involved):
    H1  A spline basis smooths groundspeed/track; dead-reckoned paths degrade badly.
    H2  Discrete fPCA on the dynamical channels also smooths; still bad.
    H3  Representing x/y directly keeps position error bounded (no integration).

Design: fit each representation on the train split, encode->decode the val split,
then measure (a) per-feature raw-unit RMSE, (b) high-frequency retention of
gs/track, and (c) the *position* error of the resulting path:
    - DYN reps: geodesic dead-reckon backward from the FAF (the canonical method,
      factor 1.0 — same as E5, so representation error is not confounded by the
      grid-convergence artifact of a planar walk), project to x/y, measure drift at
      the entry. Control row: dead-reckon with the REAL dynamics (pure integration
      floor, isolates it from representation error).
    - XY reps: decoded x/y vs true x/y directly.

Writes results/e1_smoothing/{metrics.json, table.md, *.png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    DYN,
    XY,
    hf_retention,
    load_features,
    out_dir,
    raw_rmse_named,
    save_json,
)
from src.data.reckoning import deadreckon_from_faf_geodesic, entry_drift, latlon_to_local_xy
from src.fpca import FPCA
from src.pipeline.utils import load_config

BSPLINE_NBASIS = [20, 40, 80]
FPCA_EV = [0.95, 0.99, 0.999]


def fit_rep(Xs_tr, names, kind, param):
    if kind == "bspline":
        # ev ~ 1 so PCA truncation is negligible -> isolates the spline projection
        return FPCA.fit(Xs_tr, names, explained_variance=0.9999, max_components=200,
                        basis="bspline",
                        bspline_cfg={"n_basis": param, "degree": 3, "smoothing": 1e-3})
    return FPCA.fit(Xs_tr, names, explained_variance=param, max_components=200,
                    basis="discrete")


def eval_dyn_rep(rep, d_dyn, xy_true, ll, va):
    """Reconstruction + geodesic FAF-reckoned path metrics for a DYN representation."""
    names = d_dyn["feature_names"]
    Xs_va = d_dyn["X_std"][va]
    rec_std = rep.decode(rep.encode(Xs_va))
    rec_raw = d_dyn["scaler"].inverse_transform(rec_std)
    real_raw = d_dyn["X"][va]

    ti, gi, tdi = names.index("track"), names.index("groundspeed"), names.index("timedelta")
    # geodesic reverse from the FAF (true last lat/lon), project to x/y, drift at entry
    ll_hat = deadreckon_from_faf_geodesic(rec_raw[:, :, ti], rec_raw[:, :, gi], rec_raw[:, :, tdi],
                                          ll[:, -1, :], gs_factor=1.0)
    xy_hat = latlon_to_local_xy(ll_hat, ref_latlon=ll, ref_xy=xy_true)
    return {
        "rmse_raw": raw_rmse_named(rec_std, Xs_va, d_dyn["scaler"], names),
        "hf_gs": hf_retention(rec_raw[:, :, gi], real_raw[:, :, gi]),
        "hf_track": hf_retention(rec_raw[:, :, ti], real_raw[:, :, ti]),
        "drift": entry_drift(xy_hat, xy_true),
        "m": int(rep.m),
    }, rec_raw, xy_hat


def eval_xy_rep(rep, d_xy, va):
    names = d_xy["feature_names"]
    Xs_va = d_xy["X_std"][va]
    rec_std = rep.decode(rep.encode(Xs_va))
    rec_raw = d_xy["scaler"].inverse_transform(rec_std)
    xi, yi = names.index("x"), names.index("y")
    xy_hat = rec_raw[:, :, [xi, yi]].astype(np.float64)
    xy_true = d_xy["X"][va][:, :, [xi, yi]].astype(np.float64)
    return {
        "rmse_raw": raw_rmse_named(rec_std, Xs_va, d_xy["scaler"], names),
        "drift": entry_drift(xy_hat, xy_true),      # direct decode; drift = decode error at entry
        "m": int(rep.m),
    }, rec_raw, xy_hat


def main() -> None:
    cfg = load_config()
    cfg.setdefault("paths", {}).setdefault("processed", "data/processed.npz")
    out = out_dir(cfg, "e1_smoothing")

    d_dyn = load_features(cfg, DYN)
    d_xy = load_features(cfg, XY)
    tr, va = d_dyn["train_idx"], d_dyn["val_idx"]          # identical split for both
    xi, yi = d_xy["feature_names"].index("x"), d_xy["feature_names"].index("y")
    xy_true = d_xy["X"][va][:, :, [xi, yi]].astype(np.float64)
    ll = np.load(cfg["paths"]["processed"], allow_pickle=True)["latlon"][va]   # (Nval,T,2) FAF anchor source

    rows, recons = {}, {}

    # control: geodesic FAF-reverse dead-reckon with REAL dynamics (pure integration floor)
    real_raw = d_dyn["X"][va]
    names_d = d_dyn["feature_names"]
    ti, gi, tdi = names_d.index("track"), names_d.index("groundspeed"), names_d.index("timedelta")
    ll_dr_real = deadreckon_from_faf_geodesic(real_raw[:, :, ti], real_raw[:, :, gi], real_raw[:, :, tdi],
                                              ll[:, -1, :], gs_factor=1.0)
    xy_dr_real = latlon_to_local_xy(ll_dr_real, ref_latlon=ll, ref_xy=xy_true)
    rows["dyn|real|deadreckon-control"] = {"drift": entry_drift(xy_dr_real, xy_true), "m": None}

    # dyn representations
    for kind, params in (("bspline", BSPLINE_NBASIS), ("fpca", FPCA_EV)):
        for p in params:
            rep = fit_rep(d_dyn["X_std"][tr], names_d, kind, p)
            key = f"dyn|{kind}|{p}"
            rows[key], rec_raw, xy_hat = eval_dyn_rep(rep, d_dyn, xy_true, ll, va)
            recons[key] = (rec_raw, xy_hat)
            print(f"[e1] {key:24s} m={rows[key]['m']:3d} "
                  f"entry drift={rows[key]['drift']['entry_mean_m']:.0f}m "
                  f"hf_gs={rows[key].get('hf_gs', float('nan')):.2f} hf_track={rows[key].get('hf_track', float('nan')):.2f}")

    # xy representations
    for kind, params in (("bspline", BSPLINE_NBASIS), ("fpca", FPCA_EV)):
        for p in params:
            rep = fit_rep(d_xy["X_std"][tr], d_xy["feature_names"], kind, p)
            key = f"xy|{kind}|{p}"
            rows[key], rec_raw, xy_hat = eval_xy_rep(rep, d_xy, va)
            recons[key] = (rec_raw, xy_hat)
            print(f"[e1] {key:24s} m={rows[key]['m']:3d} entry drift={rows[key]['drift']['entry_mean_m']:.0f}m")

    save_json(out / "metrics.json", rows)

    # ---- table.md ----
    lines = ["| representation | m | entry drift mean (m) | entry p90 (m) | gs RMSE (kts) | track RMSE (deg) | hf gs | hf track |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        rm = r.get("rmse_raw", {})
        lines.append(
            f"| {k} | {r.get('m') or '—'} | {r['drift']['entry_mean_m']:.0f} | {r['drift']['entry_p90_m']:.0f} "
            f"| {rm.get('groundspeed', float('nan')):.2f} | {rm.get('track', float('nan')):.2f} "
            f"| {r.get('hf_gs', float('nan')):.2f} | {r.get('hf_track', float('nan')):.2f} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n")

    # ---- figures ----
    # 1) gs & track overlays for 3 val flights (real vs bspline-40 vs fpca-0.99)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(va), 3, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for col, fi in enumerate(pick):
        for rowi, (j, nm) in enumerate([(gi, "groundspeed (kts)"), (ti, "track (deg)")]):
            ax = axes[rowi, col]
            ax.plot(real_raw[fi, :, j], color="black", lw=1.4, label="real")
            ax.plot(recons["dyn|bspline|40"][0][fi, :, j], color="#d62728", lw=1.1, label="bspline-40")
            ax.plot(recons["dyn|fpca|0.99"][0][fi, :, j], color="#1f77b4", lw=1.1, label="fPCA-0.99")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(nm)
            if rowi == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("E1: smoothing of dynamical channels (3 val flights)")
    fig.tight_layout()
    fig.savefig(out / "dyn_overlays.png", dpi=130)
    plt.close(fig)

    # 2) dead-reckoned paths for one flight
    fi = pick[0]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(xy_true[fi, :, 0], xy_true[fi, :, 1], color="black", lw=2, label="true path")
    ax.plot(xy_dr_real[fi, :, 0], xy_dr_real[fi, :, 1], "--", color="gray", lw=1.5,
            label="dead-reckon (real dyn) — control")
    ax.plot(*recons["dyn|bspline|40"][1][fi].T, color="#d62728", lw=1.3, label="dead-reckon (bspline-40)")
    ax.plot(*recons["dyn|fpca|0.99"][1][fi].T, color="#1f77b4", lw=1.3, label="dead-reckon (fPCA-0.99)")
    ax.plot(*recons["xy|fpca|0.99"][1][fi].T, color="#2ca02c", lw=1.6, label="fPCA on x/y (direct)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("E1: position error — integrated dynamics vs direct x/y")
    fig.tight_layout()
    fig.savefig(out / "path_comparison.png", dpi=130)
    plt.close(fig)

    # 3) bar chart: entry drift by representation
    keys = [k for k in rows if k != "dyn|real|deadreckon-control"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    vals = [rows[k]["drift"]["entry_mean_m"] for k in keys]
    colors = ["#d62728" if k.startswith("dyn|bspline") else
              "#1f77b4" if k.startswith("dyn|fpca") else
              "#ff9896" if k.startswith("xy|bspline") else "#2ca02c" for k in keys]
    ax.bar(range(len(keys)), vals, color=colors)
    ax.axhline(rows["dyn|real|deadreckon-control"]["drift"]["entry_mean_m"], color="gray", ls="--",
               label="control: FAF-reverse dead-reckon w/ real dynamics")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("|", "\n") for k in keys], fontsize=7)
    ax.set_ylabel("mean entry drift (m)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("E1: entry drift by representation, FAF-anchored (log scale)")
    fig.tight_layout()
    fig.savefig(out / "path_error_bars.png", dpi=130)
    plt.close(fig)

    print(f"[e1] wrote {out}/{{metrics.json, table.md, 3 figures}}")


if __name__ == "__main__":
    main()
