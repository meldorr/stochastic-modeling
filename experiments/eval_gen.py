"""Shared generative metrics — one comparable scorecard for E2/E3/E4.

    python experiments/eval_gen.py --exp e2_global  --generated results/e2_global/generated.npz  --features xy
    python experiments/eval_gen.py --exp e4_raw_dyn --generated results/e4_raw_dyn/generated.npz --features dyn

Metrics (real val split vs generated):
    ks_marginal/<feature>   distribution match per channel (pooled timesteps)
    sliced_w_xy             sliced-Wasserstein on flattened x/y paths (km units)
    ks_endpoint_xy          KS on final x and y (does it hit the FAF region?)
    within_pct              fraction inside the training envelope (+5% margin)
DYN feature sets are dead-reckoned to x/y (true start points from matched real
flights are NOT available for generated samples, so integration starts at the
real flights' mean entry ring — spatial metrics for dyn are therefore reported
on *shape* after centring both real and generated paths at their start).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from experiments.common import (
    DYN,
    XY,
    dead_reckon,
    ks_marginals,
    load_features,
    out_dir,
    save_json,
    sliced_wasserstein,
    within_pct,
    xy_flat,
)
from src.pipeline.utils import load_config, resolve


def to_xy_paths(feats, names, centre=False):
    """x/y paths (N, T, 2) in metres. DYN sets are dead-reckoned from origin."""
    if "x" in names:
        xi, yi = names.index("x"), names.index("y")
        xy = feats[:, :, [xi, yi]].astype(np.float64)
    else:
        ti, gi, tdi = names.index("track"), names.index("groundspeed"), names.index("timedelta")
        z = np.zeros(len(feats))
        xy = dead_reckon(feats[:, :, ti], feats[:, :, gi], feats[:, :, tdi], z, z)
    if centre:
        xy = xy - xy[:, :1, :]
    return xy


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared generative metrics")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--features", choices=["xy", "dyn"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n-real", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = XY if args.features == "xy" else DYN
    out = out_dir(cfg, args.exp)

    data = load_features(cfg, names)
    va = data["val_idx"]
    rng = np.random.default_rng(int(cfg["seed"]))
    ridx = va[rng.choice(len(va), min(args.n_real, len(va)), replace=False)]
    real = data["X"][ridx]

    g = np.load(resolve(args.generated), allow_pickle=True)
    gen = g["feats"]
    gnames = [str(x) for x in g["feature_names"]]
    assert gnames == names, f"feature mismatch: {gnames} vs {names}"

    # dyn has no absolute position -> centre both for spatial comparison
    centre = "x" not in names
    real_xy = to_xy_paths(real, names, centre=centre)
    gen_xy = to_xy_paths(gen, names, centre=centre)

    real_flat = real_xy[:, ::8, :].reshape(len(real_xy), -1) / 1000.0
    gen_flat = gen_xy[:, ::8, :].reshape(len(gen_xy), -1) / 1000.0

    metrics = {
        "exp": args.exp,
        "features": names,
        "n_real": int(len(real)),
        "n_gen": int(len(gen)),
        "ks_marginal": ks_marginals(real, gen, names),
        "sliced_w_xy_km": sliced_wasserstein(real_flat, gen_flat),
        "ks_endpoint_x": float(ks_2samp(real_xy[:, -1, 0], gen_xy[:, -1, 0]).statistic),
        "ks_endpoint_y": float(ks_2samp(real_xy[:, -1, 1], gen_xy[:, -1, 1]).statistic),
        "within_pct": within_pct(gen, data["bounds"], names),
        "spatial_centred": bool(centre),
    }
    save_json(out / "gen_metrics.json", metrics)
    print(f"[{args.exp}] KS marginals: "
          + ", ".join(f"{k}={v:.3f}" for k, v in metrics["ks_marginal"].items()))
    print(f"[{args.exp}] sliced-W(xy)={metrics['sliced_w_xy_km']:.3f} km  "
          f"KS endpoint x/y={metrics['ks_endpoint_x']:.3f}/{metrics['ks_endpoint_y']:.3f}  "
          f"within={metrics['within_pct'] * 100:.1f}%")

    # spatial figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, xy, ttl, c in ((axes[0], real_xy, f"Real ({len(real_xy)})", "#1f77b4"),
                           (axes[1], gen_xy, f"Generated ({len(gen_xy)})", "#d62728")):
        sel = np.random.default_rng(0).choice(len(xy), min(400, len(xy)), replace=False)
        for s in sel:
            ax.plot(xy[s, :, 0], xy[s, :, 1], color=c, lw=0.4, alpha=0.3)
        ax.set_title(ttl)
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{args.exp}: spatial ({'centred dead-reckoned' if centre else 'direct x/y'})")
    fig.tight_layout()
    fig.savefig(out / "gen_spatial.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
