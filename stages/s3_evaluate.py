"""Stage 3 — per-experiment metrics + figures (comparable across experiments).

    python stages/s3_evaluate.py --exp configs/experiments/<name>.yaml

Metrics (results/<experiment>/metrics.json):
    ks_marginal/<channel>  distribution match per channel (pooled timesteps)
    sliced_w_xy_km         sliced-Wasserstein on x/y paths
    ks_endpoint_x/y        endpoint distribution match
    within_pct             fraction inside training envelope (model channels)

Spatial source per feature set (all absolute, comparable across experiments):
    xy        modelled x/y directly
    gstrack   FAF-anchored geodesic dead-reckoning from Stage 2 (tracks_xyz)
    controls  FAF-anchored control integration from Stage 2 (tracks_xyz)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from experiments.common import ks_marginals, sliced_wasserstein
from src.pipeline.reconstruct import within_bounds
from src.pipeline.utils import experiment_dirs, load_experiment_config
from stages.common import load_experiment_data


def spatial_paths(fs, feats, names, data, gen_npz, which):
    """(N, T, 2) absolute x/y paths for 'real' or 'gen'.

    xy uses the modelled channels; gstrack/controls use the FAF-anchored
    reconstruction written by Stage 2 (``tracks_xyz``) for generated samples and
    the true x/y for real ones.
    """
    if fs in ("xy", "xyt"):
        if which == "real":
            xy = data["X"][data["val_idx"]][:, :, [names.index("x"), names.index("y")]]
        else:
            xy = feats[:, :, [names.index("x"), names.index("y")]]
        return xy.astype(np.float64)
    # gstrack / controls: real = true x/y; gen = Stage-2 FAF-reckoned tracks_xyz
    if which == "real":
        return data["aux"]["real_xy"][data["val_idx"]].astype(np.float64)
    return gen_npz["tracks_xyz"][:, :, :2].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: evaluate one experiment")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-real", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_experiment_config(args.exp)
    dirs = experiment_dirs(cfg)
    data = load_experiment_data(cfg)
    names, fs = data["names"], data["feature_set"]

    g = np.load(dirs["results"] / "generated.npz", allow_pickle=True)
    feats = g["feats"]

    rng = np.random.default_rng(int(cfg["seed"]))
    va = data["val_idx"]
    ridx = rng.choice(len(va), min(args.n_real, len(va)), replace=False)
    real = data["X"][va][ridx]

    real_xy = spatial_paths(fs, feats, names, data, g, "real")
    gen_xy = spatial_paths(fs, feats, names, data, g, "gen")
    real_xy = real_xy[ridx] if len(real_xy) > len(ridx) else real_xy

    rf = real_xy[:, ::8, :].reshape(len(real_xy), -1) / 1000.0
    gf = gen_xy[:, ::8, :].reshape(len(gen_xy), -1) / 1000.0

    metrics = {
        "experiment": dirs["name"],
        "feature_set": fs,
        "n_real": int(len(real)),
        "n_gen": int(len(feats)),
        "ks_marginal": ks_marginals(real, feats, names),
        "sliced_w_xy_km": sliced_wasserstein(rf, gf),
        "ks_endpoint_x": float(ks_2samp(real_xy[:, -1, 0], gen_xy[:, -1, 0]).statistic),
        "ks_endpoint_y": float(ks_2samp(real_xy[:, -1, 1], gen_xy[:, -1, 1]).statistic),
        "within_pct": float(within_bounds(feats, data["bounds"], names).mean()),
    }
    (dirs["results"] / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[{dirs['name']}] KS marginals: "
          + ", ".join(f"{k}={v:.3f}" for k, v in metrics["ks_marginal"].items()))
    print(f"[{dirs['name']}] sliced-W(xy)={metrics['sliced_w_xy_km']:.3f} km  "
          f"KS endpoint={metrics['ks_endpoint_x']:.3f}/{metrics['ks_endpoint_y']:.3f}  "
          f"within={metrics['within_pct'] * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, xy, ttl, c in ((axes[0], real_xy, f"Real ({len(real_xy)})", "#1f77b4"),
                           (axes[1], gen_xy, f"Generated ({len(gen_xy)})", "#d62728")):
        sel = np.random.default_rng(0).choice(len(xy), min(400, len(xy)), replace=False)
        for s in sel:
            ax.plot(xy[s, :, 0], xy[s, :, 1], color=c, lw=0.4, alpha=0.3)
        ax.set_title(ttl)
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{dirs['name']}: spatial (absolute)")
    fig.tight_layout()
    fig.savefig(dirs["results"] / "spatial.png", dpi=130)
    plt.close(fig)
    print(f"[{dirs['name']}] wrote metrics.json / spatial.png -> {dirs['results']}")


if __name__ == "__main__":
    main()
