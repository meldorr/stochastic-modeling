"""Per-cluster fPCA vs one global fPCA.

    python per_cluster_fpca.py --config configs/config.yaml

Assigns a cluster id to every flight (xy_flat / k from config), then fits a
*separate* fPCA per cluster from scratch and compares it to a single global fPCA:

    results/per_cluster_means.png          per-feature mean curves, global vs per-cluster
    results/per_cluster_spatial_means.png  x/y mean path: global (mush) vs per-cluster corridors
    results/per_cluster_recon_rmse.png     held-out reconstruction RMSE, global vs per-cluster
    results/cluster_labels.npz             per-flight cluster id

The point: pooled over all 5 approach corridors the global mean is meaningless, so
per-cluster fPCA reaches the same fidelity with a smaller per-cluster latent.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cluster import build_cluster_features, fit_clusters
from src.data.dataset import load_dataset
from src.fpca import FPCA
from src.pipeline.utils import ensure_parent, load_config, resolve

CMAP = plt.get_cmap("tab10")


def fit_fpca(X_std, names, fc):
    return FPCA.fit(
        X_std, names, explained_variance=fc["explained_variance"],
        max_components=int(fc["max_components"]), basis=fc.get("basis", "discrete"),
        bspline_cfg=fc.get("bspline"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-cluster vs global fPCA")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg, fc = cfg["cluster"], cfg["fpca"]
    seed = int(cfg["seed"])
    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(cfg)
    names = list(data["feature_names"])
    Xs, Xr = data["X_std"], data["X"]
    tr, va = data["train_idx"], data["val_idx"]

    # --- cluster every flight ---
    feats, label = build_cluster_features(data, ccfg, None)  # xy_flat needs no fpca
    k = int(ccfg["k"])
    labels, _ = fit_clusters(feats, ccfg.get("method", "kmeans"), k=k, seed=seed)
    clusters = sorted(set(labels))
    print(f"[pcfpca] cluster basis={label}, k={k}, sizes="
          f"{{{', '.join(f'{c}:{int((labels==c).sum())}' for c in clusters)}}}")

    # --- global fPCA (baseline) ---
    g = fit_fpca(Xs[tr], names, fc)
    rec_g = g.decode(g.encode(Xs[va]))
    rmse_g = np.sqrt(((rec_g - Xs[va]) ** 2).mean(axis=(0, 1)))
    print(f"[pcfpca] GLOBAL   m={g.m:3d}  RMSE={dict(zip(names, np.round(rmse_g,4)))}")

    # --- per-cluster fPCA, evaluated on each cluster's own held-out flights ---
    per = {}
    rec_pc = np.empty_like(Xs[va])
    ms = []
    for c in clusters:
        tr_c = tr[labels[tr] == c]
        va_c = va[labels[va] == c]
        fc_c = fit_fpca(Xs[tr_c], names, fc)
        per[c] = fc_c
        ms.append(fc_c.m)
        # place reconstructions back into the val array by position
        pos = np.where(labels[va] == c)[0]
        rec_pc[pos] = fc_c.decode(fc_c.encode(Xs[va_c]))
        print(f"[pcfpca] cluster {c}  m={fc_c.m:3d}  ks={dict(zip(names, fc_c.ks))}")
    rmse_pc = np.sqrt(((rec_pc - Xs[va]) ** 2).mean(axis=(0, 1)))
    print(f"[pcfpca] PER-CLUSTER avg m={np.mean(ms):.1f} (sum {sum(ms)})  "
          f"RMSE={dict(zip(names, np.round(rmse_pc,4)))}")
    print(f"[pcfpca] RMSE improvement: "
          f"{dict(zip(names, np.round((rmse_g-rmse_pc)/rmse_g*100,1)))} %")

    # --- plot 1: per-feature mean curves (raw units) ---
    gm = Xr.mean(0)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    t = np.arange(Xr.shape[1])
    for ax, j, nm in zip(axes, range(len(names)), names):
        ax.plot(t, gm[:, j], color="black", lw=2.5, label="global mean")
        for c in clusters:
            ax.plot(t, Xr[labels == c][:, :, j].mean(0), color=CMAP(c % 10), lw=1.5)
        ax.set_title(nm)
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("raw value")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean curves: global (black) vs per-cluster (coloured)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_cluster_means.png", dpi=130)
    plt.close(fig)

    # --- plot 2: spatial mean paths ---
    xi, yi = names.index("x"), names.index("y")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    rng = np.random.default_rng(0)
    for s in rng.choice(len(Xr), 500, replace=False):
        ax.plot(Xr[s, :, xi], Xr[s, :, yi], color="lightgray", lw=0.3, alpha=0.5)
    ax.plot(gm[:, xi], gm[:, yi], color="black", lw=3, label="global mean")
    for c in clusters:
        m = Xr[labels == c].mean(0)
        ax.plot(m[:, xi], m[:, yi], color=CMAP(c % 10), lw=2.5, label=f"cluster {c} mean")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Mean approach path: global mean is a mush;\nper-cluster means follow real corridors")
    fig.tight_layout()
    fig.savefig(out_dir / "per_cluster_spatial_means.png", dpi=130)
    plt.close(fig)

    # --- plot 3: reconstruction RMSE bars ---
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, rmse_g, 0.4, label=f"global (m={g.m})", color="#888")
    ax.bar(x + 0.2, rmse_pc, 0.4, label=f"per-cluster (avg m={np.mean(ms):.0f})", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out RMSE (std units)")
    ax.set_title("fPCA reconstruction: global vs per-cluster (same explained-variance)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_cluster_recon_rmse.png", dpi=130)
    plt.close(fig)

    lab_out = ensure_parent(ccfg.get("labels_out", "results/cluster_labels.npz"))
    np.savez_compressed(lab_out, labels=labels.astype(np.int32), basis=ccfg["basis"],
                        method=ccfg.get("method", "kmeans"), k=k, flight_ids=data["flight_ids"])
    print(f"[pcfpca] wrote 3 plots + {lab_out}")


if __name__ == "__main__":
    main()
