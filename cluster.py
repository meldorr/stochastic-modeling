"""Explore trajectory clustering before committing to per-cluster fPCA.

    python cluster.py --config configs/config.yaml

Sweeps k for the configured basis (and compares xy_fpca vs xy_flat), scores each
against the known `initial_flow` labels (ARI) and silhouette, writes:
    results/cluster_ksweep.png     silhouette + ARI vs k, per basis
    results/cluster_spatial.png    x/y tracks coloured by cluster vs by true flow
    results/cluster_labels.npz     chosen labels (for the next step: per-cluster fPCA)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cluster import build_cluster_features, cluster_metrics, fit_clusters
from src.data.dataset import load_dataset
from src.fpca import FPCA
from src.pipeline.utils import ensure_parent, load_config, resolve


def global_fpca(data, cfg):
    fc = cfg["fpca"]
    return FPCA.fit(
        data["X_std"], data["feature_names"],
        explained_variance=fc["explained_variance"], max_components=int(fc["max_components"]),
        basis=fc.get("basis", "discrete"), bspline_cfg=fc.get("bspline"),
    )


def sweep(feats, ks, flow, seed):
    rows = []
    for k in ks:
        labels, inertia = fit_clusters(feats, "kmeans", k=k, seed=seed)
        sil, ari = cluster_metrics(feats, labels, flow, seed=seed)
        rows.append({"k": k, "silhouette": sil, "ari": ari, "inertia": inertia})
    return rows


def plot_ksweep(by_basis, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, rows in by_basis.items():
        ks = [r["k"] for r in rows]
        axes[0].plot(ks, [r["silhouette"] for r in rows], marker="o", label=label)
        axes[1].plot(ks, [r["ari"] for r in rows], marker="o", label=label)
    axes[0].set_title("silhouette (higher = tighter)")
    axes[1].set_title("ARI vs initial_flow (1 = matches known flows)")
    for ax in axes:
        ax.set_xlabel("k (clusters)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("k-means cluster quality vs k")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial(real_xy, labels, flow, path, n_show=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_xy), min(n_show, len(real_xy)), replace=False)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    # left: by cluster
    uniq = sorted(set(labels[idx]))
    for c in uniq:
        sel = idx[labels[idx] == c]
        col = "lightgray" if c == -1 else cmap(c % 10)
        for s in sel:
            axes[0].plot(real_xy[s, :, 0], real_xy[s, :, 1], color=col, lw=0.4, alpha=0.35)
        axes[0].plot([], [], color=col, label=f"c{c} (n={int((labels == c).sum())})")
    axes[0].set_title(f"by cluster (k={len([u for u in uniq if u >= 0])})")
    axes[0].legend(fontsize=7, ncol=2)

    # right: by true flow
    flows = sorted(set(flow[idx].astype(str)))
    for i, fl in enumerate(flows):
        sel = idx[flow[idx].astype(str) == fl]
        for s in sel:
            axes[1].plot(real_xy[s, :, 0], real_xy[s, :, 1], color=cmap(i % 10), lw=0.4, alpha=0.35)
        axes[1].plot([], [], color=cmap(i % 10), label=fl)
    axes[1].set_title("by initial_flow (ground truth)")
    axes[1].legend(fontsize=7, title="flow")

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", "box")
    axes[0].set_ylabel("y (m)")
    fig.suptitle("Clustering vs known approach flows")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore trajectory clustering")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["cluster"]
    seed = int(cfg["seed"])
    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(cfg)
    names = list(data["feature_names"])
    flow = data["flow"]
    fpca = global_fpca(data, cfg)
    print(f"[cluster] global fPCA m={fpca.m}, components={dict(zip(names, fpca.ks))}")

    ks = list(ccfg.get("k_sweep", [2, 3, 4, 5, 6, 7, 8]))
    bases = ["xy_fpca", "xy_flat"] if ccfg.get("compare_bases", True) else [ccfg["basis"]]

    by_basis, feats_cache = {}, {}
    for basis in bases:
        cc = dict(ccfg, basis=basis)
        feats, label = build_cluster_features(data, cc, fpca)
        feats_cache[basis] = feats
        rows = sweep(feats, ks, flow, seed)
        by_basis[label] = rows
        print(f"\n[cluster] basis = {label}")
        print("   k :  silhouette   ARI(vs flow)")
        for r in rows:
            print(f"  {r['k']:2d} :   {r['silhouette']:.3f}       {r['ari']:.3f}")
        # HDBSCAN reference (auto count)
        hb, _ = fit_clusters(feats, "hdbscan", seed=seed, hdbscan_min=int(ccfg.get("hdbscan_min", 200)))
        _, hb_ari = cluster_metrics(feats, hb, flow, seed=seed)
        n_hb = len(set(hb[hb >= 0]))
        print(f"  HDBSCAN: {n_hb} clusters, {int((hb == -1).sum())} noise, ARI={hb_ari:.3f}")

    plot_ksweep(by_basis, out_dir / "cluster_ksweep.png")

    # --- final labels at the configured basis/method/k ---
    basis, method, k = ccfg["basis"], ccfg.get("method", "kmeans"), int(ccfg["k"])
    feats = feats_cache.get(basis)
    if feats is None:
        feats, _ = build_cluster_features(data, ccfg, fpca)
    if method == "hdbscan":
        labels, _ = fit_clusters(feats, "hdbscan", seed=seed, hdbscan_min=int(ccfg.get("hdbscan_min", 200)))
    else:
        labels, _ = fit_clusters(feats, method, k=k, seed=seed)
    sil, ari = cluster_metrics(feats, labels, flow, seed=seed)
    sizes = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
    print(f"\n[cluster] FINAL basis={basis} method={method} k={k} "
          f"-> silhouette={sil:.3f} ARI={ari:.3f} sizes={sizes}")

    xi, yi = names.index("x"), names.index("y")
    plot_spatial(data["X"][:, :, [xi, yi]], labels, flow, out_dir / "cluster_spatial.png")

    lab_out = ensure_parent(ccfg.get("labels_out", "results/cluster_labels.npz"))
    np.savez_compressed(lab_out, labels=labels.astype(np.int32),
                        basis=basis, method=method, k=k,
                        flight_ids=data["flight_ids"])
    print(f"[cluster] wrote plots + {lab_out}")


if __name__ == "__main__":
    main()
