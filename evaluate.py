"""Evaluate the pipeline: fPCA fidelity, latent match, real-vs-generated plots.

    python evaluate.py --config configs/config.yaml

Produces in results/:
    fpca_explained_variance.png   retained variance per feature
    latent_distribution.png       real vs generated latents (2D PCA view + per-dim)
    feature_profiles.png          mean +/- std profiles, real vs generated
    spatial_tracks.png            reconstructed lat/lon tracks, real vs generated
    metrics.json / metrics.txt    numbers (explained variance, recon RMSE, KS distances)
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from src.data.dataset import load_dataset
from src.pipeline.checkpoint import load_checkpoint
from src.pipeline.reconstruct import walk_latlon_backward
from src.pipeline.utils import load_config, resolve

REAL_C, GEN_C = "#1f77b4", "#d62728"


def _load_generated(cfg):
    npz = resolve(cfg["paths"]["generated"]).with_suffix(".npz")
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found — run generate.py first.")
    g = np.load(npz, allow_pickle=True)
    return g["feats"], g["W"], g["anchors"]


def plot_explained_variance(fpca, path):
    n = len(fpca.feature_names)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0))
    for ax, name, b in zip(np.atleast_1d(axes), fpca.feature_names, fpca.bases):
        cum = np.cumsum(b["evr_full"])
        ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=3, color=REAL_C)
        ax.axvline(b["k"], color=GEN_C, ls="--", lw=1, label=f"k={b['k']}")
        ax.axhline(cum[b["k"] - 1], color="gray", ls=":", lw=1)
        ax.set_title(f"{name}  ({cum[b['k'] - 1] * 100:.1f}% @ k={b['k']})")
        ax.set_xlabel("components")
        ax.set_xlim(0.5, min(len(cum), b["k"] + 8) + 0.5)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cumulative explained variance")
    fig.suptitle("fPCA per-feature explained variance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_latent(real_W_n, gen_W_n, path):
    pca = PCA(n_components=2).fit(real_W_n)
    r2, g2 = pca.transform(real_W_n), pca.transform(gen_W_n)
    ndim = min(real_W_n.shape[1], 6)

    fig = plt.figure(figsize=(2.3 * ndim, 5.4))
    gs = fig.add_gridspec(2, ndim, height_ratios=[1.5, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.scatter(r2[:, 0], r2[:, 1], s=6, alpha=0.3, color=REAL_C, label="real")
    ax0.scatter(g2[:, 0], g2[:, 1], s=6, alpha=0.3, color=GEN_C, label="generated")
    ax0.set_title("latent 2D PCA view")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.legend()
    ax0.grid(alpha=0.3)

    for j in range(ndim):
        ax = fig.add_subplot(gs[1, j])
        ax.hist(real_W_n[:, j], bins=40, density=True, alpha=0.5, color=REAL_C)
        ax.hist(gen_W_n[:, j], bins=40, density=True, alpha=0.5, color=GEN_C)
        ax.set_title(f"dim {j}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Latent weight distribution: real vs generated")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profiles(real_X, gen_feats, names, path, n_lines=40):
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2))
    t = np.arange(real_X.shape[1])
    for ax, j, name in zip(np.atleast_1d(axes), range(n), names):
        for data, c in ((real_X, REAL_C), (gen_feats, GEN_C)):
            mu, sd = data[:, :, j].mean(0), data[:, :, j].std(0)
            ax.plot(t, mu, color=c, lw=2)
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18)
        ax.plot(t, real_X[:n_lines, :, j].T, color=REAL_C, lw=0.3, alpha=0.25)
        ax.plot(t, gen_feats[:n_lines, :, j].T, color=GEN_C, lw=0.3, alpha=0.25)
        ax.set_title(name)
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("value (raw units)")
    fig.suptitle("Feature profiles: real (blue) vs generated (red), mean ± std")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fpca_reconstruction(
    fpca, X_std, feature_scaler, names, path, n_lines=8, rng_seed=0
):
    """Show original vs fPCA-reconstructed profiles for a handful of flights."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(X_std), min(n_lines, len(X_std)), replace=False)
    sample_std = X_std[idx]  # (n_lines, T, F)
    recon_std = fpca.decode(fpca.encode(sample_std))  # (n_lines, T, F)

    sample_raw = feature_scaler.inverse_transform(sample_std)
    recon_raw = feature_scaler.inverse_transform(recon_std)
    residual = sample_raw - recon_raw

    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 5.5), sharex=True)
    t = np.arange(sample_raw.shape[1])

    for j, name in enumerate(names):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]
        for i in range(n_lines):
            ax_top.plot(t, sample_raw[i, :, j], color=REAL_C, lw=0.9, alpha=0.55)
            ax_top.plot(t, recon_raw[i, :, j], color=GEN_C, lw=0.9, alpha=0.55, ls="--")
            ax_bot.plot(t, residual[i, :, j], color="gray", lw=0.7, alpha=0.6)

        ax_bot.axhline(0, color="black", lw=0.8, ls=":")
        rmse = float(np.sqrt((residual[:, :, j] ** 2).mean()))
        ax_top.set_title(f"{name}\n(k={fpca.ks[j]}, RMSE={rmse:.3g})", fontsize=9)
        ax_top.grid(alpha=0.25)
        ax_bot.set_xlabel("timestep")
        ax_bot.grid(alpha=0.25)
        if j == 0:
            ax_top.set_ylabel("raw value")
            ax_bot.set_ylabel("residual")

    # legend proxy
    from matplotlib.lines import Line2D

    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=REAL_C, label="original"),
            Line2D([0], [0], color=GEN_C, ls="--", label="fPCA recon"),
        ],
        fontsize=8,
    )
    fig.suptitle(
        f"fPCA reconstruction quality  ({n_lines} random flights)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial(real_ll, gen_ll, flows, path, n_show=400):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    ri = np.random.default_rng(0).choice(
        len(real_ll), min(n_show, len(real_ll)), replace=False
    )
    gi = np.random.default_rng(1).choice(
        len(gen_ll), min(n_show, len(gen_ll)), replace=False
    )
    for ll in real_ll[ri]:
        axes[0].plot(ll[:, 1], ll[:, 0], color=REAL_C, lw=0.4, alpha=0.3)
    for ll in gen_ll[gi]:
        axes[1].plot(ll[:, 1], ll[:, 0], color=GEN_C, lw=0.4, alpha=0.3)
    axes[0].set_title(f"Real ({len(ri)})")
    axes[1].set_title(f"Generated ({len(gi)})")
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", "box")
    axes[0].set_ylabel("latitude")
    fig.suptitle("Spatial tracks (reconstructed from track/groundspeed/timedelta)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained pipeline")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["evaluate"]
    out_dir = resolve(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_checkpoint(resolve(cfg["paths"]["checkpoint"]))
    data = load_dataset(cfg)
    fpca, names = bundle["fpca"], bundle["feature_names"]

    gen_feats, gen_W_raw, gen_anchors = _load_generated(cfg)

    # subsample real for a balanced comparison
    rng = np.random.default_rng(int(cfg["seed"]))
    ridx = rng.choice(
        len(data["X"]), min(int(ecfg["n_real"]), len(data["X"])), replace=False
    )
    real_X = data["X"][ridx]
    real_X_std = data["X_std"][ridx]
    real_ll = data["latlon"][ridx]

    # latent spaces (normalized, the space the DDPM lives in)
    real_W_n = bundle["latent_scaler"].transform(fpca.encode(real_X_std))
    gen_W_n = bundle["latent_scaler"].transform(gen_W_raw)

    # reconstruct generated spatial tracks
    ti, gi, tdi = (
        names.index("track"),
        names.index("groundspeed"),
        names.index("timedelta"),
    )
    glat, glon = walk_latlon_backward(
        gen_feats[:, :, ti], gen_feats[:, :, gi], gen_feats[:, :, tdi], gen_anchors
    )
    gen_ll = np.stack([glat, glon], axis=-1)

    # ---- plots ----
    plot_explained_variance(fpca, out_dir / "fpca_explained_variance.png")
    plot_fpca_reconstruction(
        fpca,
        real_X_std,
        bundle["feature_scaler"],
        names,
        out_dir / "fpca_reconstruction.png",
    )
    plot_latent(real_W_n, gen_W_n, out_dir / "latent_distribution.png")
    plot_profiles(real_X, gen_feats, names, out_dir / "feature_profiles.png")
    plot_spatial(real_ll, gen_ll, data["flow"][ridx], out_dir / "spatial_tracks.png")

    # ---- metrics ----
    recon_rmse = fpca.reconstruction_error(data["X_std"][data["val_idx"]])
    ks_latent = [
        float(ks_2samp(real_W_n[:, j], gen_W_n[:, j]).statistic)
        for j in range(real_W_n.shape[1])
    ]
    ks_feat = {
        name: float(
            ks_2samp(real_X[:, :, j].ravel(), gen_feats[:, :, j].ravel()).statistic
        )
        for j, name in enumerate(names)
    }
    metrics = {
        "n_real": len(real_X),
        "n_generated": len(gen_feats),
        "latent_dim_m": int(fpca.m),
        "components_per_feature": dict(zip(names, [int(k) for k in fpca.ks])),
        "explained_variance": {
            k: round(v, 4) for k, v in fpca.total_explained_variance().items()
        },
        "fpca_val_recon_rmse_std": {
            n: round(float(r), 4) for n, r in zip(names, recon_rmse)
        },
        "ks_latent_mean": round(float(np.mean(ks_latent)), 4),
        "ks_latent_max": round(float(np.max(ks_latent)), 4),
        "ks_feature_marginal": {k: round(v, 4) for k, v in ks_feat.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = [
        "=== fPCA + latent DDPM evaluation ===",
        f"latent dim m = {metrics['latent_dim_m']}  {metrics['components_per_feature']}",
        f"explained variance/feature: {metrics['explained_variance']}",
        f"fPCA val recon RMSE (std): {metrics['fpca_val_recon_rmse_std']}",
        f"KS latent (mean/max over {fpca.m} dims): "
        f"{metrics['ks_latent_mean']} / {metrics['ks_latent_max']}",
        f"KS feature marginals: {metrics['ks_feature_marginal']}",
    ]
    txt = "\n".join(lines)
    (out_dir / "metrics.txt").write_text(txt + "\n")
    print(txt)
    print(f"[evaluate] plots + metrics written to {out_dir}")


if __name__ == "__main__":
    main()
