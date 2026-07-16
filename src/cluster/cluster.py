"""Cluster whole trajectories so each group is unimodal enough that a per-cluster
fPCA mean is meaningful (Jarry 2022 Sec. 4.2; Dinh 2025 "one traffic mode per patch").

Two feature bases for clustering, both driven off the standardized profiles:

* ``xy_fpca``  — run a global fPCA on all data, then cluster on the *scores* of the
  chosen channels (default x, y). Low-dim, denoised — the "fpca then cluster" route.
* ``xy_flat``  — cluster on the raw (time-subsampled) standardized channels directly.

Algorithms: k-means / GMM (need k) or HDBSCAN (density-based, auto count, à la Jarry).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _channel_idx(feature_names, sel):
    return [list(feature_names).index(c) for c in sel]


def build_cluster_features(data: dict, cfg_cl: dict, fpca=None) -> tuple[np.ndarray, str]:
    """Return (feature matrix (N, D), human label) for clustering."""
    names = list(data["feature_names"])
    Xs = data["X_std"]
    sel = list(cfg_cl.get("features", ["x", "y"]))
    idx = _channel_idx(names, sel)
    basis = cfg_cl.get("basis", "xy_fpca")

    if basis == "xy_flat":
        stride = int(cfg_cl.get("stride", 4))
        sub = Xs[:, ::stride, :][:, :, idx]           # (N, T', |sel|)
        feats = sub.reshape(len(sub), -1)
        label = f"{'+'.join(sel)} flat (stride {stride}, D={feats.shape[1]})"
    elif basis == "xy_fpca":
        if fpca is None:
            raise ValueError("xy_fpca basis needs a fitted global FPCA")
        W = fpca.encode(Xs)
        cols = np.concatenate([np.arange(fpca.slices[i].start, fpca.slices[i].stop) for i in idx])
        feats = W[:, cols]
        label = f"{'+'.join(sel)} fPCA scores (D={feats.shape[1]})"
    else:
        raise ValueError(f"unknown cluster basis: {basis!r}")

    return StandardScaler().fit_transform(feats).astype(np.float64), label


def fit_clusters(feats: np.ndarray, method: str = "kmeans", k: int = 5,
                 seed: int = 42, hdbscan_min: int = 200):
    """Return (labels, inertia_or_None). HDBSCAN labels noise as -1."""
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
        return km.labels_, float(km.inertia_)
    if method == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(feats)
        return gm.predict(feats), None
    if method == "hdbscan":
        hb = HDBSCAN(min_cluster_size=hdbscan_min).fit(feats)
        return hb.labels_, None
    raise ValueError(f"unknown method: {method!r}")


def cluster_metrics(feats, labels, flow=None, sample=3000, seed=42):
    """(silhouette on a subsample, ARI vs flow labels). Noise (-1) excluded from silhouette."""
    labels = np.asarray(labels)
    mask = labels >= 0
    sil = float("nan")
    if len(np.unique(labels[mask])) > 1:
        rng = np.random.default_rng(seed)
        ii = np.where(mask)[0]
        idx = rng.choice(ii, min(sample, len(ii)), replace=False)
        sil = float(silhouette_score(feats[idx], labels[idx]))
    ari = None if flow is None else float(adjusted_rand_score(np.asarray(flow), labels))
    return sil, ari
