"""Shared helpers for the ablation experiments (E1-E4).

Feature sets:
    DYN = [track, groundspeed, altitude, timedelta]   (position only via dead-reckoning)
    XY  = [x, y, altitude, timedelta]                  (position modelled directly)

Metrics:
    raw_rmse            per-feature reconstruction RMSE in raw units (kts, deg, ft, m, s)
    hf_retention        mean |first difference| ratio recon/real  (1 = no smoothing)
    dead_reckon         integrate (track, gs, dt) forward from the true start point -> x/y
    path_errors         mean + final point error (metres) between two x/y paths
    ks_marginals        per-feature two-sample KS, pooled over timesteps
    sliced_wasserstein  distributional distance on flattened x/y paths (lower = closer)
    within_pct          fraction of trajectories inside the training envelope
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from src.data.dataset import load_dataset
from src.pipeline.utils import resolve

DYN = ["track", "groundspeed", "altitude", "timedelta"]
XY = ["x", "y", "altitude", "timedelta"]

KTS_TO_MS = 1852.0 / 3600.0


def load_features(cfg: dict, features: list[str]) -> dict:
    """load_dataset with a specific channel subset (same split: seed + N fixed)."""
    c = copy.deepcopy(cfg)
    c["data"]["features"] = list(features)
    return load_dataset(c)


def out_dir(cfg: dict, name: str) -> Path:
    d = resolve("results") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- representation metrics -------------------------------------------------
def raw_rmse_named(recon_std, real_std, scaler, names) -> dict:
    """Per-feature RMSE converted back to raw units via the feature std."""
    rmse_std = np.sqrt(((recon_std - real_std) ** 2).mean(axis=(0, 1)))
    return {n: float(r * s) for n, r, s in zip(names, rmse_std, scaler.std)}


def hf_retention(recon_raw: np.ndarray, real_raw: np.ndarray) -> float:
    """Ratio of mean |first difference| (recon / real). <1 means smoothed."""
    hr = np.abs(np.diff(recon_raw, axis=1)).mean()
    hx = np.abs(np.diff(real_raw, axis=1)).mean()
    return float(hr / hx) if hx > 0 else float("nan")


def dead_reckon(track_deg, gs_kts, td_s, x0, y0) -> np.ndarray:
    """Forward planar integration. All (N, T); x0/y0 (N,). Returns (N, T, 2) metres.

    Aviation convention: track measured clockwise from north ->
    dx = v sin(theta), dy = v cos(theta).
    """
    dt = np.diff(td_s, axis=1)                        # (N, T-1)
    v = gs_kts[:, :-1] * KTS_TO_MS
    th = np.radians(track_deg[:, :-1])
    dx = v * np.sin(th) * dt
    dy = v * np.cos(th) * dt
    x = np.concatenate([x0[:, None], x0[:, None] + np.cumsum(dx, axis=1)], axis=1)
    y = np.concatenate([y0[:, None], y0[:, None] + np.cumsum(dy, axis=1)], axis=1)
    return np.stack([x, y], axis=-1)


def path_errors(xy_hat: np.ndarray, xy_true: np.ndarray) -> dict:
    """Mean-over-path and final-point Euclidean error (metres)."""
    d = np.linalg.norm(xy_hat - xy_true, axis=-1)     # (N, T)
    return {"mean_m": float(d.mean()), "final_m": float(d[:, -1].mean())}


# --- generative metrics -----------------------------------------------------
def ks_marginals(real: np.ndarray, gen: np.ndarray, names: list[str]) -> dict:
    return {
        n: float(ks_2samp(real[:, :, j].ravel(), gen[:, :, j].ravel()).statistic)
        for j, n in enumerate(names)
    }


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_proj=64, seed=0) -> float:
    """Mean 1-D Wasserstein over random projections of flattened paths.

    a, b: (N, D) flattened (and pre-scaled) path vectors.
    """
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    total = 0.0
    for _ in range(n_proj):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        total += wasserstein_distance(a @ u, b @ u)
    return float(total / n_proj)


def xy_flat(feats: np.ndarray, names: list[str], stride=8, scale=1000.0) -> np.ndarray:
    """Flatten the x/y path (km units) for distributional distances."""
    xi, yi = names.index("x"), names.index("y")
    sub = feats[:, ::stride, :][:, :, [xi, yi]] / scale
    return sub.reshape(len(sub), -1)


def within_pct(feats, bounds, names, margin=0.05) -> float:
    from src.pipeline.reconstruct import within_bounds

    return float(within_bounds(feats, bounds, names, margin=margin).mean())


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {path}")
