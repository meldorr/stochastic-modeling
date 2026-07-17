"""Stage 1b: standardize trajectories and expose train/val splits.

The DDPM never consumes these arrays directly — it works in fPCA weight space.
But fPCA is fit on the *standardized* profiles produced here, so this module owns
the per-feature scaler and the raw-unit bounds used later for physics repair.
"""

from __future__ import annotations

import numpy as np

from src.data.prepare import load_processed


class FeatureScaler:
    """Per-feature z-score standardization: one (mean, std) scalar per channel.

    Applied identically to every timestep, so the temporal *shape* of each
    profile is preserved (only its offset/scale change).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float64)          # (F,)
        self.std = np.asarray(std, np.float64)
        self.std[self.std == 0] = 1.0

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureScaler":
        # X: (N, T, F) -> stats over the N*T axis, per feature.
        flat = X.reshape(-1, X.shape[-1])
        return cls(flat.mean(0), flat.std(0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(np.array(d["mean"]), np.array(d["std"]))


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic shuffle -> (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * train_ratio))
    return np.sort(perm[:k]), np.sort(perm[k:])


def _resample_to(values: np.ndarray, param: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample ``values`` (T, F) onto ``n_out`` uniform points of a
    monotonic parameter ``param`` (T,). Used for the optional arc-length mode."""
    p = np.asarray(param, np.float64)
    p = p - p[0]
    if p[-1] <= 0:
        return values.copy()
    grid = np.linspace(0.0, p[-1], n_out)
    out = np.empty((n_out, values.shape[1]), values.dtype)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(grid, p, values[:, j])
    return out


def _arclength(latlon: np.ndarray) -> np.ndarray:
    """Cumulative planar arc-length (deg) along a lat/lon track (T, 2)."""
    d = np.diff(latlon, axis=0)
    step = np.sqrt((d**2).sum(1))
    return np.concatenate([[0.0], np.cumsum(step)])


def maybe_resample(X: np.ndarray, latlon: np.ndarray, mode: str, seq_len: int) -> np.ndarray:
    """Resampling guard. ``time`` keeps the native grid (identity when already
    ``seq_len`` long). ``arclength`` re-parametrizes by spatial distance."""
    if mode == "time" and X.shape[1] == seq_len:
        return X
    out = np.empty((X.shape[0], seq_len, X.shape[2]), X.dtype)
    for i in range(X.shape[0]):
        if mode == "arclength":
            param = _arclength(latlon[i])
        else:  # time / fallback: uniform index
            param = np.arange(X.shape[1], dtype=np.float64)
        out[i] = _resample_to(X[i], param, seq_len)
    return out


def load_dataset(config: dict) -> dict:
    """Load processed arrays, resample-guard, split, and standardize (train stats).

    Returns a dict with standardized ``X_std`` (all flights), the fitted
    ``scaler``, ``train_idx``/``val_idx``, raw-unit ``bounds`` (F, 2), and the
    passthrough arrays needed downstream (latlon, anchors, flow, meta).
    """
    d = load_processed(config)
    dcfg = config["data"]
    seq_len = int(dcfg["seq_len"])

    # processed.npz stores a channel superset; subset to the configured features.
    stored = [str(f) for f in d["meta"]["feature_names"]]
    wanted = [str(f) for f in dcfg["features"]]
    missing = [f for f in wanted if f not in stored]
    if missing:
        raise KeyError(
            f"features {missing} not in processed.npz (has {stored}) — re-run prepare."
        )
    sel = [stored.index(f) for f in wanted]
    d["X"] = d["X"][:, :, sel]
    d["meta"] = dict(d["meta"], feature_names=wanted)

    X = maybe_resample(d["X"], d["latlon"], dcfg.get("resample", "time"), seq_len)
    train_idx, val_idx = split_indices(len(X), float(dcfg["train_ratio"]), int(config["seed"]))

    scaler = FeatureScaler.fit(X[train_idx])
    X_std = scaler.transform(X)

    # per-feature physical bounds from the training set (for physics repair/reject)
    flat_tr = X[train_idx].reshape(-1, X.shape[-1])
    bounds = np.stack([flat_tr.min(0), flat_tr.max(0)], axis=1).astype(np.float32)  # (F, 2)

    return {
        "X": X,                     # raw units, resampled
        "X_std": X_std,             # standardized (N, T, F)
        "scaler": scaler,
        "bounds": bounds,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "latlon": d["latlon"],
        "anchor_last": d["anchor_last"],
        "anchor_first": d["anchor_first"],
        "flow": d["flow"],
        "runway": d["runway"],
        "flight_ids": d["flight_ids"],
        "meta": d["meta"],
        "feature_names": list(d["meta"]["feature_names"]),
    }
