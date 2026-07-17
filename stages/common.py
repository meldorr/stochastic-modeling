"""Shared helpers for the staged DDPM experiments.

Feature sets an experiment can select (``features:`` in its yaml):
    xy        [x, y, altitude, timedelta]
    gstrack   [track, groundspeed, altitude, timedelta]
    controls  [turn_rate, along_accel, vert_rate, timedelta]  (derived, Section 1.4)
              + per-flight entry states (x0, y0, z0, gs0, chi0) for re-integration
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.controls import CONTROL_NAMES, derive_controls
from src.data.dataset import make_scaler, split_indices
from src.data.prepare import load_processed
from src.pipeline.utils import resolve

FEATURE_SETS = {
    "xy": ["x", "y", "altitude", "timedelta"],
    "gstrack": ["track", "groundspeed", "altitude", "timedelta"],
    "controls": CONTROL_NAMES,
}


def load_experiment_data(cfg: dict) -> dict:
    """Assemble the experiment's channel tensor X (N, T, C) in raw units, split
    train/val, fit the configured scaler on train, and carry auxiliary arrays
    (entry states for controls; real x/y for evaluation)."""
    fs = cfg["features"]
    if fs not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {fs!r} (choose {list(FEATURE_SETS)})")
    names = FEATURE_SETS[fs]

    d = load_processed(cfg)
    stored = [str(f) for f in d["meta"]["feature_names"]]
    col = {n: d["X"][:, :, stored.index(n)].astype(np.float64) for n in stored}

    aux: dict = {"real_xy": np.stack([col["x"], col["y"]], axis=-1)}
    if fs == "controls":
        der = derive_controls(col["x"], col["y"], col["altitude"], col["timedelta"], cfg["controls"])
        X = der["controls"].astype(np.float32)
        aux.update(entry=der["entry"], clip_rates=der["clip_rates"])
    else:
        X = np.stack([col[n] for n in names], axis=-1).astype(np.float32)

    tr, va = split_indices(len(X), float(cfg["data"]["train_ratio"]), int(cfg["seed"]))
    scaler = make_scaler(cfg.get("scaler", "standard"), X[tr])
    flat = X[tr].reshape(-1, X.shape[-1])
    bounds = np.stack([flat.min(0), flat.max(0)], axis=1).astype(np.float32)

    return {
        "X": X, "X_std": scaler.transform(X), "names": names, "feature_set": fs,
        "scaler": scaler, "bounds": bounds, "train_idx": tr, "val_idx": va,
        "aux": aux, "flow": d["flow"],
    }


def reference_lr(epoch: int, total: int, tcfg: dict) -> float:
    """diffusion-models-lab LR: 10% linear warmup then linear decay."""
    lr_start, lr_peak, lr_end = (float(tcfg[k]) for k in ("lr_start", "lr_peak", "lr_end"))
    warmup = max(1, total // 10)
    if epoch <= warmup:
        return lr_start + (lr_peak - lr_start) * (epoch / warmup)
    u = (epoch - warmup) / max(1, total - warmup)
    return lr_peak + (lr_end - lr_peak) * u


def make_writer(cfg: dict, runs_dir: Path):
    if not cfg.get("logging", {}).get("enabled", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(runs_dir))
    except Exception:
        return None
