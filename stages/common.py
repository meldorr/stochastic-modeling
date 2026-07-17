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
    flow = d["flow"]
    if fs == "controls":
        der = derive_controls(col["x"], col["y"], col["altitude"], col["timedelta"], cfg["controls"])
        ok = der["valid_mask"]                # drop gap-interpolation artifact flights
        print(f"[data] controls: excluding {int((~ok).sum())}/{len(ok)} artifact flights "
              f"({100 * (~ok).mean():.1f}%)")
        X = der["controls"][ok].astype(np.float32)
        aux.update(entry=der["entry"][ok], clip_rates=der["clip_rates"])
        aux["real_xy"] = aux["real_xy"][ok]
        flow = flow[ok]
    else:
        X = np.stack([col[n] for n in names], axis=-1).astype(np.float32)

    tr, va = split_indices(len(X), float(cfg["data"]["train_ratio"]), int(cfg["seed"]))
    scaler = make_scaler(cfg.get("scaler", "standard"), X[tr])
    flat = X[tr].reshape(-1, X.shape[-1])
    bounds = np.stack([flat.min(0), flat.max(0)], axis=1).astype(np.float32)

    return {
        "X": X, "X_std": scaler.transform(X), "names": names, "feature_set": fs,
        "scaler": scaler, "bounds": bounds, "train_idx": tr, "val_idx": va,
        "aux": aux, "flow": flow,
    }


def reference_lr(epoch: int, total: int, tcfg: dict) -> float:
    """diffusion-models-lab LR: 10% linear warmup then linear decay."""
    lr_start, lr_peak, lr_end = (float(tcfg[k]) for k in ("lr_start", "lr_peak", "lr_end"))
    warmup = max(1, total // 10)
    if epoch <= warmup:
        return lr_start + (lr_peak - lr_start) * (epoch / warmup)
    u = (epoch - warmup) / max(1, total - warmup)
    return lr_peak + (lr_end - lr_peak) * u


def make_progress_figure(gen_std: np.ndarray, names: list[str], epoch: int):
    """4-panel in-training diagnostic (scaled units), matching the legacy runs:
    ch0-ch1 trajectories | timedelta vs t | altitude vs t | mean±std per channel."""
    import matplotlib.pyplot as plt

    n, T, C = gen_std.shape
    t = np.arange(T)
    tdi = names.index("timedelta") if "timedelta" in names else C - 1
    ai = names.index("altitude") if "altitude" in names else 2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for i in range(n):
        ax.plot(gen_std[i, :, 0], gen_std[i, :, 1], lw=0.9, alpha=0.7)
    ax.set_xlabel(f"{names[0]} (scaled)")
    ax.set_ylabel(f"{names[1]} (scaled)")
    ax.set_title(f"Epoch {epoch} — Generated {names[0]}–{names[1]} (scaled) — {n}/{n}")
    ax.set_aspect("equal", "datalim")

    for ax, j, ttl in ((axes[0, 1], tdi, "timedelta vs time"), (axes[1, 0], ai, f"{names[ai]} vs time")):
        for i in range(n):
            ax.plot(t, gen_std[i, :, j], lw=0.9, alpha=0.7)
        ax.set_xlabel("time index")
        ax.set_ylabel(f"{names[j]} (scaled)")
        ax.set_title(f"Epoch {epoch} — Generated {ttl} (scaled)")

    ax = axes[1, 1]
    for j, nm in enumerate(names):
        mu, sd = gen_std[:, :, j].mean(0), gen_std[:, :, j].std(0)
        (line,) = ax.plot(t, mu, lw=1.8, label=f"orig:{nm}")
        ax.fill_between(t, mu - sd, mu + sd, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("time index")
    ax.set_ylabel("value (scaled)")
    ax.set_title(f"Epoch {epoch} — Generated mean±std (scaled)")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


class SafeWriter:
    """TensorBoard writer wrapper: logging failures must never kill training.

    (Motivated by a run lost to a FileNotFoundError inside the async event
    writer thread propagating out of ``add_scalar``.)
    """

    def __init__(self, writer):
        self._w = writer

    def __getattr__(self, name):
        fn = getattr(self._w, name)

        def safe(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001 — deliberately broad
                print(f"[tb] {name} failed ({type(e).__name__}: {e}) — continuing")
                return None

        return safe if callable(fn) else fn


def make_writer(cfg: dict, runs_dir: Path):
    if not cfg.get("logging", {}).get("enabled", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        Path(runs_dir).mkdir(parents=True, exist_ok=True)
        return SafeWriter(SummaryWriter(log_dir=str(runs_dir)))
    except Exception:
        return None
