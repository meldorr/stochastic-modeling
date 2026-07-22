"""Shared helpers for the staged DDPM experiments.

Feature sets an experiment can select (``features:`` in its yaml):
    xy               [x, y, altitude, timedelta]
    gstrack          [track, groundspeed, altitude, timedelta]  (raw ADS-B)
    gstrack_derived  [heading, groundspeed_ms, altitude, timedelta]  (heading+speed
                     derived & SavGol-smoothed from the x/y path; grid-referenced,
                     so it reconstructs with a self-consistent planar walk)
    controls         [turn_rate, along_accel, vert_rate, timedelta]  (derived, Sec 1.4)
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
    "xyt": ["x", "y", "timedelta"],                 # SE(2)-augmentable: x/y + time only
    "gstrack": ["track", "groundspeed", "altitude", "timedelta"],
    "gstrack_derived": ["heading", "groundspeed_ms", "altitude", "timedelta"],
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

    # FAF anchors for the fixed reckoning at generation time (the FAF is the known
    # destination): lat/lon for the geodesic gstrack walk, x/y to calibrate the
    # lat/lon->UTM offset, full state for the controls integrator.
    faf_latlon = d["anchor_last"].astype(np.float64)                       # (N, 2) lat/lon
    faf_xy = np.stack([col["x"][:, -1], col["y"][:, -1]], axis=-1)          # (N, 2) x/y
    aux: dict = {"real_xy": np.stack([col["x"], col["y"]], axis=-1),
                 "faf_latlon": faf_latlon, "faf_xy": faf_xy}
    flow = d["flow"]
    if fs in ("controls", "gstrack_derived"):
        # both are derived from the x/y path; drop the same gap-interpolation artifacts
        der = derive_controls(col["x"], col["y"], col["altitude"], col["timedelta"], cfg["controls"])
        ok = der["valid_mask"]
        print(f"[data] {fs}: excluding {int((~ok).sum())}/{len(ok)} artifact flights "
              f"({100 * (~ok).mean():.1f}%)")
        if fs == "controls":
            X = der["controls"][ok].astype(np.float32)
            aux.update(entry=der["entry"][ok], faf_state=der["faf"][ok], clip_rates=der["clip_rates"])
        else:  # gstrack_derived: [heading (rad), groundspeed (m/s), altitude (ft), timedelta (s)]
            X = np.stack([der["chi"], der["gs"], col["altitude"], col["timedelta"]],
                         axis=-1)[ok].astype(np.float32)
        aux["real_xy"] = aux["real_xy"][ok]
        aux["faf_latlon"] = aux["faf_latlon"][ok]
        aux["faf_xy"] = aux["faf_xy"][ok]
        flow = flow[ok]
    else:
        X = np.stack([col[n] for n in names], axis=-1).astype(np.float32)

    tr, va = split_indices(len(X), float(cfg["data"]["train_ratio"]), int(cfg["seed"]))
    scaler = make_scaler(cfg.get("scaler", "standard"), X[tr], names)
    flat = X[tr].reshape(-1, X.shape[-1])
    bounds = np.stack([flat.min(0), flat.max(0)], axis=1).astype(np.float32)

    return {
        "X": X, "X_std": scaler.transform(X), "names": names, "feature_set": fs,
        "scaler": scaler, "bounds": bounds, "train_idx": tr, "val_idx": va,
        "aux": aux, "flow": flow,
    }


def repair_controls(feats: np.ndarray, ccfg: dict) -> np.ndarray:
    """Clip sampled controls to the Section-1.4 envelopes; make timedelta monotone."""
    out = feats.copy()
    lim_cd = np.radians(float(ccfg["clip_turn_rate_deg_s"]))
    lim_a = float(ccfg["clip_accel_ms2"])
    vz_lo, vz_hi = [float(v) for v in ccfg["clip_vz_ms"]]
    out[:, :, 0] = np.clip(out[:, :, 0], -lim_cd, lim_cd)
    out[:, :, 1] = np.clip(out[:, :, 1], -lim_a, lim_a)
    out[:, :, 2] = np.clip(out[:, :, 2], vz_lo, vz_hi)
    td = out[:, :, 3] - out[:, :1, 3]
    out[:, :, 3] = np.maximum.accumulate(td, axis=1)
    return out


def reference_lr(epoch: int, total: int, tcfg: dict) -> float:
    """diffusion-models-lab LR: 10% linear warmup then linear decay."""
    lr_start, lr_peak, lr_end = (float(tcfg[k]) for k in ("lr_start", "lr_peak", "lr_end"))
    warmup = max(1, total // 10)
    if epoch <= warmup:
        return lr_start + (lr_peak - lr_start) * (epoch / warmup)
    u = (epoch - warmup) / max(1, total - warmup)
    return lr_peak + (lr_end - lr_peak) * u


def make_progress_figure(gen_std: np.ndarray, names: list[str], epoch: int, xy: np.ndarray | None = None):
    """4-panel in-training diagnostic (scaled units), matching the legacy runs:
    top-left trajectories | timedelta vs t | altitude vs t | mean±std per channel.

    If ``xy`` (n, T, 2) is given it is the FAF-reckoned absolute x/y for these
    samples (km); the top-left panel shows that instead of scaled ch0/ch1, so
    gstrack/controls snapshots are read as real ground tracks, not raw channels.
    """
    import matplotlib.pyplot as plt

    n, T, C = gen_std.shape
    t = np.arange(T)
    tdi = names.index("timedelta") if "timedelta" in names else C - 1
    ai = names.index("altitude") if "altitude" in names else 2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    if xy is not None:
        for i in range(n):
            ax.plot(xy[i, :, 0] / 1000.0, xy[i, :, 1] / 1000.0, lw=0.9, alpha=0.7)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(f"Epoch {epoch} — Generated ground track (FAF-reckoned) — {n}/{n}")
    else:
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
