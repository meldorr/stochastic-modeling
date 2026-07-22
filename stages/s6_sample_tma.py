"""NS1 — unconditional TMA fidelity: did the SE(2) augmentation work?

    python stages/s6_sample_tma.py [--ckpt ...ckpt_best.pt] [--n 2000]

Samples the model unconditionally and checks it reproduces the *augmented*
distribution: flyable trajectories filling the TMA from every direction. Left
panel = generated; right panel = real flights put through the same SE(2)
augmentation (the training target). Reports flyability (turn-rate/accel envelope
exceedance) and angular-coverage uniformity.

Writes results/<exp>/tma_fidelity.png + tma_fidelity.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

from src.data.controls import derive_controls
from src.data.dataset import scaler_from_dict
from src.ddpm import LatentDDPM
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.utils import get_device, load_config


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    den = build_raw_denoiser(ck["arch"], ck["channels"], ck["t_len"],
                             dropout=float(ck.get("dropout", 0.0)),
                             base_channels=int(ck.get("base_channels", 64)))
    ddpm = LatentDDPM(den, ck["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ck["model_state"]); ddpm.eval()
    return ddpm, ck


def se2_np(xy, tma_lo, tma_hi, rng):
    """Numpy SE(2): rotate about centroid + in-TMA translate (the training target)."""
    th = np.radians(rng.uniform(0, 360)); c, s = np.cos(th), np.sin(th)
    ctr = xy.mean(0)
    v = xy - ctr
    r = np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c], -1) + ctr
    lo, hi = tma_lo - r.min(0), tma_hi - r.max(0)
    sh = np.array([rng.uniform(l, h) if l <= h else 0.0 for l, h in zip(lo, hi)])
    return r + sh


def coverage_cv(xy, center):
    ang = np.degrees(np.arctan2(xy[:, :, 0].ravel() - center[0],
                                xy[:, :, 1].ravel() - center[1])) % 360
    counts = np.histogram(ang, bins=72, range=(0, 360))[0]
    return float(np.std(counts) / (np.mean(counts) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = ck.get("ddpm_cfg", {}).get("sample_clamp")
    print(f"[s6] {args.ckpt}  epoch={ck.get('epoch')}  val={ck.get('val_loss')}  n={args.n}")

    chunks = []
    with torch.no_grad():
        for start in range(0, args.n, 500):
            k = min(500, args.n - start)
            chunks.append(ddpm.sample(k, shape=(C, T), device=device, clamp=clamp).cpu().numpy())
    feats = scaler.inverse_transform(np.concatenate(chunks).transpose(0, 2, 1))     # (n,T,C) raw
    gxy = feats[:, :, [names.index("x"), names.index("y")]].astype(float)

    # flyability (horizontal): turn-rate / accel envelope exceedance; xyt has no altitude
    ccfg = load_config("configs/base.yaml")["controls"]
    der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], np.zeros_like(gxy[:, :, 0]),
                          feats[:, :, names.index("timedelta")], ccfg)
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}

    # real augmented reference (the training target)
    d = np.load("data/processed.npz", allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    realxy = d["X"][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    tma_lo, tma_hi = realxy.reshape(-1, 2).min(0), realxy.reshape(-1, 2).max(0)
    center = 0.5 * (tma_lo + tma_hi)
    rng = np.random.default_rng(0)
    aug_real = np.stack([se2_np(realxy[i], tma_lo, tma_hi, rng)
                         for i in rng.choice(len(realxy), min(args.n, 2000), replace=False)])

    cv_gen, cv_aug = coverage_cv(gxy, center), coverage_cv(aug_real, center)
    metrics = {"epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"), "n": args.n,
               "flyability_exceed": fly, "coverage_cv_generated": cv_gen, "coverage_cv_augmented": cv_aug}
    dirs = Path(args.ckpt).parent
    (dirs / "tma_fidelity.json").write_text(json.dumps(metrics, indent=2))
    print(f"[s6] flyability: turn>env={fly['turn_rate']*100:.1f}%  accel>env={fly['along_accel']*100:.1f}%  "
          f"| angular-CV gen={cv_gen:.3f} vs augmented={cv_aug:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(19, 9.5), sharex=True, sharey=True)
    for ax, xy, ttl, col in ((axes[0], gxy, f"Generated (unconditional, {len(gxy)})", "#d62728"),
                             (axes[1], aug_real, f"Real + SE(2) augmentation ({len(aug_real)})", "#1f77b4")):
        ax.add_collection(LineCollection(list(xy), colors=col, linewidths=0.3, alpha=0.05))
        ax.autoscale(); ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.set_title(ttl)
    fig.suptitle(f"NS1 TMA fidelity — epoch {ck.get('epoch')}  "
                 f"(turn>env {fly['turn_rate']*100:.0f}%, accel>env {fly['along_accel']*100:.0f}%)")
    fig.tight_layout(); fig.savefig(dirs / "tma_fidelity.png", dpi=130); plt.close(fig)
    print(f"[s6] wrote {dirs}/tma_fidelity.png + tma_fidelity.json")


if __name__ == "__main__":
    main()
