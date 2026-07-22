"""Stage 2 — sample the trained DDPM and save generated trajectories.

    python stages/s2_generate.py --exp configs/experiments/<name>.yaml [--n 1000]

Outputs in results/<experiment>/:
    generated.npz       feats (n, T, C) raw units [+ tracks_xyz for gstrack/controls]
    generated.parquet   tidy per-timestep frame
    profiles.png        real vs generated channel profiles

Samples the best-by-val checkpoint by default (``--which best``).

Feature-set specifics:
    xy               physics repair clips channels to the envelope; x/y is modelled
    gstrack          repaired, then FAF-anchored geodesic dead-reckoning (factor 1.0)
                     -> absolute x/y (tracks_xyz)
    gstrack_derived  repaired, then self-consistent planar velocity walk from the FAF
                     x/y -> absolute x/y (tracks_xyz)
    controls         clipped to the Section-1.4 envelopes, then integrated backward
                     from a FAF state drawn from real flights -> x/y (tracks_xyz)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import scaler_from_dict
from src.data.reckoning import generated_to_xy
from src.ddpm import LatentDDPM
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.reconstruct import physics_repair
from src.pipeline.utils import experiment_dirs, get_device, load_experiment_config, set_seed
from stages.common import load_experiment_data, repair_controls


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    denoiser = build_raw_denoiser(ckpt["arch"], ckpt["channels"], ckpt["t_len"],
                                  dropout=float(ckpt.get("dropout", 0.0)),
                                  base_channels=int(ckpt.get("base_channels", 64)))
    ddpm = LatentDDPM(denoiser, ckpt["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ckpt["model_state"])
    ddpm.eval()
    return ddpm, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: generate samples")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--which", choices=["best", "last"], default="best",
                    help="checkpoint to sample from (default: best-by-val)")
    args = ap.parse_args()

    cfg = load_experiment_config(args.exp)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    dirs = experiment_dirs(cfg)
    n = args.n or int(cfg["generate"]["n_samples"])

    ck = dirs["results"] / f"ckpt_{args.which}.pt"
    if not ck.exists():
        ck = dirs["results"] / "ckpt.pt"       # fallback for older runs
    print(f"[{dirs['name']}] loading {ck.name}")
    ddpm, ckpt = load_ckpt(ck, device)
    scaler = scaler_from_dict(ckpt["scaler"])
    names, fs = ckpt["features"], ckpt["feature_set"]
    C, T = ckpt["channels"], ckpt["t_len"]
    data = load_experiment_data(cfg)

    print(f"[{dirs['name']}] sampling {n} trajectories ({C}x{T}, {fs}) …")
    clamp = cfg["ddpm"].get("sample_clamp")
    chunks = []
    with torch.no_grad():
        for start in range(0, n, 250):
            k = min(250, n - start)
            xs = ddpm.sample(k, shape=(C, T), device=device, clamp=clamp)
            chunks.append(xs.cpu().numpy())
    feats = scaler.inverse_transform(np.concatenate(chunks).transpose(0, 2, 1))

    extra_npz, extra_cols = {}, {}
    rng = np.random.default_rng(int(cfg["seed"]))
    if fs == "controls":
        feats = repair_controls(feats, cfg["controls"])
    else:
        feats = physics_repair(feats, ckpt["bounds"], names)
    if fs in ("gstrack", "gstrack_derived", "controls"):
        # FAF-anchored reconstruction to absolute x/y (geodesic for gstrack, planar
        # velocity walk for gstrack_derived, controls integrator for controls); the
        # FAF is the fixed, known destination.
        xy = generated_to_xy(feats, fs, names, data["aux"], rng)       # (n, T, 2) m
        extra_npz = {"tracks_xyz": xy.astype(np.float32)}
        extra_cols = {"x": xy[:, :, 0], "y": xy[:, :, 1]}

    np.savez_compressed(dirs["results"] / "generated.npz",
                        feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object), **extra_npz)

    # tidy parquet
    frames = []
    for i in range(len(feats)):
        df = pd.DataFrame({nm: feats[i, :, j] for j, nm in enumerate(names)})
        for cname, arr in extra_cols.items():
            df[cname] = arr[i]
        df["flight_id"] = f"GEN_{i:05d}"
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(dirs["results"] / "generated.parquet", index=False)

    # profiles figure
    real = data["X"][data["val_idx"]]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    t = np.arange(T)
    for ax, j, nm in zip(np.atleast_1d(axes), range(len(names)), names):
        for arr, c in ((real, "#1f77b4"), (feats, "#d62728")):
            mu, sd = arr[:, :, j].mean(0), arr[:, :, j].std(0)
            ax.plot(t, mu, color=c, lw=2)
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18)
        ax.set_title(nm)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{dirs['name']}: real (blue) vs generated (red)")
    fig.tight_layout()
    fig.savefig(dirs["results"] / "profiles.png", dpi=130)
    plt.close(fig)

    print(f"[{dirs['name']}] wrote generated.npz / generated.parquet / profiles.png "
          f"-> {dirs['results']}")


if __name__ == "__main__":
    main()
