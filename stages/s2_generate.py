"""Stage 2 — sample the trained DDPM and save generated trajectories.

    python stages/s2_generate.py --exp configs/experiments/<name>.yaml [--n 1000]

Outputs in results/<experiment>/:
    generated.npz       feats (n, T, C) raw units [+ tracks_xyz for controls]
    generated.parquet   tidy per-timestep frame
    profiles.png        real vs generated channel profiles

Feature-set specifics:
    xy / gstrack  physics repair clips channels to the training envelope
    controls      sampled control sequences are clipped to the Section-1.4
                  envelopes, then re-integrated from entry states drawn from
                  real flights -> absolute x/y/z tracks
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

from src.data.controls import integrate_controls
from src.data.dataset import scaler_from_dict
from src.ddpm import LatentDDPM
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.reconstruct import physics_repair
from src.pipeline.utils import experiment_dirs, get_device, load_experiment_config, set_seed
from stages.common import load_experiment_data


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    denoiser = build_raw_denoiser(ckpt["arch"], ckpt["channels"], ckpt["t_len"],
                                  dropout=float(ckpt.get("dropout", 0.0)))
    ddpm = LatentDDPM(denoiser, ckpt["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ckpt["model_state"])
    ddpm.eval()
    return ddpm, ckpt


def repair_controls(feats: np.ndarray, ccfg: dict) -> np.ndarray:
    """Clip sampled controls to the Section-1.4 envelopes; monotone timedelta."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: generate samples")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.exp)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    dirs = experiment_dirs(cfg)
    n = args.n or int(cfg["generate"]["n_samples"])

    ddpm, ckpt = load_ckpt(dirs["results"] / "ckpt.pt", device)
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
    if fs == "controls":
        feats = repair_controls(feats, cfg["controls"])
        # entry states resampled from REAL flights (controls carry no absolute pose)
        rng = np.random.default_rng(int(cfg["seed"]))
        entry = data["aux"]["entry"][rng.integers(0, len(data["aux"]["entry"]), n)]
        tracks = integrate_controls(entry, feats)                      # (n, T, 3) x,y,z m
        extra_npz = {"tracks_xyz": tracks.astype(np.float32), "entry": entry.astype(np.float32)}
        extra_cols = {"x": tracks[:, :, 0], "y": tracks[:, :, 1], "z_m": tracks[:, :, 2]}
    else:
        feats = physics_repair(feats, ckpt["bounds"], names)

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
