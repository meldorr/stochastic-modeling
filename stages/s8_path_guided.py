"""Next Step 3 — path-guided sampling: feasible trajectories around a chosen track.

    python stages/s8_path_guided.py [--ckpt ...] [--n 64] [--bearing 318] [--guidance-scale 1.0]

Builds a desired path (a straight-in toward the FAF along ``bearing``, spaced by the
real distance-to-FAF profile), constrains points along it, and *softly* guides
(DPS) a bundle of samples to fly NEAR it — flyable variants around the route, not a
rigid copy. Reports mean deviation from the path and flyability.

Writes results/<exp>/path_guided.png + .json.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--path", choices=["straight", "curved"], default="straight",
                    help="straight = radial straight-in (OOD); curved = a real flow-mean "
                         "approach rotated to enter from --bearing (in-distribution shape)")
    ap.add_argument("--flow", default="90-132", help="flow whose mean approach is the curved prototype")
    ap.add_argument("--bearing", type=float, default=318.0, help="entry bearing FROM the FAF (deg)")
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=8, help="constrain every k-th point along the path")
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = ck.get("ddpm_cfg", {}).get("sample_clamp")
    xi, yi, tdi = names.index("x"), names.index("y"), names.index("timedelta")

    d = np.load("data/processed.npz", allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    realxy = d["X"][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    FAF = realxy[:, -1, :].mean(0)

    if args.path == "straight":
        # radial straight-in toward the FAF (an OOD shape — the model has ~no straights)
        dist = np.median(np.linalg.norm(realxy - FAF, axis=2), axis=0)
        b = np.radians(args.bearing)
        u = np.array([np.sin(b), np.cos(b)])
        path = FAF[None, :] + dist[:, None] * u[None, :]               # (T, 2)
    else:
        # in-distribution curved shape: a real flow-mean approach rotated so its entry
        # arrives from `bearing` — a shape the SE(2) model knows, at a gap orientation
        flow = d["flow"].astype(str)
        proto = realxy[flow == args.flow].mean(0)                     # (T, 2) smooth mean approach
        entry_brg = np.degrees(np.arctan2(proto[0, 0] - FAF[0], proto[0, 1] - FAF[1])) % 360
        th = np.radians(args.bearing - entry_brg); c, s = np.cos(th), np.sin(th)
        v = proto - FAF
        path = np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c], -1) + FAF

    raw = np.zeros((1, T, C), np.float64)
    raw[0, :, xi], raw[0, :, yi] = path[:, 0], path[:, 1]
    known = torch.from_numpy(scaler.transform(raw)[0].transpose(1, 0).astype(np.float32))
    mask = torch.zeros(C, T)
    cidx = np.arange(0, T, args.stride)
    mask[xi, cidx] = 1.0; mask[yi, cidx] = 1.0
    print(f"[s8] {args.ckpt} epoch={ck.get('epoch')} n={args.n} bearing={args.bearing} "
          f"scale={args.guidance_scale} constrained_pts={len(cidx)}")

    xs = ddpm.sample_soft_guided(args.n, (C, T), known, mask,
                                 guidance_scale=args.guidance_scale, device=device, clamp=clamp)
    feats = scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))
    gxy = feats[:, :, [xi, yi]].astype(float)

    dev = np.linalg.norm(gxy - path[None], axis=2)                     # (n, T) deviation
    ccfg = load_config("configs/base.yaml")["controls"]
    der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], np.zeros_like(gxy[:, :, 0]),
                          feats[:, :, tdi], ccfg)
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}
    metrics = {"epoch": ck.get("epoch"), "n": args.n, "bearing": args.bearing,
               "guidance_scale": args.guidance_scale, "mean_deviation_m": float(dev.mean()),
               "median_deviation_m": float(np.median(dev)), "flyability_exceed": fly}
    dirs = Path(args.ckpt).parent
    tag = f"{args.path}_b{int(args.bearing)}_s{args.guidance_scale}"
    (dirs / f"path_guided_{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[s8] mean dev={dev.mean():.0f}m  median={np.median(dev):.0f}m  "
          f"turn>env={fly['turn_rate']*100:.1f}%  accel>env={fly['along_accel']*100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(LineCollection(list(gxy / 1000.0), colors="#1f77b4", linewidths=0.6, alpha=0.4))
    ax.plot(path[:, 0] / 1000, path[:, 1] / 1000, "r-", lw=3, zorder=6, label="desired path")
    ax.scatter(FAF[0] / 1000, FAF[1] / 1000, marker="*", s=320, color="black", zorder=7, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Next Step 3 — feasible trajectories around a {args.path} path from {args.bearing:.0f}° "
                 f"(epoch {ck.get('epoch')})\nmean dev {dev.mean():.0f} m, "
                 f"turn>env {fly['turn_rate']*100:.1f}%, accel>env {fly['along_accel']*100:.1f}%")
    fig.tight_layout(); fig.savefig(dirs / f"path_guided_{tag}.png", dpi=130); plt.close(fig)
    print(f"[s8] wrote {dirs}/path_guided_{tag}.png + .json")


if __name__ == "__main__":
    main()
