"""Operational-uncertainty envelope around a standard route — two methods.

    python stages/s9_route_uncertainty.py --method guided     [--fixes 5 --guidance-scale 0.3 --n 200]
    python stages/s9_route_uncertainty.py --method montecarlo  [--n 3000 --select 100 --tail 20]

Both build a dispersion corridor around the same nominal route (a real flow-mean
approach rotated to enter from --bearing):

  guided      soft-anchor a few route *fixes* and let the model disperse freely
              between them (the between-fix spread = learned operational variability).
  montecarlo  FAF-condition a big pool, then KEEP the M trajectories closest to the
              route (rejection sampling) — every kept path is a genuine model sample,
              no guidance, so no jitter; the spread is the model's natural dispersion.

Writes results/<exp>/route_uncertainty_<method>.png + .json (corridor width, flyability).
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


def curved_route(realxy, FAF, flow_arr, flow, bearing):
    """A real flow-mean approach rotated so its entry arrives from `bearing`."""
    proto = realxy[flow_arr == flow].mean(0)
    entry = np.degrees(np.arctan2(proto[0, 0] - FAF[0], proto[0, 1] - FAF[1])) % 360
    th = np.radians(bearing - entry); c, s = np.cos(th), np.sin(th)
    v = proto - FAF
    return np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c], -1) + FAF


def lateral_corridor(trajs, route):
    """Signed perpendicular offset of each traj from the route; return p10/p50/p90 (T,)."""
    tang = np.gradient(route, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)                 # left normal
    off = np.einsum("ntd,td->nt", trajs - route[None], nrm)          # (N, T) signed metres
    return np.percentile(off, [10, 50, 90], axis=0), nrm             # (3,T), (T,2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt")
    ap.add_argument("--method", choices=["guided", "montecarlo"], required=True)
    ap.add_argument("--path", choices=["curved", "straight"], default="curved")
    ap.add_argument("--flow", default="90-132")
    ap.add_argument("--bearing", type=float, default=255.0)
    ap.add_argument("--n", type=int, default=None, help="guided: #samples; montecarlo: pool size")
    ap.add_argument("--select", type=int, default=100, help="montecarlo: keep M closest")
    ap.add_argument("--fixes", type=int, default=5, help="guided: #route fixes to anchor")
    ap.add_argument("--guidance-scale", type=float, default=0.3)
    ap.add_argument("--tail", type=int, default=20, help="montecarlo: FAF-tail conditioning length")
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
        dist = np.median(np.linalg.norm(realxy - FAF, axis=2), axis=0)
        b = np.radians(args.bearing); u = np.array([np.sin(b), np.cos(b)])
        route = FAF[None, :] + dist[:, None] * u[None, :]                 # radial straight-in (T,2)
    else:
        route = curved_route(realxy, FAF, d["flow"].astype(str), args.flow, args.bearing)

    def to_known(raw_xy_full, idxs):
        raw = np.zeros((1, T, C), np.float64)
        raw[0, idxs, xi], raw[0, idxs, yi] = raw_xy_full[idxs, 0], raw_xy_full[idxs, 1]
        known = torch.from_numpy(scaler.transform(raw)[0].transpose(1, 0).astype(np.float32))
        mask = torch.zeros(C, T); mask[xi, idxs] = 1.0; mask[yi, idxs] = 1.0
        return mask, known

    extra = {}
    if args.method == "guided":
        n = args.n or 200
        fix_idx = np.linspace(0, T - 1, args.fixes).round().astype(int)
        mask, known = to_known(route, fix_idx)
        xs = ddpm.sample_soft_guided(n, (C, T), known, mask,
                                     guidance_scale=args.guidance_scale, device=device, clamp=clamp)
        feats = scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))
        trajs = feats[:, :, [xi, yi]].astype(float)
        extra = {"fixes": int(args.fixes), "guidance_scale": args.guidance_scale}
        print(f"[s9] guided: {n} samples, {args.fixes} fixes, scale {args.guidance_scale}")
    else:  # montecarlo: FAF-conditioned pool -> keep M closest to the route
        n = args.n or 3000
        tail = np.median(realxy[:, T - args.tail:, :], axis=0)
        raw_tail = np.zeros((T, 2)); raw_tail[T - args.tail:] = tail
        mask, known = to_known(raw_tail, np.arange(T - args.tail, T))
        pool = []
        for start in range(0, n, 250):
            k = min(250, n - start)
            xs = ddpm.sample_guided(k, (C, T), mask, known, device=device, resample_u=1, clamp=clamp)
            pool.append(scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))[:, :, [xi, yi]])
        pool = np.concatenate(pool).astype(float)
        dist = np.linalg.norm(pool - route[None], axis=2).mean(1)     # (n,) mean dist to route
        keep = np.argsort(dist)[:args.select]
        trajs = pool[keep]
        extra = {"pool": n, "selected": int(args.select),
                 "kept_mean_dist_m": float(dist[keep].mean()), "cutoff_dist_m": float(dist[keep].max())}
        print(f"[s9] montecarlo: pool {n} -> kept {args.select} closest "
              f"(mean {dist[keep].mean():.0f} m, cutoff {dist[keep].max():.0f} m)")

    # corridor + flyability (need timedelta for controls -> take it from a fresh unconditional? use median real td)
    (p10, p50, p90), nrm = lateral_corridor(trajs, route)
    width = float((p90 - p10).mean())
    td_real = np.median(d["X"][:, :, rn.index("timedelta")].astype(float), axis=0)   # (T,) typical td
    td_b = np.broadcast_to(td_real, (len(trajs), T))
    der = derive_controls(trajs[:, :, 0], trajs[:, :, 1], np.zeros_like(trajs[:, :, 0]), td_b,
                          load_config("configs/base.yaml")["controls"])
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}
    metrics = {"method": args.method, "epoch": ck.get("epoch"), "bearing": args.bearing,
               "n_trajectories": int(len(trajs)), "corridor_width_mean_m": width,
               "flyability_exceed": fly, **extra}
    dirs = Path(args.ckpt).parent
    (dirs / f"route_uncertainty_{args.method}_{args.path}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[s9] corridor mean width {width:.0f} m  turn>env {fly['turn_rate']*100:.1f}%  "
          f"accel>env {fly['along_accel']*100:.1f}%")

    # figure: route + bundle + median + shaded p10-p90 corridor
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(LineCollection(list(trajs / 1000.0), colors="#1f77b4", linewidths=0.4, alpha=0.25))
    hi = (route + p90[:, None] * nrm) / 1000.0
    lo = (route + p10[:, None] * nrm) / 1000.0
    ax.fill(np.concatenate([hi[:, 0], lo[::-1, 0]]), np.concatenate([hi[:, 1], lo[::-1, 1]]),
            color="#1f77b4", alpha=0.18, zorder=2, label="p10-p90 corridor")
    ax.plot(route[:, 0] / 1000, route[:, 1] / 1000, "r-", lw=2.5, zorder=6, label="standard route")
    ax.plot((route[:, 0] + p50 * nrm[:, 0]) / 1000, (route[:, 1] + p50 * nrm[:, 1]) / 1000,
            "k--", lw=1.5, zorder=6, label="median")
    ax.scatter(FAF[0] / 1000, FAF[1] / 1000, marker="*", s=300, color="black", zorder=7, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Route uncertainty — {args.method} ({len(trajs)} trajs, epoch {ck.get('epoch')})\n"
                 f"corridor width {width:.0f} m, turn>env {fly['turn_rate']*100:.1f}%, "
                 f"accel>env {fly['along_accel']*100:.1f}%")
    fig.tight_layout(); fig.savefig(dirs / f"route_uncertainty_{args.method}_{args.path}.png", dpi=130); plt.close(fig)
    print(f"[s9] wrote {dirs}/route_uncertainty_{args.method}_{args.path}.png + .json")


if __name__ == "__main__":
    main()
