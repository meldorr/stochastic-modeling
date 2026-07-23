"""Stage 5 — waypoint-guided x/y generation via RePaint inpainting (approach 1).

    python stages/s5_waypoint_generate.py [--scenario S1_straightin] [--n 200] \
        [--channels xy|xyalt] [--resample-u 5]

Loads a trained x/y DDPM (default: the tcn_unet best checkpoint), forces generated
trajectories through the waypoints of a scenario in configs/waypoints.json, and
reports the adherence-vs-realism tradeoff:

    adherence   mean distance of the generated path to each imposed waypoint (m)
    realism     sliced-Wasserstein of generated vs real x/y paths (km)
    flyability  fraction of generated trajectories exceeding the turn-rate / accel
                envelopes (Section 1.4)

Outputs results/waypoints/<scenario>/{generated.npz, overlay.png, metrics.json}.
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

from experiments.common import sliced_wasserstein
from src.data.controls import derive_controls
from src.data.dataset import scaler_from_dict
from src.ddpm import LatentDDPM
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.utils import get_device, load_config


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    den = build_raw_denoiser(ck["arch"], ck["channels"], ck["t_len"], dropout=float(ck.get("dropout", 0.0)), base_channels=int(ck.get("base_channels", 64)))
    ddpm = LatentDDPM(den, ck["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ck["model_state"]); ddpm.eval()
    return ddpm, ck


def build_mask_known(waypoints, names, scaler, C, T, use_alt):
    """(C,T) mask + standardized known tensor for the constrained (channel, t) cells."""
    xi, yi = names.index("x"), names.index("y")
    chans = [xi, yi] + ([names.index("altitude")] if use_alt else [])
    raw = np.zeros((1, T, C), np.float64)
    mask = np.zeros((C, T), np.float32)
    for w in waypoints:
        t = int(w["t"])
        raw[0, t, xi], raw[0, t, yi] = w["x"], w["y"]
        if use_alt:
            raw[0, t, names.index("altitude")] = w["alt"]
        for ch in chans:
            mask[ch, t] = 1.0
    known_std = scaler.transform(raw)[0].transpose(1, 0)          # (C,T) standardized
    return torch.from_numpy(mask), torch.from_numpy(known_std.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_standardscaler__xy/ckpt_best.pt")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--channels", choices=["xy", "xyalt"], default="xy")
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard",
                    help="hard = RePaint replacement (through waypoints); "
                         "soft = DPS guidance (near waypoints, for the SE(2) prior)")
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="soft-mode DPS step")
    ap.add_argument("--resample-u", type=int, default=5, help="hard-mode RePaint resamples")
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = load_config("configs/base.yaml")["ddpm"].get("sample_clamp")
    ccfg = load_config("configs/base.yaml")["controls"]
    use_alt = args.channels == "xyalt"

    wp = json.loads(Path("configs/waypoints.json").read_text())
    FAF = np.array(wp["meta"]["faf"])
    scen_names = list(wp["scenarios"]) if args.scenario == "all" else [args.scenario]

    # real x/y for the realism baseline + overlay
    d = np.load("data/processed.npz", allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    real_xy = d["X"][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    rng = np.random.default_rng(0)
    real_s = real_xy[rng.choice(len(real_xy), 2000, replace=False)]
    rf = real_s[:, ::8, :].reshape(len(real_s), -1) / 1000.0

    print(f"[s5] {args.ckpt}  arch={ck['arch']}  n={args.n}  mode={args.mode}"
          f"{' scale='+str(args.guidance_scale) if args.mode=='soft' else ' U='+str(args.resample_u)}")
    for sc in scen_names:
        wps = wp["scenarios"][sc]["waypoints"]
        mask, known = build_mask_known(wps, names, scaler, C, T, use_alt)
        if args.mode == "soft":
            xs = ddpm.sample_soft_guided(args.n, (C, T), known, mask,
                                         guidance_scale=args.guidance_scale, device=device, clamp=clamp)
        else:
            with torch.no_grad():
                xs = ddpm.sample_guided(args.n, (C, T), mask, known, device=device,
                                        resample_u=args.resample_u, clamp=clamp)
        feats = scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))   # (n,T,C) raw
        gxy = feats[:, :, [names.index("x"), names.index("y")]]

        # metrics
        adh = {f"t{w['t']}": float(np.linalg.norm(gxy[:, int(w["t"]), :] - [w["x"], w["y"]], axis=1).mean())
               for w in wps}
        gf = gxy[:, ::8, :].reshape(len(gxy), -1) / 1000.0
        sw = float(sliced_wasserstein(rf, gf))
        alt = (feats[:, :, names.index("altitude")] if "altitude" in names
               else np.zeros_like(gxy[:, :, 0]))          # xyt has no altitude channel
        der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], alt,
                              feats[:, :, names.index("timedelta")], ccfg)
        fly = der["clip_rates"]
        metrics = {"scenario": sc, "n": args.n, "channels": args.channels, "mode": args.mode,
                   "guidance_scale": args.guidance_scale, "resample_u": args.resample_u,
                   "adherence_m": adh, "sliced_w_xy_km": sw, "flyability_exceed": fly}
        out = Path("results/waypoints") / sc; out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        np.savez_compressed(out / "generated.npz", feats=feats.astype(np.float32),
                            feature_names=np.array(names, dtype=object))
        print(f"  {sc:15s} adherence(m)={ {k: round(v) for k,v in adh.items()} }  "
              f"sliced-W={sw:.2f}km  turn>env={fly['turn_rate']*100:.1f}%  accel>env={fly['along_accel']*100:.1f}%")

        # overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        for s in rng.choice(len(real_s), 400, replace=False):
            ax.plot(real_s[s, :, 0], real_s[s, :, 1], color="#999999", lw=0.3, alpha=0.2)
        for i in range(min(120, len(gxy))):
            ax.plot(gxy[i, :, 0], gxy[i, :, 1], color="#d62728", lw=0.5, alpha=0.35)
        for w in wps:
            ax.scatter(w["x"], w["y"], marker="o", s=90, edgecolor="black", color="#ffcc00", zorder=6)
        ax.scatter(*FAF, marker="*", s=260, color="black", zorder=7, label="FAF")
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
        ax.set_title(f"{sc}: guided (red) vs real (grey); yellow=waypoints")
        fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=130); plt.close(fig)
    print("[s5] wrote results/waypoints/<scenario>/{generated.npz, overlay.png, metrics.json}")


if __name__ == "__main__":
    main()
