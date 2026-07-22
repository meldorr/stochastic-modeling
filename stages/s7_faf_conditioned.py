"""Next Step 2 — FAF-tail-conditioned generation: landing options from all directions.

    python stages/s7_faf_conditioned.py [--ckpt ...ckpt_best.pt] [--n 300] [--tail 20]

Pin the last ``tail`` points of the trajectory to the *real* FAF final-approach
segment (hard RePaint constraint), and let the SE(2) prior complete the rest. Since
the model learned flyable shapes from every orientation, the completions feed into
that one fixed final from many bearings — approaches to the runway from all
directions. Reports entry-bearing coverage, tail adherence, and flyability.

Writes results/<exp>/faf_conditioned.png + .json.
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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tail", type=int, default=20, help="# of final points pinned to the FAF approach")
    ap.add_argument("--resample-u", type=int, default=2)
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = ck.get("ddpm_cfg", {}).get("sample_clamp")
    xi, yi, tdi = names.index("x"), names.index("y"), names.index("timedelta")
    K = args.tail

    # target tail = real FAF final-approach segment (tight: final bearing std ~2 deg)
    d = np.load("data/processed.npz", allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    realxy = d["X"][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    tail = np.median(realxy[:, T - K:, :], axis=0)                  # (K, 2)
    FAF = tail[-1]

    raw = np.zeros((1, T, C), np.float64)
    raw[0, T - K:, xi], raw[0, T - K:, yi] = tail[:, 0], tail[:, 1]
    known = torch.from_numpy(scaler.transform(raw)[0].transpose(1, 0).astype(np.float32))   # (C,T)
    mask = torch.zeros(C, T)
    mask[xi, T - K:] = 1.0; mask[yi, T - K:] = 1.0
    print(f"[s7] {args.ckpt} epoch={ck.get('epoch')} n={args.n} tail={K} FAF=({FAF[0]:.0f},{FAF[1]:.0f})")

    chunks = []
    for start in range(0, args.n, 100):
        k = min(100, args.n - start)
        xs = ddpm.sample_guided(k, (C, T), mask, known, device=device,
                                resample_u=args.resample_u, clamp=clamp)
        chunks.append(xs.cpu().numpy())
    feats = scaler.inverse_transform(np.concatenate(chunks).transpose(0, 2, 1))
    gxy = feats[:, :, [xi, yi]].astype(float)

    # metrics
    tail_adh = float(np.linalg.norm(gxy[:, T - K:, :] - tail[None], axis=2).mean())
    ent = gxy[:, 0, :] - FAF
    ebrg = np.degrees(np.arctan2(ent[:, 0], ent[:, 1])) % 360
    counts = np.histogram(ebrg, bins=36, range=(0, 360))[0]
    cov_cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    occupied = int((counts > 0).sum())
    ccfg = load_config("configs/base.yaml")["controls"]
    der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], np.zeros_like(gxy[:, :, 0]),
                          feats[:, :, tdi], ccfg)
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}
    metrics = {"epoch": ck.get("epoch"), "n": args.n, "tail": K,
               "tail_adherence_m": tail_adh, "entry_bearing_cv": cov_cv,
               "entry_sectors_occupied_of_36": occupied, "flyability_exceed": fly}
    dirs = Path(args.ckpt).parent
    (dirs / "faf_conditioned.json").write_text(json.dumps(metrics, indent=2))
    print(f"[s7] tail adherence={tail_adh:.0f}m  entry sectors {occupied}/36 (CV {cov_cv:.2f})  "
          f"turn>env={fly['turn_rate']*100:.1f}%  accel>env={fly['along_accel']*100:.1f}%")

    # overlay: approaches coloured by entry bearing, converging on the fixed FAF final
    fig, ax = plt.subplots(figsize=(10.5, 10))
    cols = plt.cm.hsv(ebrg / 360.0)
    ax.add_collection(LineCollection(list(gxy / 1000.0), colors=cols, linewidths=0.5, alpha=0.5))
    ax.plot(tail[:, 0] / 1000, tail[:, 1] / 1000, "k-", lw=3, zorder=6, label=f"pinned FAF final ({K} pts)")
    ax.scatter(FAF[0] / 1000, FAF[1] / 1000, marker="*", s=320, color="black", zorder=7, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Next Step 2 — FAF-conditioned approaches (epoch {ck.get('epoch')})\n"
                 f"{occupied}/36 entry sectors, turn>env {fly['turn_rate']*100:.1f}% "
                 f"(colour = entry bearing)")
    fig.tight_layout(); fig.savefig(dirs / "faf_conditioned.png", dpi=130); plt.close(fig)
    print(f"[s7] wrote {dirs}/faf_conditioned.png + faf_conditioned.json")


if __name__ == "__main__":
    main()
