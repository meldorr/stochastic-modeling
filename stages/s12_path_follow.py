"""Stage 12 — corridor-guided generation: enter and follow a real corridor prototype.

Builds an in-distribution path from the pooled data itself (one airport's flights,
binned by entry bearing; the densest sector's per-timestep median = the prototype),
then samples the se2_free DDPM under tube guidance (sample_corridor_guided): stay
within ±corridor-km of the polyline, pacing free, ends attached to the path ends.

    python stages/s12_path_follow.py --airport EHAM --n 50 --guidance-scale 50

Outputs results/<exp>/path_follow/<airport>_b<sector>_gs<g>/
    {generated.npz, overlay.png, metrics.json}
Metrics: corridor RMS / p95 deviation, % timesteps inside tube, entry/end error,
path arc-length coverage, flyability (turn/accel envelope exceedance).
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
                             dropout=float(ck.get("dropout") or 0.0),
                             base_channels=int(ck.get("base_channels") or 64))
    ddpm = LatentDDPM(den, ck["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ck["model_state"]); ddpm.eval()
    return ddpm, ck


def densify(poly: np.ndarray, n_pts: int = 400) -> np.ndarray:
    """Resample a polyline to n_pts equally spaced along arc length."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    grid = np.linspace(0.0, s[-1], n_pts)
    return np.stack([np.interp(grid, s, poly[:, i]) for i in range(2)], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2free__xyt/ckpt_best.pt")
    ap.add_argument("--data", default="data/processed.npz")
    ap.add_argument("--airport", default="EHAM", help="pool airport whose corridor is the prototype")
    ap.add_argument("--sector-deg", type=float, default=None,
                    help="entry-bearing sector center (45-deg bins); default = densest sector")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--corridor-km", type=float, default=5.0, help="tube half-width")
    ap.add_argument("--guidance-scale", type=float, default=50.0)
    ap.add_argument("--entry-k", type=int, default=8)
    ap.add_argument("--pin-tail", type=int, default=20,
                    help="hard-pin the last k timesteps to the prototype's final approach (strict FAF)")
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = ck.get("ddpm_cfg", {}).get("sample_clamp")
    xi, yi, tdi = names.index("x"), names.index("y"), names.index("timedelta")

    # --- corridor prototype: densest entry-bearing sector of one airport ------------
    d = np.load(args.data, allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    ap_lab = np.array([str(s) for s in (d["airport"] if "airport" in d else d["flow"])])
    sel = np.ones(len(ap_lab), bool) if args.airport == "ALL" else (ap_lab == args.airport)
    if not sel.any():
        raise SystemExit(f"airport {args.airport!r} not in {sorted(set(ap_lab))}")
    axy = d["X"][sel][:, :, [rn.index("x"), rn.index("y")]].astype(float)       # (Na, T, 2)
    ent = axy[:, 0, :] - axy[:, -1, :].mean(0)                                  # entry rel. FAF
    ebrg = np.degrees(np.arctan2(ent[:, 0], ent[:, 1])) % 360
    bins = (ebrg // 45).astype(int)
    sector = (int(args.sector_deg // 45) if args.sector_deg is not None
              else int(np.bincount(bins, minlength=8).argmax()))
    in_sec = bins == sector
    proto = np.median(axy[in_sec], axis=0)                                      # (T, 2) median path
    print(f"[s12] {args.airport} sector {sector * 45}-{sector * 45 + 45}deg "
          f"({int(in_sec.sum())} flights) | n={args.n} tube=±{args.corridor_km}km "
          f"scale={args.guidance_scale}")

    # to standardized space: dense polyline + tube width under the isotropic x/y scale
    raw = np.zeros((1, T, C), np.float64)
    raw[0, :, xi], raw[0, :, yi] = proto[:, 0], proto[:, 1]
    std = scaler.transform(raw)[0]
    path_std = densify(np.stack([std[:, xi], std[:, yi]], axis=-1))
    width_std = args.corridor_km * 1000.0 / float(scaler.scale[xi])

    # strict FAF: hard-pin the last k timesteps' x/y to the prototype's own final
    # approach segment (median real pacing included) — RePaint projection, not attraction
    K = int(args.pin_tail)
    known = torch.zeros(C, T)
    kmask = torch.zeros(C, T)
    if K > 0:
        known[xi, T - K:] = torch.from_numpy(std[T - K:, xi].astype(np.float32))
        known[yi, T - K:] = torch.from_numpy(std[T - K:, yi].astype(np.float32))
        kmask[xi, T - K:] = 1.0
        kmask[yi, T - K:] = 1.0

    xs = ddpm.sample_corridor_guided(
        args.n, (C, T), torch.from_numpy(path_std.astype(np.float32)), xi, yi, width_std,
        guidance_scale=args.guidance_scale, entry_k=args.entry_k,
        known=known if K > 0 else None, kmask=kmask if K > 0 else None,
        device=device, clamp=clamp)
    feats = scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))       # (n, T, C) raw
    gxy = feats[:, :, [xi, yi]].astype(float)

    # --- metrics ---------------------------------------------------------------------
    dense = densify(proto, 600)                                                  # raw-metres polyline
    d2 = ((gxy[:, :, None, :] - dense[None, None]) ** 2).sum(-1)                 # (n, T, P)
    near = d2.argmin(2)                                                          # nearest path index
    dmin = np.sqrt(d2.min(2))                                                    # (n, T) metres
    inside = float((dmin <= args.corridor_km * 1000.0).mean())
    coverage = float(np.mean((near.max(1) - near.min(1)) / (dense.shape[0] - 1)))
    entry_err = float(np.linalg.norm(gxy[:, 0] - dense[0], axis=1).mean())
    end_err = float(np.linalg.norm(gxy[:, -1] - dense[-1], axis=1).mean())
    ccfg = load_config("configs/base.yaml")["controls"]
    der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], np.zeros_like(gxy[:, :, 0]),
                          feats[:, :, tdi], ccfg)
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}
    metrics = {"epoch": ck.get("epoch"), "airport": args.airport, "sector": sector * 45,
               "n": args.n, "corridor_km": args.corridor_km, "guidance_scale": args.guidance_scale,
               "pin_tail": int(args.pin_tail),
               "corridor_rms_km": float(dmin.mean() / 1e3), "corridor_p95_km": float(np.percentile(dmin, 95) / 1e3),
               "frac_inside_tube": inside, "path_coverage": coverage,
               "entry_err_km": entry_err / 1e3, "end_err_km": end_err / 1e3,
               "flyability_exceed": fly}

    out = Path(args.ckpt).parent / "path_follow" / f"{args.airport}_b{sector * 45}_gs{args.guidance_scale:g}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "generated.npz", feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object), prototype=proto)
    print(f"[s12] inside-tube {inside * 100:.0f}%  rms {metrics['corridor_rms_km']:.1f}km  "
          f"coverage {coverage * 100:.0f}%  entry {metrics['entry_err_km']:.1f}km  "
          f"end {metrics['end_err_km']:.1f}km  turn>env {fly['turn_rate'] * 100:.1f}%  "
          f"accel>env {fly['along_accel'] * 100:.1f}%")

    # --- overlay -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    rs = axy[in_sec][np.random.default_rng(0).choice(int(in_sec.sum()), min(300, int(in_sec.sum())), replace=False)]
    ax.add_collection(LineCollection([r / 1e3 for r in rs], colors="#999999", linewidths=0.3, alpha=0.2))
    ax.add_collection(LineCollection([g / 1e3 for g in gxy], colors="#1f77b4", linewidths=0.7, alpha=0.5))
    ax.plot(dense[:, 0] / 1e3, dense[:, 1] / 1e3, "r-", lw=2.5, zorder=6, label="prototype path")
    ax.plot(dense[:, 0] / 1e3, dense[:, 1] / 1e3, color="red", lw=2 * args.corridor_km / 1.0,
            alpha=0.08, zorder=1)                                       # visual tube (approx width)
    ax.scatter(*(dense[-1] / 1e3), marker="*", s=300, color="black", zorder=7, label="path end (FAF)")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"s12 corridor-guided — {args.airport} sector {sector * 45}° (epoch {ck.get('epoch')})\n"
                 f"inside ±{args.corridor_km}km: {inside * 100:.0f}%, coverage {coverage * 100:.0f}%, "
                 f"turn>env {fly['turn_rate'] * 100:.1f}%, accel>env {fly['along_accel'] * 100:.1f}%")
    fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=130); plt.close(fig)
    print(f"[s12] wrote {out}/")


if __name__ == "__main__":
    main()
