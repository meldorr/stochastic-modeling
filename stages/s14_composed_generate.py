"""Stage 14 — composed guidance: corridor tube + strict (feathered) FAF + anti-loop.

Product-of-experts sampling (sample_composed): the corridor location terms, the
RePaint FAF pin, and the loop classifier's NEGATIVE gradient are summed each reverse
step — "follow this corridor, end exactly on final, and don't orbit on the way".

    python stages/s14_composed_generate.py --airport EHAM --n 50 \
        --guidance-scale 200 --loop-scale 40 [--loop-clf results/loop_clf/clf.pt]

Outputs results/<exp>/composed/<airport>_b<sector>_gs<g>_ls<l>/
    {generated.npz, overlay.png, metrics.json}
Same corridor metrics as s12 plus generated loop fraction (geometric, s13's detector).
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
from src.ddpm.classifier import load_classifier
from src.pipeline.utils import get_device, load_config
from stages.s12_path_follow import densify, load_ckpt
from stages.s13_train_loop_clf import loop_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2free__xyt/ckpt_best.pt")
    ap.add_argument("--loop-clf", default="results/loop_clf/clf.pt")
    ap.add_argument("--data", default="data/processed.npz")
    ap.add_argument("--airport", default="EHAM")
    ap.add_argument("--sector-deg", type=float, default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--corridor-km", type=float, default=5.0)
    ap.add_argument("--guidance-scale", type=float, default=200.0, help="tube+entry scale")
    ap.add_argument("--loop-scale", type=float, default=40.0,
                    help="anti-loop strength (applied as negative classifier guidance); 0 disables")
    ap.add_argument("--entry-k", type=int, default=8)
    ap.add_argument("--pin-tail", type=int, default=20)
    ap.add_argument("--feather", type=int, default=6,
                    help="ramp the pin mask over this many steps before the tail (smooth seam)")
    ap.add_argument("--route", default=None,
                    help="synthetic route body as 'x,y;x,y;...' in km (airport frame). The real "
                         "median final approach is spliced on as the pinned tail — i.e. a NEW "
                         "routing onto the EXISTING final. Overrides the sector prototype.")
    ap.add_argument("--route-name", default="custom", help="output-dir tag for --route runs")
    ap.add_argument("--plot-n", type=int, default=0,
                    help="plot only this many trajectories (0 = all); <=15 get distinct "
                         "colors + thicker lines so individual paths can be analysed")
    ap.add_argument("--smooth", type=int, default=0,
                    help="moving-average window (route-body points) to round waypoint corners "
                         "before guiding; 0 = off. ~9 turns polyline bends into flyable arcs")
    args = ap.parse_args()

    device = get_device("auto")
    ddpm, ck = load_ckpt(args.ckpt, device)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    clamp = ck.get("ddpm_cfg", {}).get("sample_clamp")
    xi, yi, tdi = names.index("x"), names.index("y"), names.index("timedelta")

    # corridor prototype (same construction as s12)
    d = np.load(args.data, allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    ap_lab = np.array([str(s) for s in (d["airport"] if "airport" in d else d["flow"])])
    sel = np.ones(len(ap_lab), bool) if args.airport == "ALL" else (ap_lab == args.airport)
    axy = d["X"][sel][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    ent = axy[:, 0, :] - axy[:, -1, :].mean(0)
    ebrg = np.degrees(np.arctan2(ent[:, 0], ent[:, 1])) % 360
    bins = (ebrg // 45).astype(int)
    K, Fk = int(args.pin_tail), int(args.feather)
    if args.route:
        # synthetic body (never-flown routing) spliced onto the REAL median final approach:
        # body arc-length-resampled to T-(K+F) rows; tail rows keep real median pacing so the
        # pin values are genuine final-approach geometry.
        wps = np.array([[float(v) for v in p.split(",")] for p in args.route.split(";")]) * 1e3
        tail = np.median(axy[:, -(K + Fk):], axis=0)                     # (K+F, 2) real final
        body = densify(np.vstack([wps, tail[:1]]), T - (K + Fk))         # ends at tail start
        if args.smooth > 1:
            # round the waypoint corners: edge-padded moving average keeps the endpoints
            w = int(args.smooth)
            pad = np.concatenate([body[:1].repeat(w, 0), body, body[-1:].repeat(w, 0)])
            ker = np.ones(2 * w + 1) / (2 * w + 1)
            body = np.stack([np.convolve(pad[:, i], ker, mode="same")[w:-w] for i in range(2)], -1)
            body[0], body[-1] = np.vstack([wps, tail[:1]])[0], tail[0]   # re-pin the ends
        proto = np.vstack([body, tail])                                  # (T, 2) spliced route
        sector, in_sec = -1, np.ones(len(axy), bool)                     # backdrop: all flights
        tag = args.route_name
        # novelty: does any real flight fly this routing? (chunked: the full broadcast
        # over 34k flights x 200 x 400 route points would need ~44 GB and OOM the job)
        dsn = densify(proto, 400)
        md = np.empty(len(axy))
        for i in range(0, len(axy), 1000):
            blk = axy[i:i + 1000]                                        # (b, T, 2)
            d2b = ((blk[:, :, None, :] - dsn[None, None]) ** 2).sum(-1)  # (b, T, 400)
            md[i:i + 1000] = np.sqrt(d2b.min(2)).mean(1)
        novelty = {"real_flights_mean_within_5km": int((md < 5e3).sum()),
                   "closest_real_mean_dist_km": float(md.min() / 1e3)}
        print(f"[s14] route '{tag}': {novelty['real_flights_mean_within_5km']} real flights fly it; "
              f"closest mean-dist {novelty['closest_real_mean_dist_km']:.1f} km")
    else:
        sector = (int(args.sector_deg // 45) if args.sector_deg is not None
                  else int(np.bincount(bins, minlength=8).argmax()))
        in_sec = bins == sector
        proto = np.median(axy[in_sec], axis=0)
        tag = f"b{sector * 45}"
        novelty = None

    raw = np.zeros((1, T, C), np.float64)
    raw[0, :, xi], raw[0, :, yi] = proto[:, 0], proto[:, 1]
    std = scaler.transform(raw)[0]
    path_std = densify(np.stack([std[:, xi], std[:, yi]], axis=-1))
    width_std = args.corridor_km * 1000.0 / float(scaler.scale[xi])

    # strict FAF pin with a feathered seam: kmask ramps 0 -> 1 over `feather` steps
    known = torch.zeros(C, T)
    kmask = torch.zeros(C, T)
    for ch, col in ((xi, std[:, xi]), (yi, std[:, yi])):
        known[ch, T - K - Fk:] = torch.from_numpy(col[T - K - Fk:].astype(np.float32))
        kmask[ch, T - K:] = 1.0
        if Fk > 0:
            kmask[ch, T - K - Fk:T - K] = torch.linspace(1.0 / (Fk + 1), Fk / (Fk + 1.0), Fk)

    classifiers = []
    if args.loop_scale > 0:
        clf, cck = load_classifier(args.loop_clf, device)
        classifiers.append((clf, int(cck.get("loop_class", 1)), -args.loop_scale))   # negative = away
    print(f"[s14] {args.airport} {tag} n={args.n} tube=±{args.corridor_km}km "
          f"gs={args.guidance_scale} loop_scale={args.loop_scale} pin={K}+{Fk}feather")

    xs = ddpm.sample_composed(
        args.n, (C, T),
        tube=(torch.from_numpy(path_std.astype(np.float32)), xi, yi, width_std, args.guidance_scale),
        entry=(args.entry_k, torch.from_numpy(path_std[0].astype(np.float32)), xi, yi,
               args.guidance_scale),
        classifiers=classifiers, known=known, kmask=kmask, device=device, clamp=clamp)
    feats = scaler.inverse_transform(xs.cpu().numpy().transpose(0, 2, 1))
    gxy = feats[:, :, [xi, yi]].astype(float)

    # metrics (s12 set + loop fraction of the generated bundle)
    dense = densify(proto, 600)
    d2 = ((gxy[:, :, None, :] - dense[None, None]) ** 2).sum(-1)
    near = d2.argmin(2)
    dmin = np.sqrt(d2.min(2))
    inside = float((dmin <= args.corridor_km * 1000.0).mean())
    coverage = float(np.mean((near.max(1) - near.min(1)) / (dense.shape[0] - 1)))
    loop_frac = float(loop_labels(gxy).mean())
    ccfg = load_config("configs/base.yaml")["controls"]
    der = derive_controls(gxy[:, :, 0], gxy[:, :, 1], np.zeros_like(gxy[:, :, 0]),
                          feats[:, :, tdi], ccfg)
    fly = {k: der["clip_rates"][k] for k in ("turn_rate", "along_accel")}
    metrics = {"epoch": ck.get("epoch"), "airport": args.airport, "route": tag,
               "novelty": novelty,
               "n": args.n, "corridor_km": args.corridor_km, "guidance_scale": args.guidance_scale,
               "loop_scale": args.loop_scale, "pin_tail": K, "feather": Fk,
               "corridor_rms_km": float(dmin.mean() / 1e3),
               "corridor_p95_km": float(np.percentile(dmin, 95) / 1e3),
               "frac_inside_tube": inside, "path_coverage": coverage,
               "loop_frac_generated": loop_frac,
               "entry_err_km": float(np.linalg.norm(gxy[:, 0] - dense[0], axis=1).mean() / 1e3),
               "end_err_km": float(np.linalg.norm(gxy[:, -1] - dense[-1], axis=1).mean() / 1e3),
               "flyability_exceed": fly}

    out = (Path(args.ckpt).parent / "composed"
           / f"{args.airport}_{tag}_gs{args.guidance_scale:g}_ls{args.loop_scale:g}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "generated.npz", feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object), prototype=proto)
    print(f"[s14] inside-tube {inside * 100:.0f}%  rms {metrics['corridor_rms_km']:.1f}km  "
          f"loops {loop_frac * 100:.0f}%  end {metrics['end_err_km']:.2f}km  "
          f"turn>env {fly['turn_rate'] * 100:.1f}%  accel>env {fly['along_accel'] * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 10))
    rs = axy[in_sec][np.random.default_rng(0).choice(int(in_sec.sum()), min(300, int(in_sec.sum())), replace=False)]
    ax.add_collection(LineCollection([r / 1e3 for r in rs], colors="#999999", linewidths=0.3, alpha=0.2))
    # metrics always use all n samples; the overlay may show a subsample for readability
    pn = args.plot_n if args.plot_n > 0 else len(gxy)
    pick = np.random.default_rng(1).choice(len(gxy), min(pn, len(gxy)), replace=False)
    if len(pick) <= 15:
        for j, i in enumerate(pick):                      # distinct colors: traceable individuals
            ax.plot(gxy[i, :, 0] / 1e3, gxy[i, :, 1] / 1e3, lw=1.5, alpha=0.9,
                    color=plt.cm.tab20(j % 20), zorder=4)
            ax.scatter(gxy[i, 0, 0] / 1e3, gxy[i, 0, 1] / 1e3, s=25,
                       color=plt.cm.tab20(j % 20), zorder=5)          # entry dot
    else:
        ax.add_collection(LineCollection([gxy[i] / 1e3 for i in pick],
                                         colors="#1f77b4", linewidths=0.7, alpha=0.5))
    ax.plot(dense[:, 0] / 1e3, dense[:, 1] / 1e3, "r-", lw=2.5, zorder=6, label="prototype path")
    ax.scatter(*(dense[-1] / 1e3), marker="*", s=300, color="black", zorder=7, label="FAF (pinned)")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"s14 composed — {args.airport} {tag} gs={args.guidance_scale:g} "
                 f"anti-loop={args.loop_scale:g} (epoch {ck.get('epoch')})\n"
                 f"inside ±{args.corridor_km}km {inside * 100:.0f}%, loops {loop_frac * 100:.0f}%, "
                 f"turn>env {fly['turn_rate'] * 100:.1f}%, accel>env {fly['along_accel'] * 100:.1f}%")
    fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=130); plt.close(fig)
    print(f"[s14] wrote {out}/")


if __name__ == "__main__":
    main()
