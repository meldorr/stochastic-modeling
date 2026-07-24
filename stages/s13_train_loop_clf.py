"""Stage 13 — train the noise-conditioned LOOP classifier p(loop | x_t, t) on the pool.

Labels are automatic and geometric: a flight is a "loop" if its heading winds a full
turn within a short window (default: >=360 deg net within 30 steps) — closed orbits /
holds, not normal base-turn maneuvering (~6% of real pool flights). The classifier is
trained with bounded SE(2) augmentation (loops are rotation/translation invariant) at
every diffusion noise level, so its gradient composes into sampling as the ANTI-loop
expert: ``classifiers=[(clf, 1, -scale)]`` in sample_composed.

    python stages/s13_train_loop_clf.py --ckpt results/ddpm_tcn_unet_se2free__xyt/ckpt_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.augment import se2_augment
from src.data.dataset import scaler_from_dict
from src.ddpm.classifier import TrajClassifier
from src.ddpm.schedule import build_schedule


def loop_labels(xy: np.ndarray, window: int = 30, thresh_deg: float = 360.0) -> np.ndarray:
    """1 = closed loop: heading winds >= thresh within `window` steps. xy (N, T, 2)."""
    dxy = np.diff(xy, axis=1)
    hdg = np.unwrap(np.arctan2(dxy[..., 0], dxy[..., 1]), axis=1)
    net = np.abs(hdg[:, window:] - hdg[:, :-window])
    return (np.degrees(net.max(1)) >= thresh_deg).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2free__xyt/ckpt_best.pt")
    ap.add_argument("--data", default="data/processed.npz")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--thresh-deg", type=float, default=360.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="results/loop_clf")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = ck["features"]
    scaler = scaler_from_dict(ck["scaler"])
    sched = build_schedule(ck["ddpm_cfg"])
    sab = sched["sqrt_alphas_bar"].to(dev)
    s1mab = sched["sqrt_one_minus_alphas_bar"].to(dev)
    tdiff = int(ck["ddpm_cfg"]["timesteps"])

    d = np.load(args.data, allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    X = np.stack([d["X"][:, :, rn.index(n)] for n in names], -1).astype(np.float32)   # (N,T,C) raw
    xi, yi = names.index("x"), names.index("y")
    y = loop_labels(X[:, :, [xi, yi]].astype(float), args.window, args.thresh_deg)
    Xs = scaler.transform(X)
    print(f"[s13] N={len(Xs)} loops={int(y.sum())} ({100 * y.mean():.1f}%) "
          f"window={args.window} thresh={args.thresh_deg}deg dev={dev}")

    Xt = torch.from_numpy(Xs.transpose(0, 2, 1).copy()).to(dev)                       # (N,C,T)
    yt = torch.from_numpy(y).to(dev)
    hi, ni = torch.where(yt == 1)[0], torch.where(yt == 0)[0]
    C, T = Xt.shape[1], Xt.shape[2]
    nh = args.batch // 2

    clf = TrajClassifier(C, T).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        clf.train()
        bi = torch.cat([hi[torch.randint(len(hi), (nh,), device=dev)],        # balanced: half loops
                        ni[torch.randint(len(ni), (nh,), device=dev)]])
        xb = se2_augment(Xt[bi], xi, yi)                                    # bounded se2 (invariance)
        yb = yt[bi]
        t = torch.randint(0, tdiff, (xb.shape[0],), device=dev)
        x_t = sab[t][:, None, None] * xb + s1mab[t][:, None, None] * torch.randn_like(xb)
        loss = F.cross_entropy(clf(x_t, t), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            clf.eval()
            with torch.no_grad():
                vb = torch.cat([hi[torch.randint(len(hi), (512,), device=dev)],
                                ni[torch.randint(len(ni), (512,), device=dev)]])
                xv = se2_augment(Xt[vb], xi, yi)
                yv = yt[vb]
                tv = torch.randint(0, tdiff, (xv.shape[0],), device=dev)
                xvt = sab[tv][:, None, None] * xv + s1mab[tv][:, None, None] * torch.randn_like(xv)
                pred = clf(xvt, tv).argmax(1)
                acc = (pred == yv).float().mean().item()
                rec = (pred[yv == 1] == 1).float().mean().item()
            print(f"[s13] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"balacc {acc:.3f} loop-recall {rec:.3f} ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": clf.state_dict(), "features": names, "channels": C, "t_len": T,
                "scaler": ck["scaler"], "ddpm_cfg": ck["ddpm_cfg"], "loop_class": 1,
                "window": args.window, "thresh_deg": args.thresh_deg,
                "arch": {"width": 96, "tdim": 128, "n_layers": 6}}, out / "clf.pt")
    print(f"[s13] saved {out}/clf.pt")


if __name__ == "__main__":
    main()
