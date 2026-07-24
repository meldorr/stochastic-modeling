"""Stage 10 — train the noise-conditioned holding classifier p(holding | x_t, t).

Trains in the se2aug DDPM's standardized [x, y, timedelta] space, with the SAME bounded
SE(2) augmentation used to train the DDPM. Holdings are rotation/translation invariant,
so the augmentation both regularizes and (via balanced sampling) oversamples the rare
class. The classifier sees every diffusion noise level, so its gradient is usable at
each reverse step in s11 (classifier guidance).

    python stages/s10_train_holding_clf.py --ckpt results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.augment import se2_augment
from src.data.dataset import scaler_from_dict
from src.ddpm.classifier import HoldingClassifier
from src.ddpm.schedule import build_schedule


def holding_labels(labels_path: str, fids: np.ndarray) -> np.ndarray:
    lab = json.loads(Path(labels_path).read_text())
    hold = set(lab.keys()) if isinstance(lab, dict) else set(map(str, lab))
    return np.array([1 if f in hold else 0 for f in fids], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt")
    ap.add_argument("--data", default="data/processed.npz")
    ap.add_argument("--labels", default="data/holding/holding_labels.json")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--augment", choices=["auto", "se2", "none"], default="auto",
                    help="match the DDPM: se2 for se2aug models, none for the plain baseline")
    ap.add_argument("--out", default="results/holding_clf")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = ck["features"]
    if args.augment == "auto":
        use_se2 = str(ck.get("augment") or ck.get("config", {}).get("augment") or "").startswith("se2")
    else:
        use_se2 = args.augment == "se2"
    scaler = scaler_from_dict(ck["scaler"])
    sched = build_schedule(ck["ddpm_cfg"])
    sab = sched["sqrt_alphas_bar"].to(dev)
    s1mab = sched["sqrt_one_minus_alphas_bar"].to(dev)
    tdiff = int(ck["ddpm_cfg"]["timesteps"])

    d = np.load(args.data, allow_pickle=True)
    rn = [str(s) for s in d["feature_names"]]
    X = np.stack([d["X"][:, :, rn.index(n)] for n in names], -1).astype(np.float32)   # (N,T,C) raw
    Xs = scaler.transform(X)                                                          # DDPM space
    fids = np.array([str(s) for s in d["flight_ids"]])
    y = holding_labels(args.labels, fids)
    print(f"[s10] N={len(Xs)} holdings={int(y.sum())} ({100 * y.mean():.1f}%) "
          f"feat={names} T={Xs.shape[1]} se2_aug={use_se2} dev={dev}")

    def aug(xb):
        return se2_augment(xb, xi, yi) if use_se2 else xb

    Xt = torch.from_numpy(Xs.transpose(0, 2, 1).copy()).to(dev)                       # (N,C,T)
    yt = torch.from_numpy(y).to(dev)
    hi = torch.where(yt == 1)[0]
    ni = torch.where(yt == 0)[0]
    C, T = Xt.shape[1], Xt.shape[2]
    xi, yi = names.index("x"), names.index("y")
    nh = args.batch // 2

    clf = HoldingClassifier(C, T).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        clf.train()
        bi = torch.cat([hi[torch.randint(len(hi), (nh,), device=dev)],       # balanced: half holdings
                        ni[torch.randint(len(ni), (nh,), device=dev)]])
        xb = aug(Xt[bi])                                                     # bounded se2 iff model used it
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
                xv = aug(Xt[vb])
                yv = yt[vb]
                tv = torch.randint(0, tdiff, (xv.shape[0],), device=dev)
                xvt = sab[tv][:, None, None] * xv + s1mab[tv][:, None, None] * torch.randn_like(xv)
                pred = clf(xvt, tv).argmax(1)
                acc = (pred == yv).float().mean().item()
                rec = (pred[yv == 1] == 1).float().mean().item()
            print(f"[s10] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"balacc {acc:.3f} hold-recall {rec:.3f} ({time.time() - t0:.0f}s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": clf.state_dict(), "features": names, "channels": C, "t_len": T,
                "scaler": ck["scaler"], "ddpm_cfg": ck["ddpm_cfg"], "hold_class": 1,
                "arch": {"width": 96, "tdim": 128, "n_layers": 6}}, out / "clf.pt")
    print(f"[s10] saved {out}/clf.pt")


if __name__ == "__main__":
    main()
