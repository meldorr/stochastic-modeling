"""Stage 11 — classifier-guided generation of holding patterns.

Loads the frozen se2aug DDPM + the noise-conditioned holding classifier and runs guided
reverse diffusion:

    x_{t-1} ~ N( mu + s * var * grad_x log p(holding | x_t, t),  var )

Reports the fraction the classifier calls "holding" and writes an overlay of the guided
samples against real holdings.

    python stages/s11_holding_guided_generate.py --n 50 --guidance-scale 5.0
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
import torch.nn.functional as F
from matplotlib.collections import LineCollection

from src.data.dataset import scaler_from_dict
from src.ddpm import LatentDDPM
from src.ddpm.classifier import load_classifier
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.utils import get_device


def load_ddpm(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    den = build_raw_denoiser(ck["arch"], ck["channels"], ck["t_len"],
                             dropout=float(ck.get("dropout", 0.0)),
                             base_channels=int(ck.get("base_channels", 64)))
    ddpm = LatentDDPM(den, ck["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ck["model_state"])
    ddpm.eval()
    return ddpm, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ddpm_tcn_unet_se2aug__xyt/ckpt_best.pt")
    ap.add_argument("--clf", default="results/holding_clf/clf.pt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--data", default="data/processed.npz")
    ap.add_argument("--labels", default="data/holding/holding_labels.json")
    ap.add_argument("--out", default="results/holding_guided")
    args = ap.parse_args()

    dev = get_device("auto")
    ddpm, ck = load_ddpm(args.ckpt, dev)
    clf, cck = load_classifier(args.clf, dev)
    scaler = scaler_from_dict(ck["scaler"])
    names, C, T = ck["features"], ck["channels"], ck["t_len"]
    hold = int(cck.get("hold_class", 1))
    clamp = ck["ddpm_cfg"].get("sample_clamp")
    xi, yi = names.index("x"), names.index("y")
    print(f"[s11] ddpm epoch={ck.get('epoch')} n={args.n} scale={args.guidance_scale} hold_class={hold}")

    def gather(buf, t, x):
        return buf[t].view(-1, *([1] * (x.dim() - 1)))

    x = torch.randn(args.n, C, T, device=dev)
    for ti in reversed(range(ddpm.timesteps)):
        t = torch.full((args.n,), ti, device=dev, dtype=torch.long)
        with torch.no_grad():
            eps = ddpm.denoiser(x, t)
            betas_t = gather(ddpm.betas, t, x)
            alphas_t = gather(ddpm.alphas, t, x)
            abar_t = gather(ddpm.alphas_bar, t, x)
            mean = (1.0 / torch.sqrt(alphas_t + 1e-8)) * (
                x - betas_t / (torch.sqrt(1.0 - abar_t) + 1e-8) * eps)
            var = gather(ddpm.posterior_variance, t, x)
        xg = x.detach().requires_grad_(True)                       # classifier-guidance gradient
        logp = F.log_softmax(clf(xg, t), dim=1)[:, hold].sum()
        grad, = torch.autograd.grad(logp, xg)
        with torch.no_grad():
            mean = mean + args.guidance_scale * var * grad         # shift by s * Sigma * grad log p(y|x_t)
            x = mean + torch.sqrt(var) * torch.randn_like(x) if ti > 0 else mean
            if clamp is not None:
                x = x.clamp(-clamp, clamp)

    with torch.no_grad():                                          # how "holding" do the samples read?
        frac = F.softmax(clf(x, torch.zeros(args.n, device=dev, dtype=torch.long)), 1)[:, hold].mean().item()
    feats = scaler.inverse_transform(x.detach().cpu().numpy().transpose(0, 2, 1))
    gxy = feats[:, :, [xi, yi]].astype(float)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "generated.npz", feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object))
    (out / "metrics.json").write_text(json.dumps(
        {"n": args.n, "guidance_scale": args.guidance_scale,
         "mean_holding_prob": frac, "ddpm_epoch": ck.get("epoch")}, indent=2))
    print(f"[s11] mean holding-prob of guided samples = {frac:.3f}")

    d = np.load(args.data, allow_pickle=True)                      # overlay vs real holdings
    rn = [str(s) for s in d["feature_names"]]
    realxy = d["X"][:, :, [rn.index("x"), rn.index("y")]].astype(float)
    lab = json.loads(Path(args.labels).read_text())
    holdset = set(lab.keys()) if isinstance(lab, dict) else set(map(str, lab))
    fids = np.array([str(s) for s in d["flight_ids"]])
    hidx = np.where(np.array([f in holdset for f in fids]))[0]
    fig, ax = plt.subplots(figsize=(9, 9))
    for i in np.random.default_rng(0).choice(hidx, min(300, len(hidx)), replace=False):
        ax.plot(realxy[i, :, 0] / 1e3, realxy[i, :, 1] / 1e3, color="#bbbbbb", lw=0.3, alpha=0.4)
    ax.add_collection(LineCollection([g / 1e3 for g in gxy], colors="#d62728", linewidths=0.7, alpha=0.55))
    ax.autoscale()
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Classifier-guided holdings (red) vs real holdings (grey)\n"
                 f"n={args.n} scale={args.guidance_scale} mean holding-prob {frac:.2f}")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=130)
    plt.close(fig)
    print(f"[s11] wrote {out}/{{generated.npz, overlay.png, metrics.json}}")


if __name__ == "__main__":
    main()
