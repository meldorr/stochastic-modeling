"""E4 — Raw trajectory-space diffusion (no fPCA).

    python experiments/e4_raw.py --features xy  [--epochs 300] [--n-gen 1000]
    python experiments/e4_raw.py --features dyn [--epochs 300] [--n-gen 1000]

A TCN denoiser diffuses the standardized trajectory tensor (C, 200) directly —
the end-of-ablation baseline against the fPCA-latent pipelines (E2/E3).
Writes results/e4_raw_{xy,dyn}/{generated.npz, gen_metrics.json, figures, ckpt}.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import DYN, XY, load_features, out_dir, save_json
from src.ddpm import LatentDDPM, TCNUNetDenoiser, TrajTCNDenoiser
from src.ddpm.ddpm import EMA
from src.pipeline.reconstruct import physics_repair
from src.pipeline.utils import get_device, load_config, set_seed


def reference_lr(epoch, total, lr_start=5e-5, lr_peak=5e-4, lr_end=1e-6):
    """LR schedule from diffusion-models-lab/train.py: 10% linear warmup,
    then linear decay to lr_end."""
    warmup = max(1, total // 10)
    if epoch <= warmup:
        return lr_start + (lr_peak - lr_start) * (epoch / warmup)
    u = (epoch - warmup) / max(1, total - warmup)
    return lr_peak + (lr_end - lr_peak) * u


def make_writer(cfg, tag):
    if not cfg.get("logging", {}).get("enabled", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        from src.pipeline.utils import resolve

        return SummaryWriter(log_dir=str(resolve(cfg["logging"]["tensorboard_dir"]) / tag))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="E4: raw-space diffusion baseline")
    ap.add_argument("--features", choices=["xy", "dyn"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--arch", choices=["unet", "tcn"], default="unet",
                    help="unet = reference TCN U-Net (diffusion-models-lab); tcn = small flat TCN")
    ap.add_argument("--epochs", type=int, default=2000)  # converged budget (reference)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-gen", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    names = XY if args.features == "xy" else DYN
    exp = f"e4_raw_{args.features}"
    out = out_dir(cfg, exp)
    writer = make_writer(cfg, exp)

    data = load_features(cfg, names)
    Xs = data["X_std"]                      # (N, T, C)
    tr, va = data["train_idx"], data["val_idx"]
    X_tr = torch.from_numpy(Xs[tr].transpose(0, 2, 1).copy())   # (N, C, T)
    X_va = torch.from_numpy(Xs[va].transpose(0, 2, 1).copy()).to(device)
    C, T = X_tr.shape[1], X_tr.shape[2]

    # reference setup: TCN U-Net + linear beta schedule (diffusion-models-lab)
    if args.arch == "unet":
        denoiser = TCNUNetDenoiser(c=C, t_len=T)
        ddpm_cfg = dict(cfg["ddpm"], schedule="linear")
    else:
        denoiser = TrajTCNDenoiser(channels_in=C)
        ddpm_cfg = dict(cfg["ddpm"])
    ddpm = LatentDDPM(denoiser, ddpm_cfg).to(device)
    n_params = sum(p.numel() for p in ddpm.parameters())
    print(f"[{exp}] arch={args.arch} features={names} shape=({C},{T}) "
          f"params={n_params / 1e6:.2f}M schedule={ddpm_cfg['schedule']} device={device}")

    opt = torch.optim.AdamW(ddpm.parameters(), lr=reference_lr(0, args.epochs), weight_decay=1e-4)
    ema = EMA(ddpm, decay=0.999)
    loader = DataLoader(TensorDataset(X_tr), batch_size=args.batch, shuffle=True, drop_last=True)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        lr_now = reference_lr(ep, args.epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        ddpm.train()
        running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = ddpm(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            ema.update(ddpm)
            running += loss.item() * xb.size(0)
        train_loss = running / len(loader.dataset)
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, ep)
        if ep % 25 == 0 or ep == 1 or ep == args.epochs:
            ddpm.eval()
            with torch.no_grad():
                vloss = torch.stack([ddpm(X_va[:512]) for _ in range(4)]).mean().item()
            if writer:
                writer.add_scalar("val/loss", vloss, ep)
            print(f"[{exp}] epoch {ep:4d}/{args.epochs} train {train_loss:.4f} "
                  f"val {vloss:.4f} ({time.time() - t0:.0f}s)")

    # EMA weights for sampling
    ema.copy_to(ddpm)
    ddpm.eval()
    torch.save({"model_state": ddpm.state_dict(), "channels": C, "t_len": T,
                "features": names, "arch": args.arch, "ddpm_cfg": ddpm_cfg}, out / "ckpt.pt")

    # ---- sample ----
    clamp = cfg["ddpm"].get("sample_clamp")
    chunks = []
    with torch.no_grad():
        for start in range(0, args.n_gen, 250):
            k = min(250, args.n_gen - start)
            xs = ddpm.sample(k, shape=(C, T), device=device, clamp=clamp)
            chunks.append(xs.cpu().numpy())
    gen_std = np.concatenate(chunks).transpose(0, 2, 1)         # (n, T, C)
    feats = data["scaler"].inverse_transform(gen_std)
    feats = physics_repair(feats, data["bounds"], names)
    np.savez_compressed(out / "generated.npz", feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object))
    print(f"[{exp}] sampled {len(feats)} trajectories -> {out / 'generated.npz'}")

    # quick profile figure (full metrics come from eval_gen.py)
    real = data["X"][va]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    t = np.arange(T)
    for ax, j, nm in zip(np.atleast_1d(axes), range(len(names)), names):
        for arr, c in ((real, "#1f77b4"), (feats, "#d62728")):
            mu, sd = arr[:, :, j].mean(0), arr[:, :, j].std(0)
            ax.plot(t, mu, color=c, lw=2)
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18)
        ax.set_title(nm)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{exp}: real (blue) vs generated (red)")
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=130)
    plt.close(fig)
    if writer:
        writer.close()
    save_json(out / "train_info.json", {"epochs": args.epochs, "params_k": n_params / 1e3,
                                        "arch": args.arch, "features": names,
                                        "n_gen": int(len(feats))})


if __name__ == "__main__":
    main()
