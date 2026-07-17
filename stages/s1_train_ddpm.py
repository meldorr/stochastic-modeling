"""Stage 1 — train a raw-space DDPM for one named experiment.

    python stages/s1_train_ddpm.py --exp configs/experiments/ddpm_tcn_unet_standardscaler__xy.yaml
    python stages/s1_train_ddpm.py --exp ... --epochs 2      # smoke override

Reads the experiment overlay on configs/base.yaml, trains on the selected
feature set, writes results/<experiment>/ckpt.pt (+ TensorBoard runs/<experiment>).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ddpm import LatentDDPM
from src.ddpm.ddpm import EMA
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.utils import experiment_dirs, get_device, load_experiment_config, set_seed
from stages.common import load_experiment_data, make_progress_figure, make_writer, reference_lr


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: train raw-space DDPM")
    ap.add_argument("--exp", required=True, help="experiment yaml (configs/experiments/*.yaml)")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    args = ap.parse_args()

    cfg = load_experiment_config(args.exp)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    dirs = experiment_dirs(cfg)
    writer = make_writer(cfg, dirs["runs"])

    data = load_experiment_data(cfg)
    Xs = data["X_std"]
    tr, va = data["train_idx"], data["val_idx"]
    X_tr = torch.from_numpy(Xs[tr].transpose(0, 2, 1).copy())        # (N, C, T)
    X_va = torch.from_numpy(Xs[va].transpose(0, 2, 1).copy()).to(device)
    C, T = X_tr.shape[1], X_tr.shape[2]

    arch = cfg.get("arch", "tcn_unet")
    dropout = float(cfg.get("dropout", 0.0))
    denoiser = build_raw_denoiser(arch, C, T, dropout=dropout)
    ddpm = LatentDDPM(denoiser, cfg["ddpm"]).to(device)
    n_params = sum(p.numel() for p in ddpm.parameters())
    print(f"[{dirs['name']}] arch={arch} dropout={dropout} scaler={cfg.get('scaler', 'standard')} "
          f"features={data['names']} shape=({C},{T}) params={n_params / 1e6:.2f}M device={device}")

    tcfg = cfg["training"]
    epochs = int(tcfg["epochs"])
    opt = torch.optim.AdamW(ddpm.parameters(), lr=reference_lr(0, epochs, tcfg),
                            weight_decay=float(tcfg["weight_decay"]))
    ema = EMA(ddpm, decay=float(tcfg["ema_decay"]))
    loader = DataLoader(TensorDataset(X_tr), batch_size=int(tcfg["batch_size"]),
                        shuffle=True, drop_last=True)
    clip, val_every = float(tcfg["grad_clip"]), int(tcfg["val_every"])

    t0 = time.time()
    for ep in range(1, epochs + 1):
        lr_now = reference_lr(ep, epochs, tcfg)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        ddpm.train()
        running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = ddpm(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), clip)
            opt.step()
            ema.update(ddpm)
            running += loss.item() * xb.size(0)
        train_loss = running / len(loader.dataset)
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, ep)
            writer.add_scalar("train/lr", lr_now, ep)
        if ep % val_every == 0 or ep == 1 or ep == epochs:
            ddpm.eval()
            with torch.no_grad():
                vloss = torch.stack([ddpm(X_va[:512]) for _ in range(4)]).mean().item()
            if writer:
                writer.add_scalar("val/loss", vloss, ep)
            print(f"[{dirs['name']}] epoch {ep:4d}/{epochs} train {train_loss:.4f} "
                  f"val {vloss:.4f} ({time.time() - t0:.0f}s)")

        # in-training sample snapshots (legacy-style 4-panel, scaled units)
        viz_every = int(cfg.get("viz", {}).get("every", 0))
        if viz_every and (ep % viz_every == 0 or ep == epochs):
            ddpm.eval()
            n_viz = int(cfg.get("viz", {}).get("n", 12))
            with torch.no_grad():
                samp = ddpm.sample(n_viz, shape=(C, T), device=device,
                                   clamp=cfg["ddpm"].get("sample_clamp"))
            gen_std = samp.cpu().numpy().transpose(0, 2, 1)
            fig = make_progress_figure(gen_std, data["names"], ep)
            viz_dir = dirs["results"] / "viz"
            viz_dir.mkdir(exist_ok=True)
            fig.savefig(viz_dir / f"epoch_{ep:05d}.png", dpi=110)
            if writer:
                writer.add_figure("samples/progress", fig, ep)
            import matplotlib.pyplot as plt

            plt.close(fig)

    ema.copy_to(ddpm)
    ddpm.eval()
    ckpt = dirs["results"] / "ckpt.pt"
    torch.save(
        {
            "experiment": dirs["name"],
            "arch": arch,
            "dropout": dropout,
            "features": data["names"],
            "feature_set": data["feature_set"],
            "channels": C,
            "t_len": T,
            "scaler": data["scaler"].to_dict(),
            "bounds": data["bounds"],
            "ddpm_cfg": cfg["ddpm"],
            "model_state": ddpm.state_dict(),
            "config": cfg,
        },
        ckpt,
    )
    (dirs["results"] / "train_info.json").write_text(json.dumps(
        {"epochs": epochs, "params_M": n_params / 1e6, "arch": arch,
         "features": data["names"], "wall_s": round(time.time() - t0, 1)}, indent=2))
    if writer:
        writer.close()
    print(f"[{dirs['name']}] checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
