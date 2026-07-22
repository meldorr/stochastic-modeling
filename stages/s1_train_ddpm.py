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
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.augment import se2_augment
from src.data.reckoning import generated_to_xy
from src.ddpm import LatentDDPM
from src.ddpm.ddpm import EMA
from src.ddpm.registry import build_raw_denoiser
from src.pipeline.utils import experiment_dirs, get_device, load_experiment_config, set_seed
from stages.common import (
    load_experiment_data,
    make_progress_figure,
    make_writer,
    reference_lr,
    repair_controls,
)


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
    base_channels = int(cfg.get("base_channels", 64))
    denoiser = build_raw_denoiser(arch, C, T, dropout=dropout, base_channels=base_channels)
    ddpm = LatentDDPM(denoiser, cfg["ddpm"]).to(device)
    n_params = sum(p.numel() for p in ddpm.parameters())

    # on-the-fly SE(2) augmentation (rigid -> flyable); applied in normalized space
    augment = cfg.get("augment")
    x_idx = data["names"].index("x") if "x" in data["names"] else 0
    y_idx = data["names"].index("y") if "y" in data["names"] else 1

    def maybe_aug(batch):
        # returns (batch, mask); mask is None unless free SE(2) padding is on
        if augment in ("se2", "se2_free"):
            return se2_augment(batch, x_idx, y_idx, free=(augment == "se2_free"))
        return batch, None

    print(f"[{dirs['name']}] arch={arch} base_ch={base_channels} dropout={dropout} "
          f"scaler={cfg.get('scaler', 'standard')} augment={augment} "
          f"features={data['names']} shape=({C},{T}) params={n_params / 1e6:.2f}M device={device}")

    tcfg = cfg["training"]
    epochs = int(tcfg["epochs"])
    opt = torch.optim.AdamW(ddpm.parameters(), lr=reference_lr(0, epochs, tcfg),
                            weight_decay=float(tcfg["weight_decay"]))
    ema = EMA(ddpm, decay=float(tcfg["ema_decay"]))
    loader = DataLoader(TensorDataset(X_tr), batch_size=int(tcfg["batch_size"]),
                        shuffle=True, drop_last=True)
    clip, val_every = float(tcfg["grad_clip"]), int(tcfg["val_every"])

    def save_ckpt(path, state, epoch, val):
        torch.save(
            {
                "experiment": dirs["name"], "arch": arch, "dropout": dropout,
                "base_channels": base_channels, "augment": augment,
                "features": data["names"], "feature_set": data["feature_set"],
                "channels": C, "t_len": T, "scaler": data["scaler"].to_dict(),
                "bounds": data["bounds"], "ddpm_cfg": cfg["ddpm"], "config": cfg,
                "model_state": state, "epoch": epoch, "val_loss": val, "ema": True,
            },
            path,
        )

    best_val = float("inf")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        lr_now = reference_lr(ep, epochs, tcfg)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        ddpm.train()
        running = 0.0
        for (xb,) in loader:
            xb, mask = maybe_aug(xb.to(device))
            loss = ddpm(xb, mask)
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
            # evaluate the EMA weights (what we sample from), and pick best on them
            raw_state = {k: v.detach().clone() for k, v in ddpm.state_dict().items()}
            ema.copy_to(ddpm)
            ddpm.eval()
            with torch.no_grad():
                # augmented val: same transform distribution as training
                vb_all = [maybe_aug(X_va[:512]) for _ in range(4)]
                vloss = torch.stack([ddpm(vb, vm) for vb, vm in vb_all]).mean().item()
            if vloss < best_val:
                best_val = vloss
                save_ckpt(dirs["results"] / "ckpt_best.pt", ema.state_dict(), ep, vloss)
            ddpm.load_state_dict(raw_state)          # restore raw weights for training
            if writer:
                writer.add_scalar("val/loss", vloss, ep)
                writer.add_scalar("val/best_loss", best_val, ep)
            print(f"[{dirs['name']}] epoch {ep:4d}/{epochs} train {train_loss:.4f} "
                  f"val {vloss:.4f} (best {best_val:.4f}) ({time.time() - t0:.0f}s)")

        # in-training sample snapshots: sample, then FAF-reckon back to absolute
        # x/y so gstrack/controls progress is read as real ground tracks
        viz_every = int(cfg.get("viz", {}).get("every", 0))
        if viz_every and (ep % viz_every == 0 or ep == epochs):
            ddpm.eval()
            n_viz = int(cfg.get("viz", {}).get("n", 12))
            with torch.no_grad():
                samp = ddpm.sample(n_viz, shape=(C, T), device=device,
                                   clamp=cfg["ddpm"].get("sample_clamp"))
            gen_std = samp.cpu().numpy().transpose(0, 2, 1)
            feats_raw = data["scaler"].inverse_transform(gen_std)
            if data["feature_set"] == "controls":
                feats_raw = repair_controls(feats_raw, cfg["controls"])
            xy_viz = generated_to_xy(feats_raw, data["feature_set"], data["names"],
                                     data["aux"], np.random.default_rng(ep))
            fig = make_progress_figure(gen_std, data["names"], ep, xy=xy_viz)
            viz_dir = dirs["results"] / "viz"
            viz_dir.mkdir(exist_ok=True)
            fig.savefig(viz_dir / f"epoch_{ep:05d}.png", dpi=110)
            if writer:
                writer.add_figure("samples/progress", fig, ep)
            import matplotlib.pyplot as plt

            plt.close(fig)

    # final (last) checkpoint = EMA weights at end of training
    save_ckpt(dirs["results"] / "ckpt_last.pt", ema.state_dict(), epochs, float("nan"))
    # keep ckpt.pt as an alias of last for backward compat
    save_ckpt(dirs["results"] / "ckpt.pt", ema.state_dict(), epochs, float("nan"))
    (dirs["results"] / "train_info.json").write_text(json.dumps(
        {"epochs": epochs, "params_M": n_params / 1e6, "arch": arch,
         "features": data["names"], "best_val": round(best_val, 5),
         "wall_s": round(time.time() - t0, 1)}, indent=2))
    if writer:
        writer.close()
    print(f"[{dirs['name']}] saved ckpt_best.pt (val {best_val:.4f}) + ckpt_last.pt "
          f"-> {dirs['results']}")


if __name__ == "__main__":
    main()
