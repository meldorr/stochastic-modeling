"""End-to-end training: prepare -> fit fPCA -> train latent DDPM -> checkpoint.

    python train.py --config configs/config.yaml [--epochs N] [--tag NAME] [--no-tb]

Run with a `traffic`-capable interpreter, e.g.
    /Users/meldor/Desktop/git/deep-traffic-generation-paper/.venv/bin/python train.py
(Only Stage 1 needs `traffic`; it is skipped automatically once processed.npz exists.)

TensorBoard:
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import load_dataset
from src.data.prepare import prepare
from src.ddpm import LatentDDPM, MLPDenoiser, TCNDenoiser, UNetMLPDenoiser
from src.ddpm.ddpm import EMA
from src.fpca import FPCA
from src.fpca.fpca import LatentScaler
from src.pipeline.checkpoint import save_checkpoint
from src.pipeline.utils import get_device, load_config, resolve, set_seed


def make_writer(cfg: dict, tag: str | None):
    """Create a TensorBoard SummaryWriter (or None if disabled/unavailable)."""
    if not cfg.get("logging", {}).get("enabled", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:  # pragma: no cover
        print(f"[tb] tensorboard unavailable ({e}); logging disabled")
        return None
    run = tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = resolve(cfg["logging"]["tensorboard_dir"]) / run
    print(
        f"[tb] logging to {logdir}  (tensorboard --logdir {resolve(cfg['logging']['tensorboard_dir'])})"
    )
    return SummaryWriter(log_dir=str(logdir))


def fit_fpca(data: dict, cfg: dict, writer=None) -> FPCA:
    fcfg = cfg["fpca"]
    X_tr = data["X_std"][data["train_idx"]]
    fpca = FPCA.fit(
        X_tr,
        data["feature_names"],
        explained_variance=fcfg["explained_variance"],
        max_components=int(fcfg["max_components"]),
        basis=fcfg.get("basis", "discrete"),
        bspline_cfg=fcfg.get("bspline"),
    )
    ev = fpca.total_explained_variance()
    rmse = fpca.reconstruction_error(data["X_std"][data["val_idx"]])
    print(
        "[fpca] components/feature:",
        dict(zip(fpca.feature_names, fpca.ks)),
        f"-> m={fpca.m}",
    )
    print("[fpca] explained variance:", {k: round(v, 4) for k, v in ev.items()})
    print(
        "[fpca] val recon RMSE (std units):",
        {n: round(float(r), 4) for n, r in zip(fpca.feature_names, rmse)},
    )
    if writer:
        writer.add_scalar("fpca/latent_dim_m", fpca.m, 0)
        for name, b in zip(fpca.feature_names, fpca.bases):
            writer.add_scalar(
                f"fpca/explained_variance/{name}", float(b["evr"].sum()), 0
            )
            writer.add_scalar(f"fpca/n_components/{name}", int(b["k"]), 0)
        for name, r in zip(fpca.feature_names, rmse):
            writer.add_scalar(f"fpca/val_recon_rmse/{name}", float(r), 0)
    return fpca


def train_ddpm(W_tr, W_val, m, cfg, device, writer=None, denoiser_type="mlp"):
    tcfg = cfg["training"]
    if denoiser_type == "tcn":
        dcfg = cfg["tcn_denoiser"]
        denoiser_cfg = {
            "m": m,
            "channels": int(dcfg["channels"]),
            "n_blocks": int(dcfg["n_blocks"]),
            "kernel_size": int(dcfg["kernel_size"]),
            "dilations": list(dcfg["dilations"]),
            "time_dim": int(dcfg["time_dim"]),
            "dropout": float(dcfg["dropout"]),
        }
        ddpm = LatentDDPM(TCNDenoiser(**denoiser_cfg), cfg["ddpm"]).to(device)
    elif denoiser_type == "unet":
        dcfg = cfg["unet_denoiser"]
        denoiser_cfg = {
            "m": m,
            "hidden_dim": int(dcfg["hidden_dim"]),
            "depth": int(dcfg["depth"]),
            "channel_mult": tuple(int(x) for x in dcfg["channel_mult"]),
            "blocks_per_level": int(dcfg["blocks_per_level"]),
            "time_dim": int(dcfg["time_dim"]),
            "dropout": float(dcfg["dropout"]),
        }
        ddpm = LatentDDPM(UNetMLPDenoiser(**denoiser_cfg), cfg["ddpm"]).to(device)
    else:
        dcfg = cfg["denoiser"]
        denoiser_cfg = {
            "m": m,
            "hidden_dim": int(dcfg["hidden_dim"]),
            "n_blocks": int(dcfg["n_blocks"]),
            "time_dim": int(dcfg["time_dim"]),
            "dropout": float(dcfg["dropout"]),
        }
        ddpm = LatentDDPM(MLPDenoiser(**denoiser_cfg), cfg["ddpm"]).to(device)
    n_params = sum(p.numel() for p in ddpm.parameters())
    print(
        f"[ddpm] latent m={m}, denoiser={denoiser_type}, params={n_params / 1e3:.1f}k, device={device}"
    )

    opt = torch.optim.AdamW(
        ddpm.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    ema = EMA(ddpm, decay=float(tcfg["ema_decay"]))
    _warmup_epochs = int(tcfg.get("warmup_epochs", 100))
    _lr_init = float(tcfg.get("lr_init", 1e-5))
    _lr_min = float(tcfg.get("lr_min", 1e-6))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=_lr_init / float(tcfg["lr"]),
                end_factor=1.0,
                total_iters=_warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=int(tcfg["epochs"]) - _warmup_epochs,
                eta_min=_lr_min,
            ),
        ],
        milestones=[_warmup_epochs],
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(W_tr)),
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        drop_last=True,
    )
    W_val_t = torch.from_numpy(W_val).to(device)

    epochs, clip = int(tcfg["epochs"]), float(tcfg["grad_clip"])
    val_every, log_every = (
        int(tcfg["val_every"]),
        int(cfg.get("logging", {}).get("log_every", 25)),
    )
    step, t0 = 0, time.time()
    for ep in range(1, epochs + 1):
        ddpm.train()
        running = 0.0
        for (wb,) in loader:
            wb = wb.to(device)
            loss = ddpm(wb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), clip)
            opt.step()
            ema.update(ddpm)
            running += loss.item() * wb.size(0)
            step += 1
            if writer and step % log_every == 0:
                writer.add_scalar("train/loss_step", loss.item(), step)
        train_loss = running / len(loader.dataset)
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, ep)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], ep)
        if ep % val_every == 0 or ep == 1 or ep == epochs:
            ddpm.eval()
            with torch.no_grad():
                vloss = torch.stack([ddpm(W_val_t) for _ in range(8)]).mean().item()
            if writer:
                writer.add_scalar("val/loss", vloss, ep)
            print(
                f"[ddpm] epoch {ep:4d}/{epochs}  train {train_loss:.4f}  val {vloss:.4f}"
                f"  ({time.time() - t0:.0f}s)"
            )
        scheduler.step()

    # end-of-run: compare a sampled latent batch to the real one, per dim
    if writer:
        ddpm.eval()
        clamp = cfg["ddpm"].get("sample_clamp")
        with torch.no_grad():
            samp = (
                ddpm.sample(min(2000, len(W_tr)), m, device=device, clamp=clamp)
                .cpu()
                .numpy()
            )
        for j in range(min(m, 8)):
            writer.add_histogram(f"latent/real/dim_{j}", W_tr[:, j], 0)
            writer.add_histogram(f"latent/gen/dim_{j}", samp[:, j], 0)
    return ddpm, ema, denoiser_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the fPCA + latent DDPM pipeline")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    ap.add_argument(
        "--tag", default=None, help="TensorBoard run name (default: timestamp)"
    )
    ap.add_argument("--no-tb", action="store_true", help="disable TensorBoard logging")
    ap.add_argument(
        "--denoiser",
        choices=["mlp", "tcn", "unet"],
        default="mlp",
        help="denoiser architecture (default: mlp)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_tb:
        cfg.setdefault("logging", {})["enabled"] = False
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    writer = make_writer(cfg, args.tag)

    if not resolve(cfg["paths"]["processed"]).exists():
        print("[stage1] processed.npz missing -> running prepare (needs traffic lib)")
        prepare(cfg)
    data = load_dataset(cfg)
    print(
        f"[data] {len(data['X'])} flights, split "
        f"{len(data['train_idx'])}/{len(data['val_idx'])} train/val"
    )

    fpca = fit_fpca(data, cfg, writer)

    W_all = fpca.encode(data["X_std"])
    latent_scaler = LatentScaler.fit(W_all[data["train_idx"]])
    W_tr = latent_scaler.transform(W_all[data["train_idx"]])
    W_val = latent_scaler.transform(W_all[data["val_idx"]])

    if writer:
        writer.add_text("config", "```json\n" + json.dumps(cfg, indent=2) + "\n```", 0)

    ddpm, ema, denoiser_cfg = train_ddpm(
        W_tr, W_val, fpca.m, cfg, device, writer, args.denoiser
    )

    out = save_checkpoint(
        cfg["paths"]["checkpoint"],
        config=cfg,
        fpca=fpca,
        feature_scaler=data["scaler"],
        latent_scaler=latent_scaler,
        bounds=data["bounds"],
        denoiser_cfg=denoiser_cfg,
        model_state=ema.state_dict(),
    )
    if writer:
        writer.close()
    print(f"[done] checkpoint -> {out}")


if __name__ == "__main__":
    main()
