"""Save/load the full pipeline bundle in one file.

A checkpoint is self-contained: it carries the frozen fPCA basis, the feature and
latent scalers, the physical bounds, and the (EMA) DDPM weights — everything
``generate.py`` needs without re-fitting anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.dataset import FeatureScaler
from src.ddpm import LatentDDPM, MLPDenoiser
from src.fpca import FPCA
from src.fpca.fpca import LatentScaler
from src.pipeline.utils import ensure_parent, get_device


def save_checkpoint(path, *, config, fpca: FPCA, feature_scaler: FeatureScaler,
                    latent_scaler: LatentScaler, bounds: np.ndarray,
                    denoiser_cfg: dict, model_state: dict) -> Path:
    out = ensure_parent(path)
    torch.save(
        {
            "config": config,
            "feature_names": fpca.feature_names,
            "fpca": fpca.state(),
            "feature_scaler": feature_scaler.to_dict(),
            "latent_scaler": latent_scaler.state(),
            "bounds": bounds,
            "denoiser_cfg": denoiser_cfg,
            "ddpm_cfg": config["ddpm"],
            "m": fpca.m,
            "model_state": model_state,
        },
        out,
    )
    return out


def load_checkpoint(path, device=None) -> dict:
    """Rebuild every object from a checkpoint. Returns a dict ready for sampling."""
    device = device or get_device()
    # weights_only=False: bundle embeds numpy arrays / plain dicts we control.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    fpca = FPCA.from_state(ckpt["fpca"])
    denoiser = MLPDenoiser(**ckpt["denoiser_cfg"])
    ddpm = LatentDDPM(denoiser, ckpt["ddpm_cfg"]).to(device)
    ddpm.load_state_dict(ckpt["model_state"])
    ddpm.eval()

    return {
        "config": ckpt["config"],
        "feature_names": ckpt["feature_names"],
        "fpca": fpca,
        "feature_scaler": FeatureScaler.from_dict(ckpt["feature_scaler"]),
        "latent_scaler": LatentScaler.from_state(ckpt["latent_scaler"]),
        "bounds": np.asarray(ckpt["bounds"]),
        "ddpm": ddpm,
        "m": ckpt["m"],
        "device": device,
    }
