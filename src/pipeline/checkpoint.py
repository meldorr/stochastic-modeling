"""Save/load the full pipeline bundle in one file.

Two checkpoint modes:
* ``single``       — one global fPCA + one DDPM (the original pipeline).
* ``per_cluster``  — shared feature scaler + bounds + clusterer, plus one
  ``{fPCA, latent scaler, DDPM}`` per cluster.

A checkpoint is self-contained: it carries the frozen fPCA basis/bases, the
scalers, the physical bounds, and the (EMA) DDPM weights — everything
``generate.py`` needs without re-fitting anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.dataset import FeatureScaler
from src.ddpm import LatentDDPM, MLPDenoiser, TCNDenoiser, UNetMLPDenoiser
from src.fpca import FPCA
from src.fpca.fpca import LatentScaler
from src.pipeline.utils import ensure_parent, get_device


def build_denoiser(dcfg: dict):
    """Reconstruct a denoiser from its saved config (type inferred from keys)."""
    if "channels" in dcfg:
        return TCNDenoiser(**dcfg)
    if "depth" in dcfg:
        return UNetMLPDenoiser(**dcfg)
    return MLPDenoiser(**dcfg)


# --- single-model ----------------------------------------------------------
def save_checkpoint(
    path, *, config, fpca: FPCA, feature_scaler: FeatureScaler,
    latent_scaler: LatentScaler, bounds: np.ndarray, denoiser_cfg: dict, model_state: dict,
) -> Path:
    out = ensure_parent(path)
    torch.save(
        {
            "mode": "single",
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


# --- per-cluster -----------------------------------------------------------
def save_per_cluster_checkpoint(
    path, *, config, feature_names, feature_scaler: FeatureScaler, bounds: np.ndarray,
    assigner, labels, flight_ids, frequencies: dict, models: dict,
) -> Path:
    """``models[c] = {fpca_state, latent_scaler_state, denoiser_cfg, m, model_state}``."""
    out = ensure_parent(path)
    torch.save(
        {
            "mode": "per_cluster",
            "config": config,
            "feature_names": list(feature_names),
            "feature_scaler": feature_scaler.to_dict(),
            "bounds": bounds,
            "ddpm_cfg": config["ddpm"],
            "cluster": {
                "assigner": assigner.state(),
                "labels": np.asarray(labels),
                "flight_ids": flight_ids,
                "frequencies": {int(c): float(f) for c, f in frequencies.items()},
                "clusters": sorted(int(c) for c in models),
            },
            "models": {
                int(c): {
                    "fpca": m["fpca_state"],
                    "latent_scaler": m["latent_scaler_state"],
                    "denoiser_cfg": m["denoiser_cfg"],
                    "m": int(m["m"]),
                    "model_state": m["model_state"],
                }
                for c, m in models.items()
            },
        },
        out,
    )
    return out


def _load_ddpm(denoiser_cfg, ddpm_cfg, state, device):
    ddpm = LatentDDPM(build_denoiser(denoiser_cfg), ddpm_cfg).to(device)
    ddpm.load_state_dict(state)
    ddpm.eval()
    return ddpm


def load_checkpoint(path, device=None) -> dict:
    """Rebuild every object from a checkpoint. Returns a dict ready for sampling.

    ``bundle["mode"]`` is ``"single"`` or ``"per_cluster"``.
    """
    device = device or get_device()
    # weights_only=False: bundle embeds numpy arrays / sklearn objects we control.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "single")

    if mode == "per_cluster":
        from src.cluster.assigner import ClusterAssigner

        cl = ckpt["cluster"]
        models = {}
        for c, ms in ckpt["models"].items():
            models[int(c)] = {
                "fpca": FPCA.from_state(ms["fpca"]),
                "latent_scaler": LatentScaler.from_state(ms["latent_scaler"]),
                "ddpm": _load_ddpm(ms["denoiser_cfg"], ckpt["ddpm_cfg"], ms["model_state"], device),
                "m": int(ms["m"]),
            }
        return {
            "mode": "per_cluster",
            "config": ckpt["config"],
            "feature_names": ckpt["feature_names"],
            "feature_scaler": FeatureScaler.from_dict(ckpt["feature_scaler"]),
            "bounds": np.asarray(ckpt["bounds"]),
            "assigner": ClusterAssigner.from_state(cl["assigner"]),
            "labels": np.asarray(cl["labels"]),
            "flight_ids": cl["flight_ids"],
            "frequencies": {int(c): float(f) for c, f in cl["frequencies"].items()},
            "clusters": [int(c) for c in cl["clusters"]],
            "models": models,
            "device": device,
        }

    ddpm = _load_ddpm(ckpt["denoiser_cfg"], ckpt["ddpm_cfg"], ckpt["model_state"], device)
    return {
        "mode": "single",
        "config": ckpt["config"],
        "feature_names": ckpt["feature_names"],
        "fpca": FPCA.from_state(ckpt["fpca"]),
        "feature_scaler": FeatureScaler.from_dict(ckpt["feature_scaler"]),
        "latent_scaler": LatentScaler.from_state(ckpt["latent_scaler"]),
        "bounds": np.asarray(ckpt["bounds"]),
        "ddpm": ddpm,
        "m": ckpt["m"],
        "device": device,
    }
