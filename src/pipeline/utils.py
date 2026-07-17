"""Config loading, path resolution, seeding, and device selection."""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# numpy>=2 can emit spurious FP flags from the matmul SIMD path (incl. inside
# sklearn) even when results are finite. Silence just that class, project-wide.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

# Repo root = parent of the `src` package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load the YAML config, resolving it relative to the repo root if needed."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge; overlay wins. Returns a new dict."""
    out = dict(base)
    for k, v in overlay.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment_config(exp_path: str | Path, base_path: str | Path = "configs/base.yaml") -> dict:
    """Load base.yaml, overlay an experiment yaml, and resolve data paths.

    The experiment name defaults to the yaml filename stem. The data folder is a
    single knob: ``paths.data_dir`` (overridable via env ``STOCH_DATA_DIR``);
    ``paths.processed`` is derived from it unless set explicitly.
    """
    base = load_config(base_path)
    with open(resolve(exp_path)) as fh:
        overlay = yaml.safe_load(fh) or {}
    cfg = deep_merge(base, overlay)
    cfg.setdefault("experiment", Path(exp_path).stem)

    data_dir = os.environ.get("STOCH_DATA_DIR", cfg["paths"]["data_dir"])
    cfg["paths"]["data_dir"] = data_dir
    cfg["paths"].setdefault("processed", str(Path(data_dir) / "processed.npz"))
    return cfg


def experiment_dirs(cfg: dict) -> dict[str, Path]:
    """Per-experiment output folders: results/<exp> and runs/<exp>."""
    name = cfg["experiment"]
    res = resolve(cfg["paths"]["results_dir"]) / name
    runs = resolve(cfg["paths"]["runs_dir"]) / name
    res.mkdir(parents=True, exist_ok=True)
    return {"results": res, "runs": runs, "name": name}


def ensure_parent(path: str | Path) -> Path:
    """Make sure the parent directory of ``path`` exists; return the resolved path."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device. ``auto`` prefers MPS (Apple), then CUDA, then CPU."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
