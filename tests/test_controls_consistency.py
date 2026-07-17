"""Spec Section 1.4 consistency requirement (as a pytest):

Re-integrating the derived controls from each flight's entry state must
reproduce the interpolated track with mean position error < 300 m at a
60-step horizon. Run with:

    pytest tests/test_controls_consistency.py -q

Requires data/processed.npz (or STOCH_DATA_DIR pointing at the data folder).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.controls import consistency_errors, derive_controls, integrate_controls
from src.data.prepare import load_processed
from src.pipeline.utils import resolve

N_SAMPLE = 2000


@pytest.fixture(scope="module")
def flights():
    cfg = _base_cfg()
    if not resolve(cfg["paths"]["processed"]).exists():
        pytest.skip("data/processed.npz missing — run stage 0 / copy the cache first")
    d = load_processed(cfg)
    names = [str(f) for f in d["meta"]["feature_names"]]
    idx = [names.index(c) for c in ("x", "y", "altitude", "timedelta")]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(d["X"]), min(N_SAMPLE, len(d["X"])), replace=False)
    X = d["X"][sel]
    return tuple(X[:, :, j].astype(float) for j in idx), cfg["controls"]


def _base_cfg():
    import yaml

    from src.pipeline.utils import REPO_ROOT

    with open(REPO_ROOT / "configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    import os

    data_dir = os.environ.get("STOCH_DATA_DIR", cfg["paths"]["data_dir"])
    cfg["paths"]["processed"] = str(Path(data_dir) / "processed.npz")
    return cfg


def test_reintegration_error_under_300m_at_60_steps(flights):
    (x, y, alt, td), ccfg = flights
    horizon = int(ccfg.get("consistency_horizon", 60))
    max_mean = float(ccfg.get("consistency_max_mean_m", 300.0))
    r = consistency_errors(x, y, alt, td, ccfg, horizon=horizon)
    assert r["mean_m_at_horizon"] < max_mean, (
        f"mean position error {r['mean_m_at_horizon']:.0f} m at {horizon} steps "
        f"exceeds {max_mean:.0f} m — iterate on smoothing (clip rates: {r['clip_rates']})"
    )


def test_artifact_fraction_is_small(flights):
    (x, y, alt, td), ccfg = flights
    r = consistency_errors(x, y, alt, td, ccfg)
    assert r["artifact_fraction"] < 0.05, (
        f"{r['artifact_fraction']:.1%} flights flagged as gap-interpolation artifacts"
    )


def test_clip_rates_are_low(flights):
    (x, y, alt, td), ccfg = flights
    d = derive_controls(x, y, alt, td, ccfg)
    for name, rate in d["clip_rates"].items():
        assert rate < 0.05, f"{name} clip rate {rate:.3f} > 5% — envelope or smoothing wrong"


def test_integration_shapes_and_finiteness(flights):
    (x, y, alt, td), ccfg = flights
    d = derive_controls(x[:64], y[:64], alt[:64], td[:64], ccfg)
    xyz = integrate_controls(d["entry"], d["controls"])
    assert xyz.shape == (64, x.shape[1], 3)
    assert np.isfinite(xyz).all()
