"""Derived kinematic control signals (spec Section 1.4) and their re-integration.

From per-flight ``x, y`` (UTM m), ``altitude`` (ft) and ``timedelta`` (s) on the
uniform per-flight grid, derive by finite differences:

    gs (m/s), track chi (rad, unwrapped), turn rate chidot (rad/s),
    along-track acceleration a (m/s^2), vertical rate vz (m/s)

``gs`` and ``chi`` are Savitzky-Golay smoothed *before* differencing (ADS-B jitter
amplification); derived controls are clipped to sane envelopes and clip rates are
reported. ``integrate_controls`` is the inverse channel: forward-Euler integration
of (chidot, a, vz) from an entry state — the consistency requirement (mean position
error < 300 m at a 60-step horizon vs the source track) is enforced by
``tests/test_controls_consistency.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

FT_TO_M = 0.3048

CONTROL_NAMES = ["turn_rate", "along_accel", "vert_rate", "timedelta"]
ENTRY_NAMES = ["x0", "y0", "z0", "gs0", "chi0"]


def _per_flight_dt(td: np.ndarray) -> np.ndarray:
    """(N, T) timedelta -> (N,) grid step. The tilFAF grid is uniform per flight."""
    return (td[:, -1] - td[:, 0]) / (td.shape[1] - 1)


def derive_controls(x, y, alt_ft, td, cfg: dict) -> dict:
    """All inputs (N, T). Returns controls, entry states, smoothed gs/chi, clip rates.

    Robustness (motivated by ~1% of flights containing near-stationary
    interpolation artifacts): velocities come from Savitzky-Golay *derivative*
    filters (smooth-then-differentiate), and the heading is bridged by
    interpolation across spans where gs < ``gs_floor_ms`` — at near-zero speed
    arctan2 returns the angle of noise, whose fake ±180° flips otherwise unwrap
    into kilometres of drift after integration.
    """
    win = int(cfg.get("savgol_window", 5))
    order = int(cfg.get("savgol_order", 2))
    gs_floor = float(cfg.get("gs_floor_ms", 30.0))
    dt = _per_flight_dt(td)[:, None]                       # (N, 1)

    z = alt_ft * FT_TO_M
    vx = np.gradient(x, axis=1) / dt
    vy = np.gradient(y, axis=1) / dt
    gs = np.hypot(vx, vy)
    chi = np.unwrap(np.arctan2(vx, vy), axis=1)            # 0 = north, clockwise positive

    # Artifact flag (spec 1.3: gaps must be discarded, not interpolated): a real
    # approach never flies < ~gs_floor m/s, so near-stationary samples mean the
    # source data was gap-interpolated. Heading there is the angle of noise and
    # no clipped-control sequence can reproduce the fake speed cliff -> these
    # flights are unreconstructable-by-design and are flagged, not "fixed".
    valid_mask = gs.min(axis=1) >= gs_floor                # (N,)

    # smooth BEFORE differencing (jitter amplification)
    gs_s = savgol_filter(gs, win, order, axis=1)
    chi_s = savgol_filter(chi, win, order, axis=1)
    z_s = savgol_filter(z, win, order, axis=1)

    chidot = np.gradient(chi_s, axis=1) / dt
    accel = np.gradient(gs_s, axis=1) / dt
    vz = np.gradient(z_s, axis=1) / dt

    lim_cd = np.radians(float(cfg.get("clip_turn_rate_deg_s", 3.5)))
    lim_a = float(cfg.get("clip_accel_ms2", 1.5))
    vz_lo, vz_hi = [float(v) for v in cfg.get("clip_vz_ms", [-15.0, 8.0])]
    clip_rates = {
        "turn_rate": float((np.abs(chidot) > lim_cd).mean()),
        "along_accel": float((np.abs(accel) > lim_a).mean()),
        "vert_rate": float(((vz < vz_lo) | (vz > vz_hi)).mean()),
    }
    chidot = np.clip(chidot, -lim_cd, lim_cd)
    accel = np.clip(accel, -lim_a, lim_a)
    vz = np.clip(vz, vz_lo, vz_hi)

    controls = np.stack([chidot, accel, vz, td], axis=-1).astype(np.float32)   # (N, T, 4)
    entry = np.stack([x[:, 0], y[:, 0], z_s[:, 0], gs_s[:, 0], chi_s[:, 0]], axis=-1)
    return {
        "controls": controls,
        "entry": entry.astype(np.float32),                 # (N, 5)
        "gs": gs_s, "chi": chi_s, "z": z_s,
        "clip_rates": clip_rates,
        "valid_mask": valid_mask,                          # (N,) False = gap-interpolation artifact
        "dt": dt[:, 0],
    }


def integrate_controls(entry: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Forward-Euler re-integration. entry (N, 5) raw units, controls (N, T, 4)
    with channels CONTROL_NAMES. Returns (N, T, 3) x, y, z in metres."""
    chidot, accel, vz, td = (controls[..., j].astype(np.float64) for j in range(4))
    dt = np.diff(td, axis=1)
    dt = np.clip(dt, 0.0, None)

    def cumint(rate, init):
        # trapezoidal: state_t = init + sum 0.5*(rate_s + rate_{s+1}) * dt_s
        inc = 0.5 * (rate[:, :-1] + rate[:, 1:]) * dt
        return np.concatenate([init[:, None], init[:, None] + np.cumsum(inc, axis=1)], axis=1)

    chi = cumint(chidot, entry[:, 4].astype(np.float64))
    gs = np.clip(cumint(accel, entry[:, 3].astype(np.float64)), 0.0, None)
    z = cumint(vz, entry[:, 2].astype(np.float64))
    x = cumint(gs * np.sin(chi), entry[:, 0].astype(np.float64))
    y = cumint(gs * np.cos(chi), entry[:, 1].astype(np.float64))
    return np.stack([x, y, z], axis=-1)


def consistency_errors(x, y, alt_ft, td, cfg: dict, horizon: int = 60) -> dict:
    """Derive -> re-integrate -> horizontal error vs source track (metres).

    Scored on valid flights only; gap-interpolation artifacts (see
    ``valid_mask`` in :func:`derive_controls`) are counted, not scored.
    """
    d = derive_controls(x, y, alt_ft, td, cfg)
    ok = d["valid_mask"]
    xyz = integrate_controls(d["entry"][ok], d["controls"][ok])
    err = np.hypot(xyz[:, :, 0] - x[ok], xyz[:, :, 1] - y[ok])
    h = min(horizon, err.shape[1])
    return {
        "mean_m_at_horizon": float(err[:, :h].mean()),
        "final_m_at_horizon": float(err[:, h - 1].mean()),
        "mean_m_full": float(err.mean()),
        "p99_mean_m_full": float(np.percentile(err.mean(1), 99)),
        "max_mean_m_full": float(err.mean(1).max()),
        "artifact_fraction": float(1.0 - ok.mean()),
        "clip_rates": d["clip_rates"],
    }
