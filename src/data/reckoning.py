"""Canonical dead-reckoning — one implementation for every experiment.

Mirrors the reference repo (`deep-traffic-generation-paper/dtg/traffic_builder.py`,
``_walk_latlon(forward=False)``): anchor at the **FAF** (the fixed, known
destination) and integrate **backward** in time, so accumulated error lands at
the uncertain **entry** point — which is exactly where we measure drift.

Differences from the reference that are deliberate and documented:
  * we work in the projected UTM x/y metric plane (our feature space), not WGS84
    lat/lon geodesic — over TMA distances the two agree to <0.1%, and our data
    is already in x/y;
  * otherwise identical: Euler step, per-step distance
    ``gs_factor * gs[kts->m/s] * dt`` using the *start-of-interval* groundspeed,
    the 180 deg bearing reversal for the backward walk, and the ``gs_factor=0.99``
    correction constant.

``deadreckon_from_faf`` handles the (track, groundspeed) representation;
``integrate_controls_from_faf`` is the analogous backward integrator for the
derived-control representation (turn rate / accel / vertical rate).
"""

from __future__ import annotations

import numpy as np

KTS_TO_MS = 1852.0 / 3600.0
# The reference hardcodes 0.99 (geodesic lat/lon). Measured on our UTM-planar data
# it makes reconstruction worse (1057 m vs 619 m mean entry drift on real track/gs),
# so our default is 1.0; keep it a knob to reproduce the reference exactly.
FAF_GS_FACTOR = 1.0


def deadreckon_from_faf(track_deg, gs_kts, td, faf_xy, gs_factor=FAF_GS_FACTOR):
    """Backward walk from the FAF anchor. All (N, T); ``faf_xy`` (N, 2) x/y metres.

    Returns reconstructed positions ``(N, T, 2)`` in x/y metres, chronological
    order (index 0 = entry, index -1 = FAF == ``faf_xy``).
    """
    n, t = track_deg.shape
    tr = (track_deg[:, ::-1] - 180.0) % 360.0      # reverse time, flip bearing
    gsr = gs_kts[:, ::-1]
    tdr = td[:, ::-1]

    x = np.empty((n, t), np.float64)
    y = np.empty((n, t), np.float64)
    x[:, 0], y[:, 0] = faf_xy[:, 0], faf_xy[:, 1]
    for i in range(1, t):
        dt = np.clip(tdr[:, i - 1] - tdr[:, i], 0.0, None)
        d = gs_factor * gsr[:, i - 1] * KTS_TO_MS * dt
        th = np.radians(tr[:, i - 1])
        x[:, i] = x[:, i - 1] + d * np.sin(th)     # 0 deg = north (+y), 90 = east (+x)
        y[:, i] = y[:, i - 1] + d * np.cos(th)
    return np.stack([x[:, ::-1], y[:, ::-1]], axis=-1)


def integrate_controls_from_faf(faf_state, controls):
    """Backward integration of derived controls from the FAF state.

    ``faf_state`` (N, 5) = (xF, yF, zF, gsF, chiF) in SI (m, m/s, rad);
    ``controls`` (N, T, 4) = (turn_rate rad/s, along_accel m/s^2, vert_rate m/s,
    timedelta s). Returns ``(N, T, 3)`` x/y/z metres, chronological order.
    """
    chidot, accel, vz, td = (controls[..., j].astype(np.float64) for j in range(4))
    n, t = chidot.shape
    x = np.empty((n, t)); y = np.empty((n, t)); z = np.empty((n, t))
    gs = np.empty((n, t)); chi = np.empty((n, t))
    x[:, -1], y[:, -1], z[:, -1] = faf_state[:, 0], faf_state[:, 1], faf_state[:, 2]
    gs[:, -1], chi[:, -1] = faf_state[:, 3], faf_state[:, 4]
    for i in range(t - 2, -1, -1):
        dti = np.clip(td[:, i + 1] - td[:, i], 0.0, None)
        chi[:, i] = chi[:, i + 1] - 0.5 * (chidot[:, i] + chidot[:, i + 1]) * dti
        gs[:, i] = np.clip(gs[:, i + 1] - 0.5 * (accel[:, i] + accel[:, i + 1]) * dti, 0.0, None)
        z[:, i] = z[:, i + 1] - 0.5 * (vz[:, i] + vz[:, i + 1]) * dti
        x[:, i] = x[:, i + 1] - 0.5 * (gs[:, i] * np.sin(chi[:, i]) + gs[:, i + 1] * np.sin(chi[:, i + 1])) * dti
        y[:, i] = y[:, i + 1] - 0.5 * (gs[:, i] * np.cos(chi[:, i]) + gs[:, i + 1] * np.cos(chi[:, i + 1])) * dti
    return np.stack([x, y, z], axis=-1)


def entry_drift(recon_xy, true_xy):
    """Drift at the entry point (index 0) after a FAF-anchored reconstruction.

    Returns dict with mean/median/p90 of ||recon_entry - true_entry|| (metres)
    and the per-flight full-path mean error, all in metres.
    """
    d0 = np.linalg.norm(recon_xy[:, 0, :] - true_xy[:, 0, :], axis=1)
    dpath = np.linalg.norm(recon_xy - true_xy, axis=2).mean(axis=1)
    return {
        "entry_mean_m": float(d0.mean()),
        "entry_median_m": float(np.median(d0)),
        "entry_p90_m": float(np.percentile(d0, 90)),
        "path_mean_m": float(dpath.mean()),
    }
