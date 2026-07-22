"""Canonical dead-reckoning — one implementation for every experiment.

Follows the reference repo (`deep-traffic-generation-paper/dtg/traffic_builder.py`,
``_walk_latlon(forward=False)``): anchor at the **FAF** (the fixed, known
destination) and integrate **backward** in time, so accumulated error lands at
the uncertain **entry** point — which is exactly where we measure drift.

The **canonical** reconstruction is :func:`deadreckon_from_faf_geodesic` — the
reference's WGS84 geodesic Euler walk — with the single tweak ``gs_factor=1.0``
instead of the reference's 0.99 (measured on our data the 0.99 groundspeed
correction under-shoots: it triples entry drift, 874 -> 289 m mean). This is what
E5 selected to proceed with.

:func:`deadreckon_from_faf` (planar UTM x/y walk) is kept as a **documented
ablation**: walking with raw compass track in the UTM grid leaves a systematic
bearing error equal to the grid-convergence angle gamma (~-0.43 deg at LSZH,
west of zone-32's 9 deg E meridian), which alone costs ~320 m of entry drift
(610 vs 289 m). Subtracting gamma from the track recovers the geodesic result to
<10 m, confirming the mechanism — see E5.

``integrate_controls_from_faf`` is the analogous backward integrator for the
derived-control representation (turn rate / accel / vertical rate).
"""

from __future__ import annotations

import numpy as np
import pyproj

KTS_TO_MS = 1852.0 / 3600.0
# Reference hardcodes 0.99; on our data 1.0 reconstructs far better (see module
# docstring / E5). Kept a knob so the literature baseline is one call away.
FAF_GS_FACTOR = 1.0

_WGS84 = "EPSG:4326"
_UTM32N = "EPSG:32632"                       # LSZH / Zurich TMA
_GEOD = pyproj.Geod(ellps="WGS84")
_TF_LL2XY = pyproj.Transformer.from_crs(_WGS84, _UTM32N, always_xy=True)


def deadreckon_from_faf_geodesic(track_deg, gs_kts, td, faf_latlon, gs_factor=FAF_GS_FACTOR):
    """Canonical backward geodesic walk from the FAF (reference method, WGS84).

    All inputs ``(N, T)``; ``faf_latlon`` ``(N, 2)`` = (lat, lon) of the anchor.
    Returns ``(N, T, 2)`` = (lat, lon) in chronological order (index 0 = entry,
    -1 = FAF). Identical scheme to the reference ``_walk_latlon(forward=False)``:
    reverse time, flip bearing 180 deg, Euler step with start-of-interval speed
    ``gs_factor * gs[kts->m/s] * dt`` via WGS84 ``geod.fwd``, then re-reverse.
    """
    n, t = track_deg.shape
    tr = (track_deg[:, ::-1] - 180.0) % 360.0
    gsr = gs_kts[:, ::-1]
    tdr = td[:, ::-1]
    lat = np.empty((n, t), np.float64)
    lon = np.empty((n, t), np.float64)
    lat[:, 0], lon[:, 0] = faf_latlon[:, 0], faf_latlon[:, 1]
    for i in range(1, t):
        dt = np.clip(tdr[:, i - 1] - tdr[:, i], 0.0, None)
        d = gs_factor * gsr[:, i - 1] * KTS_TO_MS * dt
        lon[:, i], lat[:, i], _ = _GEOD.fwd(lon[:, i - 1], lat[:, i - 1], tr[:, i - 1], d)
    return np.stack([lat[:, ::-1], lon[:, ::-1]], axis=-1)


def geodesic_entry_drift(recon_latlon, true_latlon):
    """Entry/path drift (metres) for a geodesic reconstruction, measured as the
    WGS84 geodesic distance between reconstructed and true (lat, lon).

    ``recon_latlon``/``true_latlon`` ``(N, T, 2)`` = (lat, lon). Returns the same
    dict shape as :func:`entry_drift` plus the per-timestep mean curve.
    """
    _, _, dist = _GEOD.inv(recon_latlon[..., 1], recon_latlon[..., 0],
                           true_latlon[..., 1], true_latlon[..., 0])   # (N, T) metres
    d0 = dist[:, 0]
    return {
        "entry_mean_m": float(d0.mean()),
        "entry_median_m": float(np.median(d0)),
        "entry_p90_m": float(np.percentile(d0, 90)),
        "path_mean_m": float(dist.mean()),
        "curve_m": dist.mean(axis=0),                                  # (T,)
    }


def latlon_to_local_xy(latlon, ref_latlon=None, ref_xy=None):
    """Project ``(..., 2)`` = (lat, lon) to the dataset's UTM32N x/y frame.

    Our stored ``x/y`` is UTM32N shifted by a near-constant ~100 m offset. Pass
    matching true ``ref_latlon`` and ``ref_xy`` to calibrate that offset so the
    output lands in the same frame as the stored trajectories (for overlays).
    """
    ex, ny = _TF_LL2XY.transform(latlon[..., 1], latlon[..., 0])
    xy = np.stack([ex, ny], axis=-1)
    if ref_latlon is not None and ref_xy is not None:
        rex, rny = _TF_LL2XY.transform(ref_latlon[..., 1], ref_latlon[..., 0])
        off = np.array([(ref_xy[..., 0] - rex).mean(), (ref_xy[..., 1] - rny).mean()])
        xy = xy + off
    return xy


def generated_to_xy(feats, feature_set, names, anchors, rng):
    """Reconstruct generated samples to absolute x/y in the stored UTM frame.

    One reconstruction for the whole pipeline (in-training snapshots, final
    sampling, evaluation), using the canonical FAF-anchored reckoning per
    representation:

        xy               take the modelled x/y channels directly;
        gstrack          geodesic backward walk from the FAF (factor 1.0), lat/lon
                         -> x/y (the literature method with our one tweak);
        gstrack_derived  self-consistent planar velocity walk from the FAF x/y
                         (grid-referenced heading + speed, single integration);
        controls         backward trapezoid from the FAF state.

    ``feats`` ``(N, T, C)`` raw units. Because the FAF is the fixed, known
    destination, every generated sample is anchored at a FAF state drawn from the
    real pool (``anchors`` = the experiment's ``aux``): ``faf_latlon`` (M, 2) and
    ``faf_xy`` (M, 2) for gstrack, ``faf_state`` (M, 5) for controls. Returns
    ``(N, T, 2)``.
    """
    n = len(feats)
    if feature_set in ("xy", "xyt"):
        xi, yi = names.index("x"), names.index("y")
        return feats[:, :, [xi, yi]].astype(np.float64)
    if feature_set == "gstrack":
        ti, gi, tdi = names.index("track"), names.index("groundspeed"), names.index("timedelta")
        idx = rng.integers(0, len(anchors["faf_latlon"]), n)
        ll = deadreckon_from_faf_geodesic(feats[:, :, ti], feats[:, :, gi], feats[:, :, tdi],
                                          anchors["faf_latlon"][idx])
        return latlon_to_local_xy(ll, ref_latlon=anchors["faf_latlon"], ref_xy=anchors["faf_xy"])
    if feature_set == "gstrack_derived":
        hi, gi, tdi = names.index("heading"), names.index("groundspeed_ms"), names.index("timedelta")
        idx = rng.integers(0, len(anchors["faf_xy"]), n)
        return deadreckon_from_faf_velocity(feats[:, :, hi], feats[:, :, gi], feats[:, :, tdi],
                                            anchors["faf_xy"][idx])
    if feature_set == "controls":
        idx = rng.integers(0, len(anchors["faf_state"]), n)
        return integrate_controls_from_faf(anchors["faf_state"][idx], feats)[:, :, :2]
    raise ValueError(f"unknown feature_set {feature_set!r}")


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


def deadreckon_from_faf_velocity(chi_rad, gs_ms, td, faf_xy):
    """Backward planar walk from the FAF given a grid-referenced heading + speed.

    For the *derived* gs/track representation: ``chi_rad`` (heading) and ``gs_ms``
    (speed) come from differentiating the x/y path, so the heading is already in
    the UTM grid frame — the walk is self-consistent in the plane (no grid
    convergence term, no ``gs_factor``, single integration). Trapezoidal to match
    the central-difference derivation. All ``(N, T)``; ``faf_xy`` ``(N, 2)``.
    Returns ``(N, T, 2)`` x/y metres, chronological (index 0 = entry, -1 = FAF).
    """
    n, t = chi_rad.shape
    vx = gs_ms * np.sin(chi_rad)
    vy = gs_ms * np.cos(chi_rad)
    x = np.empty((n, t)); y = np.empty((n, t))
    x[:, -1], y[:, -1] = faf_xy[:, 0], faf_xy[:, 1]
    for i in range(t - 2, -1, -1):
        dti = np.clip(td[:, i + 1] - td[:, i], 0.0, None)
        x[:, i] = x[:, i + 1] - 0.5 * (vx[:, i] + vx[:, i + 1]) * dti
        y[:, i] = y[:, i + 1] - 0.5 * (vy[:, i] + vy[:, i + 1]) * dti
    return np.stack([x, y], axis=-1)


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
