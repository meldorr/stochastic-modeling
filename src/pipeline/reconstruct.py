"""Decoded feature profiles -> lat/lon tracks -> tidy DataFrame / parquet.

Mirrors `deep-traffic-generation-paper/dtg/traffic_builder.py`: because the data
runs *up to* the FAF, the anchor is each trajectory's last point and we integrate
(track, groundspeed, dt) backwards from it. Vectorized over the batch (one numpy
op per timestep) so reconstructing thousands of tracks is fast, and with no
`traffic`/`pitot` dependency (falls back to a spherical-earth step if `pitot`
is unavailable).

Also hosts the data-driven physics layer (bounds clip + monotone timedelta),
the lightweight analogue of the OpenAP envelope in `diffusion-models-lab`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_KTS_TO_MS = 1852.0 / 3600.0
_EARTH_R = 6_371_000.0


def _destination(lat, lon, bearing_deg, dist_m):
    """Great-circle destination point(s). Vectorized over the batch axis."""
    try:
        from pitot.geodesy import destination  # matches the reference builder

        la, lo, _ = destination(lat, lon, bearing_deg, dist_m)
        return la, lo
    except Exception:
        d = np.asarray(dist_m) / _EARTH_R
        th = np.radians(bearing_deg)
        p1 = np.radians(lat)
        l1 = np.radians(lon)
        p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
        l2 = l1 + np.arctan2(
            np.sin(th) * np.sin(d) * np.cos(p1),
            np.cos(d) - np.sin(p1) * np.sin(p2),
        )
        return np.degrees(p2), np.degrees(l2)


def walk_latlon_backward(
    track: np.ndarray, groundspeed: np.ndarray, timedelta: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate tracks backward from the FAF anchor. All inputs ``(N, T)``,
    ``anchor`` is ``(N, 2)`` lat/lon of the last point. Returns lat, lon ``(N, T)``."""
    n, t = track.shape
    track_r = (track[:, ::-1] - 180.0) % 360.0
    gs_r = groundspeed[:, ::-1]
    td_r = timedelta[:, ::-1]

    lat = np.empty((n, t), np.float64)
    lon = np.empty((n, t), np.float64)
    lat[:, 0] = anchor[:, 0]
    lon[:, 0] = anchor[:, 1]
    for i in range(1, t):
        dt = np.clip(td_r[:, i - 1] - td_r[:, i], 0.0, None)
        dist = 0.99 * gs_r[:, i - 1] * _KTS_TO_MS * dt
        lat[:, i], lon[:, i] = _destination(lat[:, i - 1], lon[:, i - 1], track_r[:, i - 1], dist)
    return lat[:, ::-1], lon[:, ::-1]


# --- data-driven physics layer --------------------------------------------
def _feat_idx(feature_names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(feature_names)}


def physics_repair(feats: np.ndarray, bounds: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Enforce physical plausibility on decoded profiles ``(N, T, F)`` in raw units:

    * groundspeed / altitude clipped to the training envelope,
    * timedelta shifted to start at 0 and forced monotonic non-decreasing,
    * track wrapped to [0, 360).
    """
    out = feats.copy()
    idx = _feat_idx(feature_names)
    for name in ("groundspeed", "altitude"):
        if name in idx:
            j = idx[name]
            out[:, :, j] = np.clip(out[:, :, j], bounds[j, 0], bounds[j, 1])
    if "timedelta" in idx:
        j = idx["timedelta"]
        td = out[:, :, j]
        td = td - td[:, :1]
        out[:, :, j] = np.maximum.accumulate(td, axis=1)
    if "track" in idx:
        j = idx["track"]
        out[:, :, j] = np.mod(out[:, :, j], 360.0)
    return out


def within_bounds(
    feats: np.ndarray, bounds: np.ndarray, feature_names: list[str], margin: float = 0.05
) -> np.ndarray:
    """Boolean mask ``(N,)``: does every timestep sit inside the (margined)
    training envelope for groundspeed and altitude? Used for rejection sampling."""
    idx = _feat_idx(feature_names)
    ok = np.ones(feats.shape[0], bool)
    for name in ("groundspeed", "altitude"):
        if name in idx:
            j = idx[name]
            lo, hi = bounds[j, 0], bounds[j, 1]
            pad = margin * (hi - lo)
            col = feats[:, :, j]
            ok &= (col >= lo - pad).all(1) & (col <= hi + pad).all(1)
    return ok


def reconstruct_to_frame(
    feats: np.ndarray,
    anchors: np.ndarray,
    feature_names: list[str],
    flight_prefix: str = "GEN",
    base_ts: pd.Timestamp | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a tidy long DataFrame (one row per timestep) with reconstructed
    lat/lon and a UTC timestamp axis derived from ``timedelta``."""
    idx = _feat_idx(feature_names)
    n, t, _ = feats.shape
    track = feats[:, :, idx["track"]]
    gs = feats[:, :, idx["groundspeed"]]
    td = feats[:, :, idx["timedelta"]]
    lat, lon = walk_latlon_backward(track, gs, td, anchors)

    if base_ts is None:
        base_ts = pd.Timestamp("2019-01-01", tz="UTC")

    frames = []
    for i in range(n):
        df = pd.DataFrame({name: feats[i, :, idx[name]] for name in feature_names})
        df["latitude"] = lat[i]
        df["longitude"] = lon[i]
        df["timestamp"] = base_ts + pd.to_timedelta(td[i], unit="s")
        fid = f"{flight_prefix}_{i:05d}"
        df["flight_id"] = fid
        if extra:
            for k, v in extra.items():
                df[k] = v[i]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
