"""Stage 1a: convert the `traffic` pickle into a portable ``processed.npz``.

This is the only step that needs the `traffic` library. It caches the expensive
445 MB load into a ~90 MB array bundle so every downstream stage (fPCA, DDPM,
evaluation) runs with plain numpy/torch and never touches `traffic` again.

Layout produced (N flights, T=200 timesteps, F=4 features):

    X            (N, T, F) float32  channels = [track, groundspeed, altitude, timedelta]
                                     NB: the 'track' channel stores the *unwrapped*
                                     heading so fPCA never sees the 0/360 jump.
    latlon       (N, T, 2) float32  real (latitude, longitude) — for evaluation plots
    anchor_last  (N, 2)    float32  lat/lon of the FAF (last point) — reconstruction anchor
    anchor_first (N, 2)    float32  lat/lon of the entry point (first point)
    flight_ids   (N,)      str
    flow         (N,)      str      initial_flow label (approach direction)
    runway       (N,)      str
    feature_names, meta_json         bookkeeping
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.pipeline.utils import ensure_parent, load_config, resolve


def prepare(config: dict) -> dict:
    from traffic.core import Traffic  # local import: only needed here

    dcfg = config["data"]
    seq_len = int(dcfg["seq_len"])
    track_col = dcfg["track_column"]
    # source columns, with the requested 'track' channel mapped to track_column
    src_cols = [track_col if f == "track" else f for f in dcfg["features"]]

    raw_path = resolve(config["paths"]["raw_pkl"])
    print(f"[prepare] loading {raw_path} …")
    df = Traffic.from_file(raw_path).data

    needed = set(src_cols) | {"flight_id", "timedelta", "latitude", "longitude"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"columns missing from data: {sorted(missing)}")

    # Contiguous, time-ordered rows per flight -> clean reshape into (N, T, ·).
    df = df.sort_values(["flight_id", "timedelta"], kind="stable")
    sizes = df.groupby("flight_id", sort=False).size()
    if not (sizes == seq_len).all():
        bad = sizes[sizes != seq_len]
        raise ValueError(
            f"{len(bad)} flights are not length {seq_len} "
            f"(min={sizes.min()}, max={sizes.max()}); resampling guard needed."
        )
    n = len(sizes)
    print(f"[prepare] {n} flights x {seq_len} steps, features={dcfg['features']}")

    X = df[src_cols].to_numpy(np.float32).reshape(n, seq_len, len(src_cols))
    latlon = df[["latitude", "longitude"]].to_numpy(np.float32).reshape(n, seq_len, 2)

    if not np.isfinite(X).all():
        # linear-interpolate any stray gaps per (flight, feature) then error if still bad
        for i in range(n):
            for j in range(X.shape[2]):
                col = X[i, :, j]
                m = ~np.isfinite(col)
                if m.any():
                    idx = np.arange(seq_len)
                    col[m] = np.interp(idx[m], idx[~m], col[~m])
        if not np.isfinite(X).all():
            raise ValueError("non-finite values remain in X after interpolation")

    g = df.groupby("flight_id", sort=False)
    flight_ids = np.array(list(g.groups.keys()), dtype=object)
    flow = g["initial_flow"].first().to_numpy(dtype=object) if "initial_flow" in df else np.array([""] * n, object)
    runway = g["runway"].first().to_numpy(dtype=object) if "runway" in df else np.array([""] * n, object)

    anchor_last = latlon[:, -1, :].copy()
    anchor_first = latlon[:, 0, :].copy()

    meta = {
        "feature_names": list(dcfg["features"]),
        "track_column": track_col,
        "seq_len": seq_len,
        "anchor_index": int(dcfg["anchor_index"]),
        "n_flights": int(n),
    }

    out = ensure_parent(config["paths"]["processed"])
    np.savez_compressed(
        out,
        X=X,
        latlon=latlon,
        anchor_last=anchor_last.astype(np.float32),
        anchor_first=anchor_first.astype(np.float32),
        flight_ids=flight_ids,
        flow=flow,
        runway=runway,
        feature_names=np.array(dcfg["features"], dtype=object),
        meta_json=json.dumps(meta),
    )
    print(f"[prepare] wrote {out}  (X {X.shape}, {X.nbytes / 1e6:.1f} MB uncompressed)")
    print(f"[prepare] flows: {dict(zip(*np.unique(flow.astype(str), return_counts=True)))}")
    return meta


def load_processed(config: dict) -> dict:
    """Load ``processed.npz`` into a dict of arrays (+ parsed meta)."""
    path = resolve(config["paths"]["processed"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run Stage 1 first (train.py runs it automatically)."
        )
    npz = np.load(path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    data["meta"] = json.loads(str(npz["meta_json"]))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1a: traffic pickle -> processed.npz")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
