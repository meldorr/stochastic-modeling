"""Generate trajectories: sample latents -> decode -> repair -> reconstruct.

    python generate.py --config configs/config.yaml [--n 2000]

Writes ``results/generated.parquet`` (tidy, one row per timestep) and
``results/generated.npz`` (arrays for evaluation).
"""

from __future__ import annotations

import argparse

import numpy as np

from src.pipeline.checkpoint import load_checkpoint
from src.pipeline.reconstruct import (
    physics_repair,
    reconstruct_to_frame_direct,
    within_bounds,
)
from src.pipeline.utils import ensure_parent, load_config, resolve, set_seed


def draw_batch(bundle, k, clamp):
    """Sample k latents and decode to raw-unit feature profiles."""
    z = bundle["ddpm"].sample(k, bundle["m"], device=bundle["device"], clamp=clamp)
    W = bundle["latent_scaler"].inverse_transform(z.cpu().numpy())
    with np.errstate(
        all="ignore"
    ):  # under-trained samples can overflow decode; expected
        X_std = bundle["fpca"].decode(W)
        feats = bundle["feature_scaler"].inverse_transform(X_std)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return W, feats


def sample_trajectories(bundle, n, reject, max_tries, clamp):
    """Draw n trajectories, optionally rejecting physics-implausible ones.

    Always returns something: if an (under-trained) model produces nothing that
    passes the filter, we fall back to repairing the last decoded batch so the
    smoke path still yields output.
    """
    names, bounds = bundle["feature_names"], bundle["bounds"]
    W_keep, F_keep, have, tries, last = [], [], 0, 0, None
    while have < n and tries <= max_tries:
        W, feats = draw_batch(bundle, n - have if have else n, clamp)
        last = (W, feats)
        mask = (
            within_bounds(feats, bounds, names) if reject else np.ones(len(feats), bool)
        )
        if mask.any():
            W_keep.append(W[mask])
            F_keep.append(feats[mask])
            have += int(mask.sum())
        tries += 1

    if have == 0:
        print(
            "[generate] WARNING: no samples passed the physics filter "
            "(model likely under-trained) — using repaired raw samples instead."
        )
        W_keep, F_keep = [last[0]], [last[1]]
    elif have < n:
        print(f"[generate] kept {have}/{n} after {tries} tries (physics filter).")

    W = np.concatenate(W_keep)[:n]
    feats = physics_repair(np.concatenate(F_keep)[:n], bounds, names)
    return W, feats


def allocate_counts(weights: dict, clusters: list, n: int) -> dict:
    """Largest-remainder allocation of n samples across clusters by weight."""
    w = np.array([max(0.0, float(weights.get(c, 0.0))) for c in clusters])
    if w.sum() == 0:
        w = np.ones(len(clusters))
    raw = w / w.sum() * n
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base))
    for i in range(int(n - base.sum())):
        base[order[i % len(base)]] += 1
    return {c: int(base[i]) for i, c in enumerate(clusters)}


def generate_per_cluster(bundle, cfg, n, clamp, reject, max_tries):
    """Sample each cluster's DDPM in proportion to its (or a configured) frequency."""
    clusters = bundle["clusters"]
    mix = cfg["generate"].get("cluster_mix")
    weights = {int(k): float(v) for k, v in (mix or bundle["frequencies"]).items()}
    counts = allocate_counts(weights, clusters, n)
    feats_all, ids_all = [], []
    for c in clusters:
        if counts[c] <= 0:
            continue
        mdl = bundle["models"][c]
        sub = {
            "ddpm": mdl["ddpm"], "m": mdl["m"], "latent_scaler": mdl["latent_scaler"],
            "fpca": mdl["fpca"], "feature_scaler": bundle["feature_scaler"],
            "bounds": bundle["bounds"], "feature_names": bundle["feature_names"],
            "device": bundle["device"],
        }
        _, feats_c = sample_trajectories(sub, counts[c], reject, max_tries, clamp)
        feats_all.append(feats_c)
        ids_all.append(np.full(len(feats_c), c, np.int32))
        print(f"[generate] cluster {c}: {len(feats_c)} trajectories (m={mdl['m']})")
    return np.concatenate(feats_all), np.concatenate(ids_all)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate trajectories from a trained pipeline"
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument(
        "--checkpoint", default=None, help="override paths.checkpoint in config"
    )
    ap.add_argument(
        "--no-reject", action="store_true",
        help="skip physics rejection sampling (much faster with under-trained models)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    gcfg = cfg["generate"]
    n = args.n or int(gcfg["n_samples"])

    ckpt_path = (
        resolve(args.checkpoint)
        if args.checkpoint
        else resolve(cfg["paths"]["checkpoint"])
    )
    bundle = load_checkpoint(ckpt_path)
    names = bundle["feature_names"]
    clamp = cfg["ddpm"].get("sample_clamp")
    reject = (not args.no_reject) and bool(gcfg["physics_reject"])
    max_tries = int(gcfg["reject_max_tries"])

    # position is modelled directly (x, y) -> no dead-reckoning, no anchors.
    if bundle["mode"] == "per_cluster":
        print(f"[generate] per-cluster sampling {n} trajectories over "
              f"{len(bundle['clusters'])} clusters …")
        feats, cluster_ids = generate_per_cluster(bundle, cfg, n, clamp, reject, max_tries)
        df = reconstruct_to_frame_direct(feats, names, flight_prefix="GEN",
                                         extra={"cluster": cluster_ids})
        npz_extra = {"cluster_ids": cluster_ids.astype(np.int32)}
    else:
        print(f"[generate] sampling {n} latents (m={bundle['m']}) …")
        W, feats = sample_trajectories(bundle, n, reject, max_tries, clamp)
        df = reconstruct_to_frame_direct(feats, names, flight_prefix="GEN")
        npz_extra = {"W": W.astype(np.float32)}

    pq = ensure_parent(cfg["paths"]["generated"])
    df.to_parquet(pq, index=False)
    npz = pq.with_suffix(".npz")
    np.savez_compressed(npz, feats=feats.astype(np.float32),
                        feature_names=np.array(names, dtype=object), **npz_extra)
    print(f"[generate] wrote {pq}  ({len(df)} rows, {len(feats)} trajectories)")
    print(f"[generate] wrote {npz}")


if __name__ == "__main__":
    main()
