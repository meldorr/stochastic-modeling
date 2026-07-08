"""Generate trajectories: sample latents -> decode -> repair -> reconstruct.

    python generate.py --config configs/config.yaml [--n 2000]

Writes ``results/generated.parquet`` (tidy, one row per timestep) and
``results/generated.npz`` (arrays for evaluation).
"""

from __future__ import annotations

import argparse

import numpy as np

from src.data.dataset import load_dataset
from src.pipeline.checkpoint import load_checkpoint
from src.pipeline.reconstruct import physics_repair, reconstruct_to_frame, within_bounds
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


def assign_anchors(gen_feats, data, feature_names, seed, k_near=20):
    """Match each generated trajectory to a real FAF anchor by nearest approach
    heading (circular), sampling among the k nearest for spatial variety."""
    rng = np.random.default_rng(seed)
    ti = feature_names.index("track")
    real_track = np.mod(data["X"][:, -1, ti], 360.0)  # (Nr,) real FAF heading
    real_anchor = data["anchor_last"]  # (Nr, 2)
    gen_track = np.mod(gen_feats[:, -1, ti], 360.0)  # (Ng,)

    out = np.empty((len(gen_feats), 2), np.float64)
    for start in range(0, len(gen_track), 512):  # chunk to bound memory
        chunk = gen_track[start : start + 512]
        diff = np.abs(chunk[:, None] - real_track[None, :])
        circ = np.minimum(diff, 360.0 - diff)  # (chunk, Nr)
        near = np.argpartition(circ, k_near, axis=1)[:, :k_near]
        pick = near[np.arange(len(chunk)), rng.integers(0, k_near, len(chunk))]
        out[start : start + 512] = real_anchor[pick]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate trajectories from a trained pipeline"
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument(
        "--checkpoint", default=None, help="override paths.checkpoint in config"
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
    data = load_dataset(cfg)
    names = bundle["feature_names"]

    print(f"[generate] sampling {n} latents (m={bundle['m']}) …")
    clamp = cfg["ddpm"].get("sample_clamp")
    W, feats = sample_trajectories(
        bundle, n, bool(gcfg["physics_reject"]), int(gcfg["reject_max_tries"]), clamp
    )
    anchors = assign_anchors(feats, data, names, int(cfg["seed"]))

    df = reconstruct_to_frame(feats, anchors, names, flight_prefix="GEN")

    pq = ensure_parent(cfg["paths"]["generated"])
    df.to_parquet(pq, index=False)
    npz = pq.with_suffix(".npz")
    np.savez_compressed(
        npz,
        feats=feats.astype(np.float32),
        W=W.astype(np.float32),
        anchors=anchors.astype(np.float32),
        feature_names=np.array(names, dtype=object),
    )
    print(f"[generate] wrote {pq}  ({len(df)} rows, {n} trajectories)")
    print(f"[generate] wrote {npz}")


if __name__ == "__main__":
    main()
