"""Design waypoint scenarios for guided trajectory generation (Phase A).

Data-driven: from processed.npz derive the FAF anchor, the runway final bearing,
the along-track radial-distance + altitude profile, and the entry-bearing coverage
(to locate empty sectors). Then define three scenarios as ordered waypoints
``(t_index, x, y, altitude)`` for the inpainting sampler:

    S0_indist   waypoints on a populated real corridor  -> sanity: should reproduce
    S1_straightin  aligned approach down the runway centerline (318 deg, ~absent)
    S2_gap      aligned approach from an uncovered bearing (255 deg WSW)

Writes configs/waypoints.json and results/waypoints/design.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WP_INDICES = [20, 100, 163]        # far, mid, near-FAF along-track positions


def bearing_xy(bearing_deg, dist_m):
    """Offset (dx_east, dy_north) at a compass bearing (0=N, cw) and distance."""
    b = np.radians(bearing_deg)
    return dist_m * np.sin(b), dist_m * np.cos(b)


def main() -> None:
    d = np.load("data/processed.npz", allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    xi, yi, ai = names.index("x"), names.index("y"), names.index("altitude")
    X = d["X"]
    xy = X[:, :, [xi, yi]].astype(float)
    alt = X[:, :, ai].astype(float)

    FAF = xy[:, -1, :].mean(0)
    dist = np.linalg.norm(xy - FAF, axis=2)
    # empirical radial distance + altitude at each waypoint index
    prof = {k: (float(np.median(dist[:, k])), float(np.median(alt[:, k]))) for k in WP_INDICES}

    # entry bearing of each flight relative to the FAF -> populated / empty sectors
    ent = xy[:, 0, :] - FAF
    ebrg = np.degrees(np.arctan2(ent[:, 0], ent[:, 1])) % 360

    def radial_scenario(bearing):
        """Aligned approach: all waypoints on one radial at the empirical dist/alt."""
        wps = []
        for k in WP_INDICES:
            dm, altm = prof[k]
            dx, dy = bearing_xy(bearing, dm)
            wps.append({"t": k, "x": float(FAF[0] + dx), "y": float(FAF[1] + dy), "alt": altm})
        return wps

    # S0: median of a populated real corridor (entry sector ~75-90 deg, E)
    sel = (ebrg >= 75) & (ebrg < 90)
    s0 = [{"t": k, "x": float(np.median(xy[sel, k, 0])), "y": float(np.median(xy[sel, k, 1])),
           "alt": float(np.median(alt[sel, k]))} for k in WP_INDICES]

    scenarios = {
        "S0_indist": {"desc": "populated real corridor (E entry ~75-90 deg) medians; sanity check",
                      "waypoints": s0},
        "S1_straightin": {"desc": "runway-aligned straight-in down the 318 deg centerline (~absent in data)",
                          "waypoints": radial_scenario(318.0)},
        "S2_gap": {"desc": "aligned approach from an uncovered bearing (255 deg WSW)",
                   "waypoints": radial_scenario(255.0)},
    }
    meta = {"faf": [float(FAF[0]), float(FAF[1])], "faf_alt": float(alt[:, -1].mean()),
            "final_bearing_deg": 138.0, "wp_indices": WP_INDICES,
            "profile": {str(k): {"dist_m": prof[k][0], "alt_ft": prof[k][1]} for k in WP_INDICES}}
    out = {"meta": meta, "scenarios": scenarios}
    Path("configs/waypoints.json").write_text(json.dumps(out, indent=2))
    print("[wp] wrote configs/waypoints.json")
    for name, sc in scenarios.items():
        pts = ", ".join(f"t{w['t']}=({w['x']:.0f},{w['y']:.0f},{w['alt']:.0f}ft)" for w in sc["waypoints"])
        print(f"  {name:15s} {pts}")

    # design figure: real cloud + the three scenarios
    out_dir = Path("results/waypoints"); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(xy), 600, replace=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    for s in pick:
        ax.plot(xy[s, :, 0], xy[s, :, 1], color="#1f77b4", lw=0.3, alpha=0.15)
    colors = {"S0_indist": "#2ca02c", "S1_straightin": "#d62728", "S2_gap": "#ff7f0e"}
    for name, sc in scenarios.items():
        wx = [w["x"] for w in sc["waypoints"]]; wy = [w["y"] for w in sc["waypoints"]]
        ax.plot(wx + [FAF[0]], wy + [FAF[1]], "-o", color=colors[name], lw=2, ms=8,
                label=name, zorder=5)
    ax.scatter(*FAF, marker="*", s=260, color="black", zorder=6, label="FAF")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("Waypoint scenarios over real approaches (S1/S2 target absent geometries)")
    fig.tight_layout(); fig.savefig(out_dir / "design.png", dpi=130); plt.close(fig)
    print(f"[wp] wrote {out_dir}/design.png")


if __name__ == "__main__":
    main()
