"""Stage 4 — collate every experiment's metrics into one master table.

    python stages/s4_collate.py

Sweeps results/, results_remote/ and archive/ for metrics.json / gen_metrics.json
(the shared scorecard: per-channel marginal KS, sliced-Wasserstein on x/y paths,
endpoint KS, envelope %) and writes:

    results/summary/master_table.csv
    results/summary/master_table.md

Rows are tagged with their source so fPCA-latent pipelines (e2/e3), raw-space
staged experiments (ddpm_*), and archived baselines sit in one comparison.
Spatial comparability caveat: rows with spatial_centred=True (gstrack/dyn) are
shape-only — no absolute position exists for those representations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.utils import REPO_ROOT

SCAN = ["results", "results_remote", "archive"]


def rows_from(root: Path):
    for p in sorted(root.rglob("*.json")):
        if p.name not in ("metrics.json", "gen_metrics.json"):
            continue
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        ks = m.get("ks_marginal") or m.get("ks_feature_marginal")
        if not isinstance(ks, dict):        # e.g. e1 representation study
            continue
        name = m.get("experiment") or m.get("exp") or p.parent.name
        yield {
            "experiment": name,
            "source": str(p.parent.relative_to(REPO_ROOT)),
            "feature_set": m.get("feature_set") or m.get("features") or "",
            "n_gen": m.get("n_gen", ""),
            **{f"ks_{k}": round(float(v), 4) for k, v in ks.items()},
            "sliced_w_xy_km": round(float(m["sliced_w_xy_km"]), 3) if "sliced_w_xy_km" in m else "",
            "ks_endpoint_x": round(float(m["ks_endpoint_x"]), 3) if "ks_endpoint_x" in m else "",
            "ks_endpoint_y": round(float(m["ks_endpoint_y"]), 3) if "ks_endpoint_y" in m else "",
            "within_pct": round(float(m["within_pct"]), 3) if "within_pct" in m else "",
            "spatial_centred": m.get("spatial_centred", ""),
        }


def main() -> None:
    rows = []
    for d in SCAN:
        root = REPO_ROOT / d
        if root.exists():
            rows.extend(rows_from(root))
    if not rows:
        print("no metrics found"); return

    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)

    out = REPO_ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)

    import csv

    with open(out / "master_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out / "master_table.md").write_text("\n".join(lines) + "\n")

    print(f"[s4] {len(rows)} experiment rows -> {out}/master_table.{{csv,md}}")
    for r in rows:
        print(f"  {r['experiment']:45s} {r['source']}")


if __name__ == "__main__":
    main()
