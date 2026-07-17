"""Stage 0 — data preparation.

Current data (tilFAF pickle, already gridded to 200 ticks): builds the
processed cache into <data_dir>/processed.npz. Only this stage needs the
`traffic` library.

    python stages/s0_prepare.py

Raw ADS-B ingest (spec Sections 1.1-1.3: UTM 32N runway-centred frame, TMA
membership R<=75 km & z<=4500 m, 10 s wall-clock grid, gap/go-around handling)
is NOT implementable against the current pickle — it requires the raw 2-month
state-vector dump with real UNIX timestamps. The config knobs already exist
under `tma:` in configs/base.yaml; wire them up here when that dump lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.data.prepare import prepare
from src.pipeline.utils import load_experiment_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0: build processed.npz")
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_experiment_config(args.base, base_path=args.base)
    out = resolve(cfg["paths"]["processed"])
    if out.exists():
        print(f"[s0] {out} already exists — nothing to do (delete it to rebuild).")
        return
    prepare(cfg)


if __name__ == "__main__":
    main()
