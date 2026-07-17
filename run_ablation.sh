#!/usr/bin/env bash
# Full ablation chain — portable one-command runner.
#
#   bash run_ablation.sh                       # everything (E1 -> E2 -> E3 -> E4)
#   PY=python3 bash run_ablation.sh            # choose interpreter
#   E4_EPOCHS=2000 bash run_ablation.sh        # E4 budget (default 2000 = converged)
#   SKIP_E2=1 SKIP_E3=1 bash run_ablation.sh   # run only what you need
#
# Requirements: pip install -r requirements.txt ;  data/processed.npz present
# (ships pre-built — no `traffic` library needed). Device is auto-detected
# (CUDA > MPS > CPU). Progress: tee'd to stdout + results/ablation_run.log;
# TensorBoard: tensorboard --logdir runs
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

PY="${PY:-python}"
E4_EPOCHS="${E4_EPOCHS:-2000}"
LOG=results/ablation_run.log
mkdir -p results

if [ ! -f data/processed.npz ]; then
  echo "ERROR: data/processed.npz missing. Copy it from the dev machine, or run:"
  echo "  $PY -m src.data.prepare --config configs/config.yaml   (needs `traffic` + raw pickle)"
  exit 1
fi

run() { echo "=================== $1 ==================="; shift; "$@"; }

{
if [ -z "${SKIP_E1:-}" ]; then
  run "E1 REPRESENTATION" $PY experiments/e1_smoothing.py
fi

if [ -z "${SKIP_E2:-}" ]; then
  run "E2 GLOBAL fPCA+DDPM" $PY train.py --config configs/e2_global.yaml --tag e2_global
  $PY generate.py --config configs/e2_global.yaml
  $PY evaluate.py --config configs/e2_global.yaml
  $PY experiments/eval_gen.py --exp e2_global --generated results/e2_global/generated.npz --features xy
fi

if [ -z "${SKIP_E3:-}" ]; then
  run "E3 PER-CLUSTER fPCA+DDPM" $PY train.py --config configs/e3_cluster.yaml --tag e3_cluster
  $PY generate.py --config configs/e3_cluster.yaml
  $PY evaluate.py --config configs/e3_cluster.yaml
  $PY experiments/eval_gen.py --exp e3_cluster --generated results/e3_cluster/generated.npz --features xy
fi

if [ -z "${SKIP_E4:-}" ]; then
  run "E4 RAW U-NET DYN (${E4_EPOCHS} ep)" \
      $PY experiments/e4_raw.py --features dyn --arch unet --epochs "$E4_EPOCHS"
  $PY experiments/eval_gen.py --exp e4_raw_dyn --generated results/e4_raw_dyn/generated.npz --features dyn
  run "E4 RAW U-NET XY (${E4_EPOCHS} ep)" \
      $PY experiments/e4_raw.py --features xy --arch unet --epochs "$E4_EPOCHS"
  $PY experiments/eval_gen.py --exp e4_raw_xy --generated results/e4_raw_xy/generated.npz --features xy
fi

echo "=================== ABLATION DONE ==================="
} 2>&1 | tee -a "$LOG"
