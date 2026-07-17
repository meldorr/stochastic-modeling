#!/usr/bin/env bash
# Staged DDPM experiments — portable runner (train -> generate -> evaluate each).
#
#   bash run_experiments.sh                                # all experiments in EXPERIMENTS
#   bash run_experiments.sh ddpm_tcn_unet_standardscaler__xy   # just one (by name)
#   PY=python3 EPOCHS=2000 N_GEN=1000 bash run_experiments.sh
#   STOCH_DATA_DIR=/path/to/data bash run_experiments.sh   # relocate the data folder
#
# Prereqs: pip install -r requirements.txt ; <data_dir>/processed.npz present
# (copy it, or build with: $PY stages/s0_prepare.py — needs the traffic lib).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

PY="${PY:-python}"
EPOCHS="${EPOCHS:-}"          # empty = use configs (base.yaml training.epochs)
N_GEN="${N_GEN:-}"            # empty = use configs (generate.n_samples)

EXPERIMENTS=(
  ddpm_fc_standardscaler__xy
  ddpm_fcn_unet_standardscaler__xy
  ddpm_tcn_unet_standardscaler__xy
  ddpm_tcn_unet_standardscaler_dropout__xy
  ddpm_tcn_unet_standardscaler__gstrack
  ddpm_tcn_unet_standardscaler__controls
)
# positional args override the list
if [ "$#" -gt 0 ]; then EXPERIMENTS=("$@"); fi

mkdir -p results
LOG=results/experiments_run.log

{
for name in "${EXPERIMENTS[@]}"; do
  exp="configs/experiments/${name}.yaml"
  [ -f "$exp" ] || { echo "SKIP: $exp not found"; continue; }
  echo "=================== ${name} ==================="
  # shellcheck disable=SC2086
  $PY stages/s1_train_ddpm.py --exp "$exp" ${EPOCHS:+--epochs $EPOCHS}
  # shellcheck disable=SC2086
  $PY stages/s2_generate.py --exp "$exp" ${N_GEN:+--n $N_GEN}
  $PY stages/s3_evaluate.py --exp "$exp"
done
echo "=================== ALL EXPERIMENTS DONE ==================="
} 2>&1 | tee -a "$LOG"
