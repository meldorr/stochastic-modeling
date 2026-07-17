#!/usr/bin/env bash
# Drive training on the SSH compute node. Code+data go over rsync, jobs run
# under nohup (survive disconnects), results rsync back. Git stays authoritative
# on the dev machine only — never commit on the remote.
#
#   REMOTE=sim@131.180.117.11 REMOTE_DIR=~/stochastic-modeling bash remote.sh <cmd>
#
# Commands:
#   check    remote GPU / python / disk sanity
#   push     rsync code + data/processed.npz to the remote (excludes results/runs/archive/.git)
#   setup    create venv + pip install -r requirements.txt on the remote
#   launch   [args...] start run_experiments.sh remotely under nohup
#            e.g.: bash remote.sh launch                       # all experiments
#                  bash remote.sh launch ddpm_tcn_unet_standardscaler__xy
#                  EPOCHS=2000 bash remote.sh launch ...       # env passed through
#   status   tail the remote experiment log + show GPU + running procs
#   fetch    rsync results/ and runs/ back to this machine
#   stop     kill remote training processes
#
# One-time prerequisite (from YOUR terminal, enters your password once):
#   ssh-copy-id sim@131.180.117.11
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-sim@131.180.117.11}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/stochastic-modeling}"
SSH="ssh -o ConnectTimeout=8 $REMOTE"

cmd="${1:-status}"; shift || true

case "$cmd" in
  check)
    $SSH "hostname; echo '--- gpu ---'; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo no-gpu; echo '--- python ---'; python3 --version; echo '--- disk ---'; df -h \$HOME | tail -1"
    ;;
  push)
    rsync -avz --progress \
      --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude 'results/' --exclude 'runs/' --exclude 'archive/' \
      --exclude 'checkpoints/' --exclude '.DS_Store' \
      ./ "$REMOTE:$REMOTE_DIR/"
    rsync -avz --progress data/processed.npz "$REMOTE:$REMOTE_DIR/data/"
    ;;
  setup)
    $SSH "cd $REMOTE_DIR && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt && .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'"
    ;;
  launch)
    ARGS="$*"
    $SSH "cd $REMOTE_DIR && mkdir -p results && PY=.venv/bin/python EPOCHS='${EPOCHS:-}' N_GEN='${N_GEN:-}' nohup bash run_experiments.sh $ARGS > results/nohup.out 2>&1 & echo \"launched pid \$!\""
    ;;
  status)
    $SSH "cd $REMOTE_DIR 2>/dev/null && { echo '--- last log ---'; tail -15 results/experiments_run.log 2>/dev/null || tail -15 results/nohup.out 2>/dev/null || echo 'no log yet'; echo '--- procs ---'; pgrep -af 'stages/s1|run_experiments' || echo 'nothing running'; echo '--- gpu ---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; }"
    ;;
  fetch)
    rsync -avz --progress "$REMOTE:$REMOTE_DIR/results/" results_remote/
    rsync -avz --progress "$REMOTE:$REMOTE_DIR/runs/" runs_remote/ 2>/dev/null || true
    echo "fetched into results_remote/ and runs_remote/ (kept separate from local results/)"
    ;;
  stop)
    $SSH "pkill -f 'stages/s1_train_ddpm' 2>/dev/null; pkill -f run_experiments.sh 2>/dev/null; echo stopped"
    ;;
  *)
    echo "unknown command: $cmd (check|push|setup|launch|status|fetch|stop)"; exit 1
    ;;
esac
