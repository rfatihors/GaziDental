#!/usr/bin/env bash
# Runs the whole Stage 5 training plan on the workstation, in order, restartably.
# Each step is skipped when its completion marker runs/<name>/DONE exists, so the script
# can simply be re-run after an interruption (e.g. overnight power loss). A failing step
# stops the script (set -e); fix, then re-run.
#
#   nohup scripts/train_all.sh > logs/train_all.out 2>&1 &     # or inside tmux
#
# 9 trainings: final, lc25, lc50, lc75, fold0..fold4; then learning-curve collection,
# test-set evaluation and test-set mask export.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs
PY=${PY:-python}
export PYTHONUNBUFFERED=1

step() {                       # step <name> <command...>
  local name=$1; shift
  if [ -f "runs/$name/DONE" ]; then
    echo "[$(date '+%F %T')] SKIP $name (DONE marker present)"
    return 0
  fi
  echo "[$(date '+%F %T')] START $name" | tee -a "logs/$name.log"
  "$@" 2>&1 | tee -a "logs/$name.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "logs/$name.log"
}

echo "[$(date '+%F %T')] git $(git rev-parse --short HEAD) — $($PY -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))')"

# 0. dataset (idempotent; symlinks + labels + lists)
$PY -m gsv4.train.prepare_yolo_dataset --no-figure 2>&1 | tail -3

# 1. final model on the main split
step final       $PY -m gsv4.train.train --name final

# 2. learning curve: three extra trainings (100 % = final), then collection
step lc25        $PY -m gsv4.train.learning_curve --fraction 0.25
step lc50        $PY -m gsv4.train.learning_curve --fraction 0.50
step lc75        $PY -m gsv4.train.learning_curve --fraction 0.75
step lc_collect  $PY -m gsv4.train.learning_curve --collect

# 3. five fold models, each predicting only its held-out fold (out-of-fold masks)
for k in 0 1 2 3 4; do
  step "fold${k}_predict" $PY -m gsv4.train.cv_predict --fold "$k"
done

# 4. final model on the fixed test set: metrics, confusion matrix, masks, boundary errors
step eval        $PY -m gsv4.train.evaluate_test

echo "[$(date '+%F %T')] ALL DONE — see scripts/README_TRAINING.md, section 'After training'"
