#!/usr/bin/env bash
# Seed spread of the reported measurement — the whole sequence, in order, restartable.
# Pre-registration: outputs/09_final_rfdetr/PLAN.md, Amendment 5. Read it before running this.
#
#   tmux new -s spread
#   bash scripts/train_seed_spread.sh 2>&1 | tee -a logs/seed_spread.out
#
# 10 trainings (5 folds x seeds 43 and 44), about 30 hours on the RTX 5090. Seed 42 is NOT re-run:
# its out-of-fold masks already exist in outputs/05_predictions/oof_rfdetr and they are the reported
# ones. Nothing here changes the final model; Amendment 5 A5.3 fixes that in advance.
#
# Each step is skipped when runs/rfdetr/<name>/DONE exists, so an interrupted run is resumed by
# simply re-running the script. A failing step stops it (set -e); fix, then re-run.
# The class check runs after every prediction and aborts on a mismatch: the label-space fault of
# PROTOCOL.md Amendment 2 was invisible in the metrics and visible only in the measurement, which is
# exactly what this script produces.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs/rfdetr outputs/10_seed_spread
PY=${PY:-python}
RFPY=${RFPY:-.venv-rfdetr/bin/python}
SEEDS=${SEEDS:-"43 44"}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*"; }
done_marker() { echo "runs/rfdetr/$1/DONE"; }

step() {                       # step <name> <command...>
  local name=$1; shift
  if [ -f "$(done_marker "$name")" ]; then
    log "SKIP $name (DONE marker present)"
    return 0
  fi
  log "START $name"
  "$@" 2>&1 | tee -a "logs/rfdetr_$name.log"
  mkdir -p "runs/rfdetr/$name"
  date '+%F %T' > "$(done_marker "$name")"
  log "END $name"
}

check_classes() {              # check_classes <mask dir> <figure>   — aborts the script on a mismatch
  log "class check on $1"
  $PY scripts/check_mask_classes.py --masks "$1" --n 6 --figure "$2"
}

log "git $(git rev-parse --short HEAD)"
$RFPY -c "import rfdetr, torch; print('rfdetr', getattr(rfdetr, '__version__', '?'), '| torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 0. datasets (idempotent; already built for seed 42)
for v in fold0 fold1 fold2 fold3 fold4; do
  step "dataset_$v" $PY scripts/build_rfdetr_dataset.py --variant "$v"
done

# ---------------------------------------------------------------- 1. five folds per seed -> its own out-of-fold set
# Same folds, same datasets, same resolution, same published defaults, same early stopping as seed 42
# (PLAN.md 2). Only --seed and --masks-out differ, and --masks-out is what keeps the seeds apart.
for s in $SEEDS; do
  MASKS="outputs/05_predictions/oof_rfdetr_s${s}"
  mkdir -p "$MASKS"
  for k in 0 1 2 3 4; do
    step "fold${k}_s${s}" $RFPY scripts/rfdetr_train_predict.py \
      --variant "fold${k}" --seed "$s" --masks-out "$MASKS"
    check_classes "$MASKS" "outputs/10_seed_spread/class_check_fold${k}_s${s}.png"
  done
  log "seed $s done: $(ls "$MASKS"/*_gingiva.png 2>/dev/null | wc -l) gingiva masks"
done

log "ALL TRAININGS DONE"
log "now, in the training venv (not .venv-rfdetr):"
log "  $PY scripts/run_seed_spread.py --seeds 42 $SEEDS   # PLAN.md Amendment 5"
log "then commit outputs/10_seed_spread/ and re-run the report if its numbers are quoted:"
log "  $PY scripts/build_report.py --stage6 09_final_rfdetr"
log "  $PY scripts/build_rebuttal.py && $PY scripts/build_manuscript_edits.py"
