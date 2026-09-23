#!/usr/bin/env bash
# Stage 6 with RF-DETR-Seg Large — the whole sequence, in order, restartable.
# Pre-registration: outputs/09_final_rfdetr/PLAN.md. Read it before running this.
#
#   tmux new -s rfdetr
#   bash scripts/train_final_rfdetr.sh 2>&1 | tee -a logs/final_rfdetr.out
#
# Each step is skipped when its completion marker runs/rfdetr/<name>/DONE exists, so the script can
# simply be re-run after an interruption. A failing step stops the script (set -e); fix, then re-run.
# The class check runs after every prediction and aborts on a mismatch, because the label space is
# the one thing that has already gone wrong once and it is invisible in the metrics.
#
# 9 trainings: fold0..4, lc25, lc50, lc75, and the final model is REUSED from the architecture
# comparison (PLAN.md §1) rather than retrained.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs/rfdetr outputs/09_final_rfdetr
PY=${PY:-python}
RFPY=${RFPY:-.venv-rfdetr/bin/python}
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

check_classes() {              # check_classes <mask dir> <figure>
  log "class check on $1"
  $PY scripts/check_mask_classes.py --masks "$1" --n 6 --figure "$2"
}

log "git $(git rev-parse --short HEAD)"
$RFPY -c "import rfdetr, torch; print('rfdetr', getattr(rfdetr, '__version__', '?'), '| torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 0. datasets (idempotent)
for v in main fold0 fold1 fold2 fold3 fold4 lc25 lc50 lc75; do
  step "dataset_$v" $PY scripts/build_rfdetr_dataset.py --variant "$v"
done
log "verifying the built datasets with the RF-DETR interpreter"
$PY scripts/build_rfdetr_dataset.py --variant main --verify-with "$RFPY" >/dev/null

# ---------------------------------------------------------------- 1. five folds -> out-of-fold masks
for k in 0 1 2 3 4; do
  step "fold${k}" $RFPY scripts/rfdetr_train_predict.py --variant "fold${k}" --seed 42
  check_classes outputs/05_predictions/oof_rfdetr "outputs/09_final_rfdetr/class_check_fold${k}.png"
done

# ---------------------------------------------------------------- 2. learning curve
for f in 25 50 75; do
  step "lc${f}" $RFPY scripts/rfdetr_train_predict.py --variant "lc${f}" --seed 42 --no-predict
done

# ---------------------------------------------------------------- 3. the final model: reuse, predict the test set
# PLAN.md §1: the seed-42 run of the architecture comparison IS the final model. It is not retrained;
# only its test-set masks and its own COCO evaluation are produced here.
step final_predict $RFPY scripts/rfdetr_train_predict.py --variant main --seed 42 --predict-only
check_classes outputs/05_predictions/test_rfdetr outputs/09_final_rfdetr/class_check_test.png

log "ALL DONE"
log "next, in THIS venv (.venv-rfdetr), the final model's own COCO evaluation (PLAN.md 4):"
log "  $RFPY scripts/rfdetr_train_predict.py --variant main --seed 42 --predict-only --evaluate"
log "then, in the training venv:"
log "  $PY scripts/run_oracle.py --masks outputs/05_predictions/oof_rfdetr --out 09_final_rfdetr/oracle   # C_p25 fallback rate, PLAN.md 7 (method and scale fixed from config; no re-selection)"
log "  $PY scripts/run_prediction_eval.py --oof-masks outputs/05_predictions/oof_rfdetr \\"
log "      --test-masks outputs/05_predictions/test_rfdetr --out 09_final_rfdetr \\"
log "      --exclude-uids outputs/08_architecture/arch_test_high_uids.csv   # PLAN.md 5"
log "  $PY scripts/run_offset_correction.py --out 09_final_rfdetr \\"
log "      --oof-masks outputs/05_predictions/oof_rfdetr --test-masks outputs/05_predictions/test_rfdetr \\"
log "      --adopted-offset-px 0   # offset re-estimation, PLAN.md 6"
