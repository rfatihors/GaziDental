# Training on the workstation (RTX 5090)

Everything below runs on the Linux workstation. The Mac side only writes and dry-runs the
code; nothing here is executed on the Mac.

## 1. One-time setup

```bash
git clone https://github.com/rfatihors/GaziDental.git    # or: cd GaziDental && git pull origin master
cd GaziDental/gummy_smile_v4
python3.11 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Blackwell (RTX 5090) needs the CUDA 12.8 build of PyTorch (`torch >= 2.7`, `cu128`). Verify
before anything else — the capability must be `(12, 0)`:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

If you see `CUDA error: no kernel image is available`, the wheel is not a cu128 build; reinstall
torch/torchvision with the index URL above.

The COCO export must be present at `../gummy_smile_v3/data/coco_dataset/` (the repo path used
by `configs/config.yaml`); the dataset is built as symlinks into `data/yolo_dataset/`, nothing is
copied. `data/manifest/` (manifest + splits) comes from git.

Internet is needed once: Ultralytics downloads the pretrained weights `yolo11x-seg.pt` on the
first training call (about 120 MB). Afterwards the file is cached in the working directory.

## 2. Dry run (no GPU needed)

```bash
python -m gsv4.train.prepare_yolo_dataset          # builds data/yolo_dataset, prints counts, writes the label-check figure
python -m gsv4.train.train --name final --dry-run
python -m gsv4.train.train --name smoke --epochs 1 --fraction 0.1 --dry-run
python -m gsv4.train.learning_curve --fraction 0.25 --dry-run
python -m gsv4.train.cv_predict --fold 0 --dry-run
python -m gsv4.train.evaluate_test --dry-run
```

Every dry run prints the resolved training arguments (from `configs/config.yaml`,
`yolo.train`) and checks that the yaml, the list files, the symlinks and the labels exist.

## 3. Smoke test (about 5 minutes, before the overnight run)

```bash
python -m gsv4.train.train --name smoke --epochs 1 --fraction 0.1 --predict-check 5
```

This trains one epoch on a random 10 % of the training list (validation on 10 % of the
validation list), then predicts five validation images with `retina_masks=True` through
the same code path that `cv_predict` uses. It proves on the real GPU that the cu128 torch
build, Ultralytics, the pretrained-weight download and the class-aware mask extraction
work. Check:

* the header printed at the start shows the RTX 5090 with capability `(12, 0)`;
* the last line reads `predict-check: 5 images, mask_source = {'yolo:masks.data': 5} -> OK`;
* `outputs/05_predictions/smoke/smoke_predictions.csv` has `mask_source = yolo:masks.data` in every row
  and `smoke/<image>_gingiva.png` / `_lip.png` exist at the original image size.

Smoke artefacts (`runs/smoke/`, `outputs/05_predictions/smoke/`, `data/yolo_dataset/data_smoke.yaml`)
are git-ignored. Delete `runs/smoke` if you want to repeat the test.

## 4. Run everything (overnight)

```bash
mkdir -p logs
tmux new -s train           # or: nohup scripts/train_all.sh > logs/train_all.out 2>&1 &
source .venv-train/bin/activate
scripts/train_all.sh
```

* 9 trainings (`final`, `lc25`, `lc50`, `lc75`, `fold0`…`fold4`), 100 epochs each, batch 16,
  imgsz 640, AdamW, cosine LR, patience 20 — roughly 1–1.5 h each on a 5090.
* **Restartable:** the `DONE` marker in `runs/<name>/` is written **by the Python step itself,
  only after it finished successfully** (training: after `best.pt` exists and the artefacts
  were copied; prediction: after all masks were written). The shell wrapper never writes it,
  and `set -o pipefail` makes a failing Python step fail the `tee` pipeline. On re-run,
  steps with a marker are skipped; a training interrupted mid-way resumes from
  `runs/<name>/weights/last.pt` with its saved arguments.
* Logs: `logs/<step>.log`. Every training starts by logging the GPU name, torch/ultralytics
  versions, git commit hash and a copy of the config; the same goes to
  `outputs/05_predictions/<name>/{environment.json, commit_hash.txt, config_used.yaml}`.
* A failing step stops the script; the error is at the end of its log.

What the steps produce:

| step | run dir | small artefacts (versioned) |
|---|---|---|
| final | `runs/final/` (best.pt, plots) | `outputs/05_predictions/final/{results.csv,args.yaml,commit_hash.txt,environment.json,config_used.yaml}` |
| lc25/50/75 | `runs/lc*/` | `outputs/05_predictions/lc*/…` + `learning_curve.csv/.png/.md` |
| fold0..4 | `runs/fold*/` | `outputs/05_predictions/fold*/…` + `oof/<image>_gingiva.png`, `oof/<image>_lip.png`, `oof/oof_predictions.csv` |
| eval | `runs/eval/` | `test_metrics.json`, `confusion_matrix.png`, `boundary_error.csv`, `test/<image>_{gingiva,lip}.png`, `test/test_predictions.csv` |

Fold models predict **only their own held-out fold**; the test set is predicted by the final
model only.

**Prediction memory.** `retina_masks=True` keeps one full-resolution float mask per instance,
so a whole list in one `predict` call runs out of CUDA memory (the 192-image test set did,
with 7–15 GB allocations). Both `cv_predict` and `evaluate_test` therefore predict in chunks
of `yolo.predict_batch` images (`configs/config.yaml`, default 8; `--batch N` overrides),
stream the results, write every mask to disk as it arrives and empty the CUDA cache after
each chunk; `evaluate_test` also releases the validation model before loading a separate
prediction model. `scripts/train_all.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
On a `CUDA out of memory` the step stops with the message *predict_batch değerini düşürün* —
lower `yolo.predict_batch` (1 is always safe) or re-run the step with `--batch 2`; masks
already written are kept and `DONE` is only written after the whole list.

**The `eval` step has three stages** (validation, prediction, boundary errors), each
restartable on its own:

1. `model.val` -> the per-class block of `test_metrics.json`, written **before** prediction
   starts, so a later failure never costs the validation pass;
2. prediction -> `test/<image>_{gingiva,lip}.png` and `test/test_predictions.csv`. A complete
   table already on disk is reused instead of predicting the 192 images again — pass
   `--repredict` to force a fresh prediction;
3. boundary errors and the summary tables -> `boundary_error.csv`, the `boundary_*` keys of
   `test_metrics.json`, and only then `runs/eval/DONE`.

```bash
python -m gsv4.train.evaluate_test --metrics-only    # stage 3 only, ~40 s, no GPU
```

`--metrics-only` recomputes `boundary_error.csv` and the summary from the predictions already
on disk, so a failure in the summary never means re-predicting the test set. It writes `DONE`
only when `test_metrics.json` already carries the `per_class` block (i.e. validation has run);
otherwise it says so and leaves the step incomplete.

Ground truth is addressed by **`uid` (= `group/stem`)**: the image stem repeats across groups
(iPhone numbering), so it is never a join key, and the group and the COCO source split are
read as `group` / `orig_split` columns from `data/manifest/dataset_manifest.csv` rather than
parsed out of a path string.

## 5. After training — what to commit

Only the small artefacts listed above go into git (`.gitignore` already allows PNG masks,
`results.csv`, `args.yaml`, json/csv/md under `outputs/`); `runs/` and `*.pt` never do.

```bash
cd GaziDental/gummy_smile_v4
git status --short outputs/05_predictions | head          # sanity: only png/csv/json/yaml/md/txt
du -sh outputs/05_predictions                             # expected: a few tens of MB
git add outputs/05_predictions
git commit -m "gummy_smile_v4 Stage 5: training outputs (final, learning curve, 5-fold OOF masks, test metrics)"
git push origin master
```

Then on the Mac: `git pull origin master` and continue with Stage 6
(`python scripts/run_prediction_eval.py`: fixed method and scale from Stage 3 on the OOF and
test masks; outputs in `outputs/06_prediction/`).

Keep `runs/final/weights/best.pt` on the workstation (and a copy on Drive); it is the model
of record for the manuscript together with `outputs/05_predictions/final/commit_hash.txt`.


## 6. Architecture comparison (Reviewer 4) — after the main run

Pre-registration: `outputs/08_architecture/PROTOCOL.md`. Read it first; it fixes the protocol, the
outcomes and the decision rule, and it was committed before any of these commands were run.

### 6.1 Re-evaluate the final model at the standard settings

The mAP reported in the manuscript must come from the standard evaluation convention, not from the
pipeline's operating point. Delete the marker and re-run; predictions already on disk are reused, so
only the two validation passes are recomputed.

```bash
rm runs/eval/DONE
python -m gsv4.train.evaluate_test          # writes per_class_standard and per_class_operating_point
```

### 6.2 YOLO families, published defaults, three seeds

```bash
python scripts/run_architecture_comparison.py --dry-run     # check paths and the 6 runs
nohup python scripts/run_architecture_comparison.py > logs/arch.out 2>&1 &
```

Six runs (yolo11x-seg and yolo26x-seg × seeds 42, 43, 44). Each is skipped once its per-image table
exists, so the script can be re-run after an interruption. Expect roughly 15 to 25 minutes per run
on the RTX 5090 (the final model of the main study took 16.4 minutes for 91 epochs).

### 6.3 RF-DETR-Seg, in its own virtualenv

A separate environment is mandatory: `rfdetr` pulls transformers 5.x, pytorch_lightning and a pinned
`torch-hungarian` release candidate, and must not be allowed to change torch under the model of
record.

```bash
python scripts/build_rfdetr_dataset.py --dry-run     # counts must match the manifest
bash scripts/rfdetr_setup.sh                         # .venv-rfdetr, install, compatibility report
python scripts/build_rfdetr_dataset.py --verify-with .venv-rfdetr/bin/python
.venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --probe    # 2 epochs, cost projection
```

The builder converts every numeric field explicitly: the v3 export writes the width and height of
each bbox as a formatted string (`[655, 525, '1300.0000', '170.0000']`), which our own pipeline never
sees because it rasterises `segmentation` and ignores `bbox`. `--verify-with` re-opens the written
files with the RF-DETR interpreter, through `pycocotools` and through the exact tensor step that
failed before, so the same fault cannot reach a training run again.

Check `outputs/08_architecture/rfdetr_environment.json` and `rfdetr_probe.json` before committing to
the full run. If the install or the probe fails, that is a result: record it as an amendment in
`PROTOCOL.md` and report the architecture as not evaluable under controlled conditions. If it
succeeds and the projected cost is acceptable:

Measured on the probe: 331 s per epoch, 9.2 h for the full 100-epoch budget, peak 15 GB, mask
mAP@50 of 0.756 after two epochs. Early stopping is on by default with the same patience as the YOLO
runs (`--patience 20`, monitoring `val/segm_mAP_50_95`), so a run normally stops well short of that
budget; see PROTOCOL.md Amendment 1.

```bash
for s in 42 43 44; do .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed $s; done
python scripts/run_architecture_comparison.py --measure-only   # measures the RF-DETR masks
```

Always check the classes before measuring, whichever predictor produced the masks:

```bash
python scripts/check_mask_classes.py --masks outputs/08_architecture/masks/rfdetr-seg-large_s42     --figure outputs/08_architecture/rfdetr_class_check.png
```

It cross-matches each predicted class against both annotated classes. A predicted gingiva that
matches the annotated **lip** better means the label space was read wrongly, and the exit code is
non-zero.

### 6.5 If the classes were read wrongly

This happened in the first comparison run and is worth knowing by shape. RF-DETR renumbers its label
space: `filter_parent_categories` drops the unannotated Roboflow grouping category and the survivors
get contiguous indices, so our export (`0 dudak-diseti`, `1 diseti`, `2 dudak`) trains a model that
emits **0 for gingiva and 1 for lip**. Reading those ids through the dataset's own category table
mapped gingiva onto the grouping name, which has no role and was dropped, and mapped lip onto
gingiva. The saved gingiva mask was the lip band, no lip mask was written at all, and the measurement
showed a 6 mm gingival edge bias while RF-DETR's own mask mAP stayed at 0.80, because the model was
never wrong — only our reading of it was.

The prediction step now takes the class list from the model (`dict(enumerate(model.class_names))`),
cross-checks it against the categories the dataset implies, and runs the adapter in strict mode, so
an instance that maps to no role aborts instead of disappearing. Every prediction table also carries
an `n_ignored` column.

**Recovery does not need retraining.** The weights are unaffected; only the masks are wrong, and the
gingiva instance was dropped before anything was written, so the masks must be produced again rather
than relabelled on disk:

```bash
rm -rf outputs/08_architecture/masks/rfdetr-seg-large_s*        outputs/08_architecture/predictions/rfdetr-seg-large_s*        outputs/08_architecture/per_image/rfdetr-seg-large_s*
for s in 42 43 44; do .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed $s --predict-only; done
python scripts/check_mask_classes.py --masks outputs/08_architecture/masks/rfdetr-seg-large_s42     --figure outputs/08_architecture/rfdetr_class_check.png
python scripts/run_architecture_comparison.py --measure-only
python scripts/run_architecture_comparison.py --aggregate-only
```

`--predict-only` loads `checkpoint_best_total.pth` from the run directory, so this is minutes rather
than the 9 hours a retraining would cost.

### 6.4 Aggregate and report

```bash
python scripts/run_architecture_comparison.py --aggregate-only
```

Writes `by_seed.csv`, `by_model.csv`, `edges_by_seed.csv`, `paired_comparisons.csv`,
`segmentation_metrics_all.csv` and `RESULTS.md`, including the pre-registered decision. Commit the
tables and the results; the masks are git-ignored and are rebuilt by re-running the comparison.


## 7. Resolution controls (Addendum 2)

Read `outputs/08_architecture/PROTOCOL_ADDENDUM_resolution.md` first. It was committed before either
control ran and fixes how the results are to be read, including which configuration becomes the final
model in each case. The controls are sensitivity analyses, not members of the comparison: §2 of the
protocol held the input resolution fixed and these runs deliberately break it, so their rows are
labelled `<model>@<size>` and kept in their own section.

The measured quantity is a vertical thickness, so vertical mask-pixel size is what matters:

| configuration | mask grid | one mask pixel, vertically |
|---|---|---|
| yolo11x-seg at 640 (comparison) | 160 | 1.00 mm |
| yolo11x-seg at 1024 (control A) | 256 | 0.63 mm |
| rfdetr-seg-large at 624 (comparison) | 156 | 0.68 mm |
| rfdetr-seg-large at 432 (control B) | 108 | 0.99 mm |

### 7.1 Control A — YOLOv11x at imgsz 1024

```bash
python scripts/run_architecture_comparison.py --models yolo11x-seg --imgsz 1024 --dry-run
nohup python scripts/run_architecture_comparison.py --models yolo11x-seg --imgsz 1024 > logs/ctlA.out 2>&1 &
```

Three seeds, roughly 40 to 60 minutes each on the RTX 5090 (compute scales with the pixel count, so
about 2.6 times the 640 runs). If batch 16 does not fit at 1024, lower it once and record the value:
a changed batch size is a deviation and belongs in the addendum.

### 7.2 Control B — RF-DETR at resolution 432

Same variant, same weights, only the input resolution changes.

```bash
for s in 42 43 44; do .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed $s --resolution 432; done
python scripts/check_mask_classes.py --masks outputs/08_architecture/masks/rfdetr-seg-large@432_s42     --figure outputs/08_architecture/rfdetr432_class_check.png
python scripts/run_architecture_comparison.py --measure-only
```

Roughly half the cost of the 624 runs (about 100 s per epoch there, and compute scales with the pixel
count). Check the classes before measuring, as always.

### 7.3 Aggregate

```bash
python scripts/run_architecture_comparison.py --aggregate-only
```

`RESULTS.md` gains a Resolution controls section with the mask-pixel geometry, both paired
comparisons and the verdict under the pre-registered rules of the addendum. The comparison tables and
the §8 decision are computed from the members only and do not change.


## 8. Stage 6 with RF-DETR (Addendum 2, rule 4)

Both resolution controls pointed to the architecture, so rule 4 of
`outputs/08_architecture/PROTOCOL_ADDENDUM_resolution.md` applies and Stage 6 is repeated with
RF-DETR-Seg Large at 624. Read `outputs/09_final_rfdetr/PLAN.md` first: it was committed before any
of this ran and fixes the configuration, the primary outcome, its pre-registered sensitivity
analysis, and the threshold below which the offset correction is dropped.

```bash
tmux new -s rfdetr
bash scripts/train_final_rfdetr.sh 2>&1 | tee -a logs/final_rfdetr.out
```

Eight trainings (five folds, three learning-curve subsets) plus a prediction pass for the final
model, which is **reused** from the architecture comparison rather than retrained (PLAN.md §1). At
roughly 100 s per epoch with early stopping the folds are the bulk of it; budget an overnight run.
Every step writes `runs/rfdetr/<name>/DONE` and is skipped on a re-run, so an interruption costs
only the step that was in flight.

The class check runs after every prediction and aborts the script on a mismatch. That is deliberate:
a wrongly read label space is invisible in the segmentation metrics and has already cost one full
comparison run.

Afterwards, in the training venv:

```bash
# 0. the final model's own COCO evaluation (PLAN.md §4) — it is reused, not retrained, and
#    --predict-only alone does not evaluate, so without this it has no segmentation metrics
.venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --variant main --seed 42 --predict-only --evaluate

# 1. measurement on the RF-DETR out-of-fold masks; fallback rate of C_p25 (PLAN.md §7)
python scripts/run_oracle.py --masks outputs/05_predictions/oof_rfdetr --out 09_final_rfdetr/oracle

# 2. primary outcome and its pre-registered sensitivity analysis (PLAN.md §5)
python scripts/run_prediction_eval.py --oof-masks outputs/05_predictions/oof_rfdetr \
    --test-masks outputs/05_predictions/test_rfdetr --out 09_final_rfdetr \
    --exclude-uids outputs/08_architecture/arch_test_high_uids.csv

# 3. offset re-estimated from scratch for this configuration (PLAN.md §6); the config offset
#    was fitted on YOLO masks, so it is not the one whose effect is reported here
python scripts/run_offset_correction.py --out 09_final_rfdetr \
    --oof-masks outputs/05_predictions/oof_rfdetr --test-masks outputs/05_predictions/test_rfdetr \
    --adopted-offset-px 0

# 4. learning curve of this model, from its own COCO evaluations (PLAN.md §3); the 100 % point
#    needs the final model evaluated on the same split as the subsets, in the RF-DETR venv:
.venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --variant main --seed 42 \
    --predict-only --evaluate --evaluate-split val
python scripts/build_rfdetr_learning_curve.py --out 09_final_rfdetr

# 5. manuscript figures and tables from THIS model (Stage 7)
python scripts/build_report.py --stage6 09_final_rfdetr

# 6. the comparison's closing table, which reads both Stage-6 runs
python scripts/run_architecture_comparison.py --aggregate-only
```

**The offset was dropped (PLAN.md Amendment 3, 23 Sep 2026).** `configs/config.yaml` now has
`bottom_edge_offset_px: 0`, so Stage 6 produces **one** result set and no uncorrected/corrected pair.
The YOLOv11x appendix stays reproducible with `python scripts/run_prediction_eval.py --offset-px -13`
(its own masks, `--out 06_prediction`), and `scripts/build_report.py --stage6 06_prediction` rebuilds
the previous final model's report.

The measurement method (`C_p25`) and the scale (16.84 px/mm) do not change: they were selected on
ground-truth masks in Stage 3 and are a property of the measurement geometry, not of the segmentation
model. `run_oracle.py --masks` therefore **takes both from `configs/config.yaml` and selects nothing**;
it never writes to the config, and `--reselect` (off by default) is the only way to re-run the selection
on predicted masks — a sensitivity analysis, never the primary result. The offset correction, by
contrast, is re-estimated from scratch for this configuration and is expected to be dropped
(PLAN.md §6); step 3 fits it on the Stage-3 dev subset with the method held fixed.

**Where the numbers come from.** `run_prediction_eval.py` reads the model behind each mask directory
out of the prediction table itself and stops if it is ambiguous (no `mask_source`, two predictor
families in one table, an RF-DETR table without its `model` column, several models where the final
model should be one). It then computes the boundary and IoU tables **from the masks it was given**,
writes the model and the mask directory into every table, and takes detection metrics only from the
evaluator that belongs to those masks — Ultralytics `val()` for YOLO, RF-DETR's own COCO evaluation
(`iouType="segm"`) for RF-DETR, never one in place of the other (PLAN.md §4). If that evaluator has
not run, `segmentation_metrics.md` stays empty and says what to run. The YOLOv11x numbers remain
available as rows explicitly marked `[reference] YOLOv11x (previous final model)`, computed on its
own masks and never merged into the new model's rows.
