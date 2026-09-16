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
(`python scripts/run_oracle.py --masks outputs/05_predictions/oof --out 06_prediction`).

Keep `runs/final/weights/best.pt` on the workstation (and a copy on Drive); it is the model
of record for the manuscript together with `outputs/05_predictions/final/commit_hash.txt`.
