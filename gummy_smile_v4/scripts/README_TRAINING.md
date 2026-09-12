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
python -m gsv4.train.learning_curve --fraction 0.25 --dry-run
python -m gsv4.train.cv_predict --fold 0 --dry-run
python -m gsv4.train.evaluate_test --dry-run
```

Every dry run prints the resolved training arguments (from `configs/config.yaml`,
`yolo.train`) and checks that the yaml, the list files, the symlinks and the labels exist.

## 3. Run everything (overnight)

```bash
mkdir -p logs
tmux new -s train           # or: nohup scripts/train_all.sh > logs/train_all.out 2>&1 &
source .venv-train/bin/activate
scripts/train_all.sh
```

* 9 trainings (`final`, `lc25`, `lc50`, `lc75`, `fold0`…`fold4`), 100 epochs each, batch 16,
  imgsz 640, AdamW, cosine LR, patience 20 — roughly 1–1.5 h each on a 5090.
* **Restartable:** each step writes `runs/<name>/DONE` when it finishes; on re-run finished
  steps are skipped. After an interruption just run `scripts/train_all.sh` again.
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

## 4. After training — what to commit

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
