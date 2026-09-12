# gummy_smile_v4

Clean rewrite of the gingival-display (gummy smile) measurement and decision-support pipeline
of `gummy_smile_v3`, built for the manuscript revision. v3 is kept read-only as a reference;
v4 fixes the measurement geometry, class handling, units and rule engine documented in
`docs/GummySmile_v3_Teknik_Audit_Raporu_v2.md`, and adds the analyses requested by the reviewers.

## What

- **Data layer** (`gsv4/io`, `gsv4/dataset`): clinical Excel parsing, name matching against the
  COCO export, cleaned dataset manifest, patient-level stratified splits with a fixed test set
  and 5-fold CV folds for the measured high-smile-line images.
- **Measurement** (`gsv4/masks`, `gsv4/measure`): class-aware binary masks (gingiva / lip,
  union of all instances, original image resolution), column-wise vertical thickness profile of
  the gingiva mask, A/B/C tooth regioning, p10/p25/median/min estimators, lip-anchored estimator,
  single unit `px_per_mm`.
- **Rule engine** (`gsv4/rules`): one engine; E1 `<4`, E2 `[3,6]`, E3 `[4,8]`, E4 `>8` mm;
  overlaps reported as combined labels (`E1-E2`, `E2-E3`); `<=0 mm -> NO_VISIBLE_GINGIVA`;
  `NaN -> UNCLASSIFIED`; no metadata-based disambiguation; output field `treatment_alternatives`.
- **Evaluation** (`gsv4/eval`): oracle validation on ground-truth masks, intra/inter-observer
  agreement (ICC, Bland-Altman, mixed models), kappa statistics with bootstrap CIs, boundary error.
- **Training** (`gsv4/train`, `scripts/train_all.sh`): YOLOv11x-seg dataset preparation, training,
  learning curve, out-of-fold prediction, test-set evaluation. Written and dry-run here, executed on
  the RTX 5090 workstation (`scripts/README_TRAINING.md`, `requirements-train.txt`).
- **Reporting** (`gsv4/report`): figures and tables for the manuscript and the revision summary.

## Why

The v3 pipeline did not measure gingival display (largest contour = lip, top-edge deviation),
merged the classes, wrote pixels into a `mean_mm` column, defaulted missing measurements to E1,
and split the data at image level although the same patient appears twice. See
`docs/OKUMA_NOTU.md` for the condensed list of errors, rules and open questions.

## Layout

```
configs/config.yaml   all paths, class names, thresholds, seed, YOLO settings
docs/                 specification documents (input) + OKUMA_NOTU.md
data/inputs/          clinical Excel/CSV inputs (versioned; no personal identifiers)
data/manifest/        generated manifest, splits
data/expert/          expert forms (arrive later; versioned)
gsv4/                 Python package
scripts/              one CLI entry point per stage
tests/                pytest, synthetic data only
outputs/              reports, figures, CSVs (not versioned)
```

## How to run

```bash
cd gummy_smile_v4
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
# Stage entry points (added stage by stage):
python scripts/build_manifest.py          # Stage 1 -> data/manifest/, outputs/01_data/
# python scripts/run_oracle.py            # Stage 3
# python scripts/run_expert_analysis.py   # Stage 4
```

Images are read from `../gummy_smile_v3/data/coco_dataset/` (path set once in
`configs/config.yaml`); nothing is copied and nothing is written under `gummy_smile_v3/`.

## Status

| Stage | Content | State |
|---|---|---|
| 0 | Reading, v3 review, skeleton | done |
| 1 | Data layer | done — `scripts/build_manifest.py` |
| 2 | Measurement + rule engine | done — `scripts/run_measurement_demo.py` |
| 3 | Oracle validation | — |
| 4 | Expert-agreement analysis (synthetic first) | — |
| 5 | Training pipeline (run on the workstation) | — |
| 6 | Accuracy on predicted masks | — |
| 7 | Reporting | — |
