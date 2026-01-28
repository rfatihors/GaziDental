# GummySmile v3

GummySmile v3 runs a **single-image** pipeline that combines YOLOv11x-seg
gingival segmentation with a legacy **V1 (XGBoost + pixel counting)** approach.
It outputs gingival visibility measurements, etiology/treatment recommendations,
and optional evaluation metrics under `gummy_smile_v3/results/`.

## Repository Layout

```
gummy_smile_v3/
├── configs/              # Config files (paths + hyper-parameters)
├── data/                 # Roboflow dataset + manual measurements
├── methods/              # V1 (XGBoost) + V3 (YOLO) method helpers
├── yolo/                 # YOLO inference helpers
├── measurement/          # Measurement extraction + metrics
├── evaluation/           # Evaluation helpers
├── master_pipeline_v3.py # Single-command pipeline
└── README.md
```

## Configuration

Edit `configs/config.yaml` for defaults:

- `weights_path`: optional (can be overridden by `--weights`).
- `px_per_mm`: pixels-per-mm calibration. If missing, pipeline reports **px-only** measurements
  and skips etiology/treatment classification (rules are defined in mm).
- `output_dir`: base output directory (run subfolders are created automatically).
- `metadata_source`: `filename`, `csv`, `yaml`, or `none`.
- `ambiguous_policy`: how to resolve the 4–6 mm overlap (E2 vs E3).

## Running the Pipeline

```bash
python master_pipeline_v3.py \
  --image data/raw_images/high/IMG_7285.jpg \
  --weights /path/to/best.pt
```

Outputs are written to `results/run_YYYYMMDD_HHMMSS/`:

- `v3_yolo_predictions.csv`
- `v1_xgboost_predictions.csv`
- `report.json`
- `evaluation.json` (if manual measurements exist)
- overlay/mask images in the same run folder

### Optional: stub mode (for smoke tests)

```bash
python master_pipeline_v3.py \
  --image data/raw_images/high/IMG_7285.jpg \
  --stub-model
```

This bypasses YOLO weights with a synthetic mask; use only for tests.

## Manual Measurements (Optional)

Place manual measurements at `data/manual_measurements/manual_measurements.csv`
or an Excel file (e.g. `data/manual_measurements/calibration-last.xlsx`). The
first sheet should include:

- `patient_id`
- `mean_mm`

If you use the calibration spreadsheets, the loader also accepts:

- `RESİM` or `image numarası` as the patient identifier.
- Numeric measurement columns named `1`–`6` (mean is computed as `mean_mm`).

If the file is missing, evaluation is skipped and `evaluation.json` reports `SKIP`.
