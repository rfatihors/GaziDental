# GummySmile v3

GummySmile v3 integrates Roboflow YOLO segmentation with V1 (XGBoost) outputs to
produce automated gingival measurements and run model comparisons. The project
now focuses on **inference + comparison**, while training is optional and driven
by the `train_yolo_seg.py` utility.

## Repository Layout

```
'gummy_smile_v3/
├── configs/              # Config files (paths + hyper-parameters)
├── data/                 # Roboflow dataset + manual measurements
├── methods/              # V1 (XGBoost) + V3 (YOLO) method helpers
├── yolo/                 # YOLO inference + optional training scripts
├── measurement/          # Measurement extraction + metrics
├── evaluation/           # Model comparison + intra-observer analysis
├── master_pipeline_v3.py # Single-command pipeline
└── README.md
```

## Data Preparation

1. **Roboflow dataset**
   - Export the Roboflow dataset in YOLO segmentation format.
   - Place it under `data/roboflow_yolo/` following the structure in
     `data/roboflow_yolo/README.md`.

2. **YOLOv11x-seg weights**
   - Copy `best.pt` into `yolo/weights/best.pt`.

3. **Manual measurements (ground truth)**
   - Save clinician measurements to `data/manual_measurements/manual_measurements.csv`.
   - Required columns: `patient_id`, `mean_mm`.

4. **V1 (XGBoost) predictions**
   - Save V1 predictions to `methods/v1/xgboost_predictions.csv`.
   - Required columns: `patient_id`, `predicted_mean_mm`.

5. **Smileline labels (optional)**
   - Save labels in `data/labels_smileline.csv` with `patient_id` and
     `smileline_type` (or `smileline`) columns.

## Running the Pipeline

```bash
python master_pipeline_v3.py --config configs/config.yaml
```

The pipeline performs:

1. **YOLOv11x-seg inference + measurements**
   - `yolo/infer_yolo_seg.py` runs inference and writes
     `results/yolo_measurements.csv`.

2. **Model comparison**
   - `evaluation/method_comparison.py` computes MAE, RMSE, ICC, and Bland-Altman
     metrics for:
     - V1 (XGBoost) vs manual
     - V3 (YOLO) vs manual
     - V1 vs V3

3. **Intra-observer analysis**
   - `evaluation/intra_observer.py` generates `results/intra_observer_report.csv`
     when calibration files exist.

## Optional: Training

```bash
python yolo/train_yolo_seg.py \
  --data data/roboflow_yolo/data.yaml \
  --weights yolo/weights/best.pt \
  --output results/yolo_training
```

## Outputs

- `results/yolo_predictions/yolo_predictions.csv`: image-to-mask mapping.
- `results/yolo_measurements.csv`: YOLO-derived measurements.
- `results/v1_vs_v3_summary.csv`: comparison summary table.
- `results/v1_vs_v3_by_smileline.csv`: comparison by smileline type.
- `results/intra_observer_report.csv`: intra-observer reliability report.
