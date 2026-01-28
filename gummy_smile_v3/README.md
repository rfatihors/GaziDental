# GummySmile v3

GummySmile v3 integrates YOLO segmentation with V1 (XGBoost) outputs to
produce automated gingival measurements, model comparisons, and clinical
recommendations. The project now focuses on **inference + comparison** only;
no retraining is required beyond supplying the `best.pt` weight file.

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

1. **YOLOv11x-seg weights**
   - Copy `best.pt` into `yolo/weights/best.pt`.

2. **Inference images**
   - Place inference images under `data/raw_images/`.

3. **Manual measurements (ground truth)**
   - Save clinician measurements to `data/manual_measurements/manual_measurements.csv`.
   - Required columns: `patient_id`, `mean_mm`.

4. **V1 (XGBoost) predictions**
   - Save V1 predictions to `methods/v1/xgboost_predictions.csv`.
   - Required columns: `patient_id`, `predicted_mean_mm`.

5. **Smileline labels (optional)**
   - Save labels in `data/labels_smileline.csv` with `patient_id` and
     `smileline_type` (or `smileline`) columns.

6. **Severity labels (optional)**
   - The pipeline can ingest severity metadata from COCO annotations (defaults
     to `data/coco_dataset/**/_annotations.coco.json`) or a CSV such as
     `data/labels_severity.csv`.
   - CSV columns: `patient_id` + one of `severity`, `severity_label`, or `label`.

## Running the Pipeline

```bash
python master_pipeline_v3.py --config configs/config.yaml
```

The pipeline performs:

1. **YOLOv11x-seg inference + measurements**
   - `yolo/infer_yolo_seg.py` runs inference and writes
     `results/yolo_measurements.csv`.
   - Images with no predicted mask are kept with `status=no_mask` and `mean_mm=NaN`.

2. **Model comparison**
   - `evaluation/method_comparison.py` computes MAE, RMSE, ICC, and Bland-Altman
     metrics for:
     - V1 (XGBoost) vs manual
     - V3 (YOLO) vs manual
     - V1 vs V3

3. **Etiology + treatment recommendation**
   - `methods/v3/diagnosis.py` assigns etiology/treatment codes (E1–E4, T1–T4)
     based on mean gingival display and writes `results/diagnosis_recommendations.csv`.
   - Rule precedence (derived from the clinical table):  
     `<4 mm → E1`, `4–6 mm → E2`, `6–8 mm → E3`, `>8 mm → E4`.
   - The diagnosis output also includes `severity_ground_truth` (if metadata is
     available) and `severity_predicted` (rule-based from `mean_mm`).

4. **Intra-observer analysis**
   - `evaluation/intra_observer.py` generates `results/intra_observer_report.csv`
     when calibration files exist.

## Outputs

- `results/yolo_predictions/yolo_predictions.csv`: image-to-mask mapping.
- `results/yolo_measurements.csv`: YOLO-derived measurements.
- `results/v1_vs_v3_summary.csv`: comparison summary table.
- `results/v1_vs_v3_by_smileline.csv`: comparison by smileline type.
- `results/diagnosis_recommendations.csv`: etiology and treatment recommendations.
- `results/intra_observer_report.csv`: intra-observer reliability report.
