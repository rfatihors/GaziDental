# Clinically Reliable & Explainable AI Pipeline for Gummy Smile Analysis

This document specifies a geometry-aware, explainable pipeline that extends the existing DeepLabV3+ gingival segmentation. It replaces pixel-count heuristics with calibrated geometric measurements, adds severity classification, and introduces rule-based clinical decision support (CDS) with explainability hooks.

## Overview
1. **Data ingestion & cleaning**
2. **Segmentation inference (DeepLabV3+)**
3. **OpenCV geometric measurement**
4. **Automatic vs. manual validation**
5. **Severity classification (LightGBM/XGBoost)**
6. **Rule-based CDS**
7. **Explainability (tabular + visual)**
8. **Statistical reporting**

## Module 1 – Data Cleaning & Feature Engineering
- Load `olcumler_duzenlenmis.xlsx` with pandas.
- Sanitize values: replace entries containing `*`, `-`, or `(mesafe yok)` with `NaN`; cast to float.
- For each patient row compute `mean_mm`, `max_mm`, `min_mm` across six gingival measurements.
- Derive severity label from folder name:
  - `Low Smile Line` → 0
  - `Normal Smile Line` → 1
  - `High Smile Line (Gummy Smile)` → 2
- Match each image to the corresponding Excel row by filename/patient ID.

## Module 2 – Segmentation (Retained DeepLabV3+)
- Use the pretrained DeepLabV3+ to produce aligned binary masks for gingiva and teeth.
- Keep existing preprocessing (resize, normalization) and output PA ≈ 0.98 as validated.

## Module 3 – OpenCV Geometric Automatic Measurement
Goal: Replace pixel counting with calibrated geometry.

1. **Mask cleanup & contours**
   - Morphological open/close on gingival mask.
   - `cv2.findContours` (external) to get gingival boundary polygons.

2. **Zenith detection (per tooth)**
   - Split tooth mask into 6 anterior regions (using tooth contours or equal-width bins anchored to teeth mask bbox).
   - For each region, pick gingival contour point with minimum Y (highest point) = gingival zenith.

3. **Lip reference detection**
   - Estimate upper lip line as the uppermost Y of gingival boundary or use a moving horizontal hull over gingival contour.
   - Optionally refine with Sobel/edge detection on the RGB image in the upper band.

4. **Probe-based scale calibration**
   - Detect periodontal probe via color/edge filtering (hue band for metal/markings) or rectangular contour aspect ratio.
   - Measure probe pixel length; compute scale: `mm_per_px = real_probe_length_mm / probe_px_length`.

5. **Distance computation**
   - For each of 6 zones: `gummy_px = zenith_y - lip_line_y`; `gummy_mm = gummy_px * mm_per_px`.
   - Store as `mm_model_1 … mm_model_6`.

## Module 4 – Validation (Manual vs. Automatic)
- Metrics: MAE, RMSE, R², ICC(3,1); optional Bland–Altman plots.
- Evaluate per zone and aggregated (mean_mm).
- Flag outliers where |error| > 0.5 mm for targeted review.

## Module 5 – Severity Classification (ML)
- Features: `mm_model_1 … mm_model_6`, `mean_mm`, `max_mm`, optionally demographics.
- Target: severity {0,1,2} derived from folder names.
- Models: LightGBMClassifier, XGBClassifier.
- Split: train/val/test; tune via GridSearch or Optuna.
- Metrics: accuracy, macro F1, confusion matrix; export class probabilities.

## Module 6 – Rule-Based Clinical Decision Support
No supervised labels; CDS must be deterministic.

Example rules (editable with config):
- `mean_mm < 2` → conservative monitoring (category 0)
- `2 ≤ mean_mm ≤ 4` → botulinum toxin (category 1)
- `mean_mm > 4` → surgical or combined (category 2/3) depending on `max_mm` and esthetic concern threshold.

Outputs: treatment category + textual rationale with thresholds used.

## Module 7 – Explainable AI
- **Tabular XAI:** SHAP for LightGBM/XGBoost; provide global importance and patient-level force plots.
- **Visual XAI:** overlay segmentation masks, zenith points, lip line, and measurement vectors on original images; render mm values.
- **Audit log:** JSON per case capturing measurements, scale factor, rules fired, and classifier probabilities.

## Module 8 – Statistical Reporting
- Measurement agreement: MAE/RMSE/R²/ICC tables, Bland–Altman plots.
- Severity model: accuracy/F1 with 95% CI; confusion matrix.
- CDS: distribution of treatment categories.
- Export figures/tables for publication-ready appendix.

## Implementation Notes
- Keep everything in `gummy_smile_guncel` under new subpackages: `geometry`, `classification`, `cds`, `xai`, `reports`.
- Prefer OpenCV/NumPy over heavy vision foundation models; avoid external GPU dependencies for inference.
- Ensure deterministic seeds and save all artifacts (metrics, plots, logs) under a run folder (`runs/<timestamp>`).

