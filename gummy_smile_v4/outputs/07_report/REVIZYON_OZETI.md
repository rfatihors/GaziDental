# REVIZYON_OZETI — reviewer items and the outputs that answer them

Legend: ✅ available now, ⏳ pending (what is needed is written in `report_status.md`). Paths are relative to `outputs/07_report/` unless absolute.

| reviewer item | answer in the revision | output |
|---|---|---|
| R1 — no system diagram | Block diagram of the corrected pipeline (Mermaid + PNG) | ✅ `figures/pipeline_block_diagram.png` |
| R1 — show segmentation outputs | Six images, GT vs predicted class masks side by side | ✅ `figures/segmentation_examples.png` |
| R2 #5 / R4 — how 1,315 images became 2,235 / 3,403 | Instances ≠ images (papillae annotated separately: 3,938 gingiva + 1,318 lip instances) and the Roboflow 2× augmentation of the train split (2,814/923 → 5,628/1,846); exact Roboflow version metadata still to be exported by the user | `tables/dataset_counts.md`, audit §5.3 |
| R2 #6 / R4 (CLAIM) — data leakage | Content-based check found 55 same-patient pairs (22 cross-split, 15 touching the original test set); one image per patient kept, patient-level re-split with a fixed test set, model retrained | ✅ `tables/dataset_counts.md` (pairs: 55) |
| R2 #10 — boundary accuracy | Boundary IoU and upper (lip-side) / lower (gingival margin) edge distance errors on the test set | ✅ `figures/boundary_error.png` |
| R2 / R4 — G*Power calculation inappropriate | Calculation removed; learning curve for data adequacy + precision-based justification for the agreement analysis | ✅ `figures/learning_curve.png`; precision note: With n = 145 reference images, an ICC of 0.86 has a 95 % CI half-width of ≈ 0.043 (Bonett 2002, k = 2); each Bland–Altman limit of agreement has a half-width of ≈ 0.20 mm for the observed between-method SD of 0.71 mm (Bland & Altman 1999). |
| R3 — demographics / sex distribution | Age/sex coverage per group on the cleaned dataset ("n with a record") | ✅ `tables/demographics.md` |
| R4 — calibration (px/mm) | Reference: per-image 1 mm probe interval in ImageJ on 2698×1799 copies; pipeline: single global scale fitted on the dev subset (C_p25), per-image scale variability and expert-entered scales reported | `outputs/03_oracle/scale_estimation.md`, `outputs/04_expert/scale_agreement.md` (expert part ⏳ real forms) |
| R4 — circular E/T validation | Blinded three-expert evaluation (majority reference, linear-weighted κ primary, OOF predictions primary set); E4 not validated (no case) | ⏳ `tables/expert_agreement.md` |
| R4 — overlap with the J Dent 2026 cohort | Name matching: 149 of 216 high-smile-line images have a reference measurement from the earlier study's measurement file; clinical confirmation pending | `outputs/01_data/parse_report.md` |
| R4 — external validity claim | Removed; single-centre, single-device limitation stated | manuscript text (Guc_analizi §2) |
| Figure 6 inconsistency | Explained as a corrected software error (lip contour top-edge deviation, pixels reported as mm; v3 vs v4 on the same masks) and replaced by the corrected measurement vs reference figure | `outputs/02_measure/v3_vs_v4.md`, ✅ `figures/measurement_gt_masks.png`, ✅ `figures/measurement_predicted_masks.png` |
| Intra-observer reliability (20 images) | ICC(2,1) 0.995 tooth level / 0.998 image level, SD 0.17 mm | ✅ `outputs/03_oracle/intra_observer.md` |
| Segmentation performance, clean split | Per-class box/mask mAP@50, mAP@50–95, P, R, F1 and confusion matrix on the fixed test set | ✅ `tables/segmentation_metrics_test.md` |
