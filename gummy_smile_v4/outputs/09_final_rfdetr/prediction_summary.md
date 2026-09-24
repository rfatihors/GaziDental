# Stage 6 — full pipeline on predicted masks

Method **C_p25**, global scale **16.84 px/mm**, both fixed in `configs/config.yaml` from Stage 3 (ground-truth masks, dev subset). Nothing was re-selected or re-fitted on predicted masks. **There is one result set: no post-hoc calibration is applied** (`measurement.bottom_edge_offset_px = 0`). The mask-level offset of the YOLOv11x pipeline was estimated for that model and is not carried over; for this one it was re-estimated and deliberately dropped after seeing the results (`outputs/09_final_rfdetr/PLAN.md`, Amendment 3). The offset analysis remains in the appendix as a finding about the YOLO family. Bootstrap CIs of κ: 2000 resamples, seed 42.

## Provenance of every number below
| what | model | masks | metrics evaluator |
|---|---|---|---|
| (a) out-of-fold masks, 145 reference images | RF-DETR-Seg Large @624, seed 42, 5 fold models | `outputs/05_predictions/oof_rfdetr` (`oof_check.md`) | — (mm accuracy is measured here, not by a framework) |
| (b) final-model masks, test high images | RF-DETR-Seg Large @624, seed 42 | `outputs/05_predictions/test_rfdetr` | RF-DETR's own COCO evaluation (pycocotools, iouType='segm') → `segmentation_metrics.md` |
| ground-truth masks (Stage 3 comparison) | COCO annotations, no model | `../gummy_smile_v3/data/coco_dataset` | — |
| [reference] rows in `boundary_by_set.md` | YOLOv11x (previous final model) | `outputs/05_predictions/boundary_error.csv` | Ultralytics val() (box/mask P, R, F1, mAP@50, mAP@50-95) |

Boundary and IoU tables are computed here from exactly these masks. An earlier final model's figures appear only in rows marked **[reference]**, never merged into this model's rows, and detection metrics are taken only from the evaluator of the model that produced the masks (RF-DETR's own COCO evaluation (pycocotools, iouType='segm')) — the two frameworks' mAP definitions are not interchangeable (PLAN.md 4).

## Primary result — (a) all 145 reference images, out-of-fold masks
n = 145: MAE 0.521 mm, RMSE 0.730 mm, r 0.892, ICC(2,1) 0.877 [0.805, 0.919], bias +0.270 mm [+0.158, +0.381], LoA -1.07 to 1.60 mm, proportional-bias slope -0.056 (p = 0.160); within 0.5 mm 61 %, within 1 mm 87 %; label agreement 75.9 %, linear-weighted κ 0.718 [0.631, 0.797].
**Segmentation failure: n = 0 of 145 (0.0 %)** — none: no gingiva instance predicted, so no measurement exists; excluded from the mm and label metrics above and reported as a separate failure category (a deployed system must flag such images for manual review rather than output a value).
Stage-3 holdout images only (the scale was never fitted on them): n = 58: MAE 0.479 mm, RMSE 0.675 mm, r 0.896, ICC(2,1) 0.876 [0.757, 0.933], bias +0.294 mm [+0.133, +0.455], LoA -0.91 to 1.50 mm, proportional-bias slope +0.045 (p = 0.478); within 0.5 mm 64 %, within 1 mm 88 %; label agreement 77.6 %, linear-weighted κ 0.733 [0.607, 0.847].

Same method and scale on the ground-truth masks (Stage 3): MAE 0.546 mm, r 0.871, ICC 0.867, bias +0.154 mm → the segmentation adds -0.025 mm MAE and +0.116 mm bias (see `error_decomposition.md`).

## Secondary — (b) final model, 29 test-set high images
n = 29: MAE 0.671 mm, RMSE 0.941 mm, r 0.804, ICC(2,1) 0.787 [0.595, 0.894], bias +0.284 mm [-0.064, +0.631], LoA -1.51 to 2.07 mm, proportional-bias slope -0.150 (p = 0.245); within 0.5 mm 38 %, within 1 mm 90 %; label agreement 69.0 %, linear-weighted κ 0.621 [0.421, 0.788].
The fold models on the same images (b'): MAE 0.611 mm, bias +0.214 mm, ICC 0.803.

## Error decomposition (a)
Segmentation part: bias +0.116 mm, MAE 0.276 mm; geometry part (Stage 3): bias +0.154 mm, MAE 0.546 mm; variance shares 24 % / 120 % / covariance -44 %. The segmentation error follows the lower gingiva edge: bias +0.07 mm (predicted gingiva extends below the annotated edge), upper edge bias +0.02 mm — `error_decomposition.md`, figure `figures/error_decomposition.png`.

## Segmentation quality (three sets; `boundary_by_set.md`) — RF-DETR-Seg Large @624, seed 42, masks `outputs/05_predictions/test_rfdetr` / `outputs/05_predictions/oof_rfdetr`
| set | n | gingiva IoU mean (median) | upper edge MAE, mm | lower edge MAE, mm | lower edge bias, mm | lip IoU |
|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | 145 | 0.821 (0.838) | 0.28 | 0.47 | +0.07 | 0.836 |
| (b) test high, final model | 29 | 0.820 (0.834) | 0.26 | 0.48 | +0.04 | 0.834 |
| (c) test all | 29 | 0.820 (0.834), defined on 29 | 0.26 | 0.48 | +0.04 | 0.834 |
| (c) test low | 0 | nan (nan) | nan | nan | +nan | nan |
| (c) test normal | 0 | nan (nan) | nan | nan | +nan | nan |

Low/normal IoU is low by construction (thin or absent gingiva: median annotated width nan columns in low vs 935 in high; undefined IoU on 0 images with no gingiva in either mask), while the edge errors there are no larger than in the high set. The pipeline is specified for the high smile line; (a) is the segmentation result that matters for the measurement.

## Fallback transparency
Zenith regioning (C_p25) fell back to equal splits on 27 of 145 OOF images (19 %; GT masks in Stage 3: 27 of 145, 19 %). Pre-registered rule: re-evaluate against A_p25 above 30 % → not triggered; the A_p25 row is in the table below either way.

## All rows (`measurement_accuracy.csv`)
| set | model | correction | n | mae | rmse | r | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | ba_bias | ba_loa_low | ba_loa_high | ba_prop_slope | ba_prop_p | threshold_agreement | threshold_kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) OOF masks, all reference images [PRIMARY] | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 145 | 0.521 | 0.730 | 0.892 | 0.877 | 0.805 | 0.919 | 0.270 | -1.065 | 1.605 | -0.056 | 0.160 | 0.759 | 0.718 |
| (a) OOF, Stage-3 holdout images only (scale never fitted on these) | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 58 | 0.479 | 0.675 | 0.896 | 0.876 | 0.757 | 0.933 | 0.294 | -0.909 | 1.496 | 0.045 | 0.478 | 0.776 | 0.733 |
| (a) OOF, Stage-3 dev images only | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 87 | 0.549 | 0.765 | 0.893 | 0.878 | 0.805 | 0.923 | 0.254 | -1.169 | 1.676 | -0.107 | 0.041 | 0.747 | 0.708 |
| (a) OOF, main-split train | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 87 | 0.469 | 0.611 | 0.928 | 0.912 | 0.836 | 0.949 | 0.258 | -0.833 | 1.350 | -0.059 | 0.167 | 0.816 | 0.789 |
| (a) OOF, main-split valid | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 29 | 0.588 | 0.836 | 0.874 | 0.854 | 0.682 | 0.932 | 0.359 | -1.147 | 1.865 | -0.015 | 0.882 | 0.655 | 0.605 |
| (b') OOF masks on the test high images (fold models) | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 29 | 0.611 | 0.921 | 0.809 | 0.803 | 0.626 | 0.902 | 0.214 | -1.572 | 2.001 | -0.092 | 0.467 | 0.690 | 0.621 |
| (a) sensitivity: without dash-zero reference rows | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 112 | 0.508 | 0.717 | 0.879 | 0.869 | 0.803 | 0.912 | 0.220 | -1.122 | 1.563 | -0.037 | 0.451 | 0.732 | 0.717 |
| (a) sensitivity: only 2698x1799 (±2 px) frames | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 123 | 0.515 | 0.710 | 0.904 | 0.891 | 0.824 | 0.930 | 0.261 | -1.040 | 1.561 | -0.039 | 0.345 | 0.764 | 0.728 |
| (a) sensitivity: without zenith-fallback images | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 118 | 0.506 | 0.728 | 0.882 | 0.867 | 0.790 | 0.913 | 0.259 | -1.080 | 1.599 | -0.050 | 0.282 | 0.763 | 0.722 |
| (a) sensitivity: without empty predictions | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 145 | 0.521 | 0.730 | 0.892 | 0.877 | 0.805 | 0.919 | 0.270 | -1.065 | 1.605 | -0.056 | 0.160 | 0.759 | 0.718 |
| (a) PRE-REGISTERED SENSITIVITY: excluding the 29 images used to choose the architecture | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 116 | 0.498 | 0.674 | 0.913 | 0.896 | 0.813 | 0.937 | 0.284 | -0.921 | 1.488 | -0.047 | 0.239 | 0.776 | 0.741 |
| (b) final model masks, test high images [secondary set] | RF-DETR-Seg Large @624, seed 42 | none | 29 | 0.671 | 0.941 | 0.804 | 0.787 | 0.595 | 0.894 | 0.284 | -1.506 | 2.073 | -0.150 | 0.245 | 0.690 | 0.621 |
| GT masks, all reference images (Stage 3, same method and scale) | COCO annotations (no model) | n/a (GT masks) | 145 | 0.546 | 0.761 | 0.871 | 0.867 | 0.818 | 0.903 | 0.154 | -1.311 | 1.619 | -0.040 | 0.367 | 0.786 | 0.742 |
| GT masks, Stage-3 holdout | COCO annotations (no model) | n/a (GT masks) | 58 | 0.527 | 0.726 | 0.860 | 0.857 | 0.770 | 0.912 | 0.125 | -1.290 | 1.540 | 0.065 | 0.378 | 0.793 | 0.733 |
| (a) OOF, A_p25 at its Stage-3 scale 17.04 px/mm (fallback sensitivity) | RF-DETR-Seg Large @624, seed 42, 5 fold models | none | 145 | 0.558 | 0.761 | 0.883 | 0.860 | 0.775 | 0.909 | 0.293 | -1.087 | 1.673 | -0.111 | 0.009 | 0.766 | 0.720 |

## Tooth level
`mixed_models.md`, `tooth_level.csv` — random intercept per patient; intercept of pipeline − reference: +0.270 mm [+0.159, +0.381] (OLS, cluster-robust SE (patient)); segmentation part alone: +0.116 mm [+0.062, +0.170].

## Deviations
See `SAPMALAR.md`.
