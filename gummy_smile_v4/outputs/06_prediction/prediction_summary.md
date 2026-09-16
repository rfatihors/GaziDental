# Stage 6 — full pipeline on predicted masks

Method **C_p25**, global scale **16.84 px/mm**, both fixed in `configs/config.yaml` from Stage 3 (ground-truth masks, dev subset). Nothing was re-selected or re-fitted on predicted masks. Masks: `outputs/05_predictions/oof` (5 fold models, each predicting only its held-out fold; `oof_check.md`) and `outputs/05_predictions/test` (final model). Bootstrap CIs of κ: 2000 resamples, seed 42.

## Primary result — (a) all 144 reference images, out-of-fold masks
n = 144: MAE 0.839 mm, RMSE 0.999 mm, r 0.875, ICC(2,1) 0.784 [0.288, 0.909], bias +0.684 mm [+0.564, +0.805], LoA -0.75 to 2.12 mm, proportional-bias slope -0.085 (p = 0.052); within 0.5 mm 26 %, within 1 mm 67 %; label agreement 62.5 %, linear-weighted κ 0.594 [0.503, 0.677].
**Segmentation failure: n = 1 of 145 (0.7 %)** — IMG_7289_jpg: no gingiva instance predicted, so no measurement exists; excluded from the mm and label metrics above and reported as a separate failure category (a deployed system must flag such images for manual review rather than output a value).
Stage-3 holdout images only (the scale was never fitted on them): n = 57: MAE 0.757 mm, RMSE 0.917 mm, r 0.894, ICC(2,1) 0.797 [0.191, 0.926], bias +0.670 mm [+0.502, +0.838], LoA -0.57 to 1.91 mm, proportional-bias slope +0.071 (p = 0.269); within 0.5 mm 30 %, within 1 mm 74 %; label agreement 64.9 %, linear-weighted κ 0.611 [0.466, 0.741].
Same method and scale on the ground-truth masks (Stage 3): MAE 0.542 mm, r 0.872, ICC 0.868, bias +0.150 mm → the segmentation adds +0.298 mm MAE and +0.534 mm bias (see `error_decomposition.md`).

## Secondary — (b) final model, 29 test-set high images
n = 29: MAE 0.953 mm, RMSE 1.108 mm, r 0.804, ICC(2,1) 0.731 [0.344, 0.884], bias +0.647 mm [+0.299, +0.995], LoA -1.15 to 2.44 mm, proportional-bias slope -0.129 (p = 0.315); within 0.5 mm 14 %, within 1 mm 62 %; label agreement 48.3 %, linear-weighted κ 0.401 [0.205, 0.588].
The fold models on the same images (b'): MAE 0.925 mm, bias +0.626 mm, ICC 0.732.

## Error decomposition (a)
Segmentation part: bias +0.534 mm, MAE 0.580 mm; geometry part (Stage 3): bias +0.150 mm, MAE 0.544 mm; variance shares 31 % / 104 % / covariance -36 %. The segmentation error follows the lower gingiva edge: bias +0.66 mm (predicted gingiva extends below the annotated edge), upper edge bias +0.16 mm — `error_decomposition.md`, figure `figures/error_decomposition.png`.

## Segmentation quality (three sets; `boundary_by_set.md`)
| set | n | gingiva IoU mean (median) | upper edge MAE, mm | lower edge MAE, mm | lower edge bias, mm | lip IoU |
|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | 145 | 0.758 (0.774) | 0.32 | 0.76 | +0.66 | 0.794 |
| (b) test high, final model | 29 | 0.763 (0.785) | 0.32 | 0.76 | +0.60 | 0.811 |
| (c) test all | 192 | 0.508 (0.548), defined on 167 | 0.30 | 0.66 | +0.58 | 0.786 |
| (c) test low | 45 | 0.280 (0.344) | 0.17 | 0.53 | +0.49 | 0.762 |
| (c) test normal | 118 | 0.492 (0.540) | 0.32 | 0.66 | +0.59 | 0.788 |

Low/normal IoU is low by construction (thin or absent gingiva: median annotated width 16 columns in low vs 935 in high; undefined IoU on 25 images with no gingiva in either mask), while the edge errors there are no larger than in the high set. The pipeline is specified for the high smile line; (a) is the segmentation result that matters for the measurement.

## Fallback transparency
Zenith regioning (C_p25) fell back to equal splits on 25 of 145 OOF images (17 %; GT masks in Stage 3: 28 of 145, 19 %). Pre-registered rule: re-evaluate against A_p25 above 30 % → not triggered; the A_p25 row is in the table below either way.

## All rows (`measurement_accuracy.csv`)
| set | masks | n | mae | rmse | r | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | ba_bias | ba_loa_low | ba_loa_high | ba_prop_slope | ba_prop_p | threshold_agreement | threshold_kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) OOF masks, all reference images [PRIMARY] | OOF (fold models) | 144 | 0.839 | 0.999 | 0.875 | 0.784 | 0.288 | 0.909 | 0.684 | -0.746 | 2.115 | -0.085 | 0.052 | 0.625 | 0.594 |
| (a) OOF, Stage-3 holdout images only (scale never fitted on these) | OOF (fold models) | 57 | 0.757 | 0.917 | 0.894 | 0.797 | 0.191 | 0.926 | 0.670 | -0.568 | 1.909 | 0.071 | 0.269 | 0.649 | 0.611 |
| (a) OOF, Stage-3 dev images only | OOF (fold models) | 87 | 0.894 | 1.049 | 0.872 | 0.779 | 0.325 | 0.905 | 0.694 | -0.857 | 2.244 | -0.171 | 0.003 | 0.609 | 0.582 |
| (a) OOF, main-split train | OOF (fold models) | 87 | 0.790 | 0.907 | 0.910 | 0.817 | 0.216 | 0.933 | 0.667 | -0.544 | 1.878 | -0.097 | 0.043 | 0.667 | 0.671 |
| (a) OOF, main-split valid | OOF (fold models) | 28 | 0.907 | 1.139 | 0.854 | 0.754 | 0.178 | 0.911 | 0.800 | -0.819 | 2.418 | -0.027 | 0.809 | 0.536 | 0.458 |
| (b') OOF masks on the test high images (fold models) | OOF (fold models) | 29 | 0.925 | 1.111 | 0.798 | 0.732 | 0.371 | 0.882 | 0.626 | -1.205 | 2.457 | -0.106 | 0.415 | 0.586 | 0.503 |
| (a) sensitivity: without dash-zero reference rows | OOF (fold models) | 111 | 0.807 | 0.948 | 0.864 | 0.783 | 0.351 | 0.904 | 0.620 | -0.792 | 2.032 | -0.073 | 0.160 | 0.559 | 0.574 |
| (a) sensitivity: only 2698x1799 (±2 px) frames | OOF (fold models) | 123 | 0.822 | 0.969 | 0.889 | 0.807 | 0.323 | 0.921 | 0.665 | -0.724 | 2.053 | -0.069 | 0.121 | 0.650 | 0.623 |
| (a) sensitivity: without zenith-fallback images | OOF (fold models) | 119 | 0.799 | 0.956 | 0.865 | 0.781 | 0.335 | 0.904 | 0.632 | -0.782 | 2.045 | -0.086 | 0.086 | 0.613 | 0.583 |
| (a) sensitivity: without empty predictions | OOF (fold models) | 144 | 0.839 | 0.999 | 0.875 | 0.784 | 0.288 | 0.909 | 0.684 | -0.746 | 2.115 | -0.085 | 0.052 | 0.625 | 0.594 |
| (b) final model masks, test high images [secondary] | final model | 29 | 0.953 | 1.108 | 0.804 | 0.731 | 0.344 | 0.884 | 0.647 | -1.147 | 2.440 | -0.129 | 0.315 | 0.483 | 0.401 |
| GT masks, all reference images (Stage 3, same method and scale) | ground truth | 145 | 0.542 | 0.756 | 0.872 | 0.868 | 0.819 | 0.904 | 0.150 | -1.307 | 1.608 | -0.044 | 0.314 | 0.786 | 0.742 |
| GT masks, Stage-3 holdout | ground truth | 58 | 0.518 | 0.718 | 0.861 | 0.858 | 0.773 | 0.914 | 0.116 | -1.284 | 1.516 | 0.054 | 0.465 | 0.793 | 0.733 |
| (a) OOF, A_p25 at its Stage-3 scale 17.04 px/mm (fallback sensitivity) | OOF (fold models) | 144 | 0.869 | 1.027 | 0.863 | 0.760 | 0.267 | 0.896 | 0.697 | -0.786 | 2.180 | -0.157 | 0.001 | 0.611 | 0.573 |

## Tooth level
`mixed_models.md`, `tooth_level.csv` — random intercept per patient; intercept of pipeline − reference: +0.684 mm [+0.565, +0.804] (MixedLM (REML)); segmentation part alone: +0.534 mm [+0.468, +0.601].

## Deviations
See `SAPMALAR.md`.
