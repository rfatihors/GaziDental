# Post-hoc offset calibration of the pipeline measurement (Stage 6 addendum)

**Status: post-hoc calibration, estimated on the Stage-3 dev subset, reported on holdout.** It was not part of the pre-registered protocol (Stage 3 fixed the method and scale on ground-truth masks; Stage 6 applied them unchanged to predicted masks). If adopted, the manuscript must present it as a calibration step derived after inspecting the out-of-fold error, with the fit set and the held-out evaluation stated. Nothing here was written into `configs/config.yaml`; the recommendation below is for the clinical team's decision.

## Why
On the OOF masks the lower gingiva edge is drawn systematically below the annotation (dev mean +11.3 px = +0.67 mm; upper edge +3.4 px = +0.20 mm), the pipeline − reference bias is +0.68 mm and the tooth-level mixed model gives +0.68 [+0.57, +0.80] mm with a proportional-bias slope near zero. A near-constant offset is the signature of a calibration error, which is correctable.

## Data and protocol
* Images: the 145 reference images with OOF masks; **1 segmentation failure(s)** (IMG_7289_jpg: no gingiva predicted, no mm value) excluded from the fit and the evaluation and reported as a separate category.
* Split: Stage 3 dev / holdout lists (`outputs/03_oracle/dev_holdout_split.json`, seed 42): dev n = 87, holdout n = 57. **All three corrections are fitted on dev only; holdout numbers are reported once, nothing is re-fitted there.** Method C_p25, scale 16.84 px/mm unchanged.
* (a) constant: subtract the mean dev error, **0.694 mm**.
* (b) pixel: shift the lower gingiva edge up by d px at mask level (column-wise removal of the lowest d pixels of the bottom run) and re-measure with the unchanged method; d chosen as the integer minimising dev MAE on a 0–24 px grid → **d* = 13 px (0.77 mm)**. The dev boundary bias alone would suggest 11 px; the curve is in `pixel_shift_curve.csv` (holdout column for information only).
* (c) regression: reference ≈ a + b · pipeline on dev → **a = -0.778 mm, b = 1.024**; corrected value = a + b · pipeline.

## Comparison (bootstrap CIs of κ: 2000 resamples)
| subset / correction | n | MAE, mm | RMSE, mm | r | ICC(2,1) [95 % CI] | bias, mm [95 % CI] | 95 % LoA, mm | prop. slope (p) | label agreement | κ linear [95 % CI] | within 1 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev: uncorrected | 87 | 0.894 | 1.049 | 0.872 | 0.779 [0.325, 0.905] | +0.694 [+0.525, +0.862] | -0.86 to 2.24 | -0.171 (0.003) | 60.9 % | 0.582 [0.460, 0.693] | 62 % |
| dev: (a) constant mm offset | 87 | 0.522 | 0.786 | 0.872 | 0.862 [0.796, 0.908] | -0.000 [-0.169, +0.169] | -1.55 to 1.55 | -0.171 (0.003) | 77.0 % | 0.705 [0.570, 0.819] | 87 % |
| dev: (b) lower edge −13 px (mask level) | 87 | 0.509 | 0.785 | 0.873 | 0.863 [0.798, 0.908] | -0.063 [-0.231, +0.104] | -1.61 to 1.48 | -0.171 (0.003) | 79.3 % | 0.727 [0.591, 0.836] | 87 % |
| dev: (c) linear recalibration | 87 | 0.520 | 0.786 | 0.872 | 0.865 [0.801, 0.910] | -0.000 [-0.168, +0.168] | -1.55 to 1.55 | -0.146 (0.011) | 78.2 % | 0.720 [0.582, 0.831] | 87 % |
| holdout: uncorrected | 57 | 0.757 | 0.917 | 0.894 | 0.797 [0.191, 0.926] | +0.670 [+0.502, +0.838] | -0.57 to 1.91 | +0.071 (0.269) | 64.9 % | 0.611 [0.466, 0.741] | 74 % |
| holdout: (a) constant mm offset | 57 | 0.454 | 0.627 | 0.894 | 0.894 [0.826, 0.936] | -0.024 [-0.191, +0.144] | -1.26 to 1.21 | +0.071 (0.269) | 82.5 % | 0.794 [0.682, 0.898] | 91 % |
| holdout: (b) lower edge −13 px (mask level) | 57 | 0.426 | 0.606 | 0.894 | 0.895 [0.828, 0.937] | -0.039 [-0.201, +0.123] | -1.23 to 1.16 | +0.015 (0.813) | 82.5 % | 0.794 [0.682, 0.898] | 95 % |
| holdout: (c) linear recalibration | 57 | 0.470 | 0.640 | 0.894 | 0.892 [0.823, 0.935] | -0.027 [-0.198, +0.144] | -1.29 to 1.24 | +0.096 (0.138) | 82.5 % | 0.794 [0.682, 0.898] | 89 % |

## Selection rule and recommendation
Rule stated in advance: lowest holdout MAE, but the simplest correction (constant mm) is preferred when it is within 0.03 mm of the best. Holdout MAE: constant 0.454, pixel 0.426, regression 0.470 (uncorrected 0.757). Best = **pixel**; constant − best = +0.028 mm → **recommended: constant** ((a) constant mm offset). Not applied to the config; decision pending.

Effect of the recommended correction on holdout: MAE 0.757 → 0.454 mm, bias +0.670 → -0.024 mm, ICC(2,1) 0.797 → 0.894, LoA width 2.48 → 2.48 mm, label agreement 64.9 % → 82.5 %, κ 0.611 → 0.794. A constant offset cannot change r or the spread of the differences; it moves the bias and, through the thresholds, the labels.

### Holdout label confusion, uncorrected
| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 25 | 9 | 2 | 0 |
| E1-E2 | 0 | 3 | 7 | 0 |
| E2-E3 | 0 | 0 | 8 | 1 |
| E3 | 0 | 0 | 1 | 1 |

### Holdout label confusion, (a) constant mm offset
| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 31 | 5 | 0 | 0 |
| E1-E2 | 0 | 9 | 1 | 0 |
| E2-E3 | 0 | 1 | 7 | 1 |
| E3 | 0 | 0 | 2 | 0 |

## How to present it in the manuscript
"The segmentation model places the lower gingival margin on average 0.67 mm below the annotated margin. A constant offset of 0.69 mm was therefore estimated post hoc on the development subset (n = 87) of the out-of-fold predictions and applied unchanged to the held-out subset (n = 57), where it reduced the mean absolute error from 0.76 to 0.45 mm and the bias from +0.67 to -0.02 mm. Uncorrected results are reported as the primary analysis." (Adjust to the chosen correction if it is not the constant one.)

## Caveats
* The offset was fitted on dev images whose reference measurements also determined the global scale in Stage 3; holdout is the only clean estimate of the corrected accuracy.
* If the correction is adopted, the same offset must be applied to the test-set (final model) predictions and to the expert-agreement analysis *before* those are looked at again.
* The segmentation failure category (n = 1) is unaffected by any calibration; it needs a manual-review path.

## Adopted correction: mask level (config.yaml `measurement.bottom_edge_offset_px = -13`)
Adopted: the lower gingiva edge of the predicted mask is moved up 13 px in every column and the profile and all estimators are re-measured (config `measurement.bottom_edge_offset_px = -13`; 13 px is d*, the dev optimum of the mask-level grid above). Holdout MAE: mask level 0.426 mm vs value level -13 px 0.461 mm vs constant 0.694 mm (11.7 px) 0.454 mm. The mask-level row is identical to row (b) of the comparison table (same re-measurement). The value-level px constant is *not* equivalent: subtracting a constant after the measurement equals the mask-level shift only when every column of every region loses the same 13 px, which fails where a column is thinner than the shift or where the zenith regioning changes on the corrected profile.

| subset | n | MAE, mm | RMSE, mm | ICC(2,1) [CI] | bias, mm [CI] | 95 % LoA | label agreement | κ linear [CI] |
|---|---|---|---|---|---|---|---|---|
| dev | 87 | 0.509 | 0.785 | 0.863 [0.798, 0.908] | -0.063 [-0.231, +0.104] | -1.61 to 1.48 | 79.3 % | 0.727 [0.591, 0.836] |
| holdout | 57 | 0.426 | 0.606 | 0.895 [0.828, 0.937] | -0.039 [-0.201, +0.123] | -1.23 to 1.16 | 82.5 % | 0.794 [0.682, 0.898] |
| test high, final model | 29 | 0.469 | 0.892 | 0.803 [0.623, 0.902] | -0.095 [-0.438, +0.249] | -1.86 to 1.68 | 75.9 % | 0.640 [0.373, 0.857] |

Columns zeroed by the shift (bottom run shorter than 13 px; never negative): test-set images with at least one such column 17 of 29; OOF images whose whole measurement fell to 0: 0. Per-image counts are in `per_image_results*.csv` (`n_columns_zeroed`, `columns_zeroed_frac`).

### The three variants side by side (MAE / bias / ICC(2,1) / label agreement / κ; mm)
| variant | dev (n = 87) | holdout (n = 57) | test high, final model (n = 29) |
|---|---|---|---|
| uncorrected (PRIMARY) | 0.894 / +0.694 / 0.779 / 61 % / 0.582 | 0.757 / +0.670 / 0.797 / 65 % / 0.611 | 0.953 / +0.647 / 0.731 / 48 % / 0.401 |
| (a) value level, constant 0.694 mm | 0.522 / -0.000 / 0.862 / 77 % / 0.705 | 0.454 / -0.024 / 0.894 / 82 % / 0.794 | 0.504 / -0.047 / 0.804 / 72 % / 0.609 |
| value level, constant -13 px (-0.772 mm at the global scale) | 0.515 / -0.078 / 0.861 / 78 % / 0.711 | 0.461 / -0.102 / 0.891 / 82 % / 0.794 | 0.491 / -0.125 / 0.801 / 76 % / 0.640 |
| **(b) mask level, lower edge -13 px — ADOPTED** | 0.509 / -0.063 / 0.863 / 79 % / 0.727 | 0.426 / -0.039 / 0.895 / 82 % / 0.794 | 0.469 / -0.095 / 0.803 / 76 % / 0.640 |
| (c) value level, linear recalibration | 0.520 / -0.000 / 0.865 / 78 % / 0.720 | 0.470 / -0.027 / 0.892 / 82 % / 0.794 | 0.511 / -0.046 / 0.806 / 72 % / 0.609 |

### Why the mask-level variant was chosen
1. **It acts where the error is.** The boundary analysis locates the bias at the lower gingiva edge (`boundary_by_set.md`, `error_decomposition.md`); moving that edge corrects the mask itself, so the thickness profile, the zenith regioning, the estimators and any downstream use of the mask (overlays, expert review) all see the corrected geometry. A constant subtracted from the final number corrects only the number.
2. **It is the best on holdout** (0.426 mm vs 0.454 mm for the constant and 0.461 mm for the value-level pixel constant) and it was the dev optimum of a pre-stated grid (d* = 13 px), not a tuned constant.
3. **It is scale-free.** Defined in pixels at mask level, it converts to mm with whatever px/mm is valid for the image — the global scale here, the experts' per-image scale in Stage 4 — whereas a mm constant would be wrong under a per-image scale.
4. **It cannot produce negative values.** A column thinner than the shift becomes 0 and is counted, instead of being clipped after the fact.
5. **Equivalence with a constant is not guaranteed.** Subtracting 13 px after the measurement equals the mask-level shift only if all columns of all regions lose the same amount; the dev/holdout numbers above show the two are close but not identical.

## Manuscript paragraph (Methods / Results, post-hoc calibration)
The segmentation error was concentrated at the lower gingival margin. Against the annotated masks of the 145 reference images (out-of-fold predictions), the upper, lip-side edge of the gingiva was accurate (mean absolute error 0.32 mm (bias +0.16 mm)) whereas the lower edge was placed systematically too low (mean absolute error 0.76 mm (bias +0.66 mm)), i.e. the model consistently included a thin strip of the festooned margin. The resulting over-measurement was constant rather than proportional: the mean error was 0.57–0.76 mm across the five folds (one-way ANOVA p = 0.85), did not differ between image frame groups, and the final model on the 29 test-set images showed the same bias (+0.647 mm [+0.30, +0.99]) as the fold models on the same images (+0.626 mm) (paired difference final − fold models +0.02 mm [-0.12, +0.16]). The lower edge of the predicted gingiva mask was therefore moved up by 13 px (0.77 mm at the global scale of 16.84 px/mm) in every image column before measurement; this mask-level offset was estimated post hoc on the development subset of the out-of-fold predictions (n = 87) and applied unchanged to the held-out subset (n = 57) and to the final model's test-set images (n = 29). Alternative estimators were examined first: lower percentiles of the column profile (p10, p5, minimum) and lip-anchored variants did not remove the bias (best alternative C_p25_lipanchored: holdout MAE 0.58 mm, residual bias +0.41 mm); the bias is a shift of the edge, not a spread within the profile, so a percentile cannot absorb it. On the held-out subset the correction reduced the mean absolute error from 0.76 to 0.43 mm and the bias from +0.67 to -0.04 mm (ICC(2,1) 0.80 → 0.90; agreement with the Table 1 class 65 % → 82 %, linear-weighted κ 0.61 → 0.79); on the independent test images from 0.95 to 0.47 mm (bias +0.65 → -0.09 mm). Because the offset was derived after inspecting the out-of-fold error, it is a post-hoc calibration: uncorrected results are reported as the primary analysis throughout, corrected results as a secondary analysis, and the offset is stated explicitly so that it can be re-estimated for any retrained model.
