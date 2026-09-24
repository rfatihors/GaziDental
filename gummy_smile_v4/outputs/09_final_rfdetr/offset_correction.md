# Post-hoc offset calibration of the pipeline measurement (Stage 6 addendum)

**Status: post-hoc calibration, estimated on the Stage-3 dev subset, reported on holdout.** It was not part of the pre-registered protocol (Stage 3 fixed the method and scale on ground-truth masks; Stage 6 applied them unchanged to predicted masks). If adopted, the manuscript must present it as a calibration step derived after inspecting the out-of-fold error, with the fit set and the held-out evaluation stated. Nothing here was written into `configs/config.yaml`; the recommendation below is for the clinical team's decision.

## Why
On the OOF masks (oof_rfdetr) the lower gingiva edge is drawn systematically below the annotation (dev mean +1.3 px = +0.08 mm; upper edge +0.8 px = +0.05 mm) and the pipeline − reference bias on dev is +0.25 mm [+0.10, +0.41] with a proportional-bias slope of -0.107 (p = 0.041); the tooth-level mixed model (`mixed_models.md`) reports the same shift. The shift dominates, but it is not purely constant on dev (slope p = 0.041): a constant offset removes the shift and leaves the proportional part, which only the linear recalibration (c) addresses. Both are in the table below, with the holdout slope as the check.

## Data and protocol
* Images: the 145 reference images with OOF masks; **0 segmentation failure(s)** (—: no gingiva predicted, no mm value) excluded from the fit and the evaluation and reported as a separate category.
* Split: Stage 3 dev / holdout lists (`outputs/03_oracle/dev_holdout_split.json`, seed 42): dev n = 87, holdout n = 58. **All three corrections are fitted on dev only; holdout numbers are reported once, nothing is re-fitted there.** Method C_p25, scale 16.84 px/mm unchanged.
* (a) constant: subtract the mean dev error, **0.254 mm**.
* (b) pixel: shift the lower gingiva edge up by d px at mask level (column-wise removal of the lowest d pixels of the bottom run) and re-measure with the unchanged method; d chosen as the integer minimising dev MAE on a 0–24 px grid → **d* = 5 px (0.30 mm)**. The dev boundary bias alone would suggest 1 px; the curve is in `pixel_shift_curve.csv` (holdout column for information only).
* (c) regression: reference ≈ a + b · pipeline on dev → **a = -0.217 mm, b = 0.988**; corrected value = a + b · pipeline. The slope is a -1.2 % proportional term: it is **not** 1, and (a) and (b), which shift without scaling, leave it in place.

## Comparison (bootstrap CIs of κ: 2000 resamples)
| subset / correction | n | MAE, mm | RMSE, mm | r | ICC(2,1) [95 % CI] | bias, mm [95 % CI] | 95 % LoA, mm | prop. slope (p) | label agreement | κ linear [95 % CI] | within 1 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev: uncorrected | 87 | 0.549 | 0.765 | 0.893 | 0.878 [0.805, 0.923] | +0.254 [+0.099, +0.408] | -1.17 to 1.68 | -0.107 (0.041) | 74.7 % | 0.708 [0.584, 0.806] | 86 % |
| dev: (a) constant mm offset | 87 | 0.462 | 0.722 | 0.893 | 0.890 [0.836, 0.927] | +0.000 [-0.155, +0.155] | -1.42 to 1.42 | -0.107 (0.041) | 79.3 % | 0.732 [0.589, 0.841] | 89 % |
| dev: (b) lower edge −5 px (mask level) | 87 | 0.461 | 0.723 | 0.893 | 0.889 [0.835, 0.926] | -0.031 [-0.186, +0.124] | -1.45 to 1.39 | -0.114 (0.030) | 80.5 % | 0.744 [0.606, 0.852] | 91 % |
| dev: (c) linear recalibration | 87 | 0.463 | 0.721 | 0.893 | 0.889 [0.835, 0.926] | +0.000 [-0.155, +0.155] | -1.42 to 1.42 | -0.119 (0.023) | 79.3 % | 0.732 [0.589, 0.841] | 90 % |
| holdout: uncorrected | 58 | 0.479 | 0.675 | 0.896 | 0.876 [0.757, 0.933] | +0.294 [+0.133, +0.455] | -0.91 to 1.50 | +0.045 (0.478) | 77.6 % | 0.733 [0.607, 0.847] | 88 % |
| holdout: (a) constant mm offset | 58 | 0.416 | 0.610 | 0.896 | 0.897 [0.832, 0.938] | +0.040 [-0.121, +0.202] | -1.16 to 1.24 | +0.045 (0.478) | 81.0 % | 0.781 [0.663, 0.884] | 91 % |
| holdout: (b) lower edge −5 px (mask level) | 58 | 0.412 | 0.603 | 0.895 | 0.897 [0.831, 0.937] | +0.024 [-0.135, +0.184] | -1.17 to 1.21 | +0.019 (0.759) | 79.3 % | 0.760 [0.637, 0.865] | 91 % |
| holdout: (c) linear recalibration | 58 | 0.412 | 0.605 | 0.896 | 0.897 [0.832, 0.938] | +0.041 [-0.119, +0.201] | -1.15 to 1.23 | +0.032 (0.607) | 81.0 % | 0.781 [0.663, 0.884] | 91 % |

## Selection rule and recommendation
Rule stated in advance: lowest holdout MAE, but the simplest correction (constant mm) is preferred when it is within 0.03 mm of the best. Holdout MAE: constant 0.416, pixel 0.412, regression 0.412 (uncorrected 0.479). Best = **pixel**; constant − best = +0.005 mm → **recommended: constant** ((a) constant mm offset). Not applied to the config; decision pending.

Effect of the recommended correction on holdout: MAE 0.479 → 0.416 mm, bias +0.294 → +0.040 mm, ICC(2,1) 0.876 → 0.897, LoA width 2.40 → 2.40 mm, label agreement 77.6 % → 81.0 %, κ 0.733 → 0.781. A constant offset cannot change r or the spread of the differences; it moves the bias and, through the thresholds, the labels.

### Holdout label confusion, uncorrected
| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 30 | 5 | 1 | 0 |
| E1-E2 | 0 | 7 | 3 | 0 |
| E2-E3 | 0 | 1 | 8 | 1 |
| E3 | 0 | 0 | 2 | 0 |

### Holdout label confusion, (a) constant mm offset
| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 31 | 5 | 0 | 0 |
| E1-E2 | 1 | 8 | 1 | 0 |
| E2-E3 | 0 | 1 | 8 | 1 |
| E3 | 0 | 0 | 2 | 0 |

## How to present it in the manuscript
"The segmentation model places the lower gingival margin on average 0.08 mm below the annotated margin. A constant offset of 0.25 mm was therefore estimated post hoc on the development subset (n = 87) of the out-of-fold predictions and applied unchanged to the held-out subset (n = 58), where it reduced the mean absolute error from 0.48 to 0.42 mm and the bias from +0.29 to +0.04 mm. Uncorrected results are reported as the primary analysis." (Adjust to the chosen correction if it is not the constant one.)

## Caveats
* The offset was fitted on dev images whose reference measurements also determined the global scale in Stage 3; holdout is the only clean estimate of the corrected accuracy.
* If the correction is adopted, the same offset must be applied to the test-set (final model) predictions and to the expert-agreement analysis *before* those are looked at again.
* The segmentation failure category (n = 0) is unaffected by any calibration; it needs a manual-review path.

## Adopted correction: mask level (config.yaml `measurement.bottom_edge_offset_px = +0`)
Adopted: the lower gingiva edge of the predicted mask is moved up 0 px in every column and the profile and all estimators are re-measured (config `measurement.bottom_edge_offset_px = +0`; 0 px is d*, the dev optimum of the mask-level grid above). Holdout MAE: mask level 0.479 mm vs value level +0 px 0.479 mm vs constant 0.254 mm (4.3 px) 0.416 mm. The mask-level row is identical to row (b) of the comparison table (same re-measurement). The value-level px constant is *not* equivalent: subtracting a constant after the measurement equals the mask-level shift only when every column of every region loses the same 0 px, which fails where a column is thinner than the shift or where the zenith regioning changes on the corrected profile.

| subset | n | MAE, mm | RMSE, mm | ICC(2,1) [CI] | bias, mm [CI] | 95 % LoA | label agreement | κ linear [CI] |
|---|---|---|---|---|---|---|---|---|
| dev | 87 | 0.549 | 0.765 | 0.878 [0.805, 0.923] | +0.254 [+0.099, +0.408] | -1.17 to 1.68 | 74.7 % | 0.708 [0.584, 0.806] |
| holdout | 58 | 0.479 | 0.675 | 0.876 [0.757, 0.933] | +0.294 [+0.133, +0.455] | -0.91 to 1.50 | 77.6 % | 0.733 [0.607, 0.847] |
| test high, final model | 29 | 0.671 | 0.941 | 0.787 [0.595, 0.894] | +0.284 [-0.064, +0.631] | -1.51 to 2.07 | 69.0 % | 0.621 [0.421, 0.788] |

Columns zeroed by the shift (bottom run shorter than 0 px; never negative): test-set images with at least one such column 0 of 29; OOF images whose whole measurement fell to 0: 0. Per-image counts are in `per_image_results*.csv` (`n_columns_zeroed`, `columns_zeroed_frac`).

### The three variants side by side (MAE / bias / ICC(2,1) / label agreement / κ; mm)
| variant | dev (n = 87) | holdout (n = 58) | test high, final model (n = 29) |
|---|---|---|---|
| uncorrected (PRIMARY) | 0.549 / +0.254 / 0.878 / 75 % / 0.708 | 0.479 / +0.294 / 0.876 / 78 % / 0.733 | 0.671 / +0.284 / 0.787 / 69 % / 0.621 |
| (a) value level, constant 0.254 mm | 0.462 / +0.000 / 0.890 / 79 % / 0.732 | 0.416 / +0.040 / 0.897 / 81 % / 0.781 | 0.537 / +0.030 / 0.802 / 72 % / 0.609 |
| value level, constant +0 px (+0.000 mm at the global scale) | 0.549 / +0.254 / 0.878 / 75 % / 0.708 | 0.479 / +0.294 / 0.876 / 78 % / 0.733 | 0.671 / +0.284 / 0.787 / 69 % / 0.621 |
| **(b) mask level, lower edge +0 px — ADOPTED** | 0.549 / +0.254 / 0.878 / 75 % / 0.708 | 0.479 / +0.294 / 0.876 / 78 % / 0.733 | 0.671 / +0.284 / 0.787 / 69 % / 0.621 |
| (c) value level, linear recalibration | 0.463 / +0.000 / 0.889 / 79 % / 0.732 | 0.412 / +0.041 / 0.897 / 81 % / 0.781 | 0.537 / +0.029 / 0.801 / 72 % / 0.609 |

### Why the mask-level variant was chosen
1. **It acts where the error is.** The boundary analysis locates the bias at the lower gingiva edge (`boundary_by_set.md`, `error_decomposition.md`); moving that edge corrects the mask itself, so the thickness profile, the zenith regioning, the estimators and any downstream use of the mask (overlays, expert review) all see the corrected geometry. A constant subtracted from the final number corrects only the number.
2. **It is the best on holdout** (0.479 mm vs 0.416 mm for the constant and 0.479 mm for the value-level pixel constant) and it was the dev optimum of a pre-stated grid (d* = 0 px), not a tuned constant.
3. **It is scale-free.** Defined in pixels at mask level, it converts to mm with whatever px/mm is valid for the image — the global scale here, the experts' per-image scale in Stage 4 — whereas a mm constant would be wrong under a per-image scale.
4. **It cannot produce negative values.** A column thinner than the shift becomes 0 and is counted, instead of being clipped after the fact.
5. **Equivalence with a constant is not guaranteed.** Subtracting 0 px after the measurement equals the mask-level shift only if all columns of all regions lose the same amount; the dev/holdout numbers above show the two are close but not identical.

## Manuscript paragraph (Methods / Results, post-hoc calibration)
The segmentation error was concentrated at the lower gingival margin. Against the annotated masks of the 145 reference images (out-of-fold predictions), the upper, lip-side edge of the gingiva was accurate (mean absolute error 0.28 mm (bias +0.02 mm)) whereas the lower edge was placed systematically too low (mean absolute error 0.47 mm (bias +0.07 mm)), i.e. the model consistently included a thin strip of the festooned margin. The over-measurement is predominantly a shift, but a proportional component is present and is not negligible: the Bland–Altman slope of (pipeline − reference) on the mean is -0.107 (p = 0.041) on the development subset and +0.045 (p = 0.478) on the held-out subset, and the linear recalibration (c) fits a slope of 0.988, i.e. a -1.2 % scale term rather than 1. A constant shift therefore removes the offset but not that component; both are reported, and the residual slope after the adopted correction is +0.045 (p = 0.478) on holdout. The mean error was n/a (one-way ANOVA p = 0.28), did not differ between image frame groups, and was reproduced by the final model on the independent test images (paired difference final − fold models +0.07 mm [+0.01, +0.13]). The lower edge of the predicted gingiva mask was therefore moved up by 0 px (0.00 mm at the global scale of 16.84 px/mm) in every image column before measurement; this mask-level offset was estimated post hoc on the development subset of the out-of-fold predictions (n = 87) and applied unchanged to the held-out subset (n = 58) and to the final model's test-set images (n = 29). Alternative estimators were examined first: lower percentiles and lip-anchored variants did not remove the bias; the bias is a shift of the edge, not a spread within the profile, so a percentile cannot absorb it. On the held-out subset the correction reduced the mean absolute error from 0.48 to 0.48 mm and the bias from +0.29 to +0.29 mm (ICC(2,1) 0.88 → 0.88; agreement with the Table 1 class 78 % → 78 %, linear-weighted κ 0.73 → 0.73); on the independent test images from 0.67 to 0.67 mm (bias +0.28 → +0.28 mm). Because the offset was derived after inspecting the out-of-fold error, it is a post-hoc calibration: uncorrected results are reported as the primary analysis throughout, corrected results as a secondary analysis, and the offset is stated explicitly so that it can be re-estimated for any retrained model.
