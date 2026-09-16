# Offset checks before adoption (Stage 6 addendum II)

Baseline: **C_p25**, 16.84 px/mm (Stage 3). Constant offset from the Stage-3 dev subset of the OOF masks: **+0.694 mm** (n = 87); segmentation failure(s) excluded throughout (1: IMG_7289_jpg). Dev n = 87, holdout n = 57, test high (final model) n = 29. Bootstrap CIs of κ: 2000 resamples. Nothing written to `configs/config.yaml`.

## 1. Alternative estimators instead of a constant (priority check)
Each candidate uses **its own scale fitted through the origin on the ground-truth masks of the dev images** (exactly the Stage-3 protocol; the scale is a property of the annotation geometry, not of the model), then is applied to the predicted masks. This is a re-selection of the estimator on predicted masks — no fitted constant. `p05` was added to the estimator set for this check. Sensitivity column `mae_oof_scale` in `offset_alternatives.csv`: MAE when the scale is instead re-fitted on the predicted dev masks (a proportional post-hoc correction; shown for information, not proposed).

### Dev (used only to fit scales and the offset; reported for completeness)
| candidate (scale) | n | MAE | RMSE | r | ICC(2,1) [CI] | bias [CI] | 95 % LoA | label agr. | κ linear [CI] | within 1 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| C_p25 (16.84 px/mm) | 87 | 0.894 | 1.049 | 0.872 | 0.779 [0.325, 0.905] | +0.694 [+0.525, +0.862] | -0.86 to 2.24 | 60.9 % | 0.582 [0.465, 0.685] | 62 % |
| C_p25_lipanchored (19.02 px/mm) | 87 | 0.736 | 0.904 | 0.873 | 0.815 [0.639, 0.897] | +0.435 [+0.265, +0.604] | -1.13 to 2.00 | 66.7 % | 0.626 [0.507, 0.734] | 78 % |
| C_p10 (16.04 px/mm) | 87 | 0.921 | 1.065 | 0.869 | 0.781 [0.324, 0.906] | +0.707 [+0.536, +0.877] | -0.86 to 2.28 | 62.1 % | 0.594 [0.475, 0.699] | 61 % |
| C_p10_lipanchored (18.18 px/mm) | 87 | 0.749 | 0.908 | 0.871 | 0.820 [0.640, 0.901] | +0.445 [+0.275, +0.614] | -1.12 to 2.01 | 67.8 % | 0.650 [0.531, 0.752] | 76 % |
| C_p05 (15.70 px/mm) | 87 | 0.939 | 1.082 | 0.864 | 0.780 [0.329, 0.905] | +0.714 [+0.540, +0.889] | -0.89 to 2.32 | 59.8 % | 0.581 [0.456, 0.685] | 59 % |
| C_p05_lipanchored (17.86 px/mm) | 87 | 0.752 | 0.914 | 0.867 | 0.821 [0.645, 0.901] | +0.444 [+0.273, +0.616] | -1.13 to 2.02 | 69.0 % | 0.651 [0.525, 0.757] | 75 % |
| C_min (15.16 px/mm) | 87 | 0.987 | 1.129 | 0.854 | 0.769 [0.316, 0.899] | +0.744 [+0.562, +0.926] | -0.93 to 2.42 | 57.5 % | 0.541 [0.409, 0.656] | 55 % |
| C_min_lipanchored (17.37 px/mm) | 87 | 0.781 | 0.938 | 0.861 | 0.816 [0.629, 0.899] | +0.465 [+0.290, +0.639] | -1.14 to 2.07 | 66.7 % | 0.614 [0.491, 0.725] | 74 % |
| **C_p25 + constant +0.694 mm (clipped at 0)** | 87 | 0.522 | 0.786 | 0.872 | 0.862 [0.796, 0.908] | +0.000 [-0.169, +0.169] | -1.55 to 1.55 | 77.0 % | 0.705 [0.572, 0.813] | 87 % |

### Holdout (the decision table)
| candidate (scale) | n | MAE | RMSE | r | ICC(2,1) [CI] | bias [CI] | 95 % LoA | label agr. | κ linear [CI] | within 1 mm | Δ MAE vs offset |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_p25 (16.84 px/mm) | 57 | 0.757 | 0.917 | 0.894 | 0.797 [0.191, 0.926] | +0.670 [+0.502, +0.838] | -0.57 to 1.91 | 64.9 % | 0.611 [0.462, 0.747] | 74 % | Δ vs offset +0.303 |
| C_p25_lipanchored (19.02 px/mm) | 57 | 0.579 | 0.740 | 0.888 | 0.847 [0.605, 0.928] | +0.415 [+0.251, +0.579] | -0.80 to 1.63 | 73.7 % | 0.693 [0.557, 0.819] | 89 % | Δ vs offset +0.124 |
| C_p10 (16.04 px/mm) | 57 | 0.825 | 0.987 | 0.893 | 0.777 [0.107, 0.921] | +0.747 [+0.574, +0.919] | -0.53 to 2.02 | 59.6 % | 0.557 [0.408, 0.697] | 70 % | Δ vs offset +0.370 |
| C_p10_lipanchored (18.18 px/mm) | 57 | 0.628 | 0.789 | 0.885 | 0.833 [0.518, 0.926] | +0.476 [+0.308, +0.645] | -0.77 to 1.72 | 68.4 % | 0.643 [0.502, 0.771] | 88 % | Δ vs offset +0.174 |
| C_p05 (15.70 px/mm) | 57 | 0.872 | 1.032 | 0.891 | 0.763 [0.066, 0.917] | +0.794 [+0.617, +0.971] | -0.51 to 2.10 | 57.9 % | 0.535 [0.384, 0.674] | 67 % | Δ vs offset +0.417 |
| C_p05_lipanchored (17.86 px/mm) | 57 | 0.657 | 0.820 | 0.882 | 0.823 [0.467, 0.924] | +0.511 [+0.339, +0.683] | -0.76 to 1.78 | 66.7 % | 0.628 [0.486, 0.758] | 84 % | Δ vs offset +0.202 |
| C_min (15.16 px/mm) | 57 | 0.960 | 1.123 | 0.884 | 0.736 [0.023, 0.907] | +0.876 [+0.688, +1.064] | -0.51 to 2.26 | 50.9 % | 0.470 [0.318, 0.612] | 56 % | Δ vs offset +0.506 |
| C_min_lipanchored (17.37 px/mm) | 57 | 0.706 | 0.874 | 0.876 | 0.806 [0.397, 0.919] | +0.563 [+0.384, +0.742] | -0.76 to 1.88 | 64.9 % | 0.587 [0.439, 0.721] | 75 % | Δ vs offset +0.251 |
| **C_p25 + constant +0.694 mm (clipped at 0)** | 57 | 0.454 | 0.627 | 0.894 | 0.894 [0.826, 0.936] | -0.024 [-0.191, +0.144] | -1.26 to 1.21 | 82.5 % | 0.794 [0.680, 0.895] | 91 % | Δ vs offset +0.000 |

Rule stated in advance: an estimator qualifies when its holdout MAE is within 0.05 mm of C_p25 + offset (0.454 mm). **Qualifying: none.**
No estimator gets within 0.05 mm of the offset-corrected MAE; the bias is a shift of the lower edge, not a spread that a lower percentile can absorb (the percentiles act on the column profile *within* each region, whereas the extra thickness is present in every column).
Note the biases: a lower percentile lowers the value everywhere, including where the segmentation is right, and its residual bias on holdout is C_p25_lipanchored +0.41, C_p10_lipanchored +0.48, C_p05_lipanchored +0.51, C_min_lipanchored +0.56 mm vs -0.02 for the offset.

## 2. Independent check: 29 test-set high images, final model
These images were used neither for the estimator choice nor for the offset, and the masks come from the final model, not from the fold models.

| masks | candidate | n | MAE | RMSE | r | ICC(2,1) [CI] | bias [CI] | 95 % LoA | label agr. | κ linear [CI] | within 1 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| final model | C_p25, uncorrected | 29 | 0.953 | 1.108 | 0.804 | 0.731 [0.344, 0.884] | +0.647 [+0.299, +0.995] | -1.15 to 2.44 | 48.3 % | 0.401 [0.205, 0.588] | 62 % |
| final model | C_p25 + constant +0.694 mm (clipped) | 29 | 0.504 | 0.900 | 0.804 | 0.804 [0.623, 0.903] | -0.047 [-0.395, +0.301] | -1.84 to 1.75 | 72.4 % | 0.609 [0.350, 0.827] | 83 % |
| final model | C_p25_lipanchored | 29 | 0.697 | 0.944 | 0.807 | 0.779 [0.576, 0.891] | +0.327 [-0.015, +0.670] | -1.44 to 2.09 | 62.1 % | 0.565 [0.375, 0.738] | 86 % |
| final model | C_p10 | 29 | 1.074 | 1.216 | 0.773 | 0.692 [0.276, 0.865] | +0.724 [+0.346, +1.102] | -1.22 to 2.67 | 41.4 % | 0.353 [0.164, 0.534] | 52 % |
| final model | C_p10_lipanchored | 29 | 0.814 | 1.035 | 0.767 | 0.743 [0.518, 0.871] | +0.350 [-0.026, +0.727] | -1.59 to 2.29 | 62.1 % | 0.512 [0.259, 0.725] | 83 % |
| final model | C_p05 | 29 | 1.112 | 1.253 | 0.768 | 0.681 [0.243, 0.862] | +0.764 [+0.380, +1.148] | -1.22 to 2.74 | 37.9 % | 0.307 [0.123, 0.487] | 48 % |
| final model | C_p05_lipanchored | 29 | 0.863 | 1.076 | 0.750 | 0.731 [0.501, 0.864] | +0.356 [-0.037, +0.749] | -1.67 to 2.38 | 58.6 % | 0.492 [0.253, 0.692] | 79 % |
| final model | C_min | 29 | 1.191 | 1.332 | 0.759 | 0.659 [0.179, 0.855] | +0.851 [+0.454, +1.248] | -1.19 to 2.90 | 31.0 % | 0.234 [0.020, 0.418] | 41 % |
| final model | C_min_lipanchored | 29 | 0.925 | 1.133 | 0.734 | 0.713 [0.470, 0.855] | +0.398 [-0.013, +0.808] | -1.72 to 2.51 | 55.2 % | 0.444 [0.212, 0.651] | 69 % |
| fold models (OOF) | C_p25, uncorrected | 29 | 0.925 | 1.111 | 0.798 | 0.732 [0.371, 0.882] | +0.626 [+0.271, +0.981] | -1.21 to 2.46 | 58.6 % | 0.503 [0.301, 0.703] | 62 % |
| fold models (OOF) | C_p25 + constant +0.694 mm (clipped) | 29 | 0.547 | 0.921 | 0.798 | 0.799 [0.615, 0.900] | -0.068 [-0.423, +0.288] | -1.90 to 1.76 | 72.4 % | 0.609 [0.350, 0.828] | 86 % |

Bias of the final model on these images: +0.647 mm [+0.299, +0.995]; fold models on the same images: +0.626 mm [+0.271, +0.981]; paired difference final − fold +0.021 mm [-0.117, +0.159]. Dev offset +0.694 mm lies inside the final model's bias CI. **No warning: the final model shows the same systematic offset as the fold models; a single constant is consistent across models.**
Clipped to 0 after correction on this set: 0.

## 3. Negative values after the correction
Rule for the corrected value: `corrected_mm = max(0, pipeline_mm +0.694)`, with a boolean `clipped` flag carried into the report (the rule engine reads 0 mm as NO_VISIBLE_GINGIVA, so a clipped image must be shown as "corrected below zero, gingiva present in the mask" rather than as no visible gingiva).
* OOF masks (n = 144): **0 image(s) below 0 mm** after the correction (dev 0, holdout 0). No image is driven to 0 (the smallest corrected value is 0.38 mm; the smallest reference 0.56 mm).
* Test high, final model (n = 29): 0 below 0 mm.
* Effect of clipping on holdout: MAE 0.454 → 0.454 mm, bias -0.024 → -0.024 mm (clipping can only move a negative value towards the reference).

## 4. Stability of the offset
Mean error (uncorrected C_p25, OOF masks) with 95 % CI per group; the constant is credible if the groups agree.

| grouping | group | n | offset_mm | ci_low | ci_high | sd_mm |
|---|---|---|---|---|---|---|
| fold | fold 0 | 29 | 0.752 | 0.457 | 1.047 | 0.776 |
| fold | fold 1 | 29 | 0.697 | 0.503 | 0.890 | 0.509 |
| fold | fold 2 | 29 | 0.574 | 0.223 | 0.926 | 0.924 |
| fold | fold 3 | 28 | 0.633 | 0.404 | 0.863 | 0.591 |
| fold | fold 4 | 29 | 0.764 | 0.459 | 1.069 | 0.802 |
| frame | 2698x1799 (±2 px) | 123 | 0.665 | 0.538 | 0.791 | 0.708 |
| frame | other frame | 21 | 0.799 | 0.410 | 1.189 | 0.856 |
| Stage-3 subset | dev | 87 | 0.694 | 0.525 | 0.862 | 0.791 |
| Stage-3 subset | holdout | 57 | 0.670 | 0.502 | 0.838 | 0.632 |
| main split | test | 29 | 0.626 | 0.271 | 0.981 | 0.934 |
| main split | train | 87 | 0.667 | 0.535 | 0.799 | 0.618 |
| main split | valid | 28 | 0.800 | 0.480 | 1.120 | 0.826 |
| test high, final model | final | 29 | 0.647 | 0.299 | 0.995 | 0.915 |

Folds: range 0.190 mm, between-fold SD 0.080 mm, one-way ANOVA p = 0.848. Frame groups: difference 0.134 mm, p = 0.438. Dev vs holdout: p = 0.850. **The offset is systematic: every fold, both frame groups, dev and holdout and the final model agree within their CIs.**

## Side-by-side summary (holdout unless stated)
| set | uncorrected MAE / bias | corrected MAE / bias |
|---|---|---|
| dev (n = 87) | 0.894 / +0.694 | 0.522 / +0.000 |
| holdout (n = 57) | 0.757 / +0.670 | 0.454 / -0.024 |
| test high, final model (n = 29) | 0.953 / +0.647 | 0.504 / -0.047 |
| test high, fold models (n = 29) | 0.925 / +0.626 | 0.547 / -0.068 |

Per-image corrected and uncorrected values: `per_image_offset.csv`. Figure: `figures/offset_checks.png`.
