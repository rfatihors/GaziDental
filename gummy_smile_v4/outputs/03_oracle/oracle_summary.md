# Oracle validation summary — ground-truth masks

## Set
Reference images (kept high with clinical measurement): 145; excluded `label_inconsistent` rows: 0 (—); measured: 145; dev 87 / holdout 58 (seed 42, lists in `dev_holdout_split.json`).

## Protocol
selection on dev MAE (n = 87); best dev MAE = 0.558 mm; 1 combination(s) within 0.02 mm of it (C_p25); the simplest of those is chosen (A > B > C, gingiva thickness > lip-anchored, p25 > median > p10 > p05 > min > max).
Scale of every combination fitted on dev only and applied unchanged to holdout; holdout metrics are reported for all combinations but were not used for selection.

## Selected method: **C_p25**, global scale **16.84 px/mm**
Holdout (n = 58): MAE 0.527 mm, RMSE 0.726 mm, r 0.860, ICC(2,1) 0.857 [0.770, 0.912], bias +0.125 mm [-0.065, +0.315], LoA -1.29 to 1.54 mm, proportional bias slope +0.065 (p = 0.378); threshold-label agreement 79.3 % (linear-weighted κ 0.733) — a *measurement* check against Table 1, not a clinical validation.
Dev (n = 87): MAE 0.558 mm, r 0.880, ICC(2,1) 0.873.

**Fallback transparency.** Regioning C could not be established on 27 of 145 images (19 %, `zenith_detection_failed`); there the measurement silently uses the equal-split regions (A) and the value is identical to `A_p25`. On the dev images where C succeeded (n = 72), dev MAE is 0.513 mm for `C_p25` vs 0.597 mm for `A_p25` — the advantage of C comes from these images, not from the fallback ones.

Fallback distribution (27 images): zenith candidates found left+right of the midline (3+3 needed): 0+1: 1, 2+0: 1, 2+2: 5, 2+3: 9, 2+4: 1, 3+2: 9, 4+2: 1. Total minima on fallback images: median 5 vs 6 on successful ones. Gingiva band width (fraction of image width, a proxy for premolar visibility): fallback 0.316 vs success 0.352; reference mm: fallback 2.60 vs success 2.90. Fallback images are not wider, so premolar visibility is not the main cause; the typical failure is one side of the midline having fewer than three detectable minima (a shallow festoon on that side).

**Sensitivity analysis — `A_p25` (no fallback, equal-split regions) side by side:** holdout MAE 0.537 vs 0.527 mm, RMSE 0.744 vs 0.726, r 0.849 vs 0.860, ICC(2,1) 0.843 vs 0.857, bias +0.187 vs +0.125 mm, scale 17.04 vs 16.84 px/mm.

**Stage 6 note:** on predicted masks the fallback rate of `C_p25` will be re-measured; if it exceeds 30 % the selection is re-evaluated against `A_p25`.

**Against the reference's own repeatability:** intra-observer SD is 0.17 mm per tooth site and 0.09 mm per image mean (`intra_observer.md`); pure observer noise would produce an expected absolute difference of ≈ 0.07 mm at image level. The holdout MAE of 0.53 mm therefore leaves ≈ 0.46 mm above the observer-noise floor, attributable to the estimator, the single global scale (per-image calibration was not recorded) and region alignment.

Pre-analysis plausibility check (audit §6, A/p25, same-data scale, n = 148): r ≈ 0.83, MAE ≈ 0.63 mm, ≈ 17 px/mm — consistent.

## All combinations (sorted by dev MAE; holdout columns for reporting only)
| combo | px_per_mm_dev | mae_dev | mae_holdout | rmse_holdout | r_holdout | icc2_1_holdout | icc2_1_ci_low_holdout | icc2_1_ci_high_holdout | ba_bias_holdout | ba_loa_low_holdout | ba_loa_high_holdout | ba_prop_slope_holdout | ba_prop_p_holdout | threshold_agreement_holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_p25 | 16.842 | 0.558 | 0.527 | 0.726 | 0.860 | 0.857 | 0.770 | 0.912 | 0.125 | -1.290 | 1.540 | 0.065 | 0.378 | 0.793 |
| C_p10 | 16.053 | 0.580 | 0.547 | 0.764 | 0.854 | 0.850 | 0.759 | 0.908 | 0.063 | -1.442 | 1.568 | 0.122 | 0.107 | 0.776 |
| C_median | 17.997 | 0.594 | 0.556 | 0.793 | 0.828 | 0.820 | 0.710 | 0.890 | 0.209 | -1.303 | 1.721 | 0.000 | 0.996 | 0.707 |
| C_p05 | 15.721 | 0.601 | 0.568 | 0.812 | 0.842 | 0.836 | 0.737 | 0.899 | 0.035 | -1.568 | 1.638 | 0.155 | 0.051 | 0.776 |
| B_p10 | 14.285 | 0.602 | 0.588 | 0.768 | 0.842 | 0.840 | 0.744 | 0.902 | 0.116 | -1.385 | 1.616 | 0.067 | 0.392 | 0.707 |
| C_p25_lipanchored | 19.012 | 0.604 | 0.516 | 0.734 | 0.870 | 0.854 | 0.743 | 0.916 | 0.272 | -1.077 | 1.620 | 0.050 | 0.481 | 0.793 |
| B_p05 | 13.069 | 0.607 | 0.629 | 0.810 | 0.831 | 0.828 | 0.726 | 0.894 | 0.091 | -1.500 | 1.681 | 0.105 | 0.199 | 0.724 |
| C_p10_lipanchored | 18.191 | 0.608 | 0.527 | 0.752 | 0.868 | 0.853 | 0.755 | 0.913 | 0.236 | -1.176 | 1.648 | 0.102 | 0.154 | 0.793 |
| B_min | 10.801 | 0.609 | 0.672 | 0.885 | 0.819 | 0.810 | 0.699 | 0.883 | 0.071 | -1.673 | 1.815 | 0.182 | 0.034 | 0.741 |
| C_p05_lipanchored | 17.868 | 0.621 | 0.545 | 0.780 | 0.861 | 0.846 | 0.749 | 0.907 | 0.218 | -1.263 | 1.698 | 0.130 | 0.079 | 0.793 |
| A_p25 | 17.036 | 0.627 | 0.537 | 0.744 | 0.849 | 0.843 | 0.746 | 0.905 | 0.187 | -1.236 | 1.610 | 0.016 | 0.836 | 0.793 |
| A_p10 | 14.642 | 0.630 | 0.562 | 0.769 | 0.844 | 0.842 | 0.747 | 0.903 | 0.091 | -1.418 | 1.601 | 0.084 | 0.283 | 0.759 |
| B_p25 | 16.734 | 0.637 | 0.544 | 0.719 | 0.858 | 0.852 | 0.759 | 0.910 | 0.183 | -1.192 | 1.558 | 0.003 | 0.964 | 0.759 |
| A_p05 | 13.393 | 0.637 | 0.607 | 0.817 | 0.830 | 0.827 | 0.724 | 0.894 | 0.037 | -1.577 | 1.652 | 0.119 | 0.147 | 0.793 |
| B_min_lipanchored | 13.534 | 0.643 | 0.627 | 0.844 | 0.840 | 0.822 | 0.707 | 0.893 | 0.260 | -1.327 | 1.848 | 0.135 | 0.091 | 0.690 |
| B_p05_lipanchored | 15.373 | 0.647 | 0.568 | 0.788 | 0.853 | 0.836 | 0.720 | 0.904 | 0.276 | -1.183 | 1.734 | 0.078 | 0.304 | 0.707 |
| C_min | 15.180 | 0.648 | 0.658 | 0.915 | 0.820 | 0.806 | 0.692 | 0.880 | 0.005 | -1.804 | 1.814 | 0.225 | 0.009 | 0.759 |
| C_median_lipanchored | 20.208 | 0.650 | 0.571 | 0.805 | 0.839 | 0.816 | 0.668 | 0.895 | 0.330 | -1.122 | 1.782 | -0.010 | 0.895 | 0.690 |
| B_p10_lipanchored | 16.503 | 0.651 | 0.559 | 0.769 | 0.860 | 0.841 | 0.717 | 0.909 | 0.297 | -1.106 | 1.700 | 0.054 | 0.463 | 0.741 |
| C_min_lipanchored | 17.386 | 0.654 | 0.597 | 0.845 | 0.846 | 0.829 | 0.725 | 0.895 | 0.200 | -1.424 | 1.824 | 0.183 | 0.020 | 0.759 |
| A_min | 10.898 | 0.663 | 0.757 | 0.990 | 0.784 | 0.771 | 0.641 | 0.858 | -0.040 | -1.995 | 1.915 | 0.223 | 0.019 | 0.741 |
| A_p10_lipanchored | 16.846 | 0.665 | 0.548 | 0.760 | 0.860 | 0.845 | 0.735 | 0.909 | 0.262 | -1.149 | 1.673 | 0.061 | 0.407 | 0.776 |
| A_p05_lipanchored | 15.634 | 0.672 | 0.590 | 0.788 | 0.850 | 0.837 | 0.733 | 0.902 | 0.233 | -1.254 | 1.721 | 0.091 | 0.234 | 0.793 |
| A_p25_lipanchored | 19.252 | 0.674 | 0.534 | 0.750 | 0.864 | 0.841 | 0.701 | 0.912 | 0.320 | -1.023 | 1.662 | 0.001 | 0.991 | 0.759 |
| B_p25_lipanchored | 18.943 | 0.677 | 0.554 | 0.747 | 0.867 | 0.843 | 0.699 | 0.914 | 0.327 | -1.002 | 1.655 | 0.004 | 0.957 | 0.741 |
| A_min_lipanchored | 13.625 | 0.680 | 0.677 | 0.880 | 0.819 | 0.808 | 0.696 | 0.881 | 0.160 | -1.550 | 1.871 | 0.156 | 0.068 | 0.707 |
| A_median | 20.652 | 0.731 | 0.525 | 0.728 | 0.847 | 0.826 | 0.715 | 0.896 | 0.223 | -1.147 | 1.592 | -0.159 | 0.042 | 0.776 |
| B_median | 20.404 | 0.745 | 0.553 | 0.750 | 0.849 | 0.821 | 0.682 | 0.898 | 0.297 | -1.064 | 1.658 | -0.137 | 0.077 | 0.759 |
| A_median_lipanchored | 22.897 | 0.758 | 0.531 | 0.754 | 0.852 | 0.818 | 0.661 | 0.898 | 0.324 | -1.022 | 1.671 | -0.152 | 0.047 | 0.741 |
| B_median_lipanchored | 22.620 | 0.776 | 0.576 | 0.800 | 0.851 | 0.805 | 0.578 | 0.900 | 0.411 | -0.944 | 1.767 | -0.118 | 0.124 | 0.793 |
| C_max | 25.093 | 0.826 | 0.824 | 1.143 | 0.612 | 0.575 | 0.362 | 0.729 | 0.399 | -1.718 | 2.516 | -0.219 | 0.098 | 0.586 |
| C_max_lipanchored | 27.356 | 0.856 | 0.797 | 1.126 | 0.651 | 0.596 | 0.357 | 0.753 | 0.483 | -1.529 | 2.495 | -0.217 | 0.080 | 0.655 |
| A_max | 34.800 | 1.002 | 0.798 | 1.002 | 0.763 | 0.620 | 0.326 | 0.784 | 0.512 | -1.191 | 2.215 | -0.526 | 0.000 | 0.552 |
| A_max_lipanchored | 37.005 | 1.018 | 0.808 | 1.009 | 0.781 | 0.629 | 0.280 | 0.802 | 0.570 | -1.077 | 2.216 | -0.489 | 0.000 | 0.638 |
| B_max | 36.671 | 1.025 | 0.837 | 1.039 | 0.749 | 0.589 | 0.279 | 0.766 | 0.545 | -1.203 | 2.294 | -0.570 | 0.000 | 0.621 |
| B_max_lipanchored | 38.882 | 1.043 | 0.848 | 1.038 | 0.778 | 0.603 | 0.235 | 0.788 | 0.602 | -1.072 | 2.275 | -0.541 | 0.000 | 0.586 |

## Bland–Altman direction and proportional bias (holdout)
Sign of `ba_bias`: positive = GT-mask value above the clinical reference. `ba_prop_slope` is the slope of (GT − ref) on the mean of both; p < 0.05 indicates a proportional (mm-dependent) bias. For the selected method: bias grows with gingival display (slope +0.065, p = 0.378).

## Sensitivity (selected method, scale fixed)
| subset | n | mae | rmse | r | r_p | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | icc3_1 | ba_bias | ba_bias_ci_low | ba_bias_ci_high | ba_sd | ba_loa_low | ba_loa_high | ba_prop_slope | ba_prop_p | threshold_agreement | threshold_kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout: all | 58 | 0.527 | 0.726 | 0.860 | 0.000 | 0.857 | 0.770 | 0.912 | 0.858 | 0.125 | -0.065 | 0.315 | 0.722 | -1.290 | 1.540 | 0.065 | 0.378 | 0.793 | 0.733 |
| holdout: without dash-zero rows | 45 | 0.485 | 0.617 | 0.878 | 0.000 | 0.878 | 0.789 | 0.931 | 0.878 | 0.089 | -0.097 | 0.274 | 0.617 | -1.121 | 1.298 | 0.043 | 0.585 | 0.800 | 0.793 |
| holdout: only 2698x1799 (+/-2 px) frames | 48 | 0.476 | 0.604 | 0.907 | 0.000 | 0.902 | 0.833 | 0.944 | 0.902 | 0.069 | -0.107 | 0.245 | 0.607 | -1.120 | 1.258 | 0.115 | 0.083 | 0.812 | 0.756 |
| holdout: frames outside 2698x1799 | 10 | 0.777 | 1.144 | 0.620 | 0.056 | 0.575 | -0.002 | 0.872 | 0.580 | 0.393 | -0.417 | 1.203 | 1.132 | -1.826 | 2.612 | -0.445 | 0.213 | 0.700 | 0.623 |
| holdout: without ambiguous 100-999 cells | 58 | 0.527 | 0.726 | 0.860 | 0.000 | 0.857 | 0.770 | 0.912 | 0.858 | 0.125 | -0.065 | 0.315 | 0.722 | -1.290 | 1.540 | 0.065 | 0.378 | 0.793 | 0.733 |
| dev + holdout (scale still from dev) | 145 | 0.546 | 0.761 | 0.871 | 0.000 | 0.867 | 0.818 | 0.903 | 0.871 | 0.154 | 0.031 | 0.277 | 0.748 | -1.311 | 1.619 | -0.040 | 0.367 | 0.786 | 0.742 |

## Lip–gingiva boundary (n = 144)
Gap between the lowest lip pixel and the gingiva top: median 8.0 px (IQR 5.0–12.0; mean 8.4; max 47). Expected ≈ 9 px (IQR 6–12) from the audit. Consistent. Boundary flag raised on 1 images.
Lip-anchored estimators (C_p25_lipanchored): dev MAE 0.604 mm vs 0.558 mm for gingiva thickness.

## Tooth level (secondary; region i left-to-right vs reference tooth i; A/B/C alignment is approximate)
| tooth_index | n | mae | bias | r |
|---|---|---|---|---|
| 1 | 145 | 0.734 | 0.174 | 0.837 |
| 2 | 145 | 0.776 | 0.225 | 0.839 |
| 3 | 145 | 0.591 | 0.174 | 0.892 |
| 4 | 145 | 0.578 | -0.068 | 0.842 |
| 5 | 145 | 0.831 | 0.131 | 0.765 |
| 6 | 145 | 0.773 | 0.288 | 0.803 |

## QC flags
zenith_detection_failed          27
frame_uncertain                  22
gingiva_multi_component           3
festoon_detection_failed          2
lip_gingiva_boundary_mismatch     1
Frame outside 2698×1799 ±2 px: 22 images (`frame_uncertain`; sensitivity rows above).

## Assumptions
- COCO frame = ImageJ frame (clinical team, screenshot); for other frame sizes the reference frame is uncertain.
- `-` cells are 0 mm (clinical decision); the sensitivity row without dash-zero rows shows the effect.
- The reference gingival display recorded at tooth site i is compared to the left-to-right region of the same index; the image-level mean is the primary endpoint.
- v1 (XGBoost) column of the original Figure 6 is not reproduced: the regressor was trained on 512×512 gingiva-only DeepLab masks and fed lip+gingiva masks at 1024 px in v3, i.e. inputs outside its training distribution (audit §3).
