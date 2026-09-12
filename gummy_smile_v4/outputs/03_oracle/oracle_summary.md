# Oracle validation summary — ground-truth masks

## Set
Reference images (kept high with clinical measurement): 145; excluded `label_inconsistent` rows: 0 (—); measured: 145; dev 87 / holdout 58 (seed 42, lists in `dev_holdout_split.json`).

## Protocol
selection on dev MAE (n = 87); best dev MAE = 0.557 mm; 1 combination(s) within 0.02 mm of it (C_p25); the simplest of those is chosen (A > B > C, gingiva thickness > lip-anchored, p25 > median > p10 > min > max).
Scale of every combination fitted on dev only and applied unchanged to holdout; holdout metrics are reported for all combinations but were not used for selection.

## Selected method: **C_p25**, global scale **16.84 px/mm**
Holdout (n = 58): MAE 0.518 mm, RMSE 0.718 mm, r 0.861, ICC(2,1) 0.858 [0.773, 0.914], bias +0.116 mm [-0.071, +0.304], LoA -1.28 to 1.52 mm, proportional bias slope +0.054 (p = 0.465); threshold-label agreement 79.3 % (linear-weighted κ 0.733) — a *measurement* check against Table 1, not a clinical validation.
Dev (n = 87): MAE 0.557 mm, r 0.881, ICC(2,1) 0.873.

**Fallback transparency.** Regioning C could not be established on 28 of 145 images (`zenith_detection_failed`); there the measurement silently uses the equal-split regions (A) and the value is identical to `A_p25`. On the dev images where C succeeded (n = 72), dev MAE is 0.512 mm for `C_p25` vs 0.597 mm for `A_p25` — the advantage of C comes from these images, not from the fallback ones.

**Against the reference's own repeatability:** intra-observer SD is 0.17 mm per tooth and 0.09 mm per image mean (`intra_observer.md`); pure observer noise would produce an expected absolute difference of ≈ 0.07 mm at image level. The holdout MAE of 0.52 mm therefore leaves ≈ 0.45 mm above the observer-noise floor, attributable to the estimator, the single global scale (per-image calibration was not recorded) and region alignment.

Pre-analysis plausibility check (audit §6, A/p25, same-data scale, n = 148): r ≈ 0.83, MAE ≈ 0.63 mm, ≈ 17 px/mm — consistent.

## All combinations (sorted by dev MAE; holdout columns for reporting only)
| combo | px_per_mm_dev | mae_dev | mae_holdout | rmse_holdout | r_holdout | icc2_1_holdout | icc2_1_ci_low_holdout | icc2_1_ci_high_holdout | ba_bias_holdout | ba_loa_low_holdout | ba_loa_high_holdout | ba_prop_slope_holdout | ba_prop_p_holdout | threshold_agreement_holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_p25 | 16.840 | 0.557 | 0.518 | 0.718 | 0.861 | 0.858 | 0.773 | 0.914 | 0.116 | -1.284 | 1.516 | 0.054 | 0.465 | 0.793 |
| C_p10 | 16.040 | 0.578 | 0.533 | 0.752 | 0.853 | 0.851 | 0.760 | 0.909 | 0.042 | -1.443 | 1.527 | 0.101 | 0.186 | 0.776 |
| C_median | 17.993 | 0.593 | 0.560 | 0.796 | 0.827 | 0.819 | 0.708 | 0.890 | 0.213 | -1.304 | 1.730 | 0.005 | 0.954 | 0.707 |
| C_p25_lipanchored | 19.015 | 0.605 | 0.508 | 0.729 | 0.870 | 0.854 | 0.746 | 0.915 | 0.263 | -1.081 | 1.608 | 0.042 | 0.555 | 0.793 |
| C_p10_lipanchored | 18.183 | 0.608 | 0.523 | 0.749 | 0.864 | 0.852 | 0.756 | 0.911 | 0.217 | -1.201 | 1.635 | 0.088 | 0.230 | 0.776 |
| B_min | 10.732 | 0.624 | 0.687 | 0.894 | 0.819 | 0.808 | 0.696 | 0.881 | 0.092 | -1.666 | 1.851 | 0.190 | 0.027 | 0.741 |
| B_p10 | 14.217 | 0.624 | 0.589 | 0.771 | 0.843 | 0.840 | 0.744 | 0.902 | 0.132 | -1.370 | 1.634 | 0.073 | 0.351 | 0.707 |
| A_p25 | 17.036 | 0.627 | 0.537 | 0.744 | 0.849 | 0.843 | 0.746 | 0.905 | 0.187 | -1.236 | 1.610 | 0.016 | 0.836 | 0.793 |
| A_p10 | 14.642 | 0.630 | 0.562 | 0.769 | 0.844 | 0.842 | 0.747 | 0.903 | 0.091 | -1.418 | 1.601 | 0.084 | 0.283 | 0.759 |
| B_p25 | 16.661 | 0.644 | 0.542 | 0.723 | 0.859 | 0.851 | 0.756 | 0.910 | 0.199 | -1.175 | 1.572 | 0.008 | 0.919 | 0.759 |
| C_min | 15.160 | 0.646 | 0.653 | 0.909 | 0.813 | 0.803 | 0.688 | 0.879 | -0.028 | -1.824 | 1.768 | 0.199 | 0.022 | 0.741 |
| B_min_lipanchored | 13.476 | 0.646 | 0.637 | 0.853 | 0.840 | 0.820 | 0.702 | 0.892 | 0.273 | -1.324 | 1.871 | 0.143 | 0.074 | 0.690 |
| C_median_lipanchored | 20.210 | 0.651 | 0.573 | 0.807 | 0.839 | 0.815 | 0.667 | 0.895 | 0.332 | -1.122 | 1.786 | -0.008 | 0.924 | 0.690 |
| C_min_lipanchored | 17.372 | 0.653 | 0.602 | 0.851 | 0.836 | 0.823 | 0.718 | 0.891 | 0.170 | -1.478 | 1.818 | 0.166 | 0.041 | 0.741 |
| B_p10_lipanchored | 16.446 | 0.658 | 0.566 | 0.775 | 0.860 | 0.839 | 0.711 | 0.909 | 0.308 | -1.099 | 1.715 | 0.060 | 0.414 | 0.741 |
| A_min | 10.898 | 0.663 | 0.757 | 0.990 | 0.784 | 0.771 | 0.641 | 0.858 | -0.040 | -1.995 | 1.915 | 0.223 | 0.019 | 0.741 |
| A_p10_lipanchored | 16.846 | 0.665 | 0.548 | 0.760 | 0.860 | 0.845 | 0.735 | 0.909 | 0.262 | -1.149 | 1.673 | 0.061 | 0.407 | 0.776 |
| A_p25_lipanchored | 19.252 | 0.674 | 0.534 | 0.750 | 0.864 | 0.841 | 0.701 | 0.912 | 0.320 | -1.023 | 1.662 | 0.001 | 0.991 | 0.759 |
| A_min_lipanchored | 13.625 | 0.680 | 0.677 | 0.880 | 0.819 | 0.808 | 0.696 | 0.881 | 0.160 | -1.550 | 1.871 | 0.156 | 0.068 | 0.707 |
| B_p25_lipanchored | 18.891 | 0.682 | 0.558 | 0.752 | 0.868 | 0.842 | 0.693 | 0.914 | 0.335 | -0.996 | 1.665 | 0.009 | 0.900 | 0.741 |
| A_median | 20.652 | 0.731 | 0.525 | 0.728 | 0.847 | 0.826 | 0.715 | 0.896 | 0.223 | -1.147 | 1.592 | -0.159 | 0.042 | 0.776 |
| B_median | 20.330 | 0.744 | 0.554 | 0.754 | 0.849 | 0.821 | 0.678 | 0.898 | 0.305 | -1.058 | 1.668 | -0.128 | 0.098 | 0.776 |
| A_median_lipanchored | 22.897 | 0.758 | 0.531 | 0.754 | 0.852 | 0.818 | 0.661 | 0.898 | 0.324 | -1.022 | 1.671 | -0.152 | 0.047 | 0.741 |
| B_median_lipanchored | 22.550 | 0.776 | 0.578 | 0.804 | 0.851 | 0.804 | 0.574 | 0.900 | 0.417 | -0.942 | 1.775 | -0.109 | 0.156 | 0.741 |
| C_max | 25.090 | 0.825 | 0.845 | 1.156 | 0.619 | 0.582 | 0.364 | 0.735 | 0.429 | -1.693 | 2.552 | -0.171 | 0.190 | 0.586 |
| C_max_lipanchored | 27.350 | 0.855 | 0.814 | 1.136 | 0.659 | 0.603 | 0.353 | 0.761 | 0.511 | -1.496 | 2.517 | -0.179 | 0.143 | 0.655 |
| A_max | 34.800 | 1.002 | 0.798 | 1.002 | 0.763 | 0.620 | 0.326 | 0.784 | 0.512 | -1.191 | 2.215 | -0.526 | 0.000 | 0.552 |
| A_max_lipanchored | 37.005 | 1.018 | 0.808 | 1.009 | 0.781 | 0.629 | 0.280 | 0.802 | 0.570 | -1.077 | 2.216 | -0.489 | 0.000 | 0.638 |
| B_max | 36.577 | 1.032 | 0.841 | 1.044 | 0.748 | 0.587 | 0.272 | 0.766 | 0.554 | -1.195 | 2.303 | -0.565 | 0.000 | 0.603 |
| B_max_lipanchored | 38.785 | 1.050 | 0.851 | 1.043 | 0.778 | 0.602 | 0.228 | 0.789 | 0.609 | -1.063 | 2.282 | -0.535 | 0.000 | 0.586 |

## Bland–Altman direction and proportional bias (holdout)
Sign of `ba_bias`: positive = GT-mask value above the clinical reference. `ba_prop_slope` is the slope of (GT − ref) on the mean of both; p < 0.05 indicates a proportional (mm-dependent) bias. For the selected method: bias grows with gingival display (slope +0.054, p = 0.465).

## Sensitivity (selected method, scale fixed)
| subset | n | mae | rmse | r | r_p | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | icc3_1 | ba_bias | ba_bias_ci_low | ba_bias_ci_high | ba_sd | ba_loa_low | ba_loa_high | ba_prop_slope | ba_prop_p | threshold_agreement | threshold_kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout: all | 58 | 0.518 | 0.718 | 0.861 | 0.000 | 0.858 | 0.773 | 0.914 | 0.860 | 0.116 | -0.071 | 0.304 | 0.714 | -1.284 | 1.516 | 0.054 | 0.465 | 0.793 | 0.733 |
| holdout: without dash-zero rows | 45 | 0.473 | 0.603 | 0.881 | 0.000 | 0.881 | 0.795 | 0.933 | 0.881 | 0.077 | -0.104 | 0.259 | 0.605 | -1.108 | 1.263 | 0.028 | 0.716 | 0.800 | 0.793 |
| holdout: only 2698x1799 (+/-2 px) frames | 48 | 0.464 | 0.591 | 0.909 | 0.000 | 0.905 | 0.837 | 0.946 | 0.904 | 0.059 | -0.114 | 0.231 | 0.594 | -1.107 | 1.224 | 0.102 | 0.121 | 0.812 | 0.756 |
| holdout: frames outside 2698x1799 | 10 | 0.777 | 1.144 | 0.620 | 0.056 | 0.575 | -0.002 | 0.872 | 0.580 | 0.394 | -0.416 | 1.204 | 1.132 | -1.825 | 2.613 | -0.445 | 0.213 | 0.700 | 0.623 |
| holdout: without ambiguous 100-999 cells | 58 | 0.518 | 0.718 | 0.861 | 0.000 | 0.858 | 0.773 | 0.914 | 0.860 | 0.116 | -0.071 | 0.304 | 0.714 | -1.284 | 1.516 | 0.054 | 0.465 | 0.793 | 0.733 |
| dev + holdout (scale still from dev) | 145 | 0.542 | 0.756 | 0.872 | 0.000 | 0.868 | 0.819 | 0.904 | 0.872 | 0.150 | 0.028 | 0.272 | 0.744 | -1.307 | 1.608 | -0.044 | 0.315 | 0.786 | 0.742 |

## Lip–gingiva boundary (n = 144)
Gap between the lowest lip pixel and the gingiva top: median 8.0 px (IQR 5.0–12.0; mean 8.4; max 47). Expected ≈ 9 px (IQR 6–12) from the audit. Consistent. Boundary flag raised on 1 images.
Lip-anchored estimators (C_p25_lipanchored): dev MAE 0.605 mm vs 0.557 mm for gingiva thickness.

## Tooth level (secondary; region i left-to-right vs reference tooth i; A/B/C alignment is approximate)
| tooth_index | n | mae | bias | r |
|---|---|---|---|---|
| 1 | 145 | 0.739 | 0.159 | 0.834 |
| 2 | 145 | 0.777 | 0.226 | 0.839 |
| 3 | 145 | 0.592 | 0.176 | 0.892 |
| 4 | 145 | 0.578 | -0.068 | 0.842 |
| 5 | 145 | 0.826 | 0.126 | 0.765 |
| 6 | 145 | 0.767 | 0.283 | 0.804 |

## QC flags
zenith_detection_failed          28
frame_uncertain                  22
gingiva_multi_component           3
festoon_detection_failed          2
lip_gingiva_boundary_mismatch     1
Frame outside 2698×1799 ±2 px: 22 images (`frame_uncertain`; sensitivity rows above).

## Assumptions
- COCO frame = ImageJ frame (clinical team, screenshot); for other frame sizes the reference frame is uncertain.
- `-` cells are 0 mm (clinical decision); the sensitivity row without dash-zero rows shows the effect.
- Reference per tooth is compared to the left-to-right region of the same index; the image-level mean is the primary endpoint.
- v1 (XGBoost) column of the original Figure 6 is not reproduced: the regressor was trained on 512×512 gingiva-only DeepLab masks and fed lip+gingiva masks at 1024 px in v3, i.e. inputs outside its training distribution (audit §3).
