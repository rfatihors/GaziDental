# Oracle validation summary — predicted masks: outputs/05_predictions/oof_rfdetr

## Set
Reference images (kept high with clinical measurement): 145; excluded `label_inconsistent` rows: 0 (—); measured: 145; dev 87 / holdout 58 (seed 42, lists in `dev_holdout_split.json`).

## Protocol
**No selection was made here.** Method `C_p25` and scale 16.8397 px/mm are taken from `configs/config.yaml`, where Stage 3 put them after selecting them on ground-truth masks; on predicted masks they are applied unchanged (the measurement geometry is a property of the method, not of the segmentation model, and re-selecting here would be a second look at the same clinical reference). `estimator_comparison.csv` is written for information only: every combination there carries a scale re-fitted on the dev subset of these masks, and the best dev MAE in it is `B_min_lipanchored` (0.515 mm vs 0.549 mm for the fixed method) — reported, not adopted. Re-run with `--reselect` to make that comparison the selected result.
The scales in the table below were re-fitted on the dev subset of these masks and are reported as a sensitivity analysis; the method and the scale used everywhere else in this file are the fixed ones. `--reselect` is the only way to turn that table into a selection, and it still writes nothing into `configs/config.yaml`.

## Fixed method: **C_p25**, global scale **16.84 px/mm** (from `configs/config.yaml`, Stage 3 — not selected here)
Holdout (n = 58): MAE 0.479 mm, RMSE 0.676 mm, r 0.896, ICC(2,1) 0.876 [0.756, 0.933], bias +0.294 mm [+0.133, +0.456], LoA -0.91 to 1.50 mm, proportional bias slope +0.045 (p = 0.477); threshold-label agreement 77.6 % (linear-weighted κ 0.733) — a *measurement* check against Table 1, not a clinical validation.
Dev (n = 87): MAE 0.549 mm, r 0.893, ICC(2,1) 0.878.

**Fallback transparency.** Regioning C could not be established on 27 of 145 images (19 %, `zenith_detection_failed`); there the measurement silently uses the equal-split regions (A) and the value is identical to `A_p25`. On the dev images where C succeeded (n = 70), dev MAE is 0.500 mm for `C_p25` vs 0.578 mm for `A_p25` — the advantage of C comes from these images, not from the fallback ones.

Fallback distribution (27 images): zenith candidates found left+right of the midline (3+3 needed): 2+1: 1, 2+2: 3, 2+3: 5, 3+2: 18. Total minima on fallback images: median 5 vs 6 on successful ones. Gingiva band width (fraction of image width, a proxy for premolar visibility): fallback 0.312 vs success 0.349; reference mm: fallback 2.87 vs success 2.84. Fallback images are not wider, so premolar visibility is not the main cause; the typical failure is one side of the midline having fewer than three detectable minima (a shallow festoon on that side).

**Sensitivity analysis — `A_p25` (no fallback, equal-split regions) side by side:** holdout MAE 0.483 vs 0.479 mm, RMSE 0.671 vs 0.676, r 0.889 vs 0.896, ICC(2,1) 0.872 vs 0.876, bias +0.273 vs +0.294 mm, scale 17.04 vs 16.84 px/mm.

**Stage 6 note:** on predicted masks the fallback rate of `C_p25` will be re-measured; if it exceeds 30 % the selection is re-evaluated against `A_p25`.

**Against the reference's own repeatability:** intra-observer SD is 0.17 mm per tooth site and 0.09 mm per image mean (`intra_observer.md`); pure observer noise would produce an expected absolute difference of ≈ 0.07 mm at image level. The holdout MAE of 0.48 mm therefore leaves ≈ 0.41 mm above the observer-noise floor, attributable to the estimator, the single global scale (per-image calibration was not recorded) and region alignment.

Pre-analysis plausibility check (audit §6, A/p25, same-data scale, n = 148): r ≈ 0.83, MAE ≈ 0.63 mm, ≈ 17 px/mm — consistent.

## All combinations, each with a scale re-fitted on these masks — SENSITIVITY ONLY, NOT A SELECTION
(sorted by dev MAE; the fixed method above is the result of this run whatever this table shows)
| combo | px_per_mm_dev | mae_dev | mae_holdout | rmse_holdout | r_holdout | icc2_1_holdout | icc2_1_ci_low_holdout | icc2_1_ci_high_holdout | ba_bias_holdout | ba_loa_low_holdout | ba_loa_high_holdout | ba_prop_slope_holdout | ba_prop_p_holdout | threshold_agreement_holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_min_lipanchored | 14.405 | 0.515 | 0.520 | 0.712 | 0.876 | 0.866 | 0.781 | 0.920 | 0.193 | -1.162 | 1.549 | 0.091 | 0.191 | 0.759 |
| C_p25 | 17.206 | 0.519 | 0.451 | 0.642 | 0.896 | 0.885 | 0.797 | 0.933 | 0.229 | -0.957 | 1.415 | 0.022 | 0.726 | 0.776 |
| B_min | 11.246 | 0.521 | 0.627 | 0.831 | 0.843 | 0.831 | 0.731 | 0.896 | 0.151 | -1.466 | 1.767 | 0.168 | 0.035 | 0.707 |
| B_p05 | 13.554 | 0.523 | 0.543 | 0.742 | 0.866 | 0.857 | 0.768 | 0.913 | 0.181 | -1.241 | 1.604 | 0.104 | 0.150 | 0.759 |
| B_p05_lipanchored | 16.065 | 0.531 | 0.509 | 0.696 | 0.878 | 0.869 | 0.781 | 0.921 | 0.205 | -1.109 | 1.519 | 0.057 | 0.403 | 0.759 |
| C_p25_lipanchored | 19.313 | 0.537 | 0.444 | 0.633 | 0.900 | 0.885 | 0.786 | 0.936 | 0.252 | -0.896 | 1.401 | -0.007 | 0.906 | 0.793 |
| C_p10 | 16.432 | 0.538 | 0.480 | 0.685 | 0.893 | 0.875 | 0.771 | 0.930 | 0.270 | -0.975 | 1.514 | 0.069 | 0.281 | 0.776 |
| B_p10 | 14.699 | 0.545 | 0.521 | 0.716 | 0.872 | 0.864 | 0.777 | 0.918 | 0.187 | -1.179 | 1.553 | 0.080 | 0.256 | 0.741 |
| B_p10_lipanchored | 17.032 | 0.546 | 0.499 | 0.683 | 0.881 | 0.871 | 0.784 | 0.923 | 0.208 | -1.078 | 1.494 | 0.040 | 0.551 | 0.793 |
| C_p10_lipanchored | 18.562 | 0.549 | 0.484 | 0.671 | 0.896 | 0.876 | 0.760 | 0.933 | 0.287 | -0.911 | 1.486 | 0.033 | 0.596 | 0.793 |
| C_median | 18.505 | 0.557 | 0.497 | 0.670 | 0.877 | 0.864 | 0.769 | 0.920 | 0.223 | -1.028 | 1.473 | -0.057 | 0.408 | 0.741 |
| C_p05 | 16.059 | 0.566 | 0.504 | 0.719 | 0.888 | 0.867 | 0.753 | 0.925 | 0.292 | -1.009 | 1.592 | 0.099 | 0.134 | 0.776 |
| C_p05_lipanchored | 18.234 | 0.566 | 0.506 | 0.694 | 0.892 | 0.871 | 0.748 | 0.930 | 0.302 | -0.933 | 1.537 | 0.057 | 0.372 | 0.776 |
| C_median_lipanchored | 20.626 | 0.568 | 0.478 | 0.661 | 0.883 | 0.866 | 0.760 | 0.923 | 0.250 | -0.961 | 1.461 | -0.083 | 0.217 | 0.741 |
| A_p10 | 15.071 | 0.571 | 0.489 | 0.675 | 0.876 | 0.871 | 0.789 | 0.922 | 0.169 | -1.123 | 1.461 | 0.018 | 0.799 | 0.776 |
| A_p05 | 13.744 | 0.576 | 0.539 | 0.720 | 0.862 | 0.857 | 0.770 | 0.913 | 0.148 | -1.245 | 1.541 | 0.051 | 0.486 | 0.759 |
| A_p25 | 17.432 | 0.577 | 0.453 | 0.640 | 0.889 | 0.879 | 0.795 | 0.929 | 0.203 | -0.996 | 1.403 | -0.029 | 0.654 | 0.810 |
| A_p10_lipanchored | 17.297 | 0.581 | 0.488 | 0.665 | 0.880 | 0.872 | 0.788 | 0.924 | 0.195 | -1.063 | 1.452 | -0.004 | 0.955 | 0.810 |
| A_min_lipanchored | 14.239 | 0.582 | 0.577 | 0.764 | 0.852 | 0.846 | 0.753 | 0.906 | 0.152 | -1.329 | 1.633 | 0.094 | 0.220 | 0.759 |
| B_p25 | 17.023 | 0.583 | 0.500 | 0.686 | 0.877 | 0.868 | 0.779 | 0.921 | 0.209 | -1.084 | 1.501 | 0.024 | 0.730 | 0.793 |
| A_p05_lipanchored | 16.167 | 0.586 | 0.519 | 0.690 | 0.872 | 0.866 | 0.782 | 0.919 | 0.172 | -1.148 | 1.493 | 0.027 | 0.705 | 0.776 |
| A_min | 11.118 | 0.589 | 0.678 | 0.891 | 0.812 | 0.804 | 0.690 | 0.879 | 0.103 | -1.646 | 1.852 | 0.164 | 0.059 | 0.724 |
| B_p25_lipanchored | 19.188 | 0.597 | 0.484 | 0.668 | 0.884 | 0.872 | 0.777 | 0.925 | 0.233 | -1.004 | 1.470 | -0.007 | 0.916 | 0.776 |
| A_p25_lipanchored | 19.570 | 0.598 | 0.448 | 0.639 | 0.892 | 0.877 | 0.782 | 0.930 | 0.234 | -0.940 | 1.409 | -0.056 | 0.383 | 0.793 |
| C_min_lipanchored | 17.763 | 0.608 | 0.542 | 0.735 | 0.887 | 0.860 | 0.723 | 0.925 | 0.330 | -0.969 | 1.630 | 0.092 | 0.166 | 0.776 |
| C_min | 15.453 | 0.639 | 0.570 | 0.800 | 0.876 | 0.845 | 0.711 | 0.914 | 0.335 | -1.100 | 1.770 | 0.153 | 0.029 | 0.741 |
| B_median | 20.755 | 0.700 | 0.550 | 0.743 | 0.861 | 0.823 | 0.656 | 0.904 | 0.339 | -0.969 | 1.647 | -0.154 | 0.038 | 0.759 |
| A_median | 20.834 | 0.706 | 0.528 | 0.708 | 0.875 | 0.835 | 0.678 | 0.910 | 0.319 | -0.931 | 1.569 | -0.186 | 0.009 | 0.741 |
| B_median_lipanchored | 22.889 | 0.706 | 0.540 | 0.733 | 0.872 | 0.827 | 0.635 | 0.910 | 0.361 | -0.900 | 1.622 | -0.172 | 0.016 | 0.741 |
| A_median_lipanchored | 22.960 | 0.712 | 0.519 | 0.706 | 0.881 | 0.835 | 0.659 | 0.913 | 0.338 | -0.886 | 1.563 | -0.199 | 0.004 | 0.741 |
| C_max_lipanchored | 26.564 | 0.798 | 0.670 | 0.955 | 0.727 | 0.693 | 0.518 | 0.810 | 0.315 | -1.468 | 2.098 | -0.229 | 0.034 | 0.672 |
| C_max | 24.465 | 0.812 | 0.706 | 0.994 | 0.699 | 0.673 | 0.498 | 0.794 | 0.292 | -1.586 | 2.170 | -0.201 | 0.077 | 0.672 |
| A_max_lipanchored | 35.263 | 0.963 | 0.749 | 0.954 | 0.807 | 0.658 | 0.342 | 0.815 | 0.514 | -1.075 | 2.103 | -0.503 | 0.000 | 0.690 |
| B_max_lipanchored | 36.737 | 0.963 | 0.761 | 0.950 | 0.814 | 0.663 | 0.336 | 0.821 | 0.523 | -1.045 | 2.091 | -0.496 | 0.000 | 0.621 |
| B_max | 34.641 | 0.971 | 0.777 | 0.969 | 0.794 | 0.647 | 0.337 | 0.807 | 0.515 | -1.106 | 2.137 | -0.505 | 0.000 | 0.603 |
| A_max | 33.187 | 0.972 | 0.763 | 0.969 | 0.791 | 0.646 | 0.342 | 0.805 | 0.510 | -1.121 | 2.140 | -0.509 | 0.000 | 0.638 |

## Bland–Altman direction and proportional bias (holdout)
Sign of `ba_bias`: positive = GT-mask value above the clinical reference. `ba_prop_slope` is the slope of (GT − ref) on the mean of both; p < 0.05 indicates a proportional (mm-dependent) bias. For the selected method: bias grows with gingival display (slope +0.045, p = 0.477).

## Sensitivity (selected method, scale fixed)
| subset | n | mae | rmse | r | r_p | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | icc3_1 | ba_bias | ba_bias_ci_low | ba_bias_ci_high | ba_sd | ba_loa_low | ba_loa_high | ba_prop_slope | ba_prop_p | threshold_agreement | threshold_kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| holdout: all | 58 | 0.479 | 0.676 | 0.896 | 0.000 | 0.876 | 0.756 | 0.933 | 0.896 | 0.294 | 0.133 | 0.456 | 0.614 | -0.908 | 1.497 | 0.045 | 0.477 | 0.776 | 0.733 |
| holdout: without dash-zero rows | 45 | 0.439 | 0.566 | 0.915 | 0.000 | 0.900 | 0.802 | 0.948 | 0.913 | 0.229 | 0.071 | 0.386 | 0.524 | -0.798 | 1.256 | 0.057 | 0.377 | 0.778 | 0.773 |
| holdout: only 2698x1799 (+/-2 px) frames | 48 | 0.448 | 0.588 | 0.923 | 0.000 | 0.906 | 0.811 | 0.951 | 0.920 | 0.246 | 0.090 | 0.403 | 0.539 | -0.811 | 1.303 | 0.086 | 0.152 | 0.771 | 0.713 |
| holdout: frames outside 2698x1799 | 10 | 0.630 | 0.995 | 0.785 | 0.007 | 0.713 | 0.214 | 0.919 | 0.756 | 0.525 | -0.112 | 1.163 | 0.891 | -1.221 | 2.272 | -0.308 | 0.236 | 0.800 | 0.804 |
| holdout: without ambiguous 100-999 cells | 58 | 0.479 | 0.676 | 0.896 | 0.000 | 0.876 | 0.756 | 0.933 | 0.896 | 0.294 | 0.133 | 0.456 | 0.614 | -0.908 | 1.497 | 0.045 | 0.477 | 0.776 | 0.733 |
| dev + holdout (scale still from dev) | 145 | 0.521 | 0.731 | 0.892 | 0.000 | 0.877 | 0.804 | 0.919 | 0.891 | 0.270 | 0.159 | 0.382 | 0.681 | -1.065 | 1.605 | -0.056 | 0.160 | 0.759 | 0.718 |

## Lip–gingiva boundary (n = 145)
Gap between the lowest lip pixel and the gingiva top: median 7.0 px (IQR 5.0–9.0; mean 6.9; max 19). Expected ≈ 9 px (IQR 6–12) from the audit. Consistent. Boundary flag raised on 0 images.
Lip-anchored estimators (C_p25_lipanchored): dev MAE 0.537 mm vs 0.549 mm for gingiva thickness.

## Tooth level (secondary; region i left-to-right vs reference tooth i; A/B/C alignment is approximate)
| tooth_index | n | mae | bias | r |
|---|---|---|---|---|
| 1 | 145 | 0.752 | 0.283 | 0.852 |
| 2 | 145 | 0.674 | 0.365 | 0.896 |
| 3 | 145 | 0.456 | 0.137 | 0.923 |
| 4 | 145 | 0.547 | 0.116 | 0.849 |
| 5 | 145 | 0.744 | 0.230 | 0.794 |
| 6 | 145 | 0.817 | 0.491 | 0.810 |

## QC flags
gingiva_multi_component     33
zenith_detection_failed     27
frame_uncertain             22
region_zero                 22
festoon_detection_failed     2
Frame outside 2698×1799 ±2 px: 22 images (`frame_uncertain`; sensitivity rows above).

## Assumptions
- COCO frame = ImageJ frame (clinical team, screenshot); for other frame sizes the reference frame is uncertain.
- `-` cells are 0 mm (clinical decision); the sensitivity row without dash-zero rows shows the effect.
- The reference gingival display recorded at tooth site i is compared to the left-to-right region of the same index; the image-level mean is the primary endpoint.
- v1 (XGBoost) column of the original Figure 6 is not reproduced: the regressor was trained on 512×512 gingiva-only DeepLab masks and fed lip+gingiva masks at 1024 px in v3, i.e. inputs outside its training distribution (audit §3).
