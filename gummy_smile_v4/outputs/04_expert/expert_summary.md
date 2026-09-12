# Expert-agreement analysis — summary

Data: SYNTHETIC forms (p_agree ≈ 0.8, expert 2 biased upward, messy cells) written to /Users/fors/Projeler/GaziDental_calisma/GaziDental/gummy_smile_v4/outputs/04_expert/synthetic_forms; real model table /Users/fors/Projeler/GaziDental_calisma/GaziDental/gummy_smile_v4/outputs/03_oracle/per_image_results.csv.

## Forms and quality control
| expert | n_rows | n_rows_empty | n_class_missing | n_scale_missing | n_mm_complete | n_confidence_missing | flags |
|---|---|---|---|---|---|---|---|
| 1 | 165 | 0 | 0 | 6 | 161 | 3 | scale_missing: 6; confidence_missing: 3; mm_12_missing: 1; mm_22_missing: 1; mm_11_missing: 1; mm_21_missing: 1 |
| 2 | 165 | 3 | 3 | 6 | 158 | 9 | confidence_missing: 9; scale_missing: 6; mm_13_missing: 5; mm_12_missing: 4; mm_23_missing: 4; mm_11_missing: 3; mm_21_missing: 3; mm_22_missing: 3; missing_class: 3; row_empty: 3 |
| 3 | 165 | 0 | 0 | 2 | 163 | 5 | confidence_missing: 5; scale_missing: 2; mm_22_missing: 1; mm_13_missing: 1 |
Rows are never dropped: empty rows, blank teeth, zero values, missing confidence and unparsable cells are flagged in `forms_long.csv` (`form_flags`) and excluded only from the statistic that needs the missing field.

## Reference standard
Majority of three primary classes; unanimous 48, majority 85, consensus (blinded) 0, **consensus pending 12** (excluded from the primary analysis; `consensus_pending.csv`; drop a `consensus.csv` with columns image,class into the forms directory to resolve), insufficient votes 0.

## Model vs expert reference — class (model = selected method mm + rule engine)
Scoring: strict = model's first candidate; lenient = agreement if the expert class is among the model's candidates; lenient2 = also the expert's second candidate (reported only). Bootstrap CIs: 2000 resamples, seed 42.

| scale | subset | scoring | n | n_consensus_pending_excluded | n_model_unclassified | kappa_linear | kappa_linear_ci_low | kappa_linear_ci_high | kappa_unweighted | observed_agreement | pabak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | test subset (primary) | strict | 27 | 2 | 0 | 0.471 | -0.145 | 0.836 | 0.357 | 0.815 | 0.753 |
| global | test subset (primary) | lenient | 27 | 2 | 0 | 0.724 | 0.276 | 1.000 | 0.676 | 0.889 | 0.852 |
| global | test subset (primary) | lenient2 | 27 | 2 | 0 | 0.724 | 0.276 | 1.000 | 0.676 | 0.889 | 0.852 |
| global | all 145 images (secondary) | strict | 133 | 12 | 0 | 0.549 | 0.388 | 0.688 | 0.499 | 0.797 | 0.729 |
| global | all 145 images (secondary) | lenient | 133 | 12 | 0 | 0.717 | 0.567 | 0.832 | 0.685 | 0.865 | 0.820 |
| global | all 145 images (secondary) | lenient2 | 133 | 12 | 0 | 0.717 | 0.567 | 0.832 | 0.685 | 0.865 | 0.820 |
| expert | test subset (primary) | strict | 27 | 2 | 0 | 0.360 | -0.110 | 0.685 | 0.282 | 0.815 | 0.753 |
| expert | test subset (primary) | lenient | 27 | 2 | 0 | 0.803 | 0.362 | 1.000 | 0.765 | 0.926 | 0.901 |
| expert | test subset (primary) | lenient2 | 27 | 2 | 0 | 0.803 | 0.362 | 1.000 | 0.765 | 0.926 | 0.901 |
| expert | all 145 images (secondary) | strict | 133 | 12 | 0 | 0.475 | 0.322 | 0.607 | 0.412 | 0.767 | 0.689 |
| expert | all 145 images (secondary) | lenient | 133 | 12 | 0 | 0.698 | 0.551 | 0.814 | 0.668 | 0.857 | 0.810 |
| expert | all 145 images (secondary) | lenient2 | 133 | 12 | 0 | 0.698 | 0.551 | 0.814 | 0.668 | 0.857 | 0.810 |

### Per class (test subset, strict, global scale; counts and Wilson 95 % CIs)
| class | n_reference | n_predicted | sensitivity | sensitivity_ci_low | sensitivity_ci_high | specificity | specificity_ci_low | specificity_ci_high |
|---|---|---|---|---|---|---|---|---|
| E1 | 23 | 22 | 0.870 | 0.679 | 0.955 | 0.500 | 0.150 | 0.850 |
| E2 | 3 | 4 | 0.333 | 0.061 | 0.792 | 0.875 | 0.690 | 0.957 |
| E3 | 1 | 1 | 1.000 | 0.207 | 1.000 | 1.000 | 0.871 | 1.000 |
| E4 | 0 | 0 |  |  |  | 1.000 | 0.875 | 1.000 |

### Disagreements by distance of the model value to the nearest clinical threshold (3, 4, 6, 8 mm)
| stratum | n | observed_agreement | kappa_linear | n_disagreements |
|---|---|---|---|---|
| >= 0.5 mm from thresholds | 72 | 0.819 | 0.613 | 13 |
| within 0.5 mm of a threshold | 61 | 0.770 | 0.471 | 14 |

## Inter-expert agreement
Fleiss κ (class, n = 142): 0.130 [0.036, 0.218]; pairwise linear κ: {'e1_e2': 0.18041175770558127, 'e1_e3': 0.45593869731800774, 'e2_e3': 0.14376119402985088}.
Image-mean mm, 3 experts: ICC(2,1) 0.991 [0.988, 0.993], ICC(2,k) 0.997 [0.996, 0.998] (n = 142).
With the clinical reference observer as 4th rater: ICC(2,1) 0.993 [0.991, 0.995], ICC(2,k) 0.998 [0.998, 0.999].

## Intra-expert agreement (20 repeated images)
| expert | n_repeats | n_class_pairs | kappa_linear | kappa_linear_ci_low | kappa_linear_ci_high | kappa_unweighted | observed_agreement | n_mm_pairs | icc2_1_image | icc2_1_image_ci_low | icc2_1_image_ci_high | mm_bias | mm_sd | mm_loa_low | mm_loa_high | icc2_1_tooth_naive | n_tooth_pairs | scale_icc2_1 | scale_cv_repeat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 20 | 0.291 | -0.154 | 0.635 | 0.188 | 0.600 | 20 | 0.993 | 0.983 | 0.997 | -0.045 | 0.208 | -0.452 | 0.362 | 0.975 | 120 | 0.604 | 0.060 |
| 2 | 20 | 19 | 0.101 | -0.245 | 0.402 | 0.000 | 0.316 | 19 | 0.991 | 0.978 | 0.997 | 0.003 | 0.241 | -0.470 | 0.475 | 0.971 | 114 | 0.304 | 0.060 |
| 3 | 20 | 20 | 0.346 | -0.058 | 0.667 | 0.310 | 0.650 | 20 | 0.992 | 0.981 | 0.997 | -0.057 | 0.222 | -0.493 | 0.378 | 0.977 | 119 | 0.222 | 0.084 |

## Model vs experts — millimetres
| scale | comparator | n | mae | rmse | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | bias | bias_ci_low | bias_ci_high | loa_low | loa_high | prop_slope | prop_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global | expert_1 | 145 | 0.555 | 0.768 | 0.862 | 0.812 | 0.900 | 0.160 | 0.036 | 0.284 | -1.317 | 1.637 | -0.035 | 0.437 |
| global | expert_2 | 142 | 0.537 | 0.746 | 0.870 | 0.822 | 0.905 | 0.139 | 0.017 | 0.262 | -1.303 | 1.582 | -0.051 | 0.250 |
| global | expert_3 | 145 | 0.546 | 0.772 | 0.863 | 0.812 | 0.900 | 0.154 | 0.030 | 0.279 | -1.333 | 1.642 | -0.047 | 0.292 |
| global | expert_mean | 145 | 0.540 | 0.754 | 0.868 | 0.819 | 0.904 | 0.155 | 0.034 | 0.277 | -1.297 | 1.607 | -0.037 | 0.394 |
| global | clinical_reference | 145 | 0.542 | 0.756 | 0.868 | 0.819 | 0.904 | 0.150 | 0.028 | 0.272 | -1.307 | 1.608 | -0.044 | 0.315 |
| expert | expert_1 | 145 | 0.574 | 0.808 | 0.847 | 0.792 | 0.887 | 0.146 | 0.015 | 0.277 | -1.417 | 1.709 | -0.042 | 0.376 |
| expert | expert_2 | 142 | 0.555 | 0.778 | 0.857 | 0.806 | 0.895 | 0.120 | -0.008 | 0.248 | -1.393 | 1.633 | -0.063 | 0.176 |
| expert | expert_3 | 145 | 0.574 | 0.816 | 0.845 | 0.791 | 0.886 | 0.140 | 0.007 | 0.272 | -1.442 | 1.722 | -0.055 | 0.254 |
| expert | expert_mean | 145 | 0.561 | 0.796 | 0.851 | 0.799 | 0.891 | 0.141 | 0.012 | 0.270 | -1.400 | 1.682 | -0.045 | 0.339 |
| expert | clinical_reference | 145 | 0.563 | 0.799 | 0.851 | 0.799 | 0.891 | 0.136 | 0.006 | 0.266 | -1.412 | 1.685 | -0.052 | 0.272 |

Tooth-level analysis: `mixed_models.md` (random intercept per patient; tooth position and `alignment_uncertain` as fixed effects). Scale comparison: `scale_agreement.md`. Numbers for the manuscript: `manuscript_numbers.md`.
