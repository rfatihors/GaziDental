# Millimetre accuracy vs clinical reference (Reviewers 2, 4; Figure 6 replacement)

| analysis | n | px_per_mm | mae_mm | rmse_mm | r | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | bias_mm | loa_low_mm | loa_high_mm | kappa_linear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT masks, holdout, C_p25 (selected) | 58 | 16.840 | 0.518 | 0.718 | 0.861 | 0.858 | 0.773 | 0.914 | 0.116 | -1.284 | 1.516 |  |
| GT masks, holdout, A_p25 (sensitivity: no fallback) | 58 | 17.036 | 0.537 | 0.744 | 0.849 | 0.843 | 0.746 | 0.905 | 0.187 | -1.236 | 1.610 |  |
| GT masks, holdout: without dash-zero rows | 45 | 16.840 | 0.473 | 0.603 | 0.881 | 0.881 | 0.795 | 0.933 | 0.077 | -1.108 | 1.263 |  |
| GT masks, holdout: only 2698x1799 (+/-2 px) frames | 48 | 16.840 | 0.464 | 0.591 | 0.909 | 0.905 | 0.837 | 0.946 | 0.059 | -1.107 | 1.224 |  |
| GT masks, dev + holdout (scale still from dev) | 145 | 16.840 | 0.542 | 0.756 | 0.872 | 0.868 | 0.819 | 0.904 | 0.150 | -1.307 | 1.608 |  |
| Predicted masks (OOF, fold models), all reference images, C_p25 (PRIMARY) | 144 | 16.840 | 0.839 | 0.999 | 0.875 | 0.784 | 0.288 | 0.909 | 0.684 | -0.746 | 2.115 | 0.594 |
| Predicted masks (OOF), Stage-3 holdout images, C_p25 | 57 | 16.840 | 0.757 | 0.917 | 0.894 | 0.797 | 0.191 | 0.926 | 0.670 | -0.568 | 1.909 | 0.611 |
| Predicted masks (final model), test-set high images, C_p25 (secondary) | 29 | 16.840 | 0.953 | 1.108 | 0.804 | 0.731 | 0.344 | 0.884 | 0.647 | -1.147 | 2.440 | 0.401 |

- source: outputs/03_oracle
- selected_method: C_p25
- prediction_rows: done
- prediction_source: outputs/06_prediction/measurement_accuracy.csv
- precision_note: With n = 145 reference images, an ICC of 0.86 has a 95 % CI half-width of ≈ 0.043 (Bonett 2002, k = 2); each Bland–Altman limit of agreement has a half-width of ≈ 0.20 mm for the observed between-method SD of 0.71 mm (Bland & Altman 1999).
