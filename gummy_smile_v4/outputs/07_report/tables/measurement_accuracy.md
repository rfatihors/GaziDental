# Millimetre accuracy vs clinical reference (Reviewers 2, 4; Figure 6 replacement)

| analysis | n | px_per_mm | mae_mm | rmse_mm | r | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | bias_mm | loa_low_mm | loa_high_mm | correction | model | masks_dir | kappa_linear | images_with_zeroed_columns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT masks, holdout, C_p25 (selected) | 58 | 16.840 | 0.518 | 0.718 | 0.861 | 0.858 | 0.773 | 0.914 | 0.116 | -1.284 | 1.516 |  |  |  |  |  |
| GT masks, holdout, A_p25 (sensitivity: no fallback) | 58 | 17.036 | 0.537 | 0.744 | 0.849 | 0.843 | 0.746 | 0.905 | 0.187 | -1.236 | 1.610 |  |  |  |  |  |
| GT masks, holdout: without dash-zero rows | 45 | 16.840 | 0.473 | 0.603 | 0.881 | 0.881 | 0.795 | 0.933 | 0.077 | -1.108 | 1.263 |  |  |  |  |  |
| GT masks, holdout: only 2698x1799 (+/-2 px) frames | 48 | 16.840 | 0.464 | 0.591 | 0.909 | 0.905 | 0.837 | 0.946 | 0.059 | -1.107 | 1.224 |  |  |  |  |  |
| GT masks, dev + holdout (scale still from dev) | 145 | 16.840 | 0.542 | 0.756 | 0.872 | 0.868 | 0.819 | 0.904 | 0.150 | -1.307 | 1.608 |  |  |  |  |  |
| Predicted masks (OOF, fold models), all reference images, C_p25 — PRIMARY (no post-hoc correction) | 145 | 16.840 | 0.521 | 0.731 | 0.892 | 0.877 | 0.804 | 0.919 | 0.270 | -1.065 | 1.605 | none | RF-DETR-Seg Large @624, seed 42, 5 fold models | outputs/05_predictions/oof_rfdetr | 0.718 | 0.000 |
| Predicted masks (OOF), Stage-3 holdout images, C_p25 | 58 | 16.840 | 0.479 | 0.676 | 0.896 | 0.876 | 0.756 | 0.933 | 0.294 | -0.908 | 1.497 | none | RF-DETR-Seg Large @624, seed 42, 5 fold models | outputs/05_predictions/oof_rfdetr | 0.733 | 0.000 |
| Predicted masks (final model), test-set high images, C_p25 | 29 | 16.840 | 0.671 | 0.941 | 0.804 | 0.787 | 0.594 | 0.894 | 0.284 | -1.505 | 2.074 | none | RF-DETR-Seg Large @624, seed 42 | outputs/05_predictions/test_rfdetr | 0.621 | 0.000 |

- source: outputs/03_oracle
- selected_method: C_p25
- prediction_rows: done
- prediction_source: outputs/09_final_rfdetr/measurement_accuracy.csv
- prediction_model: RF-DETR-Seg Large @624, seed 42, RF-DETR-Seg Large @624, seed 42, 5 fold models
- post_hoc_correction: none — single result set
- precision_note: With n = 145 reference images, an ICC of 0.86 has a 95 % CI half-width of ≈ 0.043 (Bonett 2002, k = 2); each Bland–Altman limit of agreement has a half-width of ≈ 0.20 mm for the observed between-method SD of 0.71 mm (Bland & Altman 1999).
