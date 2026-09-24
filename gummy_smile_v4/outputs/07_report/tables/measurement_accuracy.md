# Millimetre accuracy vs clinical reference (Reviewers 2, 4; Figure 6 replacement)

| analysis | n | px_per_mm | mae_mm | rmse_mm | r | icc2_1 | icc2_1_ci_low | icc2_1_ci_high | bias_mm | loa_low_mm | loa_high_mm | correction | model | masks_dir | kappa_linear | images_with_zeroed_columns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT masks, holdout, C_p25 (selected) | 58 | 16.842 | 0.527 | 0.726 | 0.860 | 0.857 | 0.770 | 0.912 | 0.125 | -1.290 | 1.540 |  |  |  |  |  |
| GT masks, holdout, A_p25 (sensitivity: no fallback) | 58 | 17.036 | 0.537 | 0.744 | 0.849 | 0.843 | 0.746 | 0.905 | 0.187 | -1.236 | 1.610 |  |  |  |  |  |
| GT masks, holdout: without dash-zero rows | 45 | 16.842 | 0.485 | 0.617 | 0.878 | 0.878 | 0.789 | 0.931 | 0.089 | -1.121 | 1.298 |  |  |  |  |  |
| GT masks, holdout: only 2698x1799 (+/-2 px) frames | 48 | 16.842 | 0.476 | 0.604 | 0.907 | 0.902 | 0.833 | 0.944 | 0.069 | -1.120 | 1.258 |  |  |  |  |  |
| GT masks, dev + holdout (scale still from dev) | 145 | 16.842 | 0.546 | 0.761 | 0.871 | 0.867 | 0.818 | 0.903 | 0.154 | -1.311 | 1.619 |  |  |  |  |  |
| Predicted masks (OOF, fold models), all reference images, C_p25 — PRIMARY (no post-hoc correction) | 145 | 16.842 | 0.521 | 0.730 | 0.892 | 0.877 | 0.805 | 0.919 | 0.270 | -1.065 | 1.605 | none | RF-DETR-Seg Large @624, seed 42, 5 fold models | outputs/05_predictions/oof_rfdetr | 0.718 | 0.000 |
| Predicted masks (OOF), Stage-3 holdout images, C_p25 | 58 | 16.842 | 0.479 | 0.675 | 0.896 | 0.876 | 0.757 | 0.933 | 0.294 | -0.909 | 1.496 | none | RF-DETR-Seg Large @624, seed 42, 5 fold models | outputs/05_predictions/oof_rfdetr | 0.733 | 0.000 |
| Predicted masks (final model), test-set high images, C_p25 | 29 | 16.842 | 0.671 | 0.941 | 0.804 | 0.787 | 0.595 | 0.894 | 0.284 | -1.506 | 2.073 | none | RF-DETR-Seg Large @624, seed 42 | outputs/05_predictions/test_rfdetr | 0.621 | 0.000 |

- source: outputs/03_oracle
- selected_method: C_p25
- prediction_rows: done
- prediction_source: outputs/09_final_rfdetr/measurement_accuracy.csv
- prediction_model: RF-DETR-Seg Large @624, seed 42, RF-DETR-Seg Large @624, seed 42, 5 fold models
- post_hoc_correction: none — single result set
- precision_note: With n = 145 reference images, an ICC of 0.86 has a 95 % CI half-width of ≈ 0.044 (Bonett 2002, k = 2); each Bland–Altman limit of agreement has a half-width of ≈ 0.20 mm for the observed between-method SD of 0.72 mm (Bland & Altman 1999).
