# Measurement-method sensitivity: every regioning x estimator combination (Appendix)

| selected | combo | regioning | estimator | lip_anchored | px_per_mm (dev fit) | MAE dev, mm | MAE holdout, mm | ICC(2,1) dev | ICC(2,1) holdout | BA bias dev, mm | BA bias holdout, mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **yes** | C_p25 | C | p25 | False | 16.840 | 0.557 | 0.518 | 0.873 | 0.858 | 0.173 | 0.116 |
|  | C_p10 | C | p10 | False | 16.040 | 0.578 | 0.533 | 0.869 | 0.851 | 0.154 | 0.042 |
|  | C_median | C | median | False | 17.993 | 0.593 | 0.560 | 0.853 | 0.819 | 0.214 | 0.213 |
|  | C_p05 | C | p05 | False | 15.704 | 0.599 | 0.561 | 0.862 | 0.835 | 0.152 | 0.008 |
|  | C_p25_lipanchored | C | p25 | True | 19.015 | 0.605 | 0.508 | 0.853 | 0.854 | 0.220 | 0.263 |
|  | C_p10_lipanchored | C | p10 | True | 18.183 | 0.608 | 0.523 | 0.851 | 0.852 | 0.208 | 0.217 |
|  | C_p05_lipanchored | C | p05 | True | 17.857 | 0.620 | 0.546 | 0.847 | 0.843 | 0.206 | 0.194 |
|  | B_min | B | min | False | 10.732 | 0.624 | 0.687 | 0.864 | 0.808 | 0.104 | 0.092 |
|  | B_p10 | B | p10 | False | 14.217 | 0.624 | 0.589 | 0.856 | 0.840 | 0.188 | 0.132 |
|  | A_p25 | A | p25 | False | 17.036 | 0.627 | 0.537 | 0.850 | 0.843 | 0.225 | 0.187 |
|  | B_p05 | B | p05 | False | 12.999 | 0.629 | 0.634 | 0.858 | 0.827 | 0.164 | 0.108 |
|  | A_p10 | A | p10 | False | 14.642 | 0.630 | 0.562 | 0.855 | 0.842 | 0.192 | 0.091 |
|  | A_p05 | A | p05 | False | 13.393 | 0.637 | 0.607 | 0.853 | 0.827 | 0.175 | 0.037 |
|  | B_p25 | B | p25 | False | 16.661 | 0.644 | 0.542 | 0.846 | 0.851 | 0.230 | 0.199 |
|  | C_min | C | min | False | 15.160 | 0.646 | 0.653 | 0.842 | 0.803 | 0.146 | -0.028 |
|  | B_min_lipanchored | B | min | True | 13.476 | 0.646 | 0.637 | 0.833 | 0.820 | 0.187 | 0.273 |
|  | C_median_lipanchored | C | median | True | 20.210 | 0.651 | 0.573 | 0.833 | 0.815 | 0.251 | 0.332 |
|  | C_min_lipanchored | C | min | True | 17.372 | 0.653 | 0.602 | 0.833 | 0.823 | 0.203 | 0.170 |
|  | B_p05_lipanchored | B | p05 | True | 15.315 | 0.653 | 0.577 | 0.830 | 0.835 | 0.220 | 0.286 |
|  | B_p10_lipanchored | B | p10 | True | 16.446 | 0.658 | 0.566 | 0.830 | 0.839 | 0.238 | 0.308 |
|  | A_min | A | min | False | 10.898 | 0.663 | 0.757 | 0.849 | 0.771 | 0.109 | -0.040 |
|  | A_p10_lipanchored | A | p10 | True | 16.846 | 0.665 | 0.548 | 0.831 | 0.845 | 0.241 | 0.262 |
|  | A_p05_lipanchored | A | p05 | True | 15.634 | 0.672 | 0.590 | 0.830 | 0.837 | 0.230 | 0.233 |
|  | A_p25_lipanchored | A | p25 | True | 19.252 | 0.674 | 0.534 | 0.825 | 0.841 | 0.266 | 0.320 |
|  | A_min_lipanchored | A | min | True | 13.625 | 0.680 | 0.677 | 0.830 | 0.808 | 0.188 | 0.160 |
|  | B_p25_lipanchored | B | p25 | True | 18.891 | 0.682 | 0.558 | 0.820 | 0.842 | 0.268 | 0.335 |
|  | A_median | A | median | False | 20.652 | 0.731 | 0.525 | 0.805 | 0.826 | 0.311 | 0.223 |
|  | B_median | B | median | False | 20.330 | 0.744 | 0.554 | 0.792 | 0.821 | 0.322 | 0.305 |
|  | A_median_lipanchored | A | median | True | 22.897 | 0.758 | 0.531 | 0.781 | 0.818 | 0.335 | 0.324 |
|  | B_median_lipanchored | B | median | True | 22.550 | 0.776 | 0.578 | 0.768 | 0.804 | 0.345 | 0.417 |
|  | C_max | C | max | False | 25.090 | 0.825 | 0.845 | 0.712 | 0.582 | 0.364 | 0.429 |
|  | C_max_lipanchored | C | max | True | 27.350 | 0.855 | 0.814 | 0.695 | 0.603 | 0.379 | 0.511 |
|  | A_max | A | max | False | 34.800 | 1.002 | 0.798 | 0.613 | 0.620 | 0.502 | 0.512 |
|  | A_max_lipanchored | A | max | True | 37.005 | 1.018 | 0.808 | 0.600 | 0.629 | 0.505 | 0.570 |
|  | B_max | B | max | False | 36.577 | 1.032 | 0.841 | 0.587 | 0.587 | 0.517 | 0.554 |
|  | B_max_lipanchored | B | max | True | 38.785 | 1.050 | 0.851 | 0.577 | 0.602 | 0.519 | 0.609 |

- source: outputs/03_oracle/estimator_comparison.csv
- n_combinations: 36
- selected: C_p25
- selected_px_per_mm: 16.84
- sorted_by: dev MAE (the quantity the selection used)
- selection_rule: selection on dev MAE (n = 87); best dev MAE = 0.557 mm; 1 combination(s) within 0.02 mm of it (C_p25); the simplest of those is chosen (A > B > C, gingiva thickness > lip-anchored, p25 > median > p10 > p05 > min > max)
- regioning_fallback: 19 % (28 of 145 images)
- fallback_check: on the 72 dev images where the regioning succeeded, dev MAE 0.512 mm for C_p25 against 0.597 mm for A_p25
- regioning_fallback_predicted_masks: 19 % (27 of 145 images)
- fallback_reeval_threshold: 30 % (PLAN.md 7), not reached
- note: the method and the scale were selected here, once, on GROUND-TRUTH masks on the dev subset; they were never re-selected or re-fitted on predicted masks (outputs/09_final_rfdetr/PLAN.md 5 and Amendment 1)
