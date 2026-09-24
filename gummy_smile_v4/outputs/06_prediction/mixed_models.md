# Tooth-level analysis with linear mixed models (a: OOF masks)

Six teeth per image are not independent: `diff ~ … + (1 | patient)` (REML; `patient` = same-patient cluster from the manifest, 145 clusters for 145 images). Region i (left-to-right, C_p25) is compared with reference tooth i (FDI [13, 12, 11, 21, 22, 23]); alignment is approximate and images where the zenith regioning fell back to equal splits are flagged `alignment_uncertain` (25 images).

## Per tooth
| comparison | tooth_index | tooth_fdi | n | mae | bias | sd | r |
|---|---|---|---|---|---|---|---|
| OOF region i vs reference tooth i | 1 | 13 | 144 | 1.089 | 0.755 | 1.077 | 0.825 |
| OOF region i vs reference tooth i | 2 | 12 | 144 | 1.017 | 0.802 | 0.925 | 0.874 |
| OOF region i vs reference tooth i | 3 | 11 | 144 | 0.691 | 0.473 | 0.721 | 0.904 |
| OOF region i vs reference tooth i | 4 | 21 | 144 | 0.770 | 0.514 | 0.924 | 0.841 |
| OOF region i vs reference tooth i | 5 | 22 | 144 | 0.981 | 0.645 | 1.205 | 0.784 |
| OOF region i vs reference tooth i | 6 | 23 | 144 | 1.147 | 0.915 | 1.077 | 0.803 |
| OOF region i vs GT-mask region i | 1 | 13 | 144 | 0.763 | 0.576 | 0.777 | 0.898 |
| OOF region i vs GT-mask region i | 2 | 12 | 144 | 0.760 | 0.577 | 0.753 | 0.920 |
| OOF region i vs GT-mask region i | 3 | 11 | 144 | 0.507 | 0.300 | 0.624 | 0.932 |
| OOF region i vs GT-mask region i | 4 | 21 | 144 | 0.635 | 0.578 | 0.481 | 0.957 |
| OOF region i vs GT-mask region i | 5 | 22 | 144 | 0.696 | 0.520 | 0.750 | 0.919 |
| OOF region i vs GT-mask region i | 6 | 23 | 144 | 0.913 | 0.630 | 1.070 | 0.787 |
| OOF region i (corrected, secondary) vs reference tooth i | 1 | 13 | 144 | 0.684 | 0.002 | 1.059 | 0.830 |
| OOF region i (corrected, secondary) vs reference tooth i | 2 | 12 | 144 | 0.623 | 0.034 | 0.916 | 0.877 |
| OOF region i (corrected, secondary) vs reference tooth i | 3 | 11 | 144 | 0.449 | -0.234 | 0.671 | 0.915 |
| OOF region i (corrected, secondary) vs reference tooth i | 4 | 21 | 144 | 0.552 | -0.200 | 0.901 | 0.842 |
| OOF region i (corrected, secondary) vs reference tooth i | 5 | 22 | 144 | 0.680 | -0.126 | 1.161 | 0.798 |
| OOF region i (corrected, secondary) vs reference tooth i | 6 | 23 | 144 | 0.678 | 0.199 | 1.051 | 0.807 |

### Pipeline (OOF, uncorrected, PRIMARY) − clinical reference
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.489, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.684 mm [+0.565, +0.803], p = 2.45e-29
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 1.000, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.473 mm [+0.355, +0.591], p = 4.23e-15
  - C(tooth)[T.12]: +0.329 mm [+0.207, +0.450], p = 1.13e-07
  - C(tooth)[T.13]: +0.282 mm [+0.118, +0.445], p = 0.00073
  - C(tooth)[T.21]: +0.041 mm [-0.091, +0.172], p = 0.544
  - C(tooth)[T.22]: +0.172 mm [-0.017, +0.362], p = 0.0751
  - C(tooth)[T.23]: +0.442 mm [+0.267, +0.617], p = 7.4e-07
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.988, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.420 mm [+0.286, +0.555], p = 8.99e-10
  - C(tooth)[T.12]: +0.329 mm [+0.207, +0.450], p = 1.15e-07
  - C(tooth)[T.13]: +0.282 mm [+0.118, +0.445], p = 0.000735
  - C(tooth)[T.21]: +0.041 mm [-0.091, +0.172], p = 0.545
  - C(tooth)[T.22]: +0.172 mm [-0.018, +0.362], p = 0.0753
  - C(tooth)[T.23]: +0.442 mm [+0.267, +0.617], p = 7.51e-07
  - alignment_uncertain[T.True]: +0.304 mm [-0.009, +0.616], p = 0.0567

### Pipeline (OOF, corrected: lower gingiva edge -13 px at mask level (-0.77 mm at 16.84 px/mm), secondary) − clinical reference
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.448, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.054 mm [-0.172, +0.063], p = 0.367
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.946, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.234 mm [-0.343, -0.124], p = 3.05e-05
  - C(tooth)[T.12]: +0.267 mm [+0.152, +0.383], p = 5.56e-06
  - C(tooth)[T.13]: +0.236 mm [+0.074, +0.397], p = 0.00421
  - C(tooth)[T.21]: +0.033 mm [-0.089, +0.155], p = 0.592
  - C(tooth)[T.22]: +0.108 mm [-0.069, +0.284], p = 0.231
  - C(tooth)[T.23]: +0.433 mm [+0.266, +0.600], p = 3.76e-07
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — MixedLM (REML); n = 864 teeth in 144 patients; var(patient) 0.424, var(residual) 0.514, ICC(patient) 0.45
  - Intercept: -0.284 mm [-0.451, -0.117], p = 0.000862
  - C(tooth)[T.12]: +0.267 mm [+0.102, +0.433], p = 0.00154
  - C(tooth)[T.13]: +0.236 mm [+0.070, +0.401], p = 0.00528
  - C(tooth)[T.21]: +0.033 mm [-0.132, +0.199], p = 0.693
  - C(tooth)[T.22]: +0.108 mm [-0.058, +0.273], p = 0.201
  - C(tooth)[T.23]: +0.433 mm [+0.267, +0.598], p = 2.95e-07
  - alignment_uncertain[T.True]: +0.289 mm [-0.019, +0.596], p = 0.0661

### Pipeline (OOF) − GT-mask measurement (segmentation part only)
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.427, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.530 mm [+0.464, +0.596], p = 2.7e-55
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.583, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.300 mm [+0.198, +0.402], p = 9.01e-09
  - C(tooth)[T.12]: +0.277 mm [+0.153, +0.401], p = 1.25e-05
  - C(tooth)[T.13]: +0.276 mm [+0.126, +0.426], p = 0.00031
  - C(tooth)[T.21]: +0.278 mm [+0.169, +0.387], p = 6.16e-07
  - C(tooth)[T.22]: +0.220 mm [+0.075, +0.365], p = 0.00303
  - C(tooth)[T.23]: +0.330 mm [+0.131, +0.530], p = 0.00115
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.584, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.301 mm [+0.192, +0.410], p = 5.65e-08
  - C(tooth)[T.12]: +0.277 mm [+0.153, +0.401], p = 1.27e-05
  - C(tooth)[T.13]: +0.276 mm [+0.126, +0.426], p = 0.000312
  - C(tooth)[T.21]: +0.278 mm [+0.169, +0.387], p = 6.25e-07
  - C(tooth)[T.22]: +0.220 mm [+0.074, +0.365], p = 0.00305
  - C(tooth)[T.23]: +0.330 mm [+0.131, +0.530], p = 0.00116
  - alignment_uncertain[T.True]: -0.007 mm [-0.195, +0.182], p = 0.946
