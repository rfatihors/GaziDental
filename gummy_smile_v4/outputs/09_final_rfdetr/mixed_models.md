# Tooth-level analysis with linear mixed models (a: OOF masks)

Six teeth per image are not independent: `diff ~ … + (1 | patient)` (REML; `patient` = same-patient cluster from the manifest, 145 clusters for 145 images). Region i (left-to-right, C_p25) is compared with reference tooth i (FDI [13, 12, 11, 21, 22, 23]); alignment is approximate and images where the zenith regioning fell back to equal splits are flagged `alignment_uncertain` (27 images).

## Per tooth
| comparison | tooth_index | tooth_fdi | n | mae | bias | sd | r |
|---|---|---|---|---|---|---|---|
| OOF region i vs reference tooth i | 1 | 13 | 145 | 0.754 | 0.281 | 1.014 | 0.852 |
| OOF region i vs reference tooth i | 2 | 12 | 145 | 0.673 | 0.364 | 0.849 | 0.896 |
| OOF region i vs reference tooth i | 3 | 11 | 145 | 0.456 | 0.137 | 0.651 | 0.923 |
| OOF region i vs reference tooth i | 4 | 21 | 145 | 0.547 | 0.115 | 0.899 | 0.849 |
| OOF region i vs reference tooth i | 5 | 22 | 145 | 0.743 | 0.229 | 1.184 | 0.794 |
| OOF region i vs reference tooth i | 6 | 23 | 145 | 0.817 | 0.491 | 1.055 | 0.810 |
| OOF region i vs GT-mask region i | 1 | 13 | 145 | 0.530 | 0.108 | 0.726 | 0.918 |
| OOF region i vs GT-mask region i | 2 | 12 | 145 | 0.486 | 0.140 | 0.593 | 0.951 |
| OOF region i vs GT-mask region i | 3 | 11 | 145 | 0.384 | -0.037 | 0.497 | 0.957 |
| OOF region i vs GT-mask region i | 4 | 21 | 145 | 0.386 | 0.183 | 0.442 | 0.962 |
| OOF region i vs GT-mask region i | 5 | 22 | 145 | 0.499 | 0.098 | 0.714 | 0.928 |
| OOF region i vs GT-mask region i | 6 | 23 | 145 | 0.652 | 0.203 | 0.982 | 0.819 |

### Pipeline (OOF, PRIMARY) − clinical reference
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.466, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.270 mm [+0.159, +0.381], p = 1.86e-06
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.916, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.137 mm [+0.030, +0.243], p = 0.0117
  - C(tooth)[T.12]: +0.228 mm [+0.125, +0.330], p = 1.3e-05
  - C(tooth)[T.13]: +0.145 mm [-0.012, +0.301], p = 0.0699
  - C(tooth)[T.21]: -0.021 mm [-0.135, +0.093], p = 0.713
  - C(tooth)[T.22]: +0.093 mm [-0.088, +0.273], p = 0.315
  - C(tooth)[T.23]: +0.354 mm [+0.194, +0.514], p = 1.45e-05
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.916, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.126 mm [+0.001, +0.251], p = 0.0476
  - C(tooth)[T.12]: +0.228 mm [+0.125, +0.330], p = 1.32e-05
  - C(tooth)[T.13]: +0.145 mm [-0.012, +0.301], p = 0.0701
  - C(tooth)[T.21]: -0.021 mm [-0.135, +0.093], p = 0.713
  - C(tooth)[T.22]: +0.093 mm [-0.088, +0.274], p = 0.315
  - C(tooth)[T.23]: +0.354 mm [+0.194, +0.514], p = 1.47e-05
  - alignment_uncertain[T.True]: +0.056 mm [-0.227, +0.338], p = 0.7

### Pipeline (OOF) − GT-mask measurement (segmentation part only)
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.360, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.116 mm [+0.062, +0.170], p = 2.59e-05
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.466, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.037 mm [-0.118, +0.044], p = 0.369
  - C(tooth)[T.12]: +0.177 mm [+0.075, +0.278], p = 0.000639
  - C(tooth)[T.13]: +0.145 mm [+0.015, +0.274], p = 0.0285
  - C(tooth)[T.21]: +0.220 mm [+0.130, +0.311], p = 1.76e-06
  - C(tooth)[T.22]: +0.136 mm [+0.011, +0.261], p = 0.0336
  - C(tooth)[T.23]: +0.240 mm [+0.060, +0.420], p = 0.00906
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.466, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.025 mm [-0.109, +0.058], p = 0.556
  - C(tooth)[T.12]: +0.177 mm [+0.075, +0.278], p = 0.000644
  - C(tooth)[T.13]: +0.145 mm [+0.015, +0.275], p = 0.0286
  - C(tooth)[T.21]: +0.220 mm [+0.130, +0.311], p = 1.78e-06
  - C(tooth)[T.22]: +0.136 mm [+0.010, +0.261], p = 0.0337
  - C(tooth)[T.23]: +0.240 mm [+0.060, +0.420], p = 0.0091
  - alignment_uncertain[T.True]: -0.065 mm [-0.220, +0.090], p = 0.411
