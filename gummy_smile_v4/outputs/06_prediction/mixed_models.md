# Tooth-level analysis with linear mixed models (a: OOF masks)

Six teeth per image are not independent: `diff ~ … + (1 | patient)` (REML; `patient` = same-patient cluster from the manifest, 145 clusters for 145 images). Region i (left-to-right, C_p25) is compared with reference tooth i (FDI [13, 12, 11, 21, 22, 23]); alignment is approximate and images where the zenith regioning fell back to equal splits are flagged `alignment_uncertain` (25 images).

## Per tooth
| comparison | tooth_index | tooth_fdi | n | mae | bias | sd | r |
|---|---|---|---|---|---|---|---|
| OOF region i vs reference tooth i | 1 | 13 | 144 | 1.089 | 0.755 | 1.077 | 0.825 |
| OOF region i vs reference tooth i | 2 | 12 | 144 | 1.017 | 0.802 | 0.925 | 0.874 |
| OOF region i vs reference tooth i | 3 | 11 | 144 | 0.691 | 0.473 | 0.721 | 0.904 |
| OOF region i vs reference tooth i | 4 | 21 | 144 | 0.770 | 0.514 | 0.924 | 0.841 |
| OOF region i vs reference tooth i | 5 | 22 | 144 | 0.982 | 0.646 | 1.205 | 0.784 |
| OOF region i vs reference tooth i | 6 | 23 | 144 | 1.147 | 0.916 | 1.077 | 0.803 |
| OOF region i vs GT-mask region i | 1 | 13 | 144 | 0.778 | 0.591 | 0.802 | 0.891 |
| OOF region i vs GT-mask region i | 2 | 12 | 144 | 0.760 | 0.576 | 0.754 | 0.920 |
| OOF region i vs GT-mask region i | 3 | 11 | 144 | 0.508 | 0.298 | 0.625 | 0.931 |
| OOF region i vs GT-mask region i | 4 | 21 | 144 | 0.635 | 0.578 | 0.482 | 0.956 |
| OOF region i vs GT-mask region i | 5 | 22 | 144 | 0.702 | 0.525 | 0.749 | 0.919 |
| OOF region i vs GT-mask region i | 6 | 23 | 144 | 0.912 | 0.635 | 1.069 | 0.787 |
| OOF region i (corrected, secondary) vs reference tooth i | 1 | 13 | 144 | 0.684 | 0.002 | 1.059 | 0.830 |
| OOF region i (corrected, secondary) vs reference tooth i | 2 | 12 | 144 | 0.624 | 0.034 | 0.916 | 0.877 |
| OOF region i (corrected, secondary) vs reference tooth i | 3 | 11 | 144 | 0.449 | -0.233 | 0.671 | 0.915 |
| OOF region i (corrected, secondary) vs reference tooth i | 4 | 21 | 144 | 0.552 | -0.200 | 0.901 | 0.842 |
| OOF region i (corrected, secondary) vs reference tooth i | 5 | 22 | 144 | 0.680 | -0.125 | 1.161 | 0.798 |
| OOF region i (corrected, secondary) vs reference tooth i | 6 | 23 | 144 | 0.678 | 0.200 | 1.051 | 0.807 |

### Pipeline (OOF, uncorrected, PRIMARY) − clinical reference
- `diff ~ 1 + (1 | patient)` — MixedLM (REML); n = 864 teeth in 144 patients; var(patient) 0.435, var(residual) 0.586, ICC(patient) 0.43
  - Intercept: +0.684 mm [+0.565, +0.804], p = 2.26e-29
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 1.000, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.473 mm [+0.355, +0.592], p = 4.02e-15
  - C(tooth)[T.12]: +0.329 mm [+0.207, +0.450], p = 1.11e-07
  - C(tooth)[T.13]: +0.282 mm [+0.118, +0.445], p = 0.000728
  - C(tooth)[T.21]: +0.041 mm [-0.091, +0.172], p = 0.544
  - C(tooth)[T.22]: +0.172 mm [-0.017, +0.362], p = 0.0748
  - C(tooth)[T.23]: +0.442 mm [+0.267, +0.617], p = 7.45e-07
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.988, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.421 mm [+0.286, +0.555], p = 8.67e-10
  - C(tooth)[T.12]: +0.329 mm [+0.207, +0.450], p = 1.13e-07
  - C(tooth)[T.13]: +0.282 mm [+0.118, +0.446], p = 0.000733
  - C(tooth)[T.21]: +0.041 mm [-0.091, +0.172], p = 0.544
  - C(tooth)[T.22]: +0.172 mm [-0.017, +0.362], p = 0.075
  - C(tooth)[T.23]: +0.442 mm [+0.267, +0.617], p = 7.56e-07
  - alignment_uncertain[T.True]: +0.304 mm [-0.009, +0.616], p = 0.0567

### Pipeline (OOF, corrected: lower gingiva edge -13 px at mask level (-0.77 mm at 16.84 px/mm), secondary) − clinical reference
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.448, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.054 mm [-0.171, +0.064], p = 0.37
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.946, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.233 mm [-0.343, -0.124], p = 3.13e-05
  - C(tooth)[T.12]: +0.268 mm [+0.152, +0.383], p = 5.47e-06
  - C(tooth)[T.13]: +0.236 mm [+0.074, +0.397], p = 0.0042
  - C(tooth)[T.21]: +0.033 mm [-0.089, +0.155], p = 0.592
  - C(tooth)[T.22]: +0.108 mm [-0.068, +0.285], p = 0.23
  - C(tooth)[T.23]: +0.433 mm [+0.266, +0.600], p = 3.78e-07
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.935, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.283 mm [-0.411, -0.156], p = 1.27e-05
  - C(tooth)[T.12]: +0.268 mm [+0.152, +0.383], p = 5.54e-06
  - C(tooth)[T.13]: +0.236 mm [+0.074, +0.397], p = 0.00422
  - C(tooth)[T.21]: +0.033 mm [-0.089, +0.156], p = 0.593
  - C(tooth)[T.22]: +0.108 mm [-0.069, +0.285], p = 0.23
  - C(tooth)[T.23]: +0.433 mm [+0.266, +0.600], p = 3.84e-07
  - alignment_uncertain[T.True]: +0.289 mm [-0.012, +0.589], p = 0.0598

### Pipeline (OOF) − GT-mask measurement (segmentation part only)
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.434, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.534 mm [+0.468, +0.601], p = 6.41e-56
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 864 teeth in 144 patients; var(patient) 0.000, var(residual) 0.589, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.298 mm [+0.196, +0.401], p = 1.13e-08
  - C(tooth)[T.12]: +0.278 mm [+0.154, +0.402], p = 1.14e-05
  - C(tooth)[T.13]: +0.293 mm [+0.138, +0.448], p = 0.000211
  - C(tooth)[T.21]: +0.280 mm [+0.170, +0.389], p = 5.52e-07
  - C(tooth)[T.22]: +0.227 mm [+0.081, +0.373], p = 0.00229
  - C(tooth)[T.23]: +0.337 mm [+0.137, +0.537], p = 0.00094
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — MixedLM (REML); n = 864 teeth in 144 patients; var(patient) 0.082, var(residual) 0.509, ICC(patient) 0.14
  - Intercept: +0.300 mm [+0.171, +0.430], p = 5.17e-06
  - C(tooth)[T.12]: +0.278 mm [+0.113, +0.443], p = 0.000943
  - C(tooth)[T.13]: +0.293 mm [+0.128, +0.458], p = 0.000495
  - C(tooth)[T.21]: +0.280 mm [+0.115, +0.445], p = 0.000872
  - C(tooth)[T.22]: +0.227 mm [+0.062, +0.392], p = 0.00697
  - C(tooth)[T.23]: +0.337 mm [+0.172, +0.502], p = 6.11e-05
  - alignment_uncertain[T.True]: -0.011 mm [-0.187, +0.165], p = 0.9
