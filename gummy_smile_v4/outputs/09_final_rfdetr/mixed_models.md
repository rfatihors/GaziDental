# Tooth-level analysis with linear mixed models (a: OOF masks)

Six teeth per image are not independent: `diff ~ … + (1 | patient)` (REML; `patient` = same-patient cluster from the manifest, 145 clusters for 145 images). Region i (left-to-right, C_p25) is compared with reference tooth i (FDI [13, 12, 11, 21, 22, 23]); alignment is approximate and images where the zenith regioning fell back to equal splits are flagged `alignment_uncertain` (27 images).

## Per tooth
| comparison | tooth_index | tooth_fdi | n | mae | bias | sd | r |
|---|---|---|---|---|---|---|---|
| OOF region i vs reference tooth i | 1 | 13 | 145 | 0.752 | 0.283 | 1.013 | 0.852 |
| OOF region i vs reference tooth i | 2 | 12 | 145 | 0.674 | 0.365 | 0.849 | 0.896 |
| OOF region i vs reference tooth i | 3 | 11 | 145 | 0.456 | 0.137 | 0.651 | 0.923 |
| OOF region i vs reference tooth i | 4 | 21 | 145 | 0.547 | 0.116 | 0.899 | 0.849 |
| OOF region i vs reference tooth i | 5 | 22 | 145 | 0.744 | 0.230 | 1.184 | 0.794 |
| OOF region i vs reference tooth i | 6 | 23 | 145 | 0.817 | 0.491 | 1.055 | 0.810 |
| OOF region i vs GT-mask region i | 1 | 13 | 145 | 0.543 | 0.124 | 0.759 | 0.910 |
| OOF region i vs GT-mask region i | 2 | 12 | 145 | 0.486 | 0.139 | 0.593 | 0.951 |
| OOF region i vs GT-mask region i | 3 | 11 | 145 | 0.386 | -0.039 | 0.498 | 0.957 |
| OOF region i vs GT-mask region i | 4 | 21 | 145 | 0.386 | 0.184 | 0.442 | 0.962 |
| OOF region i vs GT-mask region i | 5 | 22 | 145 | 0.500 | 0.104 | 0.714 | 0.927 |
| OOF region i vs GT-mask region i | 6 | 23 | 145 | 0.652 | 0.208 | 0.981 | 0.819 |

### Pipeline (OOF, PRIMARY) − clinical reference
- `diff ~ 1 + (1 | patient)` — MixedLM (REML); n = 870 teeth in 145 patients; var(patient) 0.371, var(residual) 0.559, ICC(patient) 0.40
  - Intercept: +0.270 mm [+0.160, +0.381], p = 1.75e-06
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.916, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.137 mm [+0.031, +0.243], p = 0.0115
  - C(tooth)[T.12]: +0.228 mm [+0.125, +0.330], p = 1.28e-05
  - C(tooth)[T.13]: +0.146 mm [-0.010, +0.303], p = 0.0669
  - C(tooth)[T.21]: -0.021 mm [-0.135, +0.093], p = 0.713
  - C(tooth)[T.22]: +0.093 mm [-0.088, +0.274], p = 0.314
  - C(tooth)[T.23]: +0.354 mm [+0.194, +0.514], p = 1.45e-05
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.916, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.127 mm [+0.002, +0.252], p = 0.0468
  - C(tooth)[T.12]: +0.228 mm [+0.125, +0.330], p = 1.29e-05
  - C(tooth)[T.13]: +0.146 mm [-0.010, +0.303], p = 0.067
  - C(tooth)[T.21]: -0.021 mm [-0.135, +0.093], p = 0.713
  - C(tooth)[T.22]: +0.093 mm [-0.088, +0.274], p = 0.314
  - C(tooth)[T.23]: +0.354 mm [+0.194, +0.514], p = 1.47e-05
  - alignment_uncertain[T.True]: +0.055 mm [-0.227, +0.338], p = 0.701

### Pipeline (OOF) − GT-mask measurement (segmentation part only)
- `diff ~ 1 + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.367, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: +0.120 mm [+0.066, +0.174], p = 1.49e-05
- `diff ~ C(tooth) + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.475, ICC(patient) 0.00; note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.039 mm [-0.120, +0.043], p = 0.35
  - C(tooth)[T.12]: +0.178 mm [+0.077, +0.280], p = 0.000586
  - C(tooth)[T.13]: +0.163 mm [+0.027, +0.298], p = 0.0187
  - C(tooth)[T.21]: +0.222 mm [+0.132, +0.313], p = 1.46e-06
  - C(tooth)[T.22]: +0.142 mm [+0.017, +0.267], p = 0.0258
  - C(tooth)[T.23]: +0.247 mm [+0.066, +0.427], p = 0.00742
- `diff ~ C(tooth) + alignment_uncertain + (1 | patient)` — OLS, cluster-robust SE (patient); n = 870 teeth in 145 patients; var(patient) 0.000, var(residual) 0.474, ICC(patient) 0.00; note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead
  - Intercept: -0.026 mm [-0.109, +0.058], p = 0.548
  - C(tooth)[T.12]: +0.178 mm [+0.076, +0.280], p = 0.00059
  - C(tooth)[T.13]: +0.163 mm [+0.027, +0.298], p = 0.0188
  - C(tooth)[T.21]: +0.222 mm [+0.132, +0.313], p = 1.48e-06
  - C(tooth)[T.22]: +0.142 mm [+0.017, +0.268], p = 0.0259
  - C(tooth)[T.23]: +0.247 mm [+0.066, +0.427], p = 0.00746
  - alignment_uncertain[T.True]: -0.071 mm [-0.225, +0.084], p = 0.372
