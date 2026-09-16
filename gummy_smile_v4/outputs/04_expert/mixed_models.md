# Tooth-level mixed models (model − expert mean, per tooth; random intercept per patient)

Uncorrected model values (PRIMARY) first, then the corrected values (secondary, `global_corrected`).

## `diff ~ 1 + (1 | patient)  [uncorrected, PRIMARY]`  (n_obs = 870, patients = 145, estimator = OLS, cluster-robust SE (patient), converged = True)

Note: random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead

| term | estimate | ci_low | ci_high | p |
|---|---|---|---|---|
| Intercept | 0.154 | 0.034 | 0.275 | 0.012 |

variance: between-patient 0.0000, residual 0.5767; derived tooth-level ICC(patient) = 0.000; AIC 2571.5

## `diff ~ C(tooth) + (1 | patient)  [uncorrected, PRIMARY]`  (n_obs = 870, patients = 145, estimator = OLS, cluster-robust SE (patient), converged = False)

Note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead

| term | estimate | ci_low | ci_high | p |
|---|---|---|---|---|
| Intercept | 0.159 | 0.028 | 0.290 | 0.017 |
| C(tooth)[T.12] | 0.099 | -0.047 | 0.246 | 0.184 |
| C(tooth)[T.13] | 0.009 | -0.172 | 0.190 | 0.922 |
| C(tooth)[T.21] | -0.210 | -0.343 | -0.077 | 0.002 |
| C(tooth)[T.22] | -0.021 | -0.216 | 0.175 | 0.837 |
| C(tooth)[T.23] | 0.094 | -0.096 | 0.284 | 0.333 |

variance: between-patient 0.0000, residual 1.1197; derived tooth-level ICC(patient) = 0.000; AIC 2573.3

## `diff ~ C(tooth) + alignment_uncertain + (1 | patient)  [uncorrected, PRIMARY]`  (n_obs = 870, patients = 145, estimator = OLS, cluster-robust SE (patient), converged = False)

Note: mixed model failed (Singular matrix); random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead

| term | estimate | ci_low | ci_high | p |
|---|---|---|---|---|
| Intercept | 0.146 | -0.004 | 0.295 | 0.056 |
| C(tooth)[T.12] | 0.099 | -0.047 | 0.246 | 0.184 |
| C(tooth)[T.13] | 0.009 | -0.172 | 0.190 | 0.922 |
| C(tooth)[T.21] | -0.210 | -0.343 | -0.076 | 0.002 |
| C(tooth)[T.22] | -0.021 | -0.216 | 0.175 | 0.837 |
| C(tooth)[T.23] | 0.094 | -0.097 | 0.284 | 0.334 |
| alignment_uncertain[T.True] | 0.070 | -0.251 | 0.390 | 0.671 |

variance: between-patient 0.0000, residual 1.1203; derived tooth-level ICC(patient) = 0.000; AIC 2574.7

`alignment_uncertain` = images where method C could not place the six zeniths and equal-split regions were used; its fixed effect tests whether tooth-level disagreement is larger on those images.
