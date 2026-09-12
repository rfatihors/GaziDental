"""Agreement statistics: ICC (pingouin), Bland–Altman with CIs and proportional bias,
regression through the origin for the px/mm scale."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def scale_through_origin(px: Sequence[float], mm: Sequence[float]) -> Dict[str, float]:
    """Least-squares ``px = k * mm`` (no intercept). Returns k, R² (about the origin
    model), residual SD in px and in mm."""
    px = np.asarray(px, dtype=float)
    mm = np.asarray(mm, dtype=float)
    ok = np.isfinite(px) & np.isfinite(mm)
    px, mm = px[ok], mm[ok]
    k = float(np.sum(px * mm) / np.sum(mm * mm))
    resid = px - k * mm
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((px - px.mean()) ** 2))
    return {
        "px_per_mm": k, "n": int(ok.sum()), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "resid_sd_px": float(np.std(resid, ddof=1)) if len(px) > 1 else float("nan"),
        "resid_sd_mm": float(np.std(resid / k, ddof=1)) if len(px) > 1 else float("nan"),
    }


def loo_scale(px: Sequence[float], mm: Sequence[float]) -> Dict[str, float]:
    """Leave-one-out distribution of the through-origin scale and the LOO prediction error."""
    px = np.asarray(px, dtype=float)
    mm = np.asarray(mm, dtype=float)
    n = len(px)
    ks, errs = [], []
    sxy, sxx = np.sum(px * mm), np.sum(mm * mm)
    for i in range(n):
        k = (sxy - px[i] * mm[i]) / (sxx - mm[i] * mm[i])
        ks.append(k)
        errs.append(px[i] / k - mm[i])
    ks = np.asarray(ks)
    errs = np.asarray(errs)
    return {
        "px_per_mm_mean": float(ks.mean()), "px_per_mm_sd": float(ks.std(ddof=1)),
        "px_per_mm_min": float(ks.min()), "px_per_mm_max": float(ks.max()),
        "loo_mae_mm": float(np.abs(errs).mean()), "loo_rmse_mm": float(np.sqrt(np.mean(errs**2))),
    }


def bland_altman(pred: Sequence[float], ref: Sequence[float]) -> Dict[str, float]:
    """Differences ``pred - ref``: bias with 95 % CI, SD, limits of agreement with 95 % CIs
    (Bland & Altman 1999), and proportional bias (slope of diff on the mean, with p)."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    n = len(pred)
    diff = pred - ref
    mean = (pred + ref) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    se_bias = sd / np.sqrt(n)
    se_loa = sd * np.sqrt(3 / n)
    if n > 2 and np.ptp(mean) > 0:
        lr = stats.linregress(mean, diff)
        slope, p, intercept = float(lr.slope), float(lr.pvalue), float(lr.intercept)
    else:
        slope, p, intercept = float("nan"), float("nan"), float("nan")
    return {
        "n": n, "bias": bias, "bias_ci_low": bias - t * se_bias, "bias_ci_high": bias + t * se_bias, "sd": sd,
        "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
        "loa_low_ci_low": bias - 1.96 * sd - t * se_loa, "loa_low_ci_high": bias - 1.96 * sd + t * se_loa,
        "loa_high_ci_low": bias + 1.96 * sd - t * se_loa, "loa_high_ci_high": bias + 1.96 * sd + t * se_loa,
        "prop_slope": slope, "prop_intercept": intercept, "prop_p": p,
        "mean_abs_diff": float(np.abs(diff).mean()),
    }


def icc_two_raters(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """ICC(2,1), ICC(2,k), ICC(3,1) with 95 % CIs for two raters over the same targets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    long = pd.DataFrame({
        "target": np.concatenate([np.arange(n), np.arange(n)]),
        "rater": ["a"] * n + ["b"] * n,
        "value": np.concatenate([a[ok], b[ok]]),
    })
    return icc_long(long, "target", "rater", "value")


def icc_long(df: pd.DataFrame, targets: str, raters: str, ratings: str) -> Dict[str, float]:
    """Two-way ICCs (McGraw & Wong 1996) with exact 95 % CIs, from a long table.

    ICC(2,1)/(2,k): two-way random, absolute agreement; ICC(3,1)/(3,k): two-way mixed,
    consistency. Targets with a missing rating are dropped (complete cases). Point
    estimates equal ``pingouin.intraclass_corr``; CIs are not rounded.
    """
    wide = df.pivot_table(index=targets, columns=raters, values=ratings, aggfunc="mean").dropna()
    x = wide.to_numpy(dtype=float)
    n, k = x.shape
    grand = x.mean()
    ms_r = k * np.sum((x.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_c = n * np.sum((x.mean(axis=0) - grand) ** 2) / (k - 1)
    ss_e = np.sum((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand) ** 2)
    ms_e = ss_e / ((n - 1) * (k - 1))
    alpha = 0.05
    out: Dict[str, float] = {"n_targets": int(n), "k_raters": int(k), "ms_rows": float(ms_r), "ms_cols": float(ms_c), "ms_error": float(ms_e)}
    # ICC(3,1), ICC(3,k)
    icc31 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
    icc3k = (ms_r - ms_e) / ms_r
    f = ms_r / ms_e if ms_e > 0 else np.inf
    fl = f / stats.f.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    fu = f * stats.f.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    out.update({"icc3_1": float(icc31), "icc3_1_ci_low": float((fl - 1) / (fl + k - 1)), "icc3_1_ci_high": float((fu - 1) / (fu + k - 1)),
                "icc3_k": float(icc3k), "icc3_k_ci_low": float(1 - 1 / fl), "icc3_k_ci_high": float(1 - 1 / fu)})
    # ICC(2,1), ICC(2,k)
    icc21 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icc2k = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    if 0 < icc21 < 1:
        a = k * icc21 / (n * (1 - icc21))
        b = 1 + k * icc21 * (n - 1) / (n * (1 - icc21))
        v = (a * ms_c + b * ms_e) ** 2 / ((a * ms_c) ** 2 / (k - 1) + (b * ms_e) ** 2 / ((n - 1) * (k - 1)))
        f_l = stats.f.ppf(1 - alpha / 2, n - 1, v)
        f_u = stats.f.ppf(1 - alpha / 2, v, n - 1)
        low = n * (ms_r - f_l * ms_e) / (f_l * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r)
        high = n * (f_u * ms_r - ms_e) / (k * ms_c + (k * n - k - n) * ms_e + n * f_u * ms_r)
    else:
        low, high = float("nan"), float("nan")
    out.update({"icc2_1": float(icc21), "icc2_1_ci_low": float(low), "icc2_1_ci_high": float(high),
                "icc2_k": float(icc2k), "icc2_k_ci_low": float(low * k / (1 + low * (k - 1))), "icc2_k_ci_high": float(high * k / (1 + high * (k - 1)))})
    return out


def pearson(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    r, p = stats.pearsonr(a[ok], b[ok])
    return {"r": float(r), "r_p": float(p)}


def mixed_model_diff(df: pd.DataFrame, formula_rhs: str = "1", group: str = "patient", diff: str = "diff") -> Dict[str, object]:
    """Linear mixed model ``diff ~ <rhs> + (1 | group)`` via statsmodels MixedLM (REML).

    Returns fixed effects with 95 % CIs and p-values, variance components (between-
    patient, residual) and the derived tooth-level ICC = var_patient / (var_patient +
    var_resid). Tooth-level analyses must use this instead of a naive ICC because the
    six teeth of one patient are not independent.
    """
    import warnings

    import statsmodels.formula.api as smf

    d = df.dropna(subset=[diff]).copy()
    d[group] = d[group].astype(str)
    note = ""
    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm(f"{diff} ~ {formula_rhs}", d, groups=d[group]).fit(reml=True, method=["lbfgs", "powell"])
        except Exception as exc:  # noqa: BLE001 - singular at the boundary; handled below
            note = f"mixed model failed ({exc}); "
            fit = None
    boundary = fit is None or not np.isfinite(np.asarray(fit.bse[fit.model.exog_names], dtype=float)).all() \
        or float(fit.cov_re.iloc[0, 0]) < 1e-8 or (np.asarray(fit.bse[fit.model.exog_names], dtype=float) > 1e3).any()
    if fit is not None and not boundary:
        fe = fit.params[fit.model.exog_names]
        ci = fit.conf_int().loc[fit.model.exog_names]
        var_patient = float(fit.cov_re.iloc[0, 0])
        var_resid = float(fit.scale)
        return {
            "formula": f"{diff} ~ {formula_rhs} + (1 | {group})", "estimator": "MixedLM (REML)", "note": "",
            "n_obs": int(fit.nobs), "n_groups": int(d[group].nunique()),
            "fixed_effects": pd.DataFrame({"estimate": fe, "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues[fit.model.exog_names]}),
            "var_patient": var_patient, "var_resid": var_resid,
            "icc_patient": var_patient / (var_patient + var_resid) if (var_patient + var_resid) > 0 else float("nan"),
            "converged": bool(fit.converged), "aic": float(fit.aic) if fit.aic is not None else float("nan"),
        }
    # boundary / singular case: between-patient variance is (numerically) zero. Report OLS
    # with cluster-robust standard errors (clusters = patients) so CIs remain valid.
    ols = smf.ols(f"{diff} ~ {formula_rhs}", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[group])[0]})
    ci = ols.conf_int()
    var_patient = float(fit.cov_re.iloc[0, 0]) if fit is not None else 0.0
    var_resid = float(fit.scale) if fit is not None else float(ols.mse_resid)
    return {
        "formula": f"{diff} ~ {formula_rhs} + (1 | {group})", "estimator": "OLS, cluster-robust SE (patient)",
        "note": note + "random-intercept variance at the boundary (≈ 0): mixed-model CIs undefined, cluster-robust OLS reported instead",
        "n_obs": int(ols.nobs), "n_groups": int(d[group].nunique()),
        "fixed_effects": pd.DataFrame({"estimate": ols.params, "ci_low": ci[0], "ci_high": ci[1], "p": ols.pvalues}),
        "var_patient": var_patient, "var_resid": var_resid,
        "icc_patient": var_patient / (var_patient + var_resid) if (var_patient + var_resid) > 0 else float("nan"),
        "converged": bool(fit.converged) if fit is not None else False, "aic": float(ols.aic),
    }
