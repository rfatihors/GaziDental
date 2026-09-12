import numpy as np
import pandas as pd
import pytest

from gsv4.eval.agreement import mixed_model_diff


def _data(bias=0.3, tooth_effect=0.0, sd_patient=0.4, sd_resid=0.3, n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n):
        u = rng.normal(0, sd_patient)
        for t in range(1, 7):
            rows.append({"patient": f"p{p}", "tooth": t, "diff": bias + u + (tooth_effect if t == 6 else 0) + rng.normal(0, sd_resid),
                         "alignment_uncertain": p % 4 == 0})
    return pd.DataFrame(rows)


def test_intercept_only_recovers_bias_and_variance_components():
    d = _data()
    r = mixed_model_diff(d, "1")
    fe = r["fixed_effects"]
    assert fe.loc["Intercept", "estimate"] == pytest.approx(0.3, abs=0.12)
    assert fe.loc["Intercept", "ci_low"] < 0.3 < fe.loc["Intercept", "ci_high"]
    assert r["var_patient"] == pytest.approx(0.16, abs=0.06)
    assert r["var_resid"] == pytest.approx(0.09, abs=0.03)
    assert 0.5 < r["icc_patient"] < 0.8
    assert r["n_obs"] == 480 and r["n_groups"] == 80 and r["converged"]


def test_tooth_and_alignment_fixed_effects():
    d = _data(tooth_effect=0.5)
    r = mixed_model_diff(d, "C(tooth)")
    fe = r["fixed_effects"]
    assert fe.loc["C(tooth)[T.6]", "estimate"] == pytest.approx(0.5, abs=0.15)
    assert fe.loc["C(tooth)[T.3]", "estimate"] == pytest.approx(0.0, abs=0.15)
    r2 = mixed_model_diff(d, "C(tooth) + alignment_uncertain")
    assert "alignment_uncertain[T.True]" in r2["fixed_effects"].index


def test_boundary_case_falls_back_to_cluster_robust_ols():
    d = _data(bias=0.2, sd_patient=0.0, sd_resid=0.3, n=60, seed=2)   # no patient effect at all
    for rhs in ("1", "C(tooth)", "C(tooth) + alignment_uncertain"):
        r = mixed_model_diff(d, rhs)
        fe = r["fixed_effects"]
        assert np.isfinite(fe[["estimate", "ci_low", "ci_high"]].to_numpy()).all()
        assert (fe["ci_high"] - fe["ci_low"] < 1.0).all()
        assert fe.loc["Intercept", "ci_low"] < 0.2 < fe.loc["Intercept", "ci_high"]
        assert "estimator" in r and "note" in r
