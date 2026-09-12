"""Agreement statistics on data with known answers."""
import numpy as np
import pytest

from gsv4.eval.agreement import bland_altman, icc_two_raters, loo_scale, scale_through_origin


def test_scale_through_origin_recovers_known_factor():
    rng = np.random.default_rng(0)
    mm = rng.uniform(0.5, 8, 80)
    px = 17.0 * mm + rng.normal(0, 0.5, 80)
    s = scale_through_origin(px, mm)
    assert s["px_per_mm"] == pytest.approx(17.0, abs=0.15)
    assert s["r2"] > 0.99
    assert s["n"] == 80
    loo = loo_scale(px, mm)
    assert abs(loo["px_per_mm_mean"] - 17.0) < 0.15 and loo["px_per_mm_sd"] < 0.1


def test_bland_altman_bias_and_proportional_slope():
    ref = np.linspace(1, 7, 50)
    pred = ref + 0.3                      # constant bias, no proportional bias
    ba = bland_altman(pred, ref)
    assert ba["bias"] == pytest.approx(0.3)
    assert ba["sd"] == pytest.approx(0.0, abs=1e-9)
    assert abs(ba["prop_slope"]) < 1e-9
    pred2 = ref * 1.2                     # proportional bias: diff grows with mean
    ba2 = bland_altman(pred2, ref)
    assert ba2["prop_slope"] > 0 and ba2["prop_p"] < 0.001
    assert ba2["loa_low"] < ba2["bias"] < ba2["loa_high"]
    assert ba2["bias_ci_low"] <= ba2["bias"] <= ba2["bias_ci_high"]


def test_icc_two_raters_known_values():
    rng = np.random.default_rng(1)
    a = rng.uniform(1, 7, 40)
    r = icc_two_raters(a, a + rng.normal(0, 1e-4, 40))   # (identical raters make MSE = 0, undefined F)
    assert r["icc2_1"] == pytest.approx(1.0, abs=1e-6)
    b = a + rng.normal(0, 0.1, 40)
    r2 = icc_two_raters(a, b)
    assert 0.95 < r2["icc2_1"] <= 1.0
    assert r2["icc2_1_ci_low"] <= r2["icc2_1"] <= r2["icc2_1_ci_high"]
    assert 0.95 < r2["icc3_1"] <= 1.0 and 0.95 < r2["icc2_k"] <= 1.0
    c = a + 2.0 + rng.normal(0, 1e-4, 40)  # constant offset lowers ICC(2,1) but not ICC(3,1)
    r3 = icc_two_raters(a, c)
    assert r3["icc3_1"] == pytest.approx(1.0, abs=1e-6)
    assert r3["icc2_1"] < r3["icc3_1"]


def test_icc_matches_pingouin_point_estimates():
    import pandas as pd
    import pingouin as pg
    from gsv4.eval.agreement import icc_long

    rng = np.random.default_rng(3)
    a = rng.uniform(1, 7, 30)
    long = pd.DataFrame({"t": list(range(30)) * 3, "r": ["x"] * 30 + ["y"] * 30 + ["z"] * 30,
                         "v": np.concatenate([a, a + rng.normal(0, 0.3, 30), a * 1.1 + rng.normal(0, 0.3, 30)])})
    mine = icc_long(long, "t", "r", "v")
    ref = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="v").set_index("Type")
    for key, name in (("icc2_1", "ICC2"), ("icc2_k", "ICC2k"), ("icc3_1", "ICC3"), ("icc3_k", "ICC3k")):
        assert mine[key] == pytest.approx(float(ref.loc[name, "ICC"]), abs=1e-9)
        lo, hi = ref.loc[name, "CI95%"]
        assert mine[f"{key}_ci_low"] == pytest.approx(float(lo), abs=0.006)
        assert mine[f"{key}_ci_high"] == pytest.approx(float(hi), abs=0.006)
    assert mine["k_raters"] == 3
