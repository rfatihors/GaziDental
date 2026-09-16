"""Post-hoc offset calibration helpers."""
import numpy as np
import pandas as pd
import pytest

from gsv4.eval.offset import (
    apply_constant, apply_regression, best_pixel_shift, fit_constant, fit_regression, select_correction, shift_lower_edge_up,
)


def test_shift_lower_edge_up_removes_only_the_bottom_run():
    m = np.zeros((12, 3), dtype=bool)
    m[2:8, 0] = True                      # one run rows 2..7
    m[1:3, 1] = True; m[6:10, 1] = True   # two runs: 1..2 and 6..9 (gap of 3)
    m[9:11, 2] = True                     # short run 9..10
    out = shift_lower_edge_up(m, 3)
    assert out[:, 0].nonzero()[0].tolist() == [2, 3, 4]                 # 3 bottom pixels gone
    assert out[:, 1].nonzero()[0].tolist() == [1, 2, 6]                 # upper run untouched, lower run 6..9 -> 6
    assert out[:, 2].sum() == 0                                         # run shorter than d vanishes
    assert shift_lower_edge_up(m, 0).sum() == m.sum() and not np.shares_memory(shift_lower_edge_up(m, 0), m)
    out5 = shift_lower_edge_up(m, 5)
    assert out5[:, 1].nonzero()[0].tolist() == [1, 2]                   # d larger than the gap: upper run still untouched


def test_constant_and_regression_fit_and_apply():
    rng = np.random.default_rng(0)
    ref = rng.uniform(1, 7, 200)
    pred = 0.2 + 1.1 * ref + rng.normal(0, 0.05, 200)
    pred[0] = np.nan
    c = fit_constant(pred, ref)
    assert c["n_fit"] == 199 and c["offset_mm"] == pytest.approx(0.2 + 0.1 * ref[1:].mean(), abs=0.03)
    assert np.isnan(apply_constant(pred, c)[0]) and np.abs(apply_constant(pred, c)[1:] - ref[1:]).mean() < 0.3
    r = fit_regression(pred, ref)
    assert r["b"] == pytest.approx(1 / 1.1, abs=0.02) and r["a"] == pytest.approx(-0.2 / 1.1, abs=0.05)
    assert np.abs(apply_regression(pred, r)[1:] - ref[1:]).mean() < 0.06


def test_best_pixel_shift_prefers_smaller_d_on_ties():
    curve = pd.DataFrame({"d": [0, 2, 4, 6, 8], "mae_dev": [0.8, 0.6, 0.5, 0.5, 0.55]})
    assert best_pixel_shift(curve) == 4


def test_select_correction_rule():
    t = pd.DataFrame({"subset": ["holdout"] * 4 + ["dev"] * 4, "correction": ["none", "constant", "pixel", "regression"] * 2,
                      "mae": [0.80, 0.55, 0.53, 0.54, 0.7, 0.5, 0.4, 0.5]})
    s = select_correction(t, 0.03)
    assert s["chosen"] == "constant" and s["best_holdout"] == "pixel" and s["delta_mm"] == pytest.approx(0.02)
    t.loc[(t.subset == "holdout") & (t.correction == "pixel"), "mae"] = 0.50
    assert select_correction(t, 0.03)["chosen"] == "pixel"
