import math

import numpy as np
import pytest

from gsv4.eval.kappa import (
    bootstrap_ci, cohen_kappa, fleiss_kappa, kappa_bundle, observed_agreement, pabak, per_class_metrics, wilson_ci,
)


def test_cohen_kappa_hand_computed_2x2():
    # classic example: 20 agree yes, 15 agree no, 5 + 10 disagree -> po = 0.7, pe = 0.5, kappa = 0.4
    a = ["E1"] * 25 + ["E2"] * 25
    b = ["E1"] * 20 + ["E2"] * 5 + ["E1"] * 10 + ["E2"] * 15
    assert cohen_kappa(a, b, ["E1", "E2"]) == pytest.approx(0.4)
    assert observed_agreement(a, b) == pytest.approx(0.7)
    assert pabak(a, b, ["E1", "E2"]) == pytest.approx(2 * 0.7 - 1)


def test_linear_weights_penalise_distance():
    ref = ["E1", "E2", "E3", "E4"] * 10
    near = ["E2", "E1", "E4", "E3"] * 10       # every case off by one class
    far = ["E4", "E3", "E2", "E1"] * 10        # reversed
    assert cohen_kappa(ref, near, weights="linear") > cohen_kappa(ref, far, weights="linear")
    assert cohen_kappa(ref, ref, weights="linear") == pytest.approx(1.0)
    assert cohen_kappa(ref, near) == cohen_kappa(ref, far)  # unweighted does not distinguish


def test_zero_count_class_is_safe():
    ref = ["E1"] * 30 + ["E2"] * 10 + ["E3"] * 5
    pred = ["E1"] * 28 + ["E2"] * 2 + ["E2"] * 9 + ["E3"] * 1 + ["E3"] * 5
    k = cohen_kappa(ref, pred, weights="linear")
    assert 0 < k <= 1
    pc = {r["class"]: r for r in per_class_metrics(ref, pred)}
    assert pc["E4"]["n_reference"] == 0 and math.isnan(pc["E4"]["sensitivity"])
    assert pc["E4"]["specificity"] == 1.0 and pc["E4"]["n_predicted"] == 0
    assert pc["E1"]["sensitivity"] == pytest.approx(28 / 30)
    lo, hi = pc["E1"]["sensitivity_ci_low"], pc["E1"]["sensitivity_ci_high"]
    assert lo < 28 / 30 < hi
    # single-class raters: kappa undefined, not an exception
    assert math.isnan(cohen_kappa(["E1"] * 5, ["E1"] * 5))


def test_fleiss_kappa_known_example():
    # Fleiss (1971) toy: 4 subjects, 3 raters
    ratings = [["E1", "E1", "E1"], ["E1", "E1", "E2"], ["E2", "E2", "E2"], ["E1", "E2", "E2"]]
    counts = np.array([[3, 0], [2, 1], [0, 3], [1, 2]], dtype=float)
    p_i = (counts * (counts - 1)).sum(axis=1) / 6
    p_j = counts.sum(axis=0) / 12
    expected = (p_i.mean() - (p_j**2).sum()) / (1 - (p_j**2).sum())
    assert fleiss_kappa(ratings, ["E1", "E2"]) == pytest.approx(expected)
    assert fleiss_kappa([["E1", "E1", "E1"]] * 5 + [["E2", "E2", "E2"]] * 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fleiss_kappa([["E1", "E1"], ["E1"]])


def test_wilson_ci():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35
    assert wilson_ci(0, 0) == (math.nan, math.nan) or all(math.isnan(v) for v in wilson_ci(0, 0))


def test_bootstrap_ci_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(0)
    ref = rng.choice(["E1", "E2", "E3"], 120, p=[0.6, 0.3, 0.1])
    pred = np.where(rng.random(120) < 0.8, ref, rng.choice(["E1", "E2", "E3"], 120))
    b1 = kappa_bundle(ref, pred, n_boot=300, seed=42)
    b2 = kappa_bundle(ref, pred, n_boot=300, seed=42)
    assert b1 == b2
    assert b1["kappa_linear_ci_low"] <= b1["kappa_linear"] <= b1["kappa_linear_ci_high"]
    assert b1["n"] == 120
    ci = bootstrap_ci(lambda idx: float("nan"), 10, n_boot=5)
    assert math.isnan(ci["ci_low"]) and ci["n_undefined"] == 5
