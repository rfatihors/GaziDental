"""Oracle harness logic on a synthetic per-image table with a known scale."""
import zlib

import numpy as np
import pandas as pd
import pytest

from gsv4.eval.oracle import (
    COMBOS,
    combo_name,
    dev_holdout_split,
    evaluate_combos,
    evaluate_fixed,
    select_method,
)


def _table(n=100, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(0.5, 8, n)
    df = pd.DataFrame({"uid": [f"high/IMG_{i}" for i in range(n)], "ref_mm": ref})
    for reg, est, anch in COMBOS:
        k = 17.0 if not anch else 19.0
        noise = {"p25": 0.3, "median": 0.3, "p10": 0.4, "p05": 0.5, "min": 0.6, "max": 1.0}[est]
        noise += {"A": 0.0, "B": 0.05, "C": 0.1}[reg]
        # per-combination stream so that adding an estimator does not reshuffle the noise of the others
        crng = np.random.default_rng(zlib.crc32(combo_name(reg, est, anch).encode()) + seed)
        df[f"{combo_name(reg, est, anch)}_px"] = k * ref + crng.normal(0, noise * k, n)
    df["frame_ok"] = True
    df["has_dash_zero"] = False
    df["has_ambiguous_100_999"] = False
    df["label_inconsistent"] = False
    return df


def test_dev_holdout_split_is_deterministic_and_60_40():
    uids = [f"u{i}" for i in range(145)]
    s1 = dev_holdout_split(uids, seed=42, dev_frac=0.6)
    s2 = dev_holdout_split(uids, seed=42, dev_frac=0.6)
    assert s1 == s2
    assert len(s1["dev"]) == 87 and len(s1["holdout"]) == 58
    assert set(s1["dev"]).isdisjoint(s1["holdout"])


def test_evaluate_combos_fits_scale_on_dev_only_and_recovers_it():
    df = _table()
    split = dev_holdout_split(list(df["uid"]), seed=42, dev_frac=0.6)
    res = evaluate_combos(df, split)
    row = res[(res.regioning == "A") & (res.estimator == "p25") & (~res.anchored)].iloc[0]
    assert row["px_per_mm_dev"] == pytest.approx(17.0, abs=0.3)
    assert row["n_dev"] == 60 and row["n_holdout"] == 40
    assert row["mae_holdout"] < 0.5 and row["r_holdout"] > 0.95 and row["icc2_1_holdout"] > 0.9
    # scale is the same number on both subsets (no refit on holdout)
    assert "px_per_mm_holdout" not in res.columns
    assert set(res.columns) >= {"mae_dev", "rmse_dev", "mae_holdout", "rmse_holdout", "ba_bias_holdout", "ba_prop_slope_holdout", "threshold_agreement_holdout"}


def test_select_method_prefers_simpler_within_tolerance():
    df = _table()
    split = dev_holdout_split(list(df["uid"]), seed=42, dev_frac=0.6)
    res = evaluate_combos(df, split)
    chosen, why = select_method(res, tolerance_mm=0.02)
    # A/p25 and A/median have equal noise; A is simplest, gingiva-thickness preferred
    assert chosen["regioning"] == "A" and chosen["anchored"] is False
    assert chosen["estimator"] in ("p25", "median")
    assert "0.02" in why
    # with a huge tolerance the simplest of all wins
    chosen2, _ = select_method(res, tolerance_mm=10.0)
    assert (chosen2["regioning"], chosen2["estimator"], chosen2["anchored"]) == ("A", "p25", False)


def test_evaluate_fixed_uses_the_given_scale_and_fits_nothing():
    df = _table()
    split = dev_holdout_split(list(df["uid"]), seed=42, dev_frac=0.6)
    res = evaluate_combos(df, split).set_index("combo")
    k_fitted = float(res.loc["C_p25", "px_per_mm_dev"])
    fixed = evaluate_fixed(df, split, "C_p25", k_fitted)
    # same scale as the dev fit -> the same numbers as evaluate_combos, no selection involved
    assert fixed["px_per_mm_dev"] == pytest.approx(k_fitted) and fixed["scale_fitted_here"] is False
    assert (fixed["regioning"], fixed["estimator"], fixed["anchored"]) == ("C", "p25", False)
    assert fixed["mae_dev"] == pytest.approx(res.loc["C_p25", "mae_dev"])
    assert fixed["mae_holdout"] == pytest.approx(res.loc["C_p25", "mae_holdout"])
    # a scale 10 % off must move the numbers: nothing here re-fits it away
    off = evaluate_fixed(df, split, "C_p25", k_fitted * 1.1)
    assert off["ba_bias_holdout"] < fixed["ba_bias_holdout"] - 0.1
    assert evaluate_fixed(df, split, "C_p25_lipanchored", 19.0)["anchored"] is True
