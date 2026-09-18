"""Paired architecture comparison and the pre-registered decision rule (PROTOCOL.md §5, §8)."""
import numpy as np
import pandas as pd
import pytest

from gsv4.eval.architecture import (
    DECISION_THRESHOLD_MM, decision, edge_table, model_table, paired_difference, per_image_errors, seed_averaged, seed_table,
)


def _long(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(1, 7, n)
    rows = []
    for model, off in (("A", 0.20), ("B", 0.50)):
        for s in (42, 43, 44):
            r = rng.normal(0, 0.05, n)
            for i in range(n):
                rows.append({"model": model, "seed": s, "uid": f"u{i}", "ref_mm": ref[i], "selected_mm": ref[i] + off + r[i],
                             "gingiva_top_edge_mae_px": 5.0 + off, "gingiva_bottom_edge_bias_px": 11.0 + off,
                             "gingiva_top_edge_bias_px": 2.0, "gingiva_bottom_edge_mae_px": 12.0})
    return pd.DataFrame(rows)


def test_per_image_errors_and_seed_table():
    d = per_image_errors(_long())
    assert {"error_mm", "abs_error_mm"} <= set(d.columns)
    st = seed_table(d)
    assert len(st) == 6 and set(st["model"]) == {"A", "B"}
    a = st[st.model == "A"]; b = st[st.model == "B"]
    assert a["mae_mm"].mean() == pytest.approx(0.20, abs=0.03) and b["mae_mm"].mean() == pytest.approx(0.50, abs=0.03)
    mt = model_table(st).set_index("model")
    assert mt.loc["A", "n_seeds"] == 3 and mt.loc["A", "mae_mm_sd"] < 0.05
    assert mt.loc["B", "bias_mm_mean"] == pytest.approx(0.50, abs=0.03)


def test_seed_averaged_gives_one_row_per_model_and_image():
    sa = seed_averaged(per_image_errors(_long()))
    assert len(sa) == 80 and set(sa["n_seeds"]) == {3}


def test_paired_difference_is_signed_and_paired():
    d = seed_averaged(per_image_errors(_long()))
    a, b = d[d.model == "A"], d[d.model == "B"]
    r = paired_difference(a, b, n_boot=300)
    assert r["n"] == 40
    assert r["diff_mae_mm"] == pytest.approx(r["mae_a"] - r["mae_b"], abs=1e-9)
    assert r["diff_mae_mm"] < 0                      # A is better, so the difference is negative
    assert r["ci_low"] < r["diff_mae_mm"] < r["ci_high"] and r["excludes_zero"]
    # the paired SD is far below the SD of either model's own error against the reference
    assert r["sd_paired_mm"] < 0.2
    assert paired_difference(a, b.assign(uid=b.uid + "x"), n_boot=50) == {"n": 0}


def test_decision_needs_both_conditions():
    big = {"model_b": "X", "n": 29, "diff_mae_mm": 0.30, "ci_low": 0.10, "ci_high": 0.50, "excludes_zero": True}
    small = {"model_b": "Y", "n": 29, "diff_mae_mm": 0.10, "ci_low": 0.02, "ci_high": 0.18, "excludes_zero": True}
    wide = {"model_b": "Z", "n": 29, "diff_mae_mm": 0.40, "ci_low": -0.05, "ci_high": 0.85, "excludes_zero": False}
    assert decision([big], "YOLOv11x")["change_final_model"] and decision([big], "YOLOv11x")["winner"] == "X"
    assert not decision([small], "YOLOv11x")["change_final_model"]      # below the threshold
    assert not decision([wide], "YOLOv11x")["change_final_model"]       # interval contains zero
    assert not decision([], "YOLOv11x")["change_final_model"]
    r = decision([small, wide, big], "YOLOv11x")
    assert r["winner"] == "X" and r["threshold_mm"] == DECISION_THRESHOLD_MM and "stays" not in r["outcome"]
    assert "stays YOLOv11x" in decision([small], "YOLOv11x")["outcome"]


def test_edge_table_converts_pixels_to_mm():
    t = edge_table(per_image_errors(_long()), 16.84).set_index(["model", "seed"])
    assert t.loc[("A", 42), "gingiva_top_edge_mae_mm"] == pytest.approx(5.20 / 16.84, abs=1e-6)
    assert t.loc[("B", 44), "gingiva_bottom_edge_bias_mm"] == pytest.approx(11.50 / 16.84, abs=1e-6)


def test_integrity_check_flags_a_wrongly_read_label_space():
    """The RF-DETR fault leaves three traces at once: no lip mask anywhere, dropped instances, and
    an edge error that is entirely systematic because every image was shifted the same way."""
    from gsv4.eval.architecture import integrity_check

    good = _long(n=10)
    good["n_lip_instances"], good["n_ignored"] = 1, 0
    good["gingiva_top_edge_mae_px"] = [5.0, 3.0] * (len(good) // 2)      # MAE above |bias|
    good["gingiva_top_edge_bias_px"] = [1.0, -1.0] * (len(good) // 2)
    t = integrity_check(good).set_index("model")
    assert not t["suspect"].any() and (t["problems"] == "").all()

    bad = good.copy()
    bad.loc[bad.model == "B", "n_lip_instances"] = 0
    bad.loc[bad.model == "B", "n_ignored"] = 1
    bad.loc[bad.model == "B", "gingiva_top_edge_mae_px"] = 6.15
    bad.loc[bad.model == "B", "gingiva_top_edge_bias_px"] = -6.15
    t = integrity_check(bad).set_index("model")
    assert not t.loc["A", "suspect"] and t.loc["B", "suspect"]
    assert t.loc["B", "images_without_lip"] == t.loc["B", "n_rows"]
    assert t.loc["B", "instances_ignored"] == 30 and bool(t.loc["B", "edge_error_fully_systematic"])
    for phrase in ("no lip mask on any image", "dropped for having no role", "entirely systematic"):
        assert phrase in t.loc["B", "problems"]


def test_bias_scatter_table_separates_a_shift_from_scatter():
    from gsv4.eval.architecture import bias_scatter_table, error_correlation

    rng = np.random.default_rng(7)
    ref = rng.uniform(1, 7, 30)
    noise = rng.normal(0, 0.30, 30)
    rows = []
    for model, off in (("shifted", 0.60), ("centred", 0.00)):
        for s in (42, 43):
            for i in range(30):                       # same noise, different constant
                rows.append({"model": model, "seed": s, "uid": f"u{i}", "ref_mm": ref[i], "selected_mm": ref[i] + off + noise[i]})
    d = pd.DataFrame(rows)
    t = bias_scatter_table(d).set_index("model")
    assert t.loc["shifted", "bias_mm"] == pytest.approx(0.60, abs=0.08) and abs(t.loc["centred", "bias_mm"]) < 0.08
    assert t.loc["shifted", "mae_mm"] > t.loc["centred", "mae_mm"]
    # once each model's own bias is removed the two are the same data, so the residual MAE matches
    assert t.loc["shifted", "mae_without_own_bias_mm"] == pytest.approx(t.loc["centred", "mae_without_own_bias_mm"], abs=1e-9)
    assert t.loc["shifted", "removable_by_calibration_mm"] > t.loc["centred", "removable_by_calibration_mm"]
    assert t.loc["shifted", "sd_of_error_mm"] == pytest.approx(t.loc["centred", "sd_of_error_mm"], abs=1e-9)
    c = error_correlation(d)
    assert c.loc["shifted", "centred"] == pytest.approx(1.0, abs=1e-9)   # identical scatter, differing only by a shift
