"""Mask-level lower-edge correction, its bookkeeping, the config key and the value-level comparison variant."""
import numpy as np
import pytest

from gsv4.measure.calibration import (
    apply_bottom_edge_offset, bottom_edge_offset_from_config, offset_mm_at, shift_lower_edge_up, value_level_mm,
)


def test_shift_lower_edge_up_removes_only_the_bottom_run():
    m = np.zeros((12, 3), dtype=bool)
    m[2:8, 0] = True                      # one run rows 2..7
    m[1:3, 1] = True; m[6:10, 1] = True   # two runs: 1..2 and 6..9 (gap of 3)
    m[9:11, 2] = True                     # short run 9..10
    out = shift_lower_edge_up(m, 3)
    assert out[:, 0].nonzero()[0].tolist() == [2, 3, 4]
    assert out[:, 1].nonzero()[0].tolist() == [1, 2, 6]
    assert out[:, 2].sum() == 0
    assert shift_lower_edge_up(m, 0).sum() == m.sum() and not np.shares_memory(shift_lower_edge_up(m, 0), m)


def test_apply_bottom_edge_offset_counts_zeroed_columns_and_never_goes_negative():
    m = np.zeros((30, 5), dtype=bool)
    m[5:25, 0] = True      # 20 px thick -> 7 after -13
    m[10:20, 1] = True     # 10 px -> zeroed
    m[0:30, 2] = True      # 30 px -> 17
    # column 3 empty; column 4: 13 px exactly -> zeroed
    m[10:23, 4] = True
    r = apply_bottom_edge_offset(m, -13)
    assert r["shift_px"] == 13 and r["n_columns_before"] == 4 and r["n_columns_after"] == 2 and r["n_columns_zeroed"] == 2
    assert r["columns_zeroed_frac"] == pytest.approx(0.5)
    assert r["mask"][:, 0].sum() == 7 and r["mask"][:, 2].sum() == 17 and r["mask"][:, 1].sum() == 0 and r["mask"][:, 4].sum() == 0
    assert r["mask"][:, 0].nonzero()[0].tolist() == list(range(5, 12))          # top rows kept, bottom removed
    z = apply_bottom_edge_offset(m, 0)
    assert z["n_columns_zeroed"] == 0 and z["mask"].sum() == m.sum()


def test_config_key_and_deprecated_key():
    assert bottom_edge_offset_from_config({"measurement": {"bottom_edge_offset_px": -13}}) == -13.0
    assert bottom_edge_offset_from_config({"measurement": {}}) == 0.0 and bottom_edge_offset_from_config({}) == 0.0
    with pytest.raises(KeyError):
        bottom_edge_offset_from_config({"measurement": {"offset_px": -13}})


def test_value_level_variant_and_mm_helper():
    r = value_level_mm([100.0, 10.0, np.nan], [16.84, 16.84, 16.84], -13)
    assert r["mm"][0] == pytest.approx(87 / 16.84) and r["mm"][1] == 0.0 and r["n_clipped"] == 1 and np.isnan(r["mm"][2])
    assert offset_mm_at(16.84, -13) == pytest.approx(-0.772, abs=1e-3)


def test_model_table_uses_the_stage6_corrected_pixel_columns():
    import pandas as pd

    from gsv4.eval.expert import model_table

    per = pd.DataFrame({"image": ["a", "b"], "selected_method": ["C_p25"] * 2, "selected_px_per_mm": [16.84] * 2,
                        "C_p25_px": [100.0, 50.0], "C_p25_px_corrected": [87.0, 37.0], "bottom_edge_offset_px": [-13.0] * 2,
                        "qc_flags": ["", ""], **{f"C_p25_region_{i}_px": [100.0, 50.0] for i in range(1, 7)},
                        **{f"C_p25_region_{i}_px_corrected": [87.0, 37.0] for i in range(1, 7)}})
    per["selected_mm"] = per["C_p25_px"] / 16.84
    m = model_table(per, expert_scale=pd.Series({"a": 20.0, "b": 10.0}))
    assert m["model_has_correction"].all() and m["model_bottom_edge_offset_px"].iloc[0] == -13.0
    assert m["model_mm_global_corrected"].tolist() == pytest.approx([87 / 16.84, 37 / 16.84])
    assert m["model_mm_expert"].tolist() == pytest.approx([5.0, 5.0]) and m["model_mm_expert_corrected"].tolist() == pytest.approx([4.35, 3.7])
    assert m["model_region_3_mm_expert_corrected"].tolist() == pytest.approx([4.35, 3.7])
    assert set(m["model_label_global_corrected"]) <= {"E1", "E1-E2", "E2-E3", "E3", "E4", "NO_VISIBLE_GINGIVA", "UNCLASSIFIED"}
    gt = model_table(per.drop(columns=[c for c in per.columns if c.endswith("_corrected") or c == "bottom_edge_offset_px"]))
    assert not gt["model_has_correction"].any() and gt["model_mm_global_corrected"].tolist() == gt["model_mm_global"].tolist()
