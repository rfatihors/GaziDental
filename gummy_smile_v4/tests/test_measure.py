"""Measurement module tests — spec §5 tests 1–9 on synthetic masks."""
import math

import numpy as np
import pytest

from gsv4.measure.gingival_display import measure_gingival_display
from gsv4.measure.profile import column_profile, lip_anchor
from gsv4.measure.qc import QCFlag
from tests.synth_masks import festooned_band, lip_band, papilla_triangles, rect_band

SHAPE = (600, 1800)
CFG = {
    "regions": 6,
    "median_filter_frac": 0.01,
    "zenith_window_frac": 0.01,
    "zenith_min_distance_frac": 0.03,
    "festoon_min_distance_frac": 0.03,
    "boundary_gap_max_frac": 0.02,
    "implausible_mm": 20.0,
    "default_method": {"regioning": "A", "estimator": "p25", "anchored": False},
}


def test_1_rectangular_band_measures_exact_height():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    for key, vals in r.region_values.items():
        if key[0] == "A" and not key[2]:
            assert vals == pytest.approx([50.0] * 6)
    assert r.gingival_display_px == pytest.approx(50.0)
    assert r.unit == "px" and r.gingival_display_mm is None
    assert (r.x0, r.x1) == (300, 1500)
    assert QCFlag.NO_LIP_MASK in r.flags


def test_2_festooned_band_min_near_zenith_max_near_papilla_and_method_c_finds_zeniths():
    g, zeniths, _ = festooned_band(SHAPE, 300, 1500, 200, zenith_t=30, papilla_t=90)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    a_min = r.region_values[("A", "min", False)]
    a_max = r.region_values[("A", "max", False)]
    assert all(abs(v - 30) <= 1 for v in a_min)
    assert all(abs(v - 90) <= 2 for v in a_max)
    assert all(v < m for v, m in zip(r.region_values[("A", "p10", False)], a_max))
    found = r.zeniths_c
    assert len(found) == 6
    assert all(abs(f - z) <= 6 for f, z in zip(found, zeniths))
    # C window = ±1 % of width around the zenith: min equals the zenith thickness, the
    # median sits slightly above it on the festoon slope
    assert all(abs(v - 30) <= 1 for v in r.region_values[("C", "min", False)])
    assert all(30 <= v <= 40 for v in r.region_values[("C", "median", False)])
    assert QCFlag.ZENITH_DETECTION_FAILED not in r.flags
    # festoon regioning (B) also succeeds and gives 6 regions
    assert QCFlag.FESTOON_DETECTION_FAILED not in r.flags
    assert len(r.regions["B"]) == 6


def test_3_two_separate_pieces_both_counted():
    g = rect_band(SHAPE, 300, 800, 200, 40) | rect_band(SHAPE, 1000, 1500, 200, 40)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert r.n_components == 2
    assert QCFlag.GINGIVA_MULTI_COMPONENT in r.flags
    # the gap columns are zero; regions fully inside a piece read 40
    vals = r.region_values[("A", "median", False)]
    assert vals[0] == 40 and vals[5] == 40
    assert (r.x0, r.x1) == (300, 1500)
    t, _, _ = column_profile(g)
    assert t[900] == 0 and t[500] == 40 and t[1200] == 40


def test_4_lip_mask_does_not_change_gingiva_value():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    lip = lip_band(SHAPE, 200, 1600, bottom=199, thickness=60)
    r_without = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    r_with = measure_gingival_display(g, lip, px_per_mm=None, cfg=CFG)
    assert r_with.gingival_display_px == r_without.gingival_display_px == pytest.approx(50.0)
    for key in r_without.region_values:
        if not key[2]:
            assert r_with.region_values[key] == pytest.approx(r_without.region_values[key])
    # lip intersects the x-window: here lip is wider, so the window stays the gingiva extent
    assert (r_with.x0, r_with.x1) == (300, 1500)
    assert QCFlag.NO_LIP_MASK not in r_with.flags


def test_5_hole_uses_longest_run_not_ymax_minus_ymin():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    g[220:225, :] = False          # 5-row hole -> runs of 20 and 25
    t, top, bottom = column_profile(g)
    assert t[900] == 25 and top[900] == 225 and bottom[900] == 249
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert r.gingival_display_px == pytest.approx(25.0)


def test_6_only_papillae_gives_zero_zeniths_and_low_image_value():
    centers = [300 + int((k) * 200) for k in range(7)]   # 7 papillae, 6 teeth
    g = papilla_triangles(SHAPE, centers, top=200, height=60, half_w=25)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    c_vals = r.region_values[("C", "median", False)]
    assert len(c_vals) == 6 and all(v == 0 for v in c_vals)
    assert QCFlag.REGION_ZERO in r.flags
    assert r.image_values[("A", "min", False)] == 0
    assert r.image_values[("A", "median", False)] < 60 / 4
    t, _, _ = column_profile(g)
    assert t[400] == 0 and not np.isnan(t).any()


def test_7_lip_anchored_equals_thickness_plus_gap():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    gap = 9
    lip = lip_band(SHAPE, 300, 1500, bottom=200 - gap - 1, thickness=40)
    r = measure_gingival_display(g, lip, px_per_mm=None, cfg=CFG)
    assert r.region_values[("A", "median", True)] == pytest.approx([50 + gap] * 6)
    assert r.gap_median == gap
    assert QCFlag.LIP_GINGIVA_BOUNDARY_MISMATCH not in r.flags
    t, top, bottom = column_profile(g)
    lb = lip_anchor(lip, top)
    assert lb[900] == 200 - gap - 1
    # a large gap raises the boundary flag
    lip_far = lip_band(SHAPE, 300, 1500, bottom=200 - 100, thickness=40)
    r2 = measure_gingival_display(g, lip_far, px_per_mm=None, cfg=CFG)
    assert QCFlag.LIP_GINGIVA_BOUNDARY_MISMATCH in r2.flags


def test_8_units():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    r = measure_gingival_display(g, None, px_per_mm=10.0, cfg=CFG)
    assert r.gingival_display_mm == pytest.approx(5.0) and r.unit == "mm"
    r2 = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert r2.gingival_display_mm is None and r2.unit == "px"
    row = r.to_row()
    assert row["gingival_display_mm"] == pytest.approx(5.0)
    assert "mean_mm" not in row and "mm_per_pixel" not in row
    assert row["A_p25_px"] == pytest.approx(50.0)
    assert row["A_p25_region_3_px"] == pytest.approx(50.0)
    assert row["A_p25_lipanchored_px"] is None or math.isnan(row["A_p25_lipanchored_px"])
    assert isinstance(row["qc_flags"], str)


def test_9_shape_mismatch_is_an_error():
    g = rect_band(SHAPE, 300, 1500, 200, 50)
    lip = np.zeros((SHAPE[0] + 1, SHAPE[1]), dtype=bool)
    with pytest.raises(ValueError):
        measure_gingival_display(g, lip, px_per_mm=None, cfg=CFG)


def test_empty_mask_flags_and_nan():
    g = np.zeros(SHAPE, dtype=bool)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert QCFlag.NO_GINGIVA_MASK in r.flags
    assert math.isnan(r.gingival_display_px)
    assert r.to_row()["qc_flags"].startswith("no_gingiva_mask")


def test_zenith_spacing_scales_with_width():
    # same geometry at half resolution must still find six zeniths
    shape = (300, 900)
    g, zeniths, _ = festooned_band(shape, 150, 750, 100, zenith_t=15, papilla_t=45)
    r = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert len(r.zeniths_c) == 6 and all(abs(f - z) <= 4 for f, z in zip(r.zeniths_c, zeniths))


def test_midline_from_lip_when_present():
    g, _, _ = festooned_band(SHAPE, 300, 1500, 200, zenith_t=30, papilla_t=90)
    lip = lip_band(SHAPE, 200, 1800, bottom=199, thickness=40)  # lip median x ≈ 1000
    r = measure_gingival_display(g, lip, px_per_mm=None, cfg=CFG)
    assert abs(r.midline_x - 1000) <= 1
    assert QCFlag.ZENITH_DETECTION_FAILED not in r.flags
    r2 = measure_gingival_display(g, None, px_per_mm=None, cfg=CFG)
    assert r2.midline_x == SHAPE[1] // 2
