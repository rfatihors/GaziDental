"""Pixel offset -> mm with the image's own scale, clipping and the config key."""
import numpy as np
import pytest

from gsv4.measure.calibration import corrected_mm, offset_mm_at, offset_px_from_config


def test_corrected_mm_uses_each_scale_and_clips():
    r = corrected_mm([100.0, 10.0, np.nan, 13.0], [16.84, 16.84, 16.84, 20.0], -13)
    assert r["mm"][0] == pytest.approx(87 / 16.84) and r["mm"][3] == 0.0 and np.isnan(r["mm"][2])
    assert r["clipped"].tolist() == [False, True, False, False] and r["n_clipped"] == 1
    r2 = corrected_mm(np.array([100.0, 50.0]), 16.84, 0)
    assert r2["mm"].tolist() == pytest.approx([100 / 16.84, 50 / 16.84]) and r2["n_clipped"] == 0


def test_offset_helpers():
    assert offset_px_from_config({"measurement": {"offset_px": -13}}) == -13.0
    assert offset_px_from_config({"measurement": {}}) == 0.0 and offset_px_from_config({}) == 0.0
    assert offset_mm_at(16.84, -13) == pytest.approx(-0.772, abs=1e-3)
