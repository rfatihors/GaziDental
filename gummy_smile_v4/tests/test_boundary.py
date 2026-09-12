import numpy as np
import pytest

from gsv4.eval.boundary import boundary_iou, boundary_report, edge_distance_errors, mask_iou
from tests.synth_masks import rect_band


def test_identical_masks_are_perfect():
    g = rect_band((300, 900), 100, 800, 100, 40)
    r = boundary_report(g, g)
    assert r["mask_iou"] == 1.0 and r["boundary_iou"] == 1.0
    assert r["top_edge_mae_px"] == 0 and r["bottom_edge_mae_px"] == 0 and r["columns_missed_frac"] == 0


def test_shifted_bottom_edge_is_measured_separately():
    gt = rect_band((300, 900), 100, 800, 100, 40)
    pred = rect_band((300, 900), 100, 800, 100, 46)   # bottom edge 6 px lower, top identical
    e = edge_distance_errors(pred, gt)
    assert e["top_edge_mae_px"] == 0
    assert e["bottom_edge_mae_px"] == 6 and e["bottom_edge_bias_px"] == 6
    assert e["thickness_mae_px"] == 6
    assert 0 < boundary_iou(pred, gt) < 1 and 0.8 < mask_iou(pred, gt) < 0.9


def test_missed_and_spurious_columns():
    gt = rect_band((300, 900), 100, 800, 100, 40)
    pred = rect_band((300, 900), 300, 850, 100, 40)
    e = edge_distance_errors(pred, gt)
    assert e["columns_missed_frac"] == pytest.approx(200 / 700)
    assert e["columns_spurious_frac"] == pytest.approx(50 / 550)
    empty = np.zeros((300, 900), dtype=bool)
    e2 = edge_distance_errors(empty, gt)
    assert e2["columns_missed_frac"] == 1.0 and np.isnan(e2["top_edge_mae_px"])
