"""Stage 6 table logic on synthetic per-image and boundary tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsv4.eval.prediction import (
    agreement_metrics, boundary_set_summary, boundary_sets_table, check_oof, error_decomposition, fallback_rate,
    label_confusion, mask_provenance, measurement_table, presence_categories, seg_error_vs_boundary, tooth_long,
    tooth_table, weighted_kappa,
)


def _oof(n=6, source="yolo:masks.data"):
    return pd.DataFrame({"image": [f"IMG_{i}" for i in range(n)], "mask_source": [source] * n, "fold": [i % 5 for i in range(n)],
                         "n_gingiva": [1] * (n - 1) + [0]})


def _ref(n=6):
    return pd.DataFrame({"image": [f"IMG_{i}" for i in range(n)], "uid": [f"high/IMG_{i}" for i in range(n)], "cv_fold": [i % 5 for i in range(n)]})


def test_check_oof_passes_on_a_consistent_table(tmp_path):
    oof, ref = _oof(), _ref()
    for i in oof["image"]:
        (tmp_path / f"{i}_gingiva.png").write_bytes(b"x")
    c = check_oof(oof, ref, tmp_path)
    assert c["ok"] and c["problems"] == [] and c["n_rows"] == 6 and c["mask_source"] == {"yolo:masks.data": 6}
    assert c["empty_gingiva_prediction"] == ["IMG_5"] and c["folds"] == {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}


def test_check_oof_accepts_every_known_predictor_mask_source(tmp_path):
    for source in ("yolo:masks.data", "yolo:masks.xy", "rfdetr:masks"):
        oof, ref = _oof(source=source), _ref()
        for i in oof["image"]:
            (tmp_path / f"{i}_gingiva.png").write_bytes(b"x")
        c = check_oof(oof, ref, tmp_path)
        assert c["ok"] and c["mask_source"] == {source: 6}
    # mixed sources are fine as long as every one of them is known
    mixed = _oof(); mixed.loc[0, "mask_source"] = "rfdetr:masks"
    assert check_oof(mixed, _ref(), tmp_path)["ok"]


def test_check_oof_stops_on_an_unexpected_mask_source(tmp_path):
    oof, ref = _oof(source="rfdetr:masks"), _ref()
    for i in oof["image"]:
        (tmp_path / f"{i}_gingiva.png").write_bytes(b"x")
    oof.loc[0, "mask_source"] = "rfdetr:masks:none"      # no instance predicted
    oof.loc[1, "mask_source"] = "png"                    # not a predictor output at all
    c = check_oof(oof, ref, tmp_path)
    assert not c["ok"]
    problem = next(p for p in c["problems"] if "mask_source" in p)
    listed = problem.split(": ", 1)[1]          # only the offending sources, not the accepted ones
    assert "'rfdetr:masks:none': 1" in listed and "'png': 1" in listed and "'rfdetr:masks':" not in listed


def test_check_oof_reports_every_problem(tmp_path):
    oof = _oof(); oof.loc[0, "mask_source"] = "detections"; oof.loc[1, "fold"] = 4
    oof = pd.concat([oof, oof.iloc[[2]]], ignore_index=True)
    ref = _ref(7)
    c = check_oof(oof, ref, tmp_path)
    assert not c["ok"]
    joined = " | ".join(c["problems"])
    assert "7 OOF rows but 7 reference images" not in joined and "duplicated OOF images: ['IMG_2']" in joined
    assert "without OOF prediction: ['IMG_6']" in joined and "'detections': 1" in joined and "PNG missing for 7" in joined and "fold of 1 images" in joined


def test_agreement_metrics_recovers_bias_and_labels():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 7.5, 80)
    pred = ref + 0.2 + rng.normal(0, 0.1, 80)
    pred[3] = np.nan
    m = agreement_metrics(pred, ref, n_boot=200)
    assert m["n"] == 79 and m["n_segmentation_failure"] == 1
    assert m["ba_bias"] == pytest.approx(0.2, abs=0.05) and m["mae"] < 0.3 and m["icc2_1"] > 0.95
    assert 0 <= m["threshold_agreement"] <= 1 and m["threshold_kappa_linear_ci_low"] <= m["threshold_kappa_linear"] <= m["threshold_kappa_linear_ci_high"]
    assert m["within_1_mm"] == 1.0
    assert agreement_metrics([1.0, np.nan], [1.0, 2.0]) == {"n": 1, "n_segmentation_failure": 1}


def test_weighted_kappa_and_confusion_use_the_combined_label_order():
    ref = ["E1", "E1-E2", "E2-E3", "E3", "E1"]
    assert weighted_kappa(ref, ref) == pytest.approx(1.0)
    assert np.isnan(weighted_kappa(["E1", "E1"], ["E1", "E1"]))
    ct = label_confusion(ref, ["E1", "E1", "E2-E3", "E3", "E1-E2"])
    assert list(ct.index) == ["E1", "E1-E2", "E2-E3", "E3"] and ct.loc["E1", "E1"] == 1 and ct.loc["E1-E2", "E1"] == 1


def test_measurement_table_one_row_per_set_small_sets_give_n_only():
    rng = np.random.default_rng(1)
    per = pd.DataFrame({"ref_mm": rng.uniform(1, 7, 30)}); per["selected_mm"] = per["ref_mm"] + rng.normal(0, 0.3, 30)
    t = measurement_table(per, {"all": pd.Series(True, index=per.index), "few": pd.Series([True] * 3 + [False] * 27, index=per.index)}, n_boot=50)
    assert list(t["set"]) == ["all", "few"] and t.loc[0, "n"] == 30 and t.loc[1, "n"] == 3 and pd.isna(t.loc[1, "mae"])


def test_error_decomposition_splits_total_into_seg_and_meas():
    rng = np.random.default_rng(2)
    ref = rng.uniform(1, 7, 100)
    gt = ref + rng.normal(0.1, 0.3, 100)          # geometry error
    pred = gt + rng.normal(0.5, 0.2, 100)         # segmentation error
    per = pd.DataFrame({"ref_mm": ref, "gt_mm": gt, "selected_mm": pred})
    per.loc[0, "selected_mm"] = np.nan
    d = error_decomposition(per)
    s = d["summary"]
    assert s["n"] == 99
    assert s["e_seg_bias"] == pytest.approx(0.5, abs=0.08) and s["e_meas_bias"] == pytest.approx(0.1, abs=0.1)
    assert s["e_total_bias"] == pytest.approx(s["e_seg_bias"] + s["e_meas_bias"], abs=1e-9)
    assert s["share_seg"] + s["share_meas"] + s["share_cov"] == pytest.approx(1.0, abs=1e-9)
    pi = d["per_image"]
    assert np.allclose((pi["e_seg"] + pi["e_meas"]).dropna(), pi["e_total"].dropna())


def test_seg_error_vs_boundary_converts_px_columns_to_mm():
    idx = [f"u{i}" for i in range(40)]
    rng = np.random.default_rng(3)
    bias_px = rng.normal(10, 4, 40)
    dec = pd.DataFrame({"e_seg": bias_px / 17.0 + rng.normal(0, 0.02, 40)}, index=idx)
    b = pd.DataFrame({"gingiva_mask_iou": rng.uniform(0.6, 0.9, 40), "gingiva_top_edge_bias_px": rng.normal(0, 2, 40),
                      "gingiva_bottom_edge_bias_px": bias_px, "gingiva_thickness_mae_px": np.abs(bias_px),
                      "gingiva_columns_missed_frac": rng.uniform(0, 0.2, 40), "gingiva_columns_spurious_frac": rng.uniform(0, 0.2, 40)}, index=idx)
    t = seg_error_vs_boundary(dec, b, 17.0).set_index("metric")
    assert t.loc["gingiva_bottom_edge_bias_mm", "slope"] == pytest.approx(1.0, abs=0.1) and t.loc["gingiva_bottom_edge_bias_mm", "r"] > 0.95
    assert "gingiva_mask_iou" in t.index and t.loc["gingiva_mask_iou", "n"] == 40


def test_tooth_long_and_table():
    per = pd.DataFrame({"uid": ["a", "b"], "alignment_uncertain": [False, True],
                        **{f"selected_region_{i}_mm": [1.0 + i, 2.0 + i] for i in range(1, 7)},
                        **{f"ref_mm_{i}": [1.0 + i - 0.5, np.nan if i == 6 else 2.0 + i] for i in range(1, 7)}})
    long = tooth_long(per, [13, 12, 11, 21, 22, 23])
    assert len(long) == 11 and set(long["tooth"]) == {13, 12, 11, 21, 22, 23}
    assert long[long.patient == "a"]["diff"].unique().tolist() == [0.5] and long[long.patient == "b"]["alignment_uncertain"].all()
    t = tooth_table(long)
    assert len(t) == 6 and t.loc[t.tooth_index == 6, "n"].item() == 1 and t.loc[t.tooth_index == 1, "bias"].item() == pytest.approx(0.25)


def test_fallback_rate_rule():
    per = pd.DataFrame({"qc_flags": ["", "zenith_detection_failed", "zenith_detection_failed,region_zero", np.nan]})
    r = fallback_rate(per)
    assert r == {"n": 4, "n_fallback": 2, "frac": 0.5, "reevaluate": True, "threshold": 0.30}
    assert not fallback_rate(pd.DataFrame({"qc_flags": [""] * 9 + ["zenith_detection_failed"]}))["reevaluate"]


def _boundary():
    return pd.DataFrame({
        "uid": ["a", "b", "c", "d", "e"],
        "gingiva_n_columns_gt": [900, 800, 0, 0, 50], "gingiva_n_columns_pred": [850, 0, 0, 30, 60],
        "gingiva_mask_iou": [0.8, 0.0, np.nan, 0.0, 0.4], "gingiva_boundary_iou": [0.3, 0.0, np.nan, 0.0, 0.2],
        "gingiva_top_edge_mae_px": [4.0, np.nan, np.nan, np.nan, 6.0], "gingiva_top_edge_bias_px": [-1.0, np.nan, np.nan, np.nan, 1.0],
        "gingiva_bottom_edge_mae_px": [10.0, np.nan, np.nan, np.nan, 12.0], "gingiva_bottom_edge_bias_px": [10.0, np.nan, np.nan, np.nan, 8.0],
        "gingiva_thickness_mae_px": [11.0, np.nan, np.nan, np.nan, 12.0],
        "gingiva_columns_missed_frac": [0.05, 1.0, 0.0, 0.0, 0.1], "gingiva_columns_spurious_frac": [0.0, 0.0, 0.0, 1.0, 0.2],
        "lip_mask_iou": [0.8, 0.7, 0.9, 0.6, np.nan], "lip_boundary_iou": [0.3, 0.2, 0.4, 0.1, np.nan],
    })


def test_presence_categories_and_set_summary():
    b = _boundary()
    assert presence_categories(b).tolist() == ["both", "gt_only", "neither", "pred_only", "both"]
    s = boundary_set_summary(b, "x", 17.0)
    assert (s["n"], s["n_gt_gingiva"], s["n_both"], s["n_missed"], s["n_spurious"], s["n_neither"]) == (5, 3, 2, 1, 1, 1)
    assert s["gingiva_mask_iou_n"] == 4 and s["gingiva_mask_iou_mean"] == pytest.approx(0.3)
    assert s["gingiva_top_edge_mae_px_n"] == 2 and s["gingiva_top_edge_mae_px_mean"] == pytest.approx(5.0)
    assert s["gingiva_bottom_edge_bias_mm_mean"] == pytest.approx(9.0 / 17.0) and s["lip_mask_iou_n"] == 4
    t = boundary_sets_table({"x": b, "y": b.iloc[:2]}, 17.0)
    assert list(t["set"]) == ["x", "y"] and t.loc[1, "n"] == 2


CFG = {"yolo": {"model": "yolo11x-seg.pt", "imgsz": 640}}


def _yolo_pred(n=6, folds=True):
    df = _oof(n)
    df["weights"] = [f"runs/fold{i % 5}/weights/best.pt" for i in range(n)]
    if not folds:
        df = df.drop(columns=["fold"])
        df["weights"] = "runs/final/weights/best.pt"
    return df


def _rfdetr_pred(n=6, folds=True):
    df = _oof(n, source="rfdetr:masks")
    df["model"] = "rfdetr-seg-large"
    df["seed"] = 42
    if not folds:
        df = df.drop(columns=["fold"])
    return df


def test_mask_provenance_names_the_model_behind_each_mask_directory():
    yolo = mask_provenance(_yolo_pred(), Path("outputs/05_predictions/oof"), CFG, "oof", single_model=False)
    assert yolo["family"] == "yolo" and yolo["label"] == "yolo11x-seg @640, 5 fold models" and yolo["n_models"] == 5
    assert "Ultralytics" in yolo["evaluator"] and yolo["short"] == "oof"
    rf = mask_provenance(_rfdetr_pred(folds=False), Path("outputs/05_predictions/test_rfdetr"), CFG, "test", single_model=True)
    assert rf["family"] == "rfdetr" and rf["label"] == "RF-DETR-Seg Large @624, seed 42" and rf["n_models"] == 1
    assert "COCO" in rf["evaluator"]        # never Ultralytics for RF-DETR masks
    assert rf["evaluator"] != yolo["evaluator"]


def test_mask_provenance_stops_when_the_source_is_ambiguous():
    mixed = _rfdetr_pred()
    mixed.loc[0, "mask_source"] = "yolo:masks.data"
    with pytest.raises(SystemExit, match="mixes predictor families"):
        mask_provenance(mixed, Path("d"), CFG, "oof", single_model=False)
    with pytest.raises(SystemExit, match="no mask_source column"):
        mask_provenance(_rfdetr_pred().drop(columns=["mask_source"]), Path("d"), CFG, "oof", single_model=False)
    with pytest.raises(SystemExit, match="without a `model` column"):
        mask_provenance(_rfdetr_pred().drop(columns=["model"]), Path("d"), CFG, "oof", single_model=False)
    # a set that must come from one final model may not carry five fold models
    with pytest.raises(SystemExit, match="expected one model"):
        mask_provenance(_yolo_pred(), Path("d"), CFG, "final-model test masks", single_model=True)
