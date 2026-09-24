import json

import pytest
from pathlib import Path

import pandas as pd

from gsv4.report import figures as F
from gsv4.report import tables as T


def test_block_diagram_and_placeholder(tmp_path):
    st = F.block_diagram(tmp_path / "d.md", tmp_path / "d.png")
    assert st["status"] == "done" and (tmp_path / "d.png").exists() and "mermaid" in (tmp_path / "d.md").read_text()
    st2 = F.copy_or_placeholder(tmp_path / "missing.png", tmp_path / "out.png", "x", "y")
    assert st2["status"] == "pending" and (tmp_path / "out.png").exists()
    st3 = F.boundary_error_figure(tmp_path / "none.csv", tmp_path / "b.png")
    assert st3["status"] == "pending"


def test_dataset_counts_and_demographics():
    m = pd.DataFrame({
        "image": ["a", "b", "c", "d"], "group": ["high", "high", "low", "low"], "keep": [True, False, True, True],
        "drop_reason": ["", "duplicate_of:a", "", ""], "has_reference_measurement": [True, False, False, False],
        "split": ["train", "", "test", "valid"], "age": [30, None, None, 40], "sex": ["F", None, "M", None],
    })
    pairs = pd.DataFrame({"image_a": ["a"], "image_b": ["b"], "cross_split": ["False"], "involves_test": ["True"]})
    df, st = T.dataset_counts(m, {"seed": 42, "cv_folds": {"n_folds": 5}}, pairs)
    tot = df[df.group == "total"].iloc[0]
    assert tot["images_roboflow_export"] == 4 and tot["kept"] == 3 and tot["same_patient_duplicate"] == 1 and tot["test"] == 1
    assert st["same_patient_pairs_detected"] == 1 and st["pairs_involving_original_test"] == 1
    d, _ = T.demographics(m)
    assert d[d.group == "all"].iloc[0]["age_recorded"] == 2 and d[d.group == "all"].iloc[0]["female"] == 1


def test_pending_tables(tmp_path):
    assert T.segmentation_metrics(tmp_path)[1]["status"] == "pending"
    st = T.learning_curve(tmp_path, "scripts/build_rfdetr_learning_curve.py --out 09_final_rfdetr")[1]
    assert st["status"] == "pending" and "build_rfdetr_learning_curve.py" in st["needs"] and "learning_curve.csv" in st["needs"]
    assert T.expert_agreement(tmp_path)[1]["status"] == "pending"
    (tmp_path / "expert_summary.md").write_text("Data: SYNTHETIC forms")
    assert T.expert_agreement(tmp_path)[1]["status"] == "pending"
    assert T.measurement_accuracy(tmp_path, None)[1]["status"] == "pending"


def _rfdetr_curve(dir_: Path, values, verdict="not plateaued (report as limitation)"):
    """A curve as scripts/build_rfdetr_learning_curve.py writes it: no `done` column, metric named in the file."""
    pd.DataFrame({"fraction": [0.25, 0.5, 0.75, 1.0], "run": ["lc25", "lc50", "lc75", "final (reused)"],
                  "n_train_images": [211, 423, 635, 846], "split": "val", "evaluator": "rfdetr model.evaluate",
                  "metric": "val/segm_mAP_50", "source": "m.json", "val/segm_mAP_50": values}).to_csv(dir_ / "learning_curve.csv", index=False)
    (dir_ / "learning_curve.md").write_text(
        f"val/segm_mAP_50 gain 25\u219250 %: +0.0269; 75\u2192100 %: +0.0274 \u2192 **{verdict}** (rule: 75\u2192100 gain < 1/4 of the 25\u219250 gain).\n",
        encoding="utf-8")


def test_learning_curve_rfdetr_points_and_verdict(tmp_path, monkeypatch):
    """The RF-DETR curve is a complete table (it is written only once all four points exist), and the
    metric, the split and the verdict travel with it into the report so the rebuttal cannot invent them."""
    _rfdetr_curve(tmp_path, [0.7599, 0.7868, 0.7763, 0.8037])
    df, st = T.learning_curve(tmp_path)
    assert st["status"] == "done" and st["metric"] == "val/segm_mAP_50" and st["split"] == "val"
    assert st["verdict"].startswith("not plateaued") and len(df) == 4

    tab = tmp_path / "tab"; tab.mkdir()
    df.to_csv(tab / "learning_curve.csv", index=False)
    (tab / "learning_curve.md").write_text("\n".join(f"- {k}: {v}" for k, v in st.items() if k != "status"), encoding="utf-8")
    facts = T.learning_curve_facts(tab, tmp_path / "missing.md")
    assert facts["plateau"] is False and facts["points"] == "0.760, 0.787, 0.776 and 0.804"
    assert facts["sizes"] == "211, 423, 635 and 846" and "gain 25" in facts["rule"] and facts["rule"].endswith("+0.0274")


def test_learning_curve_yolo_curve_still_waits_for_its_runs(tmp_path):
    pd.DataFrame({"fraction": [0.25, 1.0], "run": ["lc25", "final"], "n_train_images": [211, 846],
                  "done": [True, False], "diseti_seg_map50": [0.40, 0.44]}).to_csv(tmp_path / "learning_curve.csv", index=False)
    assert T.learning_curve(tmp_path)[1]["status"] == "pending"


def _write_table(dir_: Path, name: str, df: pd.DataFrame, meta: dict):
    df.to_csv(dir_ / f"{name}.csv", index=False)
    (dir_ / f"{name}.md").write_text(f"# {name}\n\n" + T.md_table(df) + "\n\n"
                                     + "\n".join(f"- {k}: {v}" for k, v in meta.items()) + "\n", encoding="utf-8")


def test_segmentation_metrics_facts_reads_both_table_shapes(tmp_path):
    """Stage 7 writes one table per evaluator and they are not the same quantities: Ultralytics gives
    a row per class, RF-DETR's COCO evaluation gives metric/value rows pooled over both classes. The
    reader must say which one it has, so that no answer quotes a per-class mAP that does not exist."""
    per_class = pd.DataFrame({"settings": ["standard"] * 3, "class": ["diseti", "dudak", "all"],
                              "seg_map50": [0.41, 0.98, 0.70], "seg_map50_95": [0.16, 0.69, 0.43],
                              "box_map50": [0.57, 0.99, 0.78]})
    _write_table(tmp_path, "segmentation_metrics_test", per_class, {"source": "outputs/05_predictions/test_metrics.json"})
    F = T.segmentation_metrics_facts(tmp_path)
    assert F["per_class"] and F["format"] == "per_class" and F["settings_kind"] == "standard"
    assert F["classes"]["gingiva"]["seg_map50"] == 0.41 and F["pooled"]["seg_map50"] == 0.70
    assert "Ultralytics" in F["evaluator"]

    coco = pd.DataFrame({"metric": ["test/mAP_50", "test/mAP_50_95", "test/segm_mAP_50", "test/segm_mAP_50_95",
                                    "test/mAP_75", "test/mAR", "test/precision", "test/recall", "test/F1"],
                         "value": [0.84, 0.53, 0.81, 0.47, 0.57, 0.68, 0.85, 0.81, 0.83]})
    _write_table(tmp_path, "segmentation_metrics_test", coco,
                 {"source": "outputs/08_architecture/rfdetr_metrics_rfdetr-seg-large_s42.json",
                  "model": "RF-DETR-Seg Large @624, seed 42",
                  "evaluator": "RF-DETR's own COCO evaluation (pycocotools, iouType='segm')"})
    F = T.segmentation_metrics_facts(tmp_path)
    assert not F["per_class"] and F["format"] == "coco_pooled" and F["classes"] == {}
    assert F["pooled"]["seg_map50"] == 0.81 and F["pooled"]["box_map50"] == 0.84   # split prefix stripped, box != mask
    assert F["model"].startswith("RF-DETR") and "pooled over the two classes" in F["provenance"]


def test_segmentation_metrics_facts_refuses_a_pending_table(tmp_path):
    (tmp_path / "segmentation_metrics_test.md").write_text("# x\n\nPENDING — needs: Stage 6 for this model\n", encoding="utf-8")
    (tmp_path / "segmentation_metrics_test.csv").write_text("class,seg_map50\ndiseti,0.41\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        T.segmentation_metrics_facts(tmp_path)


def _estimator_csv(dir_: Path):
    """An estimator_comparison.csv as scripts/run_oracle.py writes it (three combinations)."""
    pd.DataFrame({
        "combo": ["A_p25", "C_p25", "C_max"], "regioning": ["A", "C", "C"], "estimator": ["p25", "p25", "max"],
        "anchored": [False, False, False], "px_per_mm_dev": [17.04, 16.84, 25.09],
        "mae_dev": [0.627, 0.557, 0.825], "mae_holdout": [0.537, 0.518, 0.845],
        "icc2_1_dev": [0.850, 0.873, 0.712], "icc2_1_holdout": [0.843, 0.858, 0.582],
        "ba_bias_dev": [0.225, 0.173, 0.364], "ba_bias_holdout": [0.187, 0.116, 0.429],
    }).to_csv(dir_ / "estimator_comparison.csv", index=False)


def test_estimator_sensitivity_is_a_sensitivity_table_not_a_selection(tmp_path):
    assert T.estimator_sensitivity(tmp_path, "C_p25", 16.84)[1]["status"] == "pending"
    _estimator_csv(tmp_path)
    (tmp_path / "oracle_summary.md").write_text(
        "selection on dev MAE (n = 87); best dev MAE = 0.557 mm; 1 combination(s) within 0.02 mm of it (C_p25); "
        "the simplest of those is chosen (A > B > C).\n\n"
        "**Fallback transparency.** Regioning C could not be established on 28 of 145 images (19 %, `zenith_detection_failed`); "
        "there the measurement silently uses the equal-split regions (A) and the value is identical to `A_p25`. "
        "On the dev images where C succeeded (n = 72), dev MAE is 0.512 mm for `C_p25` vs 0.597 mm for `A_p25` — ok.\n",
        encoding="utf-8")
    df, st = T.estimator_sensitivity(tmp_path, "C_p25", 16.8397)
    assert st["status"] == "done" and st["n_combinations"] == 3 and st["selected"] == "C_p25" and st["selected_px_per_mm"] == "16.84"
    # sorted by the quantity the selection actually used, and only the configured combination is marked
    assert list(df["combo"]) == ["C_p25", "A_p25", "C_max"]
    assert list(df["selected"]) == ["**yes**", "", ""]
    # every column the appendix promises is present
    for c in ("MAE dev, mm", "MAE holdout, mm", "ICC(2,1) dev", "ICC(2,1) holdout", "BA bias dev, mm", "BA bias holdout, mm"):
        assert c in df.columns
    assert "dev MAE" in st["selection_rule"] and not st["selection_rule"].endswith(".")
    assert st["regioning_fallback"] == "19 % (28 of 145 images)"
    assert "0.512" in st["fallback_check"] and "C_p25" in st["fallback_check"]
    assert "never re-selected" in st["note"]
    # the predicted-mask rate is only reported when that directory exists
    assert "regioning_fallback_predicted_masks" not in st
    pred = tmp_path / "pred_oracle"
    pred.mkdir()
    (pred / "oracle_summary.md").write_text("Regioning C could not be established on 27 of 145 images (19 %, x).\n", encoding="utf-8")
    st2 = T.estimator_sensitivity(tmp_path, "C_p25", 16.8397, predicted_oracle_dir=pred)[1]
    assert st2["regioning_fallback_predicted_masks"] == "19 % (27 of 145 images)"
    assert "30 %" in st2["fallback_reeval_threshold"] and "not reached" in st2["fallback_reeval_threshold"]

