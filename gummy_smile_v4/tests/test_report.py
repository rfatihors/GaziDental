import json
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
