import json

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
    assert T.learning_curve(tmp_path)[1]["status"] == "pending"
    assert T.expert_agreement(tmp_path)[1]["status"] == "pending"
    (tmp_path / "expert_summary.md").write_text("Data: SYNTHETIC forms")
    assert T.expert_agreement(tmp_path)[1]["status"] == "pending"
    assert T.measurement_accuracy(tmp_path, None)[1]["status"] == "pending"
