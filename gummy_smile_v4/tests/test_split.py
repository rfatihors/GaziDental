"""Patient-level stratified splits with a fixed test set and CV folds for measured high images."""
import pandas as pd

from gsv4.dataset.split import make_splits

DATASET_CFG = {
    "split_ratios": {
        "high": {"train": 0.60, "valid": 0.20, "test": 0.20},
        "low": {"train": 0.70, "valid": 0.15, "test": 0.15},
        "normal": {"train": 0.70, "valid": 0.15, "test": 0.15},
    },
    "cv_folds": 5,
}


def _manifest():
    rows = []
    labels = ["E1"] * 60 + ["E1-E2"] * 20 + ["E2-E3"] * 15 + ["E3"] * 5
    for i, lab in enumerate(labels):
        rows.append((f"H{i}", "high", True, "", f"h{i}", True, lab, ""))
    for i in range(200):
        rows.append((f"L{i}", "low", True, "", f"l{i}", False, None, ""))
    for i in range(400):
        rows.append((f"N{i}", "normal", True, "", f"n{i}", False, None, ""))
    # dropped images must not appear in any split
    rows.append(("H_drop", "high", False, "no_reference_measurement", "h0", False, None, ""))
    # train-only twin of H1
    rows.append(("H_twin", "high", True, "", "h1", False, None, "train_only"))
    df = pd.DataFrame(rows, columns=["image", "group", "keep", "drop_reason", "patient_id", "has_reference_measurement", "reference_label", "split_constraint"])
    df["uid"] = df["group"] + "/" + df["image"]
    return df


def test_make_splits_sizes_and_disjointness():
    m = _manifest()
    s = make_splits(m, DATASET_CFG, seed=42)
    assign = {k.split("/", 1)[1]: v for k, v in s["images"].items()}
    assert "H_drop" not in assign
    high = {k: v for k, v in assign.items() if k.startswith("H") and k != "H_twin"}
    counts = pd.Series(high).value_counts()
    assert counts["test"] == 20 and counts["valid"] == 20 and counts["train"] == 60
    low = pd.Series({k: v for k, v in assign.items() if k.startswith("L")}).value_counts()
    assert low["test"] == 30 and low["valid"] == 30 and low["train"] == 140
    # stratified by reference label: every label appears in test
    test_labels = m.set_index("image").loc[[k for k, v in high.items() if v == "test"], "reference_label"]
    assert set(test_labels) == {"E1", "E1-E2", "E2-E3", "E3"}
    # train-only twin lands in train and shares patient with H1 which must also be in train
    assert assign["H_twin"] == "train"
    assert assign["H1"] == "train"
    assert s["train_only"] == ["high/H_twin"]


def test_make_splits_is_deterministic():
    m = _manifest()
    assert make_splits(m, DATASET_CFG, seed=42)["images"] == make_splits(m, DATASET_CFG, seed=42)["images"]
    assert make_splits(m, DATASET_CFG, seed=42)["images"] != make_splits(m, DATASET_CFG, seed=7)["images"]


def test_cv_folds_cover_measured_high_once():
    m = _manifest()
    s = make_splits(m, DATASET_CFG, seed=42)
    folds = {k.split("/", 1)[1]: v for k, v in s["cv_folds"]["assignments"].items()}
    measured = set(m[(m.keep) & (m.has_reference_measurement)]["image"])
    assert set(folds) == measured
    assert set(folds.values()) == {0, 1, 2, 3, 4}
    sizes = pd.Series(folds).value_counts()
    assert sizes.max() - sizes.min() <= 1
    # each fold has every label
    lab = m.set_index("image")["reference_label"]
    for f in range(5):
        assert set(lab[[k for k, v in folds.items() if v == f]]) == {"E1", "E1-E2", "E2-E3", "E3"}
    assert s["cv_folds"]["n_folds"] == 5
