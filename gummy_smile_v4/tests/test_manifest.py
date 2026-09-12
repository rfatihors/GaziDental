"""Dataset cleaning rules (spec §4.8 + task decisions of 12 Sep 2026) on synthetic tables."""
import math

import pandas as pd
import pytest

from gsv4.dataset.manifest import build_manifest

DATASET_CFG = {
    "keep_unmeasured_high_in_train": False,
    "row_ambiguous_exclude": ["IMG_7366"],
}


def _inventory():
    rows = [
        # image, group, split, key, w, h
        ("IMG_1000_jpg", "high", "train", "img1000", 2698, 1799),   # measured, no pair
        ("IMG_1001_jpg", "high", "test", "img1001", 2698, 1799),    # measured, pair with 1002 (unmeasured)
        ("IMG_1002-_jpg", "high", "train", "img1002~", 2698, 1799), # unmeasured (66 list)
        ("IMG_1003_jpg", "high", "train", "img1003", 2698, 1799),   # measured pair, both measured; 1003 in expert set
        ("IMG_1004-_jpg", "high", "train", "img1004~", 2698, 1799), # measured pair, not in expert set
        ("IMG_7366_jpg", "high", "valid", "img7366", 2698, 1799),   # row ambiguous
        ("IMG_2000_jpg", "low", "train", "img2000", 2698, 1799),    # pair with 2001 (normal), 2000 has age
        ("IMG_2001_jpg", "normal", "train", "img2001", 2698, 1799),
        ("IMG_2002_jpg", "normal", "train", "img2002", 2698, 1799), # pair neither has demo -> image_a kept
        ("IMG_2003_jpg", "normal", "valid", "img2003", 2698, 1799),
        ("IMG_2004_jpg", "normal", "train", "img2004", 2698, 1799),
    ]
    inv = pd.DataFrame(rows, columns=["image", "group", "split", "key", "width", "height"])
    inv["base"] = inv["key"].str.replace("~", "", regex=False)
    inv["dot"] = inv["key"].str.endswith("~")
    inv["age_prefix"] = None
    inv["frame_ok"] = True
    inv["n_gingiva"] = 1
    inv["n_lip"] = 1
    inv["file_name"] = inv["image"] + ".rf.x.jpg"
    inv["path"] = "/x/" + inv["file_name"]
    return inv


def _high_matches():
    # image -> (match_kind, excel_row, mean_mm, label, age, sex)
    rows = [
        ("IMG_1000_jpg", "exact", 0, 2.5, "E1", 30, "F"),
        ("IMG_1001_jpg", "exact", 1, 4.5, "E2-E3", None, None),
        ("IMG_1002-_jpg", "unmatched", None, math.nan, None, None, None),
        ("IMG_1003_jpg", "exact", 2, 1.2, "E1", None, "M"),
        ("IMG_1004-_jpg", "dash_base_fallback", 3, 1.4, "E1", 22, "F"),
        ("IMG_7366_jpg", "row_ambiguous", None, math.nan, None, None, None),
    ]
    return pd.DataFrame(rows, columns=["image", "match_kind", "excel_row", "reference_mean_mm", "reference_label", "age", "sex"])


def _demo():
    low = pd.DataFrame({"key": ["img2000"], "age": [40], "sex": ["F"]})
    normal = pd.DataFrame({"key": ["img2004"], "age": [33], "sex": ["M"]})
    return low, normal


def _pairs():
    return pd.DataFrame(
        [
            ("IMG_1001_jpg", "IMG_1002-_jpg"),
            ("IMG_1004-_jpg", "IMG_1003_jpg"),   # image_b is the expert-set one
            ("IMG_2001_jpg", "IMG_2000_jpg"),    # image_b has demographics
            ("IMG_2002_jpg", "IMG_2003_jpg"),    # neither -> image_a
        ],
        columns=["image_a", "image_b"],
    )


def _unmeasured():
    return pd.DataFrame({"image": ["IMG_1002-_jpg"]})


def _expert_set():
    return pd.DataFrame({"image": ["IMG_1000_jpg", "IMG_1001_jpg", "IMG_1003_jpg"], "mean_mm": [2.5, 4.5, 1.2], "tablo_sinifi": ["E1", "E2-E3", "E1"]})


def test_build_manifest_drop_mode():
    low, normal = _demo()
    m, summary = build_manifest(_inventory(), _high_matches(), low, normal, _pairs(), _unmeasured(), _expert_set(), DATASET_CFG)
    m = m.set_index("image")
    assert m.loc["IMG_1000_jpg", "uid"] == "high/IMG_1000_jpg"
    assert bool(m.loc["IMG_1000_jpg", "keep"]) is True
    assert m.loc["IMG_1002-_jpg", "keep"] == False and m.loc["IMG_1002-_jpg", "drop_reason"] == "no_reference_measurement"
    assert m.loc["IMG_1002-_jpg", "patient_id"] == m.loc["IMG_1001_jpg", "patient_id"] == "high/IMG_1001_jpg"
    assert m.loc["IMG_7366_jpg", "drop_reason"] == "row_ambiguous"
    # both measured: the expert-set image wins even when it is image_b
    assert bool(m.loc["IMG_1003_jpg", "keep"]) is True
    assert m.loc["IMG_1004-_jpg", "drop_reason"] == "duplicate_of:IMG_1003_jpg"
    assert m.loc["IMG_1004-_jpg", "patient_id"] == "high/IMG_1003_jpg"
    # non-high: demographics decide, across groups
    assert bool(m.loc["IMG_2000_jpg", "keep"]) is True
    assert m.loc["IMG_2001_jpg", "drop_reason"] == "duplicate_of:IMG_2000_jpg"
    # neither has demographics -> image_a kept
    assert bool(m.loc["IMG_2002_jpg", "keep"]) is True and m.loc["IMG_2003_jpg", "drop_reason"] == "duplicate_of:IMG_2002_jpg"
    # demographics propagated
    assert m.loc["IMG_2004_jpg", "age"] == 33 and m.loc["IMG_2004_jpg", "sex"] == "M"
    assert m.loc["IMG_1000_jpg", "age"] == 30
    assert bool(m.loc["IMG_1000_jpg", "has_reference_measurement"]) is True
    assert bool(m.loc["IMG_2004_jpg", "has_reference_measurement"]) is False
    assert m.loc["IMG_1000_jpg", "reference_mean_mm"] == pytest.approx(2.5)
    assert summary["kept"]["high"] == 3 and summary["kept"]["low"] == 1 and summary["kept"]["normal"] == 2
    assert summary["expert_set_mismatch"] == []
    assert m.loc["IMG_1000_jpg", "split_constraint"] == "" or pd.isna(m.loc["IMG_1000_jpg", "split_constraint"])


def test_build_manifest_train_only_mode():
    low, normal = _demo()
    cfg = dict(DATASET_CFG, keep_unmeasured_high_in_train=True)
    m, summary = build_manifest(_inventory(), _high_matches(), low, normal, _pairs(), _unmeasured(), _expert_set(), cfg)
    m = m.set_index("image")
    # 1002 is kept, train-only, twin of 1001, no reference, not in any analysis
    assert bool(m.loc["IMG_1002-_jpg", "keep"]) is True
    assert m.loc["IMG_1002-_jpg", "split_constraint"] == "train_only"
    assert m.loc["IMG_1002-_jpg", "twin_of"] == "IMG_1001_jpg"
    assert bool(m.loc["IMG_1002-_jpg", "has_reference_measurement"]) is False
    assert summary["train_only"]["high"] == 1
