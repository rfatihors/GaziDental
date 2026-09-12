"""Table 1 threshold logic (single source for parser checks and the rule engine)."""
import math

import pytest

from gsv4.rules.thresholds import DEFAULT_THRESHOLDS, label_for_mm, matching_classes


@pytest.mark.parametrize(
    "mm,label",
    [
        (float("nan"), "UNCLASSIFIED"),
        (None, "UNCLASSIFIED"),
        (0.0, "NO_VISIBLE_GINGIVA"),
        (-0.3, "NO_VISIBLE_GINGIVA"),
        (0.5, "E1"),
        (2.0, "E1"),
        (2.999, "E1"),
        (3.0, "E1-E2"),
        (3.5, "E1-E2"),
        (3.999, "E1-E2"),
        (4.0, "E2-E3"),
        (5.0, "E2-E3"),
        (6.0, "E2-E3"),
        (6.001, "E3"),
        (8.0, "E3"),
        (8.001, "E4"),
        (14.1, "E4"),
    ],
)
def test_label_for_mm(mm, label):
    assert label_for_mm(mm) == label


def test_matching_classes_lists_all_candidates():
    assert matching_classes(3.5) == ["E1", "E2"]
    assert matching_classes(5.0) == ["E2", "E3"]
    assert matching_classes(7.0) == ["E3"]
    assert matching_classes(9.0) == ["E4"]
    assert matching_classes(0.0) == []
    assert matching_classes(math.nan) == []


def test_thresholds_from_config_dict_match_defaults():
    cfg = {
        "E1": {"min_mm": 0.0, "min_inclusive": False, "max_mm": 4.0, "max_inclusive": False},
        "E2": {"min_mm": 3.0, "min_inclusive": True, "max_mm": 6.0, "max_inclusive": True},
        "E3": {"min_mm": 4.0, "min_inclusive": True, "max_mm": 8.0, "max_inclusive": True},
        "E4": {"min_mm": 8.0, "min_inclusive": False, "max_mm": None, "max_inclusive": False},
    }
    assert label_for_mm(4.0, cfg) == "E2-E3"
    assert DEFAULT_THRESHOLDS["E1"]["max_mm"] == 4.0
