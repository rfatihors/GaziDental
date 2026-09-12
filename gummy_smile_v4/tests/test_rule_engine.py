"""Rule engine (spec §3.5, test 10 + user-specified boundary points)."""
import math

import pytest

from gsv4.rules.rule_engine import classify


@pytest.mark.parametrize(
    "mm,cls,ambiguous",
    [
        (float("nan"), "UNCLASSIFIED", False),
        (None, "UNCLASSIFIED", False),
        (0.0, "NO_VISIBLE_GINGIVA", False),
        (-0.1, "NO_VISIBLE_GINGIVA", False),
        (0.5, "E1", False),
        (2.0, "E1", False),
        (3.0, "E1-E2", True),
        (3.5, "E1-E2", True),
        (4.0, "E2-E3", True),
        (5.0, "E2-E3", True),
        (6.0, "E2-E3", True),
        (7.0, "E3", False),
        (8.0, "E3", False),
        (8.1, "E4", False),
    ],
)
def test_classes(mm, cls, ambiguous):
    r = classify(mm)
    assert r.etiology_class == cls
    assert r.ambiguous is ambiguous


def test_candidates_and_treatments_are_combined_for_overlaps():
    r = classify(3.5)
    assert r.etiology_candidates == ["E1", "E2"]
    assert r.treatment_class == "T1-T2"
    assert any("ingivectomy" in t for t in r.treatment_alternatives)
    assert any("repositioning" in t for t in r.treatment_alternatives)
    assert "treatment_recommendations" not in r.to_dict()
    assert "treatment_alternatives" in r.to_dict()


def test_special_states_have_no_class_or_treatment():
    r = classify(0.0)
    assert r.etiology_candidates == [] and r.treatment_alternatives == [] and r.treatment_class is None
    assert "no visible gingiva" in r.notes
    u = classify(math.nan)
    assert u.etiology_candidates == [] and "missing" in u.notes.lower()


def test_metadata_never_changes_the_result():
    base = classify(5.0)
    for meta in ("high", "normal", "low", None):
        r = classify(5.0, metadata=meta)
        assert r.etiology_class == base.etiology_class == "E2-E3" and r.ambiguous
    r2 = classify(5.0, metadata="high", policy={"use_metadata": True})
    assert r2.etiology_class == "E2-E3" and r2.ambiguous
    assert "disabled" in r2.notes


def test_applicability_and_unit_fields():
    r = classify(2.0)
    d = r.to_dict()
    assert "high smile line" in d["applicability"]
    assert d["value_mm"] == 2.0
    r_px = classify(50.0, unit="px")
    assert r_px.etiology_class == "UNCLASSIFIED" and "px" in r_px.notes
