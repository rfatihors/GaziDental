"""Table 1 threshold logic — the single place where E1..E4 bounds are defined.

Bounds (image-level mean gingival display, mm):
    E1 : 0 < x < 4      (0 mm itself is "no visible gingiva", not a class)
    E2 : 3 <= x <= 6
    E3 : 4 <= x <= 8
    E4 : x > 8
Overlapping bands are reported together (``E1-E2``, ``E2-E3``) — the same label
format the clinicians used in the reference Excel sheet.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

DEFAULT_THRESHOLDS: Dict[str, Dict[str, object]] = {
    "E1": {"min_mm": 0.0, "min_inclusive": False, "max_mm": 4.0, "max_inclusive": False},
    "E2": {"min_mm": 3.0, "min_inclusive": True, "max_mm": 6.0, "max_inclusive": True},
    "E3": {"min_mm": 4.0, "min_inclusive": True, "max_mm": 8.0, "max_inclusive": True},
    "E4": {"min_mm": 8.0, "min_inclusive": False, "max_mm": None, "max_inclusive": False},
}

CLASS_ORDER = ["E1", "E2", "E3", "E4"]
NO_VISIBLE_GINGIVA = "NO_VISIBLE_GINGIVA"
UNCLASSIFIED = "UNCLASSIFIED"


def _is_missing(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _in_band(value: float, band: Dict[str, object]) -> bool:
    lo, hi = band.get("min_mm"), band.get("max_mm")
    if lo is not None:
        ok = value >= float(lo) if band.get("min_inclusive", True) else value > float(lo)
        if not ok:
            return False
    if hi is not None:
        ok = value <= float(hi) if band.get("max_inclusive", True) else value < float(hi)
        if not ok:
            return False
    return True


def matching_classes(value_mm: Optional[float], thresholds: Optional[Dict[str, Dict[str, object]]] = None) -> List[str]:
    """All classes whose band contains ``value_mm`` (empty for missing or <= 0)."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if _is_missing(value_mm) or value_mm <= 0:
        return []
    return [c for c in CLASS_ORDER if c in thresholds and _in_band(float(value_mm), thresholds[c])]


def label_for_mm(value_mm: Optional[float], thresholds: Optional[Dict[str, Dict[str, object]]] = None) -> str:
    """Combined label: ``E1``, ``E1-E2``, ``E2-E3`` ... or the two special states."""
    if _is_missing(value_mm):
        return UNCLASSIFIED
    if value_mm <= 0:
        return NO_VISIBLE_GINGIVA
    classes = matching_classes(value_mm, thresholds)
    return "-".join(classes) if classes else UNCLASSIFIED
