from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EtiologyResult:
    etiology_class: str
    treatment_class: str
    etiology_candidates: List[str]
    treatment_recommendations: List[str]
    ambiguous: bool
    notes: str


RULES: Dict[str, Dict[str, object]] = {
    "E1": {
        "min_mm": None,
        "max_mm": 4.0,
        "etiology": [
            "Gecikmiş pasif erüpsiyon",
            "Gingival hipertrofi",
            "Kalın gingival biyotip",
        ],
        "treatment_class": "T1",
        "treatment": ["Gingivektomi", "Gingivoplasti"],
    },
    "E2": {
        "min_mm": 3.0,
        "max_mm": 6.0,
        "etiology": [
            "Hiperaktif (hipermobil) üst dudak",
            "Kısa üst dudak (<20 mm)",
        ],
        "treatment_class": "T2",
        "treatment": ["Lip repositioning", "Botox"],
    },
    "E3": {
        "min_mm": 4.0,
        "max_mm": 8.0,
        "etiology": ["Dentoalveolar ekstrüzyon", "Derin kapanış"],
        "treatment_class": "T3",
        "treatment": ["Ortodontik tedavi", "Osteotomi"],
    },
    "E4": {
        "min_mm": 8.0,
        "max_mm": None,
        "etiology": ["Artmış vertikal maxiller yükseklik"],
        "treatment_class": "T4",
        "treatment": ["LeFort I osteotomi"],
    },
}


def _matches(value_mm: float, rule: Dict[str, object]) -> bool:
    min_mm = rule["min_mm"]
    max_mm = rule["max_mm"]
    lower_ok = True if min_mm is None else value_mm >= float(min_mm)
    upper_ok = True if max_mm is None else value_mm <= float(max_mm)
    return lower_ok and upper_ok


def _resolve_overlap(
    value_mm: float,
    metadata: Optional[str],
    ambiguous_policy: Dict[str, object],
) -> Tuple[str, bool, str]:
    default_choice = ambiguous_policy.get("default", "E2")
    use_metadata = bool(ambiguous_policy.get("use_metadata", True))
    metadata_map = ambiguous_policy.get("metadata_map", {})
    if use_metadata and metadata:
        mapped = metadata_map.get(str(metadata).lower())
        if mapped in {"E2", "E3"}:
            return mapped, False, f"Ambiguity resolved using metadata={metadata}."
    return default_choice, True, "Ambiguity unresolved; reporting both E2/E3."


def assign_etiology(
    gum_visibility_value: Optional[float],
    metadata: Optional[str],
    ambiguous_policy: Dict[str, object],
    value_unit: str = "mm",
) -> EtiologyResult:
    if gum_visibility_value is None or (isinstance(gum_visibility_value, float) and np.isnan(gum_visibility_value)):
        return EtiologyResult(
            etiology_class="E1",
            treatment_class="T1",
            etiology_candidates=RULES["E1"]["etiology"],
            treatment_recommendations=RULES["E1"]["treatment"],
            ambiguous=True,
            notes="Gum visibility missing; defaulting to E1/T1.",
        )

    notes = ""
    if value_unit != "mm":
        notes = f"Value interpreted in {value_unit}; thresholds are defined in mm."

    if _matches(gum_visibility_value, RULES["E1"]):
        return _build_result("E1", False, notes)
    if _matches(gum_visibility_value, RULES["E4"]):
        return _build_result("E4", False, notes)

    in_e2 = _matches(gum_visibility_value, RULES["E2"])
    in_e3 = _matches(gum_visibility_value, RULES["E3"])
    if in_e2 and in_e3:
        chosen, ambiguous, overlap_note = _resolve_overlap(gum_visibility_value, metadata, ambiguous_policy)
        combined_notes = "; ".join([note for note in [notes, overlap_note] if note])
        return _build_result(
            chosen,
            ambiguous,
            combined_notes,
            etiology_candidates=RULES["E2"]["etiology"] + RULES["E3"]["etiology"],
            treatment_recommendations=RULES["E2"]["treatment"] + RULES["E3"]["treatment"],
        )
    if in_e2:
        return _build_result("E2", False, notes)
    if in_e3:
        return _build_result("E3", False, notes)

    return _build_result("E1", True, f"No rule matched; defaulting to E1. {notes}".strip())


def _build_result(
    code: str,
    ambiguous: bool,
    notes: str,
    etiology_candidates: Optional[List[str]] = None,
    treatment_recommendations: Optional[List[str]] = None,
) -> EtiologyResult:
    rule = RULES[code]
    return EtiologyResult(
        etiology_class=code,
        treatment_class=rule["treatment_class"],
        etiology_candidates=etiology_candidates or rule["etiology"],
        treatment_recommendations=treatment_recommendations or rule["treatment"],
        ambiguous=ambiguous,
        notes=notes,
    )
