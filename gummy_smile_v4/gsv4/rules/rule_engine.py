"""Single rule engine mapping image-level gingival display (mm) to the etiology /
treatment framework of Table 1 (spec §3.5).

Applicability: the framework is defined for high-smile-line photographs in which
gingiva is visible. A value <= 0 mm means no visible gingiva -> ``NO_VISIBLE_GINGIVA``
(no class, no treatment). Missing values -> ``UNCLASSIFIED``. Values inside two bands
are reported with the combined label (``E1-E2``, ``E2-E3``), all candidate classes and
all treatment *alternatives*; nothing is forced to one class, and the smile-line
metadata is never used (no causal link between smile line and etiology).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from gsv4.rules.thresholds import (
    CLASS_ORDER,
    DEFAULT_THRESHOLDS,
    NO_VISIBLE_GINGIVA,
    UNCLASSIFIED,
    matching_classes,
)

APPLICABILITY = "high smile line only (visible gingiva); 0 mm -> NO_VISIBLE_GINGIVA; not validated for E4"

ETIOLOGY: Dict[str, List[str]] = {
    "E1": ["Altered/delayed passive eruption", "Gingival hypertrophy", "Thick gingival phenotype"],
    "E2": ["Hypermobile (hyperactive) upper lip", "Short upper lip (< 20 mm)"],
    "E3": ["Dentoalveolar extrusion", "Deep bite"],
    "E4": ["Vertical maxillary excess"],
}
TREATMENT_CLASS: Dict[str, str] = {"E1": "T1", "E2": "T2", "E3": "T3", "E4": "T4"}
TREATMENT: Dict[str, List[str]] = {
    "T1": ["Gingivectomy", "Gingivoplasty"],
    "T2": ["Lip repositioning", "Botulinum toxin injection"],
    "T3": ["Orthodontic intrusion", "Segmental osteotomy"],
    "T4": ["Le Fort I osteotomy (impaction)"],
}


@dataclass
class RuleResult:
    value_mm: Optional[float]
    etiology_class: str
    etiology_candidates: List[str]
    etiology_descriptions: List[str]
    treatment_class: Optional[str]
    treatment_alternatives: List[str]
    ambiguous: bool
    notes: str
    applicability: str = APPLICABILITY
    thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify(
    value_mm: Optional[float],
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    policy: Optional[Dict[str, Any]] = None,
    metadata: Optional[str] = None,
    unit: str = "mm",
) -> RuleResult:
    """Classify one image-level value. ``metadata`` is accepted for API compatibility
    and deliberately ignored."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    policy = policy or {}
    notes: List[str] = []
    if policy.get("use_metadata"):
        notes.append("metadata-based disambiguation is disabled by design (no causal link between smile line and etiology)")
    if unit != "mm":
        return RuleResult(value_mm, UNCLASSIFIED, [], [], None, [], False,
                          f"value is in {unit}, thresholds are in mm; calibration (px_per_mm) missing", thresholds=thresholds)
    if value_mm is None or (isinstance(value_mm, float) and math.isnan(value_mm)):
        return RuleResult(None, UNCLASSIFIED, [], [], None, [], False,
                          "; ".join(["measurement missing (NaN)"] + notes), thresholds=thresholds)
    v = float(value_mm)
    if v <= 0:
        return RuleResult(v, NO_VISIBLE_GINGIVA, [], [], None, [], False,
                          "; ".join(["no visible gingiva; framework applies to high smile line only"] + notes), thresholds=thresholds)
    classes = matching_classes(v, thresholds)
    if not classes:
        return RuleResult(v, UNCLASSIFIED, [], [], None, [], False,
                          "; ".join(["no rule matched"] + notes), thresholds=thresholds)
    classes = [c for c in CLASS_ORDER if c in classes]
    t_classes = [TREATMENT_CLASS[c] for c in classes]
    ambiguous = len(classes) > 1
    if ambiguous:
        notes.append(f"value {v:.3f} mm lies in overlapping bands {'/'.join(classes)}; all candidates reported")
    return RuleResult(
        value_mm=v,
        etiology_class="-".join(classes),
        etiology_candidates=classes,
        etiology_descriptions=[d for c in classes for d in ETIOLOGY[c]],
        treatment_class="-".join(t_classes),
        treatment_alternatives=[t for tc in t_classes for t in TREATMENT[tc]],
        ambiguous=ambiguous,
        notes="; ".join(notes),
        thresholds=thresholds,
    )
