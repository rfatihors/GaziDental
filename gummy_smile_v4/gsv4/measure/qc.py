"""Quality-control flags attached to every measurement (spec §3.4). Never swallowed."""
from __future__ import annotations

from enum import Enum
from typing import Iterable, List


class QCFlag(str, Enum):
    NO_GINGIVA_MASK = "no_gingiva_mask"
    NO_LIP_MASK = "no_lip_mask"
    GINGIVA_MULTI_COMPONENT = "gingiva_multi_component"
    REGION_ZERO = "region_zero"
    REGION_EMPTY_WINDOW = "region_empty_window"
    FESTOON_DETECTION_FAILED = "festoon_detection_failed"
    ZENITH_DETECTION_FAILED = "zenith_detection_failed"
    LIP_GINGIVA_BOUNDARY_MISMATCH = "lip_gingiva_boundary_mismatch"
    MASK_SHAPE_MISMATCH = "mask_shape_mismatch"
    IMPLAUSIBLE_VALUE = "implausible_value"
    FRAME_UNCERTAIN = "frame_uncertain"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def flags_to_str(flags: Iterable[QCFlag]) -> str:
    """Comma-joined, stable order (enum declaration order), for CSV output."""
    order = list(QCFlag)
    present = set(flags)
    return ",".join(f.value for f in order if f in present)


def flags_from_str(s: str) -> List[QCFlag]:
    return [QCFlag(x) for x in s.split(",") if x]
