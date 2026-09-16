"""Post-hoc pixel offset of the gingival-display measurement (adopted 16 Sep 2026).

The segmentation model includes a fixed number of pixels too many at the *lower* gingiva
edge (thin festooned margin), while the upper (lip-side) edge is nearly unbiased. The
correction is therefore defined in **pixels** (``measurement.offset_px`` in config.yaml,
estimated on the Stage-3 dev subset of the out-of-fold predictions) and converted to mm
with whatever ``px_per_mm`` is valid for the image: the global scale in Stages 3–6, the
experts' per-image scale in Stage 4. A corrected value below 0 mm is clipped to 0 and
flagged ``clipped`` — the rule engine would otherwise read it as NO_VISIBLE_GINGIVA.

Uncorrected values are the primary result; corrected values are secondary and every table
carries both.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray, float]


def offset_px_from_config(cfg: Dict[str, Any]) -> float:
    """``measurement.offset_px`` (0 when absent)."""
    v = (cfg.get("measurement") or {}).get("offset_px", 0.0)
    return float(v) if v is not None else 0.0


def corrected_mm(px: ArrayLike, px_per_mm: ArrayLike, offset_px: float) -> Dict[str, Any]:
    """``max(0, (px + offset_px) / px_per_mm)`` element-wise with a ``clipped`` flag.

    ``px_per_mm`` may be a scalar (global scale) or an array of per-image scales. NaN
    inputs stay NaN and are never flagged.
    """
    p = np.asarray(px, dtype=float)
    k = np.asarray(px_per_mm, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mm = (p + float(offset_px)) / k
    mm = np.where(np.isfinite(mm), mm, np.nan)
    clipped = np.isfinite(mm) & (mm < 0)
    out = mm.copy()
    out[clipped] = 0.0
    return {"mm": out, "clipped": clipped, "n_clipped": int(clipped.sum()), "offset_px": float(offset_px)}


def offset_mm_at(px_per_mm: float, offset_px: float) -> float:
    """The offset expressed in mm at one scale (for reporting only)."""
    return float(offset_px) / float(px_per_mm)
