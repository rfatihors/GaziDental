"""Post-hoc calibration of the gingival-display measurement (adopted 16 Sep 2026).

The segmentation model includes a fixed number of pixels too many at the *lower* gingiva
edge (thin festooned margin), while the upper (lip-side) edge is nearly unbiased. The
adopted correction therefore acts **at mask level, where the error is**: the lower edge of
the predicted gingiva mask is moved up by ``|bottom_edge_offset_px|`` pixels in every column
(``measurement.bottom_edge_offset_px`` in config.yaml, estimated on the Stage-3 dev subset of
the out-of-fold predictions), and the thickness profile and every estimator are computed
from the corrected mask. A column whose bottom run is shorter than the shift becomes 0
(never negative); the number of such columns is reported.

Because the correction is in pixels, mm conversion uses whatever ``px_per_mm`` is valid for
the image (global scale in Stages 3–6, the experts' per-image scale in Stage 4).

Uncorrected values are the primary result; corrected values are secondary and every table
carries both. The value-level variants (constant mm, constant px) stay available for the
comparison table only (``value_level_mm``).
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray, float]
DEPRECATED_KEYS = ("offset_px",)   # value-level pixel offset, replaced by bottom_edge_offset_px


def bottom_edge_offset_from_config(cfg: Dict[str, Any]) -> float:
    """``measurement.bottom_edge_offset_px`` (0 when absent; negative = lower edge moved up)."""
    m = cfg.get("measurement") or {}
    for k in DEPRECATED_KEYS:
        if k in m:
            raise KeyError(f"measurement.{k} is deprecated (value-level offset); use measurement.bottom_edge_offset_px (mask level)")
    v = m.get("bottom_edge_offset_px", 0.0)
    return float(v) if v is not None else 0.0


def shift_lower_edge_up(mask: np.ndarray, d: int) -> np.ndarray:
    """Remove the lowest ``d`` pixels of the bottom-most vertical run in every column.

    Only contiguous foreground pixels counted upward from the column's lowest foreground
    pixel are cleared, so a run shorter than ``d`` disappears without touching a run
    further up. ``d <= 0`` returns a copy.
    """
    m = np.asarray(mask).astype(bool).copy()
    if d <= 0 or not m.any():
        return m
    h, w = m.shape
    cols = np.nonzero(m.any(axis=0))[0]
    bottom = h - 1 - np.argmax(m[::-1, :][:, cols], axis=0)    # lowest foreground row per column
    alive = np.ones(len(cols), dtype=bool)
    for j in range(d):
        rows = bottom - j
        ok = alive & (rows >= 0)
        ok[ok] &= m[rows[ok], cols[ok]]
        m[rows[ok], cols[ok]] = False
        alive = ok
        if not alive.any():
            break
    return m


def apply_bottom_edge_offset(gingiva: np.ndarray, bottom_edge_offset_px: float) -> Dict[str, Any]:
    """Corrected gingiva mask plus bookkeeping: columns with gingiva before / after and the
    number of columns that became empty because their run was shorter than the shift."""
    d = int(round(abs(float(bottom_edge_offset_px))))
    before = np.asarray(gingiva).astype(bool)
    after = shift_lower_edge_up(before, d)
    had = before.any(axis=0)
    has = after.any(axis=0)
    return {"mask": after, "shift_px": d, "n_columns_before": int(had.sum()), "n_columns_after": int(has.sum()),
            "n_columns_zeroed": int((had & ~has).sum()),
            "columns_zeroed_frac": float((had & ~has).sum() / had.sum()) if had.any() else 0.0}


def value_level_mm(px: ArrayLike, px_per_mm: ArrayLike, offset_px: float) -> Dict[str, Any]:
    """Comparison variant only (not adopted): ``max(0, (px + offset_px) / px_per_mm)`` with a
    ``clipped`` flag. Kept so the three variants can be tabulated side by side."""
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
    """A pixel offset expressed in mm at one scale (reporting only)."""
    return float(offset_px) / float(px_per_mm)
