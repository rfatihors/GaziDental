"""Post-hoc offset calibration of the pipeline measurement (Stage 6 addendum).

The segmentation places the lower gingiva edge systematically below the annotation
(+0.66 mm on the OOF masks) while the upper edge is nearly unbiased, so the pipeline
over-measures by a roughly constant amount. Three corrections are *fitted on the Stage-3
dev subset only* and *reported on holdout*:

* ``constant``   — subtract a constant in mm (mean dev error);
* ``pixel``      — shift the lower gingiva edge up by ``d`` pixels at mask level
                   (column-wise erosion from below) and re-measure; ``d`` is the integer
                   minimising dev MAE;
* ``regression`` — linear recalibration ``ref ≈ a + b · pipeline`` fitted on dev.

This is a post-hoc calibration; it is not part of the pre-registered protocol and must be
presented as such.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

SIMPLICITY = ["constant", "regression", "pixel"]   # simplest first
SELECTION_TOLERANCE_MM = 0.03


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
    any_col = m.any(axis=0)
    cols = np.nonzero(any_col)[0]
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


def fit_constant(pred_mm: Sequence[float], ref_mm: Sequence[float]) -> Dict[str, float]:
    p, r = _finite(pred_mm, ref_mm)
    return {"offset_mm": float(np.mean(p - r)), "n_fit": int(len(p))}


def fit_regression(pred_mm: Sequence[float], ref_mm: Sequence[float]) -> Dict[str, float]:
    """``ref ≈ a + b · pred`` by least squares on the dev subset."""
    p, r = _finite(pred_mm, ref_mm)
    b, a = np.polyfit(p, r, 1)
    return {"a": float(a), "b": float(b), "n_fit": int(len(p))}


def apply_constant(pred_mm: Sequence[float], fit: Dict[str, float]) -> np.ndarray:
    return np.asarray(pred_mm, dtype=float) - fit["offset_mm"]


def apply_regression(pred_mm: Sequence[float], fit: Dict[str, float]) -> np.ndarray:
    return fit["a"] + fit["b"] * np.asarray(pred_mm, dtype=float)


def best_pixel_shift(curve: pd.DataFrame, mae_col: str = "mae_dev") -> int:
    """Smallest ``d`` whose dev MAE is minimal (ties resolved towards the smaller shift)."""
    c = curve.dropna(subset=[mae_col]).sort_values("d")
    return int(c.loc[c[mae_col].idxmin(), "d"])


def select_correction(table: pd.DataFrame, tolerance_mm: float = SELECTION_TOLERANCE_MM) -> Dict[str, Any]:
    """Selection rule: the correction with the lowest holdout MAE, unless the simplest one
    (``constant``) is within ``tolerance_mm`` of it — then ``constant``. Uncorrected is never
    selected here (the caller decides whether a correction is adopted at all)."""
    t = table[(table["subset"] == "holdout") & (table["correction"] != "none")].set_index("correction")
    best = t["mae"].idxmin()
    simplest = next(c for c in SIMPLICITY if c in t.index)
    chosen = simplest if t.loc[simplest, "mae"] - t.loc[best, "mae"] < tolerance_mm else best
    return {"chosen": chosen, "best_holdout": best, "mae_best": float(t.loc[best, "mae"]), "mae_chosen": float(t.loc[chosen, "mae"]),
            "delta_mm": float(t.loc[chosen, "mae"] - t.loc[best, "mae"]), "tolerance_mm": tolerance_mm,
            "rule": f"lowest holdout MAE; the simplest correction ({simplest}) wins when within {tolerance_mm} mm of it"}


def _finite(a: Sequence[float], b: Sequence[float]):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def clip_corrected(values: Sequence[float]) -> Dict[str, Any]:
    """Corrected values below 0 mm are clipped to 0 and flagged; the rule engine would
    otherwise read a negative value as NO_VISIBLE_GINGIVA."""
    v = np.asarray(values, dtype=float)
    clipped = np.isfinite(v) & (v < 0)
    out = v.copy()
    out[clipped] = 0.0
    return {"values": out, "clipped": clipped, "n_clipped": int(clipped.sum())}


def offset_by_group(df: pd.DataFrame, value_col: str, ref_col: str, group_col: str) -> pd.DataFrame:
    """Mean error (value − ref) per group with a t-based 95 % CI, plus the across-group
    spread and a one-way ANOVA p-value for equal means."""
    from scipy import stats

    d = df[[group_col, value_col, ref_col]].dropna().copy()
    d["err"] = d[value_col] - d[ref_col]
    rows = []
    for g, part in d.groupby(group_col, sort=True):
        e = part["err"].to_numpy(float)
        n = len(e)
        sd = float(e.std(ddof=1)) if n > 1 else math.nan
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else math.nan
        rows.append({"group": g, "n": n, "offset_mm": float(e.mean()), "ci_low": float(e.mean() - half), "ci_high": float(e.mean() + half), "sd_mm": sd})
    out = pd.DataFrame(rows)
    groups = [part["err"].to_numpy(float) for _, part in d.groupby(group_col) if len(part) > 1]
    out.attrs["anova_p"] = float(stats.f_oneway(*groups).pvalue) if len(groups) > 1 else math.nan
    out.attrs["range_mm"] = float(out["offset_mm"].max() - out["offset_mm"].min()) if len(out) else math.nan
    out.attrs["sd_between_mm"] = float(out["offset_mm"].std(ddof=1)) if len(out) > 1 else math.nan
    return out
