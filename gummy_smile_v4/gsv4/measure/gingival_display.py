"""Gingival display from class-separated masks (spec §0, §3.2–3.4).

The quantity measured is the vertical thickness of the **gingiva** mask: the band
bounded above by the lower lip edge and below by the gingival margin. The lip mask is
never measured; it only (a) intersects the x-window, (b) provides the midline for
method C, (c) checks the lip–gingiva boundary and (d) feeds the lip-anchored estimator.

Pure function: arrays in, dataclass out. File I/O lives in ``gsv4.masks``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gsv4.measure.profile import (
    column_profile,
    connected_components,
    lip_anchor,
    odd_window,
    smooth_profile,
)
from gsv4.measure.qc import QCFlag, flags_to_str
from gsv4.measure.regions import (
    Region,
    regions_equal,
    regions_festoon,
    windows_around,
    zeniths_midline,
)

ESTIMATORS = ("p10", "p25", "median", "min", "max")
REGIONINGS = ("A", "B", "C")
MethodKey = Tuple[str, str, bool]  # (regioning, estimator, lip_anchored)

DEFAULT_CFG: Dict[str, Any] = {
    "regions": 6,
    "median_filter_frac": 0.01,
    "zenith_window_frac": 0.01,
    "zenith_min_distance_frac": 0.03,
    "festoon_min_distance_frac": 0.03,
    "boundary_gap_max_frac": 0.02,
    "implausible_mm": 20.0,
    "default_method": {"regioning": "A", "estimator": "p25", "anchored": False},
}


def _estimate(values: np.ndarray, estimator: str) -> float:
    if values.size == 0:
        return math.nan
    if estimator == "p10":
        return float(np.percentile(values, 10))
    if estimator == "p25":
        return float(np.percentile(values, 25))
    if estimator == "median":
        return float(np.median(values))
    if estimator == "min":
        return float(values.min())
    if estimator == "max":
        return float(values.max())
    raise KeyError(estimator)


@dataclass
class MeasurementResult:
    height: int
    width: int
    x0: int
    x1: int
    midline_x: int
    n_components: int
    regions: Dict[str, List[Region]]
    zeniths_c: List[int]
    region_values: Dict[MethodKey, List[float]]
    image_values: Dict[MethodKey, float]
    image_medians: Dict[MethodKey, float]
    method: Dict[str, Any]
    gingival_display_px: float
    gingival_display_mm: Optional[float]
    px_per_mm: Optional[float]
    unit: str
    gap_median: float
    gap_iqr: Tuple[float, float]
    flags: List[QCFlag] = field(default_factory=list)
    profile: Optional[Dict[str, np.ndarray]] = None

    def to_row(self) -> Dict[str, Any]:
        """Flat record for CSV: one column per (regioning, estimator, anchoring) and region."""
        row: Dict[str, Any] = {
            "height": self.height, "width": self.width, "x0": self.x0, "x1": self.x1,
            "midline_x": self.midline_x, "n_components": self.n_components,
            "method": f"{self.method['regioning']}_{self.method['estimator']}" + ("_lipanchored" if self.method.get("anchored") else ""),
            "gingival_display_px": self.gingival_display_px, "gingival_display_mm": self.gingival_display_mm,
            "px_per_mm": self.px_per_mm, "unit": self.unit,
            "gap_median_px": self.gap_median, "gap_q1_px": self.gap_iqr[0], "gap_q3_px": self.gap_iqr[1],
            "qc_flags": flags_to_str(self.flags),
        }
        for reg in REGIONINGS:
            for est in ESTIMATORS:
                for anchored in (False, True):
                    key = (reg, est, anchored)
                    name = f"{reg}_{est}" + ("_lipanchored" if anchored else "")
                    row[f"{name}_px"] = self.image_values.get(key, math.nan)
                    row[f"{name}_median_px"] = self.image_medians.get(key, math.nan)
                    vals = self.region_values.get(key, [math.nan] * len(self.regions.get("A", []) or [0] * 6))
                    for i, v in enumerate(vals, start=1):
                        row[f"{name}_region_{i}_px"] = v
        return row


def _empty_result(h: int, w: int, px_per_mm: Optional[float], cfg: Dict[str, Any], flags: List[QCFlag]) -> MeasurementResult:
    n = int(cfg["regions"])
    nanlist = [math.nan] * n
    keys = [(r, e, a) for r in REGIONINGS for e in ESTIMATORS for a in (False, True)]
    return MeasurementResult(
        height=h, width=w, x0=-1, x1=-1, midline_x=w // 2, n_components=0,
        regions={"A": [], "B": [], "C": []}, zeniths_c=[],
        region_values={k: list(nanlist) for k in keys}, image_values={k: math.nan for k in keys},
        image_medians={k: math.nan for k in keys}, method=dict(cfg["default_method"]),
        gingival_display_px=math.nan, gingival_display_mm=None if px_per_mm is None else math.nan,
        px_per_mm=px_per_mm, unit="mm" if px_per_mm else "px", gap_median=math.nan, gap_iqr=(math.nan, math.nan), flags=flags,
    )


def measure_gingival_display(
    gingiva: np.ndarray,
    lip: Optional[np.ndarray],
    px_per_mm: Optional[float],
    cfg: Optional[Dict[str, Any]] = None,
    method: Optional[Dict[str, Any]] = None,
    keep_profile: bool = False,
) -> MeasurementResult:
    """Measure gingival display on binary masks at original image resolution.

    ``method`` selects which (regioning, estimator, anchored) combination fills
    ``gingival_display_px``; every combination is computed and stored regardless.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    method = dict(method or cfg["default_method"])
    g = np.asarray(gingiva).astype(bool)
    if g.ndim != 2:
        raise ValueError(f"gingiva mask must be 2-D, got shape {g.shape}")
    h, w = g.shape
    flags: List[QCFlag] = []
    L: Optional[np.ndarray] = None
    if lip is not None:
        L = np.asarray(lip).astype(bool)
        if L.shape != g.shape:
            raise ValueError(f"lip mask shape {L.shape} != gingiva mask shape {g.shape}")
    else:
        flags.append(QCFlag.NO_LIP_MASK)

    if not g.any():
        flags.insert(0, QCFlag.NO_GINGIVA_MASK)
        return _empty_result(h, w, px_per_mm, cfg, flags)

    n = int(cfg["regions"])
    t, top, bottom = column_profile(g)
    cols = np.nonzero(t > 0)[0]
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    if L is not None and L.any():
        lcols = np.nonzero(L.any(axis=0))[0]
        x0, x1 = max(x0, int(lcols.min())), min(x1, int(lcols.max()) + 1)
        if x1 <= x0:
            flags.append(QCFlag.NO_GINGIVA_MASK)
            return _empty_result(h, w, px_per_mm, cfg, flags)
    width_win = x1 - x0
    n_comp = connected_components(g)
    if n_comp > 1:
        flags.append(QCFlag.GINGIVA_MULTI_COMPONENT)

    t_s = smooth_profile(t, odd_window(width_win, float(cfg["median_filter_frac"])))

    # lip-anchored distance and boundary gap
    d = np.full(w, np.nan)
    gap_median, gap_iqr = math.nan, (math.nan, math.nan)
    if L is not None and L.any():
        lb = lip_anchor(L, top)
        valid = (top >= 0) & ~np.isnan(lb)
        d[valid] = bottom[valid] - lb[valid]
        gap = top[valid] - lb[valid] - 1
        if gap.size:
            gap_median = float(np.median(gap))
            gap_iqr = (float(np.percentile(gap, 25)), float(np.percentile(gap, 75)))
            if gap_median > float(cfg["boundary_gap_max_frac"]) * h:
                flags.append(QCFlag.LIP_GINGIVA_BOUNDARY_MISMATCH)
    d_filled = np.where(np.isnan(d), 0.0, d)  # columns without gingiva: 0, like t

    # midline: lip x-median, else image centre
    if L is not None and L.any():
        lx = np.nonzero(L.any(axis=0))[0]
        midline = int(np.median(lx))
    else:
        midline = w // 2

    # regionings
    regions: Dict[str, List[Region]] = {"A": regions_equal(x0, x1, n)}
    min_dist_f = max(1, int(round(width_win * float(cfg["festoon_min_distance_frac"]))))
    b = regions_festoon(t_s, x0, x1, n, min_dist_f)
    if b is None:
        flags.append(QCFlag.FESTOON_DETECTION_FAILED)
        b = regions["A"]
    regions["B"] = b
    min_dist_c = max(1, int(round(w * float(cfg["zenith_min_distance_frac"]))))
    z = zeniths_midline(t_s, x0, x1, midline, n // 2, min_dist_c)
    half = max(1, int(round(w * float(cfg["zenith_window_frac"]))))
    if z is None:
        flags.append(QCFlag.ZENITH_DETECTION_FAILED)
        regions["C"] = regions["A"]
        zeniths = []
    else:
        regions["C"] = windows_around(z, half, x0, x1)
        zeniths = z

    region_values: Dict[MethodKey, List[float]] = {}
    image_values: Dict[MethodKey, float] = {}
    image_medians: Dict[MethodKey, float] = {}
    for reg, regs in regions.items():
        for anchored, series in ((False, t.astype(float)), (True, d_filled if L is not None and L.any() else None)):
            for est in ESTIMATORS:
                key = (reg, est, anchored)
                if series is None:
                    region_values[key] = [math.nan] * n
                    image_values[key] = math.nan
                    image_medians[key] = math.nan
                    continue
                vals = []
                for s, e in regs:
                    seg = series[s:e]
                    if seg.size == 0:
                        vals.append(math.nan)
                        if QCFlag.REGION_EMPTY_WINDOW not in flags:
                            flags.append(QCFlag.REGION_EMPTY_WINDOW)
                    else:
                        vals.append(_estimate(seg, est))
                region_values[key] = vals
                arr = np.asarray(vals, dtype=float)
                image_values[key] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
                image_medians[key] = float(np.nanmedian(arr)) if np.isfinite(arr).any() else math.nan
    prim_key: MethodKey = (method["regioning"], method["estimator"], bool(method.get("anchored", False)))
    if any(v == 0 for v in region_values[prim_key] if not math.isnan(v)):
        flags.append(QCFlag.REGION_ZERO)
    value_px = image_values[prim_key]
    value_mm: Optional[float]
    if px_per_mm:
        value_mm = value_px / float(px_per_mm)
        unit = "mm"
        if not math.isnan(value_mm) and (value_mm < 0 or value_mm > float(cfg["implausible_mm"])):
            flags.append(QCFlag.IMPLAUSIBLE_VALUE)
    else:
        value_mm = None
        unit = "px"
    return MeasurementResult(
        height=h, width=w, x0=x0, x1=x1, midline_x=midline, n_components=n_comp, regions=regions,
        zeniths_c=zeniths, region_values=region_values, image_values=image_values, image_medians=image_medians,
        method={"regioning": prim_key[0], "estimator": prim_key[1], "anchored": prim_key[2]},
        gingival_display_px=value_px, gingival_display_mm=value_mm, px_per_mm=px_per_mm, unit=unit,
        gap_median=gap_median, gap_iqr=gap_iqr, flags=flags,
        profile={"t": t, "top": top, "bottom": bottom, "t_smooth": t_s, "d": d} if keep_profile else None,
    )
