"""Tooth regioning of the thickness profile (spec §3.2 step 4): A equal, B festoon, C zenith.

All spacings are fractions of the image width read from the config — no fixed pixel
constants (v3's 30 px assumed one resolution).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

Region = Tuple[int, int]  # [start, end) in columns


def regions_equal(x0: int, x1: int, n: int) -> List[Region]:
    edges = np.linspace(x0, x1, n + 1)
    return [(int(round(edges[i])), int(round(edges[i + 1]))) for i in range(n)]


def regions_festoon(t_smooth: np.ndarray, x0: int, x1: int, n: int, min_distance: int) -> Optional[List[Region]]:
    """Papillae = local maxima of the profile; regions lie between consecutive maxima.

    Exactly ``n`` regions are needed, i.e. ``n - 1`` interior maxima plus the window
    edges. If more candidates exist, the ``n - 1`` most prominent are used; if fewer,
    None is returned (caller falls back to A and flags it).
    """
    seg = t_smooth[x0:x1]
    if seg.size < 2 * min_distance or not np.any(seg > 0):
        return None
    peaks, props = find_peaks(seg, distance=max(1, min_distance), prominence=max(1.0, 0.05 * float(seg.max())))
    # drop peaks hugging the window edges (they are not interior papillae)
    keep = (peaks > min_distance // 2) & (peaks < seg.size - min_distance // 2)
    peaks, prom = peaks[keep], props["prominences"][keep]
    if len(peaks) < n - 1:
        return None
    if len(peaks) > n - 1:
        idx = np.argsort(prom)[::-1][: n - 1]
        peaks = np.sort(peaks[idx])
    edges = [x0] + [int(x0 + p) for p in peaks] + [x1]
    regions = [(edges[i], edges[i + 1]) for i in range(n)]
    if any(e - s <= 0 for s, e in regions):
        return None
    return regions


def zeniths_midline(
    t_smooth: np.ndarray, x0: int, x1: int, midline: int, n_per_side: int, min_distance: int
) -> Optional[List[int]]:
    """Zenith candidates = local minima of the profile; the ``n_per_side`` nearest to the
    midline on each side are taken. Zero-thickness plateaus (normal smile line) count as
    minima. Returns sorted x positions or None when a side has too few candidates."""
    seg = t_smooth[x0:x1].astype(float)
    if seg.size == 0:
        return None
    inv = seg.max() - seg
    mins, props = find_peaks(inv, distance=max(1, min_distance), plateau_size=(1, None))
    # use plateau centres
    centres = np.asarray([(l + r) // 2 for l, r in zip(props["left_edges"], props["right_edges"])], dtype=int) if len(mins) else np.array([], dtype=int)
    xs = centres + x0
    left = np.sort(xs[xs < midline])[::-1][:n_per_side]
    right = np.sort(xs[xs >= midline])[:n_per_side]
    if len(left) < n_per_side or len(right) < n_per_side:
        return None
    return sorted(int(v) for v in np.concatenate([left, right]))


def windows_around(centres: List[int], half_width: int, x0: int, x1: int) -> List[Region]:
    return [(max(x0, c - half_width), min(x1, c + half_width + 1)) for c in centres]
