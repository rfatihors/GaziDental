"""Tooth regioning of the thickness profile (spec §3.2 step 4): A equal, B festoon, C zenith.

All spacings are fractions of the image width read from the config — no fixed pixel
constants (v3's 30 px assumed one resolution).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences

Region = Tuple[int, int]  # [start, end) in columns


def select_by_distance(positions: np.ndarray, priority: np.ndarray, distance: int, prefer: np.ndarray) -> np.ndarray:
    """``find_peaks(distance=…)``'s filter, with the tie between equal priorities decided here.

    Why this exists. ``scipy.signal.find_peaks`` resolves its ``distance`` constraint greedily from
    the highest priority down, and it orders the candidates with ``np.argsort(priority)``, whose
    default sort is not stable. When two candidates have *exactly* the same priority and sit closer
    together than ``distance``, exactly one of them survives and which one is decided by the sort
    implementation rather than by the data — so the same code on the same mask gives a different
    answer on a machine with a different numpy build. That is not hypothetical: it moved one of 145
    measurements between the workstation and a laptop (``outputs/09_final_rfdetr/PLAN.md``,
    Amendment 6). Equal priorities are common here because both regionings key on plateaus of a
    quantised profile: two zero-thickness stretches are *exactly* equally deep, not nearly.

    The greedy itself is scipy's, unchanged, so nothing but the tie moves: walk the candidates from
    the highest priority down, and for each one still alive drop every neighbour nearer than
    ``distance``. ``prefer`` supplies the order among equal priorities — the smaller value wins —
    and each caller passes the criterion its own method already uses.

    ``positions`` must be ascending, as ``find_peaks`` returns them.
    """
    positions = np.asarray(positions)
    order = np.lexsort((np.asarray(prefer, dtype=float), -np.asarray(priority, dtype=float)))
    keep = np.ones(len(positions), dtype=bool)
    for j in order:
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and positions[j] - positions[k] < distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < len(positions) and positions[k] - positions[j] < distance:
            keep[k] = False
            k += 1
    return keep


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
    # find_peaks' own order, reproduced step by step so that only the tie changes: all maxima, then
    # the minimum separation (priority = the profile height at the peak, exactly as find_peaks uses
    # it), then the prominence threshold on the survivors. The separation step is ours because
    # find_peaks would leave two equally high papillae to an unstable sort (see select_by_distance);
    # among equally high candidates the leftmost is kept, left to right across the arch.
    peaks, _ = find_peaks(seg)
    if len(peaks):
        alive = select_by_distance(peaks, seg[peaks], max(1, min_distance), peaks.astype(float))
        peaks = peaks[alive]
    prom = peak_prominences(seg, peaks)[0] if len(peaks) else np.zeros(0)
    above = prom >= max(1.0, 0.05 * float(seg.max()))
    peaks, prom = peaks[above], prom[above]
    # drop peaks hugging the window edges (they are not interior papillae)
    keep = (peaks > min_distance // 2) & (peaks < seg.size - min_distance // 2)
    peaks, prom = peaks[keep], prom[keep]
    if len(peaks) < n - 1:
        return None
    if len(peaks) > n - 1:
        idx = np.lexsort((peaks.astype(float), -prom))[: n - 1]
        peaks = np.sort(peaks[idx])
    edges = [x0] + [int(x0 + p) for p in peaks] + [x1]
    regions = [(edges[i], edges[i + 1]) for i in range(n)]
    if any(e - s <= 0 for s, e in regions):
        return None
    return regions


def zeniths_midline(
    t_smooth: np.ndarray, x0: int, x1: int, midline: int, n_per_side: int, min_distance: int
) -> Tuple[Optional[List[int]], int, int]:
    """Zenith candidates = local minima of the profile; the ``n_per_side`` nearest to the
    midline on each side are taken. Zero-thickness plateaus (normal smile line) count as
    minima. Returns ``(sorted x positions or None, n_candidates_left, n_candidates_right)``;
    None when a side has fewer than ``n_per_side`` candidates."""
    seg = t_smooth[x0:x1].astype(float)
    if seg.size == 0:
        return None, 0, 0
    inv = seg.max() - seg
    mins, props = find_peaks(inv, plateau_size=(1, None))
    # use plateau centres
    centres = np.asarray([(l + r) // 2 for l, r in zip(props["left_edges"], props["right_edges"])], dtype=int) if len(mins) else np.array([], dtype=int)
    # The minimum separation is enforced here rather than by find_peaks. Zero-thickness stretches
    # are *exactly* equally deep, so ties are the rule and not the exception, and scipy would leave
    # them to an unstable sort (see select_by_distance). Among equally deep minima the one nearer the
    # dental midline is kept, which is the criterion this method already uses to pick its zeniths.
    if len(mins):
        alive = select_by_distance(mins, inv[mins], max(1, min_distance), np.abs(centres + x0 - midline).astype(float))
        mins, centres = mins[alive], centres[alive]
    xs = centres + x0
    n_left, n_right = int((xs < midline).sum()), int((xs >= midline).sum())
    left = np.sort(xs[xs < midline])[::-1][:n_per_side]
    right = np.sort(xs[xs >= midline])[:n_per_side]
    if len(left) < n_per_side or len(right) < n_per_side:
        return None, n_left, n_right
    return sorted(int(v) for v in np.concatenate([left, right])), n_left, n_right


def windows_around(centres: List[int], half_width: int, x0: int, x1: int) -> List[Region]:
    return [(max(x0, c - half_width), min(x1, c + half_width + 1)) for c in centres]
