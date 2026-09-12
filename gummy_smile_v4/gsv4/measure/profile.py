"""Column-wise vertical thickness profile of a binary mask (spec §3.2 steps 2–3, 6).

For every column the *longest uninterrupted vertical run* of mask pixels is used, so a
hole or a stray pixel never inflates the thickness. Columns without mask pixels have
thickness 0 (never NaN): in normal/low smile lines this is a real zero.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter


def column_profile(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, top, bottom)`` arrays of length W.

    ``t[x]`` is the length of the longest run in column x (0 if empty); ``top``/``bottom``
    are the first/last row of that run (-1 where t == 0).
    """
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    t = np.zeros(w, dtype=np.int64)
    top = np.full(w, -1, dtype=np.int64)
    bottom = np.full(w, -1, dtype=np.int64)
    if not m.any():
        return t, top, bottom
    padded = np.zeros((h + 2, w), dtype=bool)
    padded[1:-1] = m
    d = np.diff(padded.astype(np.int8), axis=0)          # +1 at run start, -1 after run end
    start_r, start_c = np.nonzero(d == 1)                # row index in original = start_r
    end_r, end_c = np.nonzero(d == -1)                   # run covers rows [start_r, end_r - 1]
    # nonzero returns row-major order; regroup by column so starts and ends pair up
    s_order = np.lexsort((start_r, start_c))
    e_order = np.lexsort((end_r, end_c))
    start_r, start_c = start_r[s_order], start_c[s_order]
    end_r = end_r[e_order]
    lengths = end_r - start_r
    # longest run per column
    best = {}
    for c, s, L in zip(start_c, start_r, lengths):
        if L > best.get(c, (0, -1))[0]:
            best[c] = (L, s)
    for c, (L, s) in best.items():
        t[c] = L
        top[c] = s
        bottom[c] = s + L - 1
    return t, top, bottom


def smooth_profile(t: np.ndarray, window: int) -> np.ndarray:
    """Odd-window median filter; window <= 1 returns a copy."""
    if window <= 1:
        return t.astype(float).copy()
    if window % 2 == 0:
        window += 1
    return median_filter(t.astype(float), size=window, mode="nearest")


def odd_window(width: int, frac: float, minimum: int = 3) -> int:
    w = max(minimum, int(round(width * frac)))
    return w if w % 2 == 1 else w + 1


def lip_anchor(lip: np.ndarray, top: np.ndarray) -> np.ndarray:
    """For every column: lowest lip row strictly above the gingiva run top; NaN where undefined."""
    lip = np.asarray(lip).astype(bool)
    h, w = lip.shape
    out = np.full(w, np.nan)
    rows = np.arange(h)
    for x in np.nonzero(top >= 0)[0]:
        col = lip[: top[x], x]
        if col.any():
            out[x] = rows[: top[x]][col].max()
    return out


def connected_components(mask: np.ndarray) -> int:
    from scipy.ndimage import label

    _, n = label(np.asarray(mask).astype(bool))
    return int(n)
