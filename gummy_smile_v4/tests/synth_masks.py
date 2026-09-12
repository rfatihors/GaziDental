"""Synthetic gingiva / lip masks with known geometry for the measurement tests."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def rect_band(shape: Tuple[int, int], x0: int, x1: int, top: int, thickness: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[top : top + thickness, x0:x1] = True
    return m


def festooned_band(
    shape: Tuple[int, int], x0: int, x1: int, top: int, zenith_t: int, papilla_t: int, n_teeth: int = 6
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Band whose lower edge is thin at tooth centres (zeniths) and thick between teeth (papillae).

    Returns the mask, the zenith x positions and the papilla x positions.
    """
    m = np.zeros(shape, dtype=bool)
    width = x1 - x0
    tooth_w = width / n_teeth
    zeniths, papillae = [], []
    for x in range(x0, x1):
        # position within the tooth: 0 at tooth boundary (papilla), 0.5 at centre (zenith)
        u = ((x - x0) % tooth_w) / tooth_w
        tri = abs(u - 0.5) * 2          # 1 at papilla, 0 at zenith
        t = int(round(zenith_t + (papilla_t - zenith_t) * tri))
        m[top : top + t, x] = True
    for k in range(n_teeth):
        zeniths.append(int(x0 + (k + 0.5) * tooth_w))
    for k in range(n_teeth + 1):
        papillae.append(int(min(x1 - 1, x0 + k * tooth_w)))
    return m, zeniths, papillae


def papilla_triangles(shape: Tuple[int, int], centers: Sequence[int], top: int, height: int, half_w: int) -> np.ndarray:
    """Normal-smile-line imitation: only downward triangles at the papillae, zero between."""
    m = np.zeros(shape, dtype=bool)
    for c in centers:
        for dy in range(height):
            hw = int(round(half_w * (1 - dy / height)))
            m[top + dy, max(0, c - hw) : c + hw + 1] = True
    return m


def lip_band(shape: Tuple[int, int], x0: int, x1: int, bottom: int, thickness: int) -> np.ndarray:
    """Upper-lip vermilion band whose lowest row is ``bottom`` (inclusive)."""
    m = np.zeros(shape, dtype=bool)
    m[bottom - thickness + 1 : bottom + 1, x0:x1] = True
    return m
