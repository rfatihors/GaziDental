from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GumVisibilityMeasurement:
    patient_id: str
    gum_visibility_px: float
    gum_visibility_mm: Optional[float]
    px_per_mm: Optional[float]
    per_region_px: Dict[str, float]


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask could not be read: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _find_main_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in gingival mask.")
    return max(contours, key=cv2.contourArea)


def _split_regions(bbox: Tuple[int, int, int, int], regions: int) -> List[Tuple[int, int]]:
    x, _, w, _ = bbox
    step = w / regions
    return [(int(x + i * step), int(x + (i + 1) * step)) for i in range(regions)]


def _find_zenith_points(contour: np.ndarray, bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    zeniths: List[Tuple[int, int]] = []
    contour_points = contour.reshape(-1, 2)
    for start, end in bounds:
        region_points = contour_points[(contour_points[:, 0] >= start) & (contour_points[:, 0] < end)]
        if region_points.size == 0:
            top_idx = int(np.argmin(contour_points[:, 1]))
            fallback = contour_points[top_idx]
            zeniths.append((int((start + end) / 2), int(fallback[1])))
            continue
        top_idx = int(np.argmin(region_points[:, 1]))
        top_pt = region_points[top_idx]
        zeniths.append((int(top_pt[0]), int(top_pt[1])))
    return zeniths


def measure_gum_visibility(
    image_path: Path,
    mask_path: Path,
    regions: int = 6,
    px_per_mm: Optional[float] = None,
) -> GumVisibilityMeasurement:
    binary_mask = _load_binary_mask(mask_path)
    contour = _find_main_contour(binary_mask)

    lip_line_y = float(contour[:, :, 1].min())
    x, y, w, h = cv2.boundingRect(contour)
    bounds = _split_regions((x, y, w, h), regions=regions)
    zenith_points = _find_zenith_points(contour, bounds)

    per_region_px: Dict[str, float] = {}
    for idx, (_, zenith_y) in enumerate(zenith_points, start=1):
        gummy_px = float(zenith_y - lip_line_y)
        per_region_px[f"region_{idx}_px"] = round(gummy_px, 3)

    px_values = list(per_region_px.values())
    gum_visibility_px = float(np.mean(px_values)) if px_values else float("nan")
    gum_visibility_mm = None
    if px_per_mm:
        gum_visibility_mm = float(gum_visibility_px / px_per_mm)

    return GumVisibilityMeasurement(
        patient_id=image_path.stem,
        gum_visibility_px=gum_visibility_px,
        gum_visibility_mm=gum_visibility_mm,
        px_per_mm=px_per_mm,
        per_region_px=per_region_px,
    )
