from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class YoloMeasurement:
    patient_id: str
    mean_mm: float
    mm_per_pixel: float
    measurements: Dict[str, float]


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


def measure_from_mask(
    image_path: Path,
    mask_path: Path,
    regions: int = 6,
    mm_per_pixel: float = 1.0,
) -> YoloMeasurement:
    binary_mask = _load_binary_mask(mask_path)
    contour = _find_main_contour(binary_mask)

    lip_line_y = float(contour[:, :, 1].min())
    x, y, w, h = cv2.boundingRect(contour)
    bounds = _split_regions((x, y, w, h), regions=regions)
    zenith_points = _find_zenith_points(contour, bounds)

    measurements: Dict[str, float] = {}
    for idx, (_, zenith_y) in enumerate(zenith_points, start=1):
        gummy_px = float(zenith_y - lip_line_y)
        measurements[f"mm_model_{idx}"] = round(gummy_px * mm_per_pixel, 3)

    measurement_values = list(measurements.values())
    mean_mm = float(np.mean(measurement_values)) if measurement_values else float("nan")

    return YoloMeasurement(
        patient_id=image_path.stem,
        mean_mm=mean_mm,
        mm_per_pixel=mm_per_pixel,
        measurements=measurements,
    )
