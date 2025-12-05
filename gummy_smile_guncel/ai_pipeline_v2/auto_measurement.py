"""
Automatic geometric measurement module for gummy smile severity pipeline (v2).

This module processes DeepLab gingival masks alongside original photographs to
extract zenith points, estimate lip reference lines, calibrate pixel-to-mm
scales using a periodontal probe, and compute six regional gingival display
measurements. Results are merged with cleaned manual measurements and saved
inside the ai_pipeline_v2 workspace.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class MeasurementResult:
    """Container for automatic measurement outputs."""

    case_id: str
    mm_per_pixel: float
    lip_line_y: float
    zenith_points: List[Tuple[int, int]]
    measurements_mm: Dict[str, float]


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs: Sequence[Path] = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask could not be read: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def _find_main_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in gingival mask.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def _estimate_lip_line_y(image: np.ndarray, contour: np.ndarray) -> float:
    topmost = contour[:, :, 1].min()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    _, edges = cv2.threshold(abs_sobel_y, 50, 255, cv2.THRESH_BINARY)
    search_band = edges[max(topmost - 40, 0): topmost + 5, :]
    edge_points = np.argwhere(search_band == 255)
    if edge_points.size == 0:
        return float(topmost)
    lip_y_local = edge_points[:, 0].min()
    return float(max(topmost - 40, 0) + lip_y_local)


def _split_regions(bbox: Tuple[int, int, int, int], regions: int = 6) -> List[Tuple[int, int]]:
    x, y, w, h = bbox
    step = w / regions
    bounds: List[Tuple[int, int]] = []
    for i in range(regions):
        start = int(x + i * step)
        end = int(x + (i + 1) * step)
        bounds.append((start, end))
    return bounds


def _find_zenith_points(contour: np.ndarray, region_bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    zeniths: List[Tuple[int, int]] = []
    contour_points = contour.reshape(-1, 2)
    for start, end in region_bounds:
        region_pts = contour_points[(contour_points[:, 0] >= start) & (contour_points[:, 0] < end)]
        if region_pts.size == 0:
            zeniths.append((int((start + end) / 2), int(contour_points[:, 1].min())))
            continue
        top_idx = np.argmin(region_pts[:, 1])
        top_pt = region_pts[top_idx]
        zeniths.append((int(top_pt[0]), int(top_pt[1])))
    return zeniths


def _estimate_probe_scale(image: np.ndarray, expected_mm: float = 10.0) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    probe_length_px = 0.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, 1) / max(h, 1)
        area = cv2.contourArea(cnt)
        if 0.05 < aspect < 0.4 and area > 50:
            probe_length_px = max(probe_length_px, float(h))
    if probe_length_px == 0:
        raise ValueError("Unable to detect periodontal probe for scaling.")
    return expected_mm / probe_length_px


def _compute_measurements(image_path: Path, mask_path: Path) -> MeasurementResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    binary_mask = _load_binary_mask(mask_path)
    cleaned_mask = _clean_mask(binary_mask)
    contour = _find_main_contour(cleaned_mask)

    lip_line_y = _estimate_lip_line_y(image, contour)
    x, y, w, h = cv2.boundingRect(contour)
    bounds = _split_regions((x, y, w, h), regions=6)
    zenith_points = _find_zenith_points(contour, bounds)
    mm_per_pixel = _estimate_probe_scale(image)

    measurements: Dict[str, float] = {}
    for idx, (_, zenith_y) in enumerate(zenith_points, start=1):
        gummy_px = float(zenith_y - lip_line_y)
        measurements[f"mm_model_{idx}"] = round(gummy_px * mm_per_pixel, 3)

    case_id = image_path.stem
    return MeasurementResult(
        case_id=case_id,
        mm_per_pixel=mm_per_pixel,
        lip_line_y=lip_line_y,
        zenith_points=zenith_points,
        measurements_mm=measurements,
    )


def _pair_images_and_masks(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    mask_lookup = {path.stem: path for path in masks_dir.glob("*.png")}
    for image_path in images_dir.glob("*.png"):
        stem = image_path.stem
        if stem in mask_lookup:
            pairs.append((image_path, mask_lookup[stem]))
    return pairs


def _results_to_dataframe(results: List[MeasurementResult]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for res in results:
        record: Dict[str, float] = {"case_id": res.case_id, "mm_per_pixel": res.mm_per_pixel}
        record.update(res.measurements_mm)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _guess_merge_key(clean_df: pd.DataFrame) -> Optional[str]:
    priority_cols = ["case_id", "patient_id", "image_id", "filename", "image"]
    for col in priority_cols:
        if col in clean_df.columns:
            return col
    for col in clean_df.columns:
        if any(token in col.lower() for token in ["case", "patient", "image", "file"]):
            return col
    return None


def run_auto_measurement(
    images_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
    cleaned_measurements_path: Optional[Path] = None,
) -> Path:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    images_dir = images_dir or base_dir / "data" / "images"
    masks_dir = masks_dir or base_dir / "data" / "masks"
    cleaned_measurements_path = cleaned_measurements_path or base_dir / "data" / "clean_measurements.csv"
    output_path = base_dir / "data" / "auto_measurements.csv"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Images or masks directory is missing for auto measurement.")

    pairs = _pair_images_and_masks(images_dir, masks_dir)
    if not pairs:
        raise ValueError("No matching image-mask pairs found.")

    results: List[MeasurementResult] = []
    for image_path, mask_path in pairs:
        results.append(_compute_measurements(image_path, mask_path))

    auto_df = _results_to_dataframe(results)

    if cleaned_measurements_path.exists():
        clean_df = pd.read_csv(cleaned_measurements_path)
        merge_key = _guess_merge_key(clean_df)
        if merge_key and merge_key in clean_df.columns:
            clean_df[merge_key] = clean_df[merge_key].astype(str)
            auto_df["case_id"] = auto_df["case_id"].astype(str)
            merged_df = clean_df.merge(auto_df, left_on=merge_key, right_on="case_id", how="left")
        else:
            merged_df = pd.concat([clean_df, auto_df], axis=1)
    else:
        merged_df = auto_df

    merged_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    try:
        output_file = run_auto_measurement()
        print(f"Automatic measurements saved to: {output_file}")
        print("Sample usage: python auto_measurement.py")
    except Exception as exc:  # noqa: BLE001 - surfaced for CLI visibility
        print(f"Auto measurement failed: {exc}")
