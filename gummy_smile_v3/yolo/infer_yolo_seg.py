from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from gummy_smile_v3.measurement.yolo_measurements import measure_from_mask


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _collect_images(images_dir: Path) -> List[Path]:
    return [path for path in images_dir.glob("**/*") if path.suffix.lower() in IMAGE_EXTENSIONS]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    output_path.write_bytes(cv2.imencode(".png", mask_uint8)[1].tobytes())


def run_inference(
    weights_path: Path,
    images_dir: Path,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    max_det: int = 5,
) -> Path:
    _ensure_dir(output_dir)
    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        save=False,
        verbose=False,
    )

    records: List[Dict[str, str | None]] = []
    for result, image_path in zip(results, image_paths):
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            records.append(
                {
                    "patient_id": image_path.stem,
                    "image_path": str(image_path),
                    "mask_path": None,
                    "status": "no_mask",
                }
            )
            continue
        combined_mask = result.masks.data.max(dim=0).values.cpu().numpy()
        mask_path = output_dir / "masks" / f"{image_path.stem}.png"
        _save_mask(combined_mask, mask_path)
        records.append(
            {
                "patient_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "status": "ok",
            }
        )

    df = pd.DataFrame(records)
    output_csv = output_dir / "yolo_predictions.csv"
    df.to_csv(output_csv, index=False)
    return output_csv


def predict_and_measure(
    weights_path: Path,
    images_dir: Path,
    output_dir: Path,
    measurement_output: Path,
    mm_per_pixel: float = 1.0,
    regions: int = 6,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    max_det: int = 5,
) -> Path:
    predictions_csv = run_inference(
        weights_path=weights_path,
        images_dir=images_dir,
        output_dir=output_dir,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
    )

    predictions_df = pd.read_csv(predictions_csv)
    measurement_rows: List[Dict[str, object]] = []
    for _, row in predictions_df.iterrows():
        image_path = Path(row["image_path"])
        mask_value = row.get("mask_path")
        status = row.get("status", "ok")
        if pd.isna(mask_value) or mask_value in ("", None):
            measurement_rows.append(
                {
                    "patient_id": image_path.stem,
                    "mean_mm": float("nan"),
                    "mm_per_pixel": mm_per_pixel,
                    "status": "no_mask",
                }
            )
            continue

        mask_path = Path(mask_value)
        measurement = measure_from_mask(
            image_path=image_path,
            mask_path=mask_path,
            regions=regions,
            mm_per_pixel=mm_per_pixel,
        )
        row_data: Dict[str, object] = {
            "patient_id": measurement.patient_id,
            "mean_mm": measurement.mean_mm,
            "mm_per_pixel": measurement.mm_per_pixel,
            "status": status,
        }
        row_data.update(measurement.measurements)
        measurement_rows.append(row_data)

    measurement_df = pd.DataFrame(measurement_rows)
    measurement_output.parent.mkdir(parents=True, exist_ok=True)
    measurement_df.to_csv(measurement_output, index=False)
    return measurement_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv11x-seg inference for gummy smile v3.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-output", type=Path, required=True)
    parser.add_argument("--mm-per-pixel", type=float, default=1.0)
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=5)
    args = parser.parse_args()

    predict_and_measure(
        weights_path=args.weights,
        images_dir=args.images,
        output_dir=args.output,
        measurement_output=args.measurement_output,
        mm_per_pixel=args.mm_per_pixel,
        regions=args.regions,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )
