from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    output_path.write_bytes(cv2.imencode(".png", mask_uint8)[1].tobytes())


def _overlay_mask(
    image_path: Path,
    mask: np.ndarray,
    output_path: Path,
    mask_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = mask_color
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def _stub_mask(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    height, width = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, width // 6), max(1, height // 10))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def run_yolo_segmentation(
    image_path: Path,
    weights_path: Optional[Path],
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    max_det: int = 5,
    mask_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    use_stub: bool = False,
) -> Dict[str, Optional[str]]:
    _ensure_dir(output_dir)

    if use_stub:
        mask = _stub_mask(image_path)
    else:
        if weights_path is None:
            raise FileNotFoundError("weights_path is required for YOLO inference.")
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            save=False,
            verbose=False,
        )
        result = results[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return {
                "status": "no_mask",
                "mask_path": None,
                "overlay_path": None,
            }
        mask = result.masks.data.max(dim=0).values.cpu().numpy()

    mask_path = output_dir / "masks" / f"{image_path.stem}.png"
    _save_mask(mask, mask_path)
    overlay_path = output_dir / "overlays" / f"{image_path.stem}_overlay.png"
    _overlay_mask(image_path, mask.astype(np.uint8), overlay_path, mask_color=mask_color, alpha=alpha)

    return {
        "status": "ok",
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }
