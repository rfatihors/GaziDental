"""Class-separated binary masks from YOLO results, COCO polygons or PNG files (spec §3.1).

Every source yields the same ``ClassMasks`` at the *original image resolution*: one
gingiva mask and one lip mask, each the union of all instances of that class. No
"largest contour" selection and no merging of classes anywhere.

``from_yolo`` has only been exercised against a fake ``Results`` object (no weights
are available on the development machine); validation against real Ultralytics
output is part of Stage 6. Ultralytics itself is not imported here — the caller
runs ``model.predict(..., retina_masks=True)`` and passes the result in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ClassMasks:
    gingiva: np.ndarray
    lip: Optional[np.ndarray]
    n_gingiva_instances: int
    n_lip_instances: int
    source: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.gingiva.shape)  # type: ignore[return-value]

    def check_shape(self, image_shape: Tuple[int, int]) -> None:
        if tuple(self.gingiva.shape) != tuple(image_shape):
            raise ValueError(f"mask shape {self.gingiva.shape} != image shape {tuple(image_shape)}")
        if self.lip is not None and tuple(self.lip.shape) != tuple(image_shape):
            raise ValueError(f"lip mask shape {self.lip.shape} != image shape {tuple(image_shape)}")


def rasterize_polygons(polygons: Iterable[Sequence[float]], shape: Tuple[int, int]) -> np.ndarray:
    """Fill every polygon (flat ``[x0, y0, x1, y1, ...]`` or ``(N, 2)`` array) into one mask."""
    h, w = int(shape[0]), int(shape[1])
    canvas = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64)
        if pts.ndim == 1:
            if pts.size < 6:
                continue
            pts = pts.reshape(-1, 2)
        if len(pts) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas.astype(bool)


def from_coco(
    annotations: Iterable[Dict[str, Any]],
    image_id: int,
    shape: Tuple[int, int],
    category_ids: Dict[str, int],
) -> ClassMasks:
    """Union of all COCO polygon instances per class for one image."""
    gid, lid = int(category_ids["gingiva"]), int(category_ids["lip"])
    g_polys: List[Sequence[float]] = []
    l_polys: List[Sequence[float]] = []
    n_g = n_l = 0
    for a in annotations:
        if int(a["image_id"]) != int(image_id):
            continue
        seg = a["segmentation"]
        if not isinstance(seg, list):
            raise TypeError("RLE segmentations are not expected in this export")
        cid = int(a["category_id"])
        if cid == gid:
            n_g += 1
            g_polys.extend(seg)
        elif cid == lid:
            n_l += 1
            l_polys.extend(seg)
    gingiva = rasterize_polygons(g_polys, shape)
    lip = rasterize_polygons(l_polys, shape) if n_l else None
    return ClassMasks(gingiva, lip, n_g, n_l, source="coco")


def _read_png(path: Path, expected_shape: Optional[Tuple[int, int]]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if expected_shape is not None and tuple(m.shape) != tuple(expected_shape):
        raise ValueError(f"{path.name}: mask shape {m.shape} != expected {tuple(expected_shape)}")
    return m > 127


def from_png(gingiva_path: Path, lip_path: Optional[Path], expected_shape: Optional[Tuple[int, int]] = None) -> ClassMasks:
    """Load previously saved class masks (``*_gingiva.png`` / ``*_lip.png``)."""
    g = _read_png(Path(gingiva_path), expected_shape)
    lip = _read_png(Path(lip_path), expected_shape) if lip_path is not None and Path(lip_path).exists() else None
    from gsv4.measure.profile import connected_components

    return ClassMasks(g, lip, connected_components(g), connected_components(lip) if lip is not None else 0, source="png")


def save_masks(masks: ClassMasks, out_dir: Path, stem: str) -> Dict[str, Path]:
    """Write separate binary PNGs; never a combined mask."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"gingiva": out_dir / f"{stem}_gingiva.png"}
    cv2.imwrite(str(paths["gingiva"]), (masks.gingiva.astype(np.uint8) * 255))
    if masks.lip is not None:
        paths["lip"] = out_dir / f"{stem}_lip.png"
        cv2.imwrite(str(paths["lip"]), (masks.lip.astype(np.uint8) * 255))
    return paths


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def from_yolo(result: Any, class_names: Dict[str, str], image_shape: Tuple[int, int]) -> ClassMasks:
    """Class-aware masks from one Ultralytics ``Results`` object.

    Class ids come from ``result.boxes.cls`` and are resolved through ``result.names``;
    the model's names must contain the configured ``class_names`` (``diseti``/``dudak``)
    or a ``ValueError`` is raised. Masks are taken from ``masks.data`` when it already
    is at the original resolution (``retina_masks=True``); otherwise ``masks.xy``
    polygons (always in original coordinates) are rasterised. The final shape is
    asserted against ``image_shape``.
    """
    names = {int(k): str(v) for k, v in dict(result.names).items()}
    wanted = {role: name for role, name in class_names.items()}
    missing = [n for n in wanted.values() if n not in names.values()]
    if missing:
        raise ValueError(f"model class names {sorted(names.values())} do not contain {missing}")
    role_by_id = {cid: role for role, name in wanted.items() for cid, n in names.items() if n == name}
    h, w = int(image_shape[0]), int(image_shape[1])
    orig = getattr(result, "orig_shape", None)
    if orig is not None and tuple(int(v) for v in orig[:2]) != (h, w):
        raise ValueError(f"result.orig_shape {tuple(orig[:2])} != image shape {(h, w)}")
    gingiva = np.zeros((h, w), dtype=bool)
    lip = np.zeros((h, w), dtype=bool)
    n_g = n_l = 0
    source = "yolo:none"
    if result.masks is not None and result.boxes is not None and len(result.boxes.cls) > 0:
        cls = _to_numpy(result.boxes.cls).astype(int)
        data = _to_numpy(result.masks.data) if getattr(result.masks, "data", None) is not None else None
        use_data = data is not None and data.ndim == 3 and tuple(data.shape[1:]) == (h, w)
        source = "yolo:masks.data" if use_data else "yolo:masks.xy"
        for i, cid in enumerate(cls):
            role = role_by_id.get(int(cid))
            if role is None:
                continue
            if use_data:
                inst = data[i] > 0.5
            else:
                inst = rasterize_polygons([np.asarray(result.masks.xy[i])], (h, w))
            if role == "gingiva":
                gingiva |= inst
                n_g += 1
            else:
                lip |= inst
                n_l += 1
    cm = ClassMasks(gingiva, lip if n_l else None, n_g, n_l, source=source)
    cm.check_shape((h, w))
    return cm
