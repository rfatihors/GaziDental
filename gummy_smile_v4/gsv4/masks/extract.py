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
    # Predicted instances whose class id mapped to neither role and were therefore left out of both
    # masks. A silently dropped instance is how a mismatched label space hides itself, so the count
    # is carried out of the extractor and written into every prediction table.
    n_ignored_instances: int = 0

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


def role_by_class_id(id_to_name: Dict[int, str], class_names: Dict[str, str]) -> Dict[int, str]:
    """``class id -> role`` ("gingiva" / "lip") through the model's own class names.

    ``id_to_name`` is the predictor's class list; ``class_names`` the configured role -> name
    map (``diseti`` / ``dudak``). Every configured name must be present, or the prediction
    came from a model that cannot answer this question and a ``ValueError`` is raised.
    """
    names = {int(k): str(v) for k, v in dict(id_to_name).items()}
    missing = [n for n in class_names.values() if n not in names.values()]
    if missing:
        raise ValueError(f"model class names {sorted(names.values())} do not contain {missing}")
    return {cid: role for role, name in class_names.items() for cid, n in names.items() if n == name}


def from_instances(instance_masks: Any, class_ids: Sequence[int], id_to_name: Dict[int, str],
                   class_names: Dict[str, str], image_shape: Tuple[int, int], source: str = "instances",
                   strict: bool = False) -> ClassMasks:
    """Union the per-instance masks of one image into one mask per class.

    This is the shared core of every predictor adapter: any model that can produce a stack of
    instance masks at the original image resolution plus a class id per instance can be measured
    by this pipeline. ``instance_masks`` is indexable per instance and each element must be, or
    convert to, an ``(H, W)`` array matching ``image_shape``; values above 0.5 are foreground.

    An instance whose class id maps to neither role is left out of both masks and counted in
    ``n_ignored_instances``. With ``strict=True`` it raises instead. Pass ``strict=True`` whenever
    the predictor is known to emit only the two roles: a dropped instance then means the class ids
    do not mean what the caller thinks, which is otherwise invisible in the output.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    role_by_id = role_by_class_id(id_to_name, class_names)
    gingiva = np.zeros((h, w), dtype=bool)
    lip = np.zeros((h, w), dtype=bool)
    n_g = n_l = 0
    ignored: List[int] = []
    for i, cid in enumerate(np.asarray(class_ids).astype(int)):
        role = role_by_id.get(int(cid))
        if role is None:
            ignored.append(int(cid))
            continue
        inst = _to_numpy(instance_masks[i])
        if inst.shape != (h, w):
            raise ValueError(f"instance {i} mask shape {inst.shape} != image shape {(h, w)}")
        inst = inst > 0.5
        if role == "gingiva":
            gingiva |= inst
            n_g += 1
        else:
            lip |= inst
            n_l += 1
    if ignored and strict:
        raise ValueError(
            f"{len(ignored)} predicted instance(s) have a class id that maps to no role: ids {sorted(set(ignored))}. "
            f"The map given was {dict(sorted((int(k), str(v)) for k, v in dict(id_to_name).items()))} and the roles "
            f"wanted are {class_names}. This is what a label space mismatched with the predictor's own looks like; "
            "pass the predictor's own class names, not the dataset's category list.")
    cm = ClassMasks(gingiva, lip if n_l else None, n_g, n_l, source=source, n_ignored_instances=len(ignored))
    cm.check_shape((h, w))
    return cm


def from_yolo(result: Any, class_names: Dict[str, str], image_shape: Tuple[int, int], strict: bool = False) -> ClassMasks:
    """Class-aware masks from one Ultralytics ``Results`` object.

    Class ids come from ``result.boxes.cls`` and are resolved through ``result.names``;
    the model's names must contain the configured ``class_names`` (``diseti``/``dudak``)
    or a ``ValueError`` is raised. Masks are taken from ``masks.data`` when it already
    is at the original resolution (``retina_masks=True``); otherwise ``masks.xy``
    polygons (always in original coordinates) are rasterised, and the union per class is
    formed by ``from_instances``. The final shape is asserted against ``image_shape``.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    role_by_class_id(result.names, class_names)   # fail early on a model with the wrong classes
    orig = getattr(result, "orig_shape", None)
    if orig is not None and tuple(int(v) for v in orig[:2]) != (h, w):
        raise ValueError(f"result.orig_shape {tuple(orig[:2])} != image shape {(h, w)}")
    if result.masks is None or result.boxes is None or len(result.boxes.cls) == 0:
        cm = ClassMasks(np.zeros((h, w), dtype=bool), None, 0, 0, source="yolo:none")
        cm.check_shape((h, w))
        return cm
    cls = _to_numpy(result.boxes.cls).astype(int)
    data = _to_numpy(result.masks.data) if getattr(result.masks, "data", None) is not None else None
    use_data = data is not None and data.ndim == 3 and tuple(data.shape[1:]) == (h, w)
    if use_data:
        instances: Any = data
    else:
        instances = [rasterize_polygons([np.asarray(result.masks.xy[i])], (h, w)) for i in range(len(cls))]
    return from_instances(instances, cls, result.names, class_names, (h, w),
                          source="yolo:masks.data" if use_data else "yolo:masks.xy", strict=strict)


def from_detections(detections: Any, id_to_name: Dict[int, str], class_names: Dict[str, str],
                    image_shape: Tuple[int, int], source: str = "detections", strict: bool = False) -> ClassMasks:
    """Class masks from a ``supervision.Detections``-shaped object (RF-DETR and anything else
    that follows that convention): ``.mask`` is an ``(K, H, W)`` boolean array at the original
    image resolution and ``.class_id`` holds one class id per instance.

    RF-DETR upsamples its masks to the image size by default
    (``PostProcess.upsample_masks_to_image_size``); a predictor left at mask-head resolution
    would fail the shape check in ``from_instances`` rather than be silently resized here.

    ``id_to_name`` must be the predictor's OWN class list, not the category list of the dataset it
    was trained on. RF-DETR renumbers its label space: it drops unannotated grouping categories and
    assigns contiguous indices to the rest, so a Roboflow export whose categories are
    ``0 grouping, 1 diseti, 2 dudak`` yields a model that emits ``0`` for gingiva and ``1`` for lip.
    Reading those ids through the dataset's own table maps gingiva onto the grouping name (dropped)
    and lip onto gingiva. Use ``dict(enumerate(model.class_names))`` and pass ``strict=True``.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    masks = getattr(detections, "mask", None)
    class_ids = getattr(detections, "class_id", None)
    if masks is None or class_ids is None or len(np.asarray(class_ids)) == 0:
        cm = ClassMasks(np.zeros((h, w), dtype=bool), None, 0, 0, source=f"{source}:none")
        cm.check_shape((h, w))
        return cm
    return from_instances(masks, class_ids, id_to_name, class_names, (h, w), source=source, strict=strict)
