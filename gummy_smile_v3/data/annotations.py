from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    bbox: List[float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CocoDataset:
    images: Dict[int, CocoImage]
    annotations: List[CocoAnnotation]
    categories: Dict[int, CocoCategory]


@dataclass(frozen=True)
class SegmentationMetadata:
    image_id: int
    file_name: str
    width: int
    height: int
    category_id: int
    category_name: str
    segmentation: List[List[float]]
    bbox: List[float]
    area: float
    iscrowd: int


_REQUIRED_TOP_LEVEL_FIELDS = ("images", "annotations", "categories")
_REQUIRED_IMAGE_FIELDS = ("id", "file_name", "width", "height")
_REQUIRED_ANNOTATION_FIELDS = (
    "id",
    "image_id",
    "category_id",
    "segmentation",
    "bbox",
    "area",
    "iscrowd",
)
_REQUIRED_CATEGORY_FIELDS = ("id", "name")


def _validate_required_fields(
    item: Mapping[str, Any],
    required: Iterable[str],
    item_label: str,
) -> None:
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"Missing {item_label} field(s): {', '.join(missing)}")


def _ensure_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def load_coco_annotations(annotation_path: Path) -> CocoDataset:
    """Load COCO annotations from a _annotations.coco.json file with validation."""
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for field in _REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            raise ValueError(f"Missing top-level field '{field}' in COCO annotations")

    images = _ensure_list(data["images"], "images")
    annotations = _ensure_list(data["annotations"], "annotations")
    categories = _ensure_list(data["categories"], "categories")

    image_index: Dict[int, CocoImage] = {}
    for image in images:
        _validate_required_fields(image, _REQUIRED_IMAGE_FIELDS, "image")
        coco_image = CocoImage(
            id=int(image["id"]),
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
        image_index[coco_image.id] = coco_image

    category_index: Dict[int, CocoCategory] = {}
    for category in categories:
        _validate_required_fields(category, _REQUIRED_CATEGORY_FIELDS, "category")
        coco_category = CocoCategory(
            id=int(category["id"]),
            name=str(category["name"]),
        )
        category_index[coco_category.id] = coco_category

    annotation_items: List[CocoAnnotation] = []
    for annotation in annotations:
        _validate_required_fields(annotation, _REQUIRED_ANNOTATION_FIELDS, "annotation")
        segmentation = annotation["segmentation"]
        if segmentation is None:
            raise ValueError("Annotation segmentation cannot be null")
        segmentation_list: List[List[float]] = []
        for segment in _ensure_list(segmentation, "segmentation"):
            segment_list = [float(value) for value in _ensure_list(segment, "segment")]
            if not segment_list:
                raise ValueError("Segmentation segments cannot be empty")
            segmentation_list.append(segment_list)
        if not segmentation_list:
            raise ValueError("Segmentation segments cannot be empty")

        bbox = [float(value) for value in _ensure_list(annotation["bbox"], "bbox")]
        if len(bbox) != 4:
            raise ValueError("Annotation bbox must contain 4 values")

        coco_annotation = CocoAnnotation(
            id=int(annotation["id"]),
            image_id=int(annotation["image_id"]),
            category_id=int(annotation["category_id"]),
            segmentation=segmentation_list,
            bbox=bbox,
            area=float(annotation["area"]),
            iscrowd=int(annotation["iscrowd"]),
        )
        annotation_items.append(coco_annotation)

    return CocoDataset(
        images=image_index,
        annotations=annotation_items,
        categories=category_index,
    )


def build_segmentation_metadata(dataset: CocoDataset) -> List[SegmentationMetadata]:
    """Expose segmentation-centric metadata for preprocessing or training."""
    metadata: List[SegmentationMetadata] = []
    for annotation in dataset.annotations:
        image = dataset.images.get(annotation.image_id)
        if image is None:
            raise ValueError(
                "Annotation references missing image_id "
                f"{annotation.image_id}"
            )
        category = dataset.categories.get(annotation.category_id)
        if category is None:
            raise ValueError(
                "Annotation references missing category_id "
                f"{annotation.category_id}"
            )
        metadata.append(
            SegmentationMetadata(
                image_id=image.id,
                file_name=image.file_name,
                width=image.width,
                height=image.height,
                category_id=category.id,
                category_name=category.name,
                segmentation=annotation.segmentation,
                bbox=annotation.bbox,
                area=annotation.area,
                iscrowd=annotation.iscrowd,
            )
        )
    return metadata
