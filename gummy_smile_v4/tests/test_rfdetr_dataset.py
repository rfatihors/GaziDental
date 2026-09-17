"""COCO type conversion and validation for the RF-DETR dataset builder.

The v3 export writes the width and height of every bbox as a formatted string
(``[655, 525, '1300.0000', '170.0000']``); our own pipeline never reads bbox, so RF-DETR's loader
was the first consumer to hit it. These tests pin the conversion and the validation that catch it.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("build_rfdetr_dataset", Path(__file__).resolve().parent.parent / "scripts" / "build_rfdetr_dataset.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _source_annotation():
    """An annotation shaped exactly like the v3 export: string bbox extent, int area, mixed polygon."""
    return {"id": 7, "image_id": 3, "category_id": 1, "bbox": [655, 525, "1300.0000", "170.0000"],
            "area": 221000, "iscrowd": 0,
            "segmentation": [[655, 650, 762.5, 617.5, 907.5, 575, 655, 650]]}


def test_convert_annotation_turns_the_string_bbox_into_floats():
    a = B.convert_annotation(_source_annotation(), ann_id=1, image_id=2, where="t")
    assert a["bbox"] == [655.0, 525.0, 1300.0, 170.0] and all(isinstance(v, float) for v in a["bbox"])
    assert isinstance(a["area"], float) and a["area"] == 221000.0
    assert (a["id"], a["image_id"], a["category_id"], a["iscrowd"]) == (1, 2, 1, 0)
    assert all(isinstance(t, int) for t in (a["id"], a["image_id"], a["category_id"], a["iscrowd"]))
    assert all(isinstance(v, float) for p in a["segmentation"] for v in p)


def test_convert_annotation_rejects_what_it_cannot_convert():
    for bad, match in (
        ({"bbox": [1, 2, 3]}, "4 values"),
        ({"bbox": [1, 2, "x", 4]}, "cannot convert"),
        ({"bbox": [1, 2, 0, 4]}, "non-positive"),
        ({"segmentation": []}, "segmentation is empty"),
        ({"segmentation": [[1, 2, 3, 4]]}, "at least 6"),
        ({"category_id": 1.5}, "not an integer"),
    ):
        a = {**_source_annotation(), **bad}
        with pytest.raises((ValueError, TypeError), match=match):
            B.convert_annotation(a, 1, 2, "t")


def test_convert_image_and_categories():
    im = B.convert_image({"width": "2698", "height": 1799.0}, 5, "high__IMG_1.jpg", {"uid": "high/IMG_1"}, "t")
    assert im == {"id": 5, "file_name": "high__IMG_1.jpg", "width": 2698, "height": 1799, "uid": "high/IMG_1"}
    cats = B.convert_categories([{"id": "0", "name": "dudak-diseti", "supercategory": "none"}, {"id": 1, "name": "diseti"}])
    assert cats[0]["id"] == 0 and cats[1] == {"id": 1, "name": "diseti", "supercategory": "none"}


def _split(n_images=2):
    images = [B.convert_image({"width": 100, "height": 80}, i + 1, f"im{i}.jpg", {}, "t") for i in range(n_images)]
    anns = [B.convert_annotation(_source_annotation(), i + 1, i + 1, "t") for i in range(n_images)]
    cats = B.convert_categories([{"id": 0, "name": "dudak-diseti"}, {"id": 1, "name": "diseti"}, {"id": 2, "name": "dudak"}])
    return images, anns, cats


def test_validate_split_accepts_a_well_formed_split_and_names_every_problem():
    images, anns, cats = _split()
    assert B.validate_split(images, anns, cats, "test") == []
    assert any("not in this split" in p for p in B.validate_split(images, [{**anns[0], "image_id": 99}], cats, "test"))
    assert any("category_id 9" in p for p in B.validate_split(images, [{**anns[0], "category_id": 9}], cats, "test"))
    assert any("not four floats" in p for p in B.validate_split(images, [{**anns[0], "bbox": [1, 2, "3", 4]}], cats, "test"))
    assert any("area is int" in p for p in B.validate_split(images, [{**anns[0], "area": 3}], cats, "test"))
    assert any("carry no annotation" in p for p in B.validate_split(images, [anns[0]], cats, "test"))
    assert any("duplicate annotation id" in p for p in B.validate_split(images, [anns[0], anns[0]], cats, "test"))


def test_check_categories_confirms_the_configured_ids_and_reports_the_dummy_class():
    _, _, cats = _split()
    r = B.check_categories(cats, {"diseti": 1, "dudak": 2})
    assert r["problems"] == [] and r["dummy_class_0"] == "dudak-diseti"
    bad = B.check_categories(cats, {"diseti": 2, "dudak": 1})
    assert len(bad["problems"]) == 2 and "should be 'diseti'" in bad["problems"][0]


def test_verify_written_reports_a_missing_split_and_skips_absent_packages(tmp_path):
    images, anns, cats = _split()
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / B.ANNOTATION_FILE).write_text(json.dumps({"categories": cats, "images": images, "annotations": anns}))
    r = B.verify_written(tmp_path)
    assert r["valid"] == "MISSING" and r["test"] == "MISSING"
    assert isinstance(r["train"], (dict, str))
