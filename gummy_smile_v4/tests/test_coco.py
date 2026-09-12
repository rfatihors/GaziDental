"""COCO inventory on a synthetic export mimicking the Roboflow layout."""
import json

from gsv4.io.coco import coco_inventory, load_annotations

COCO_CFG = {
    "groups": ["high", "normal"],
    "splits": ["train", "valid"],
    "annotation_file": "_annotations.coco.json",
    "category_ids": {"gingiva": 1, "lip": 2},
    "reference_frame": {"width": 2698, "height": 1799, "tolerance_px": 2},
}


def _write(root, group, split, images):
    d = root / group / split
    d.mkdir(parents=True)
    cats = [
        {"id": 0, "name": "dudak-diseti", "supercategory": "none"},
        {"id": 1, "name": "diseti", "supercategory": "dudak-diseti"},
        {"id": 2, "name": "dudak", "supercategory": "dudak-diseti"},
    ]
    imgs, anns = [], []
    aid = 0
    for i, (fn, w, h, n_g, n_l) in enumerate(images):
        imgs.append({"id": i, "file_name": fn, "width": w, "height": h})
        (d / fn).write_bytes(b"")
        for cat, n in ((1, n_g), (2, n_l)):
            for _ in range(n):
                anns.append({"id": aid, "image_id": i, "category_id": cat, "segmentation": [[0, 0, 1, 0, 1, 1]], "area": 1, "bbox": [0, 0, 1, 1], "iscrowd": 0})
                aid += 1
    (d / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))


def test_inventory(tmp_path):
    _write(tmp_path, "high", "train", [("IMG_2544-_jpeg.rf.a.jpeg", 2698, 1799, 1, 1), ("IMG_2633_jpg.rf.b.jpg", 2593, 1729, 1, 1)])
    _write(tmp_path, "high", "valid", [("IMG_2633-_jpg.rf.c.jpg", 2699, 1799, 1, 1)])
    _write(tmp_path, "normal", "train", [("23-IMG_4080_JPG.rf.d.jpg", 2700, 1800, 4, 1)])
    _write(tmp_path, "normal", "valid", [])
    inv = coco_inventory(tmp_path, COCO_CFG)
    assert len(inv) == 4
    assert set(inv.columns) >= {"image", "file_name", "group", "split", "width", "height", "n_gingiva", "n_lip", "key", "base", "dot", "age_prefix", "path", "frame_ok"}
    row = inv.set_index("image").loc["IMG_2544-_jpeg"]
    assert row["key"] == "img2544~" and row["group"] == "high" and row["split"] == "train"
    assert bool(row["frame_ok"]) is True
    assert row["path"] == "high/train/IMG_2544-_jpeg.rf.a.jpeg"
    assert bool(inv.set_index("image").loc["IMG_2633_jpg"]["frame_ok"]) is False
    assert bool(inv.set_index("image").loc["23-IMG_4080_JPG"]["frame_ok"]) is True  # within 2 px
    assert inv.set_index("image").loc["23-IMG_4080_JPG"]["n_gingiva"] == 4
    assert inv.set_index("image").loc["23-IMG_4080_JPG"]["age_prefix"] == 23
    assert inv["uid"].is_unique and inv.set_index("image").loc["IMG_2544-_jpeg"]["uid"] == "high/IMG_2544-_jpeg"
    d = load_annotations(tmp_path, "high", "train", COCO_CFG)
    assert len(d["images"]) == 2
