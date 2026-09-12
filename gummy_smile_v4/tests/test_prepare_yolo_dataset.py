"""COCO -> YOLO-seg conversion and list/yaml generation on a synthetic export."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from gsv4.train.prepare_yolo_dataset import build_dataset, coco_to_yolo_lines, nested_fractions, yolo_file_name
from tests.test_coco import COCO_CFG, _write


def test_coco_to_yolo_lines_normalised_and_class_mapped():
    anns = [
        {"image_id": 1, "category_id": 1, "segmentation": [[0, 0, 100, 0, 100, 50], [200, 0, 300, 0, 300, 50]]},
        {"image_id": 1, "category_id": 2, "segmentation": [[0, 0, 400, 0, 400, 100]]},
        {"image_id": 1, "category_id": 1, "segmentation": [[0, 0, 1, 1]]},   # degenerate
        {"image_id": 2, "category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]]},
    ]
    lines, dropped = coco_to_yolo_lines(anns, 1, 400, 100, {1: 0, 2: 1})
    assert len(lines) == 3 and dropped == 1
    assert lines[0].startswith("0 ") and lines[2].startswith("1 ")
    vals = np.asarray(lines[2].split()[1:], dtype=float)
    assert vals.max() <= 1.0 and vals.min() >= 0.0 and vals[2] == pytest.approx(1.0)
    assert yolo_file_name("high", "IMG_1_jpg", "IMG_1_jpg.rf.x.JPG") == "high__IMG_1_jpg.jpg"
    assert yolo_file_name("high", "111 (1)_jpg", "111 (1)_jpg.rf.x.jpg") == "high__111__1__jpg.jpg"


def test_nested_fractions_are_nested_and_stratified():
    df = pd.DataFrame({"yolo_name": [f"h{i}" for i in range(40)] + [f"n{i}" for i in range(80)], "group": ["high"] * 40 + ["normal"] * 80})
    n = nested_fractions(df, [0.25, 0.5, 0.75], seed=42)
    assert set(n[0.25]) <= set(n[0.5]) <= set(n[0.75])
    assert len(n[0.25]) == 30 and sum(x.startswith("h") for x in n[0.25]) == 10


def test_build_dataset_end_to_end(tmp_path):
    root = tmp_path / "coco"
    imgs_high = [(f"IMG_{i}_jpg.rf.a.jpg", 200, 100, 1, 1) for i in range(12)]
    imgs_norm = [(f"IMG_{i}_jpg.rf.b.jpg", 200, 100, 3, 1) for i in range(100, 110)]
    _write(root, "high", "train", imgs_high)
    _write(root, "high", "valid", [])
    _write(root, "normal", "train", imgs_norm)
    _write(root, "normal", "valid", [])
    man_dir = tmp_path / "manifest"; man_dir.mkdir()
    rows = []
    for fn, w, h, _, _ in imgs_high + imgs_norm:
        g = "high" if fn in [x[0] for x in imgs_high] else "normal"
        stem = fn.split(".rf.")[0]
        rows.append({"uid": f"{g}/{stem}", "image": stem, "group": g, "orig_split": "train", "keep": True, "patient_id": f"{g}/{stem}",
                     "has_reference_measurement": g == "high", "file_name": fn, "path": f"{g}/train/{fn}"})
    man = pd.DataFrame(rows)
    man.to_csv(man_dir / "dataset_manifest.csv", index=False)
    high = [r["uid"] for r in rows if r["group"] == "high"]
    norm = [r["uid"] for r in rows if r["group"] == "normal"]
    images = {u: ("train" if i < 8 else "valid" if i < 10 else "test") for i, u in enumerate(high)}
    images.update({u: ("train" if i < 7 else "valid" if i < 9 else "test") for i, u in enumerate(norm)})
    folds = {u: i % 2 for i, u in enumerate(high)}
    (man_dir / "splits.json").write_text(json.dumps({"images": images, "cv_folds": {"assignments": folds}}))
    cfg = {"_root": tmp_path, "seed": 42, "paths": {"coco_root": str(root), "manifest_dir": str(man_dir), "yolo_dataset": str(tmp_path / "yolo")},
           "coco": COCO_CFG, "class_names": {"gingiva": "diseti", "lip": "dudak"},
           "dataset": {"cv_folds": 2}, "yolo": {"learning_curve_fractions": [0.5, 1.0]}}
    s = build_dataset(cfg)
    ds = tmp_path / "yolo"
    assert s["n_images"] == 22 and s["n_images_without_labels"] == 0
    assert (ds / "images" / "all" / "high__IMG_0_jpg.jpg").is_symlink()
    assert (ds / "labels" / "all" / "high__IMG_0_jpg.txt").read_text().count("\n") == 2
    d = yaml.safe_load((ds / "data_main.yaml").read_text())
    assert d["names"] == {0: "diseti", 1: "dudak"} and d["train"] == "lists/main_train.txt" and "test" in d
    assert s["configs"]["main"] == {"train": 15, "valid": 4, "test": 3}
    assert s["configs"]["lc50"]["train"] == 8
    f0 = s["configs"]["fold0"]
    assert f0["heldout"] == 6
    held = set((ds / "lists" / "fold0_heldout.txt").read_text().splitlines())
    train0 = set((ds / "lists" / "fold0_train.txt").read_text().splitlines())
    valid0 = set((ds / "lists" / "fold0_valid.txt").read_text().splitlines())
    assert held.isdisjoint(train0) and held.isdisjoint(valid0) and train0.isdisjoint(valid0)
    assert all(n.startswith("images/all/") for n in held)
    # low/normal images are always in training for every fold
    assert all(f"images/all/normal__IMG_{i}_jpg.jpg" in train0 for i in range(100, 107))
