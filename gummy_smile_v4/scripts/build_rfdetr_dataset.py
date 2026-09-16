#!/usr/bin/env python
"""Build the COCO dataset RF-DETR expects, from our participant-level partition.

    python scripts/build_rfdetr_dataset.py [--out data/rfdetr_dataset] [--dry-run]

RF-DETR reads a directory with ``train/``, ``valid/`` and ``test/`` subdirectories, each holding
the images and one ``_annotations.coco.json`` (the Roboflow export layout). Our source export is
split by smile-line group and by the ORIGINAL splits, so it cannot be used directly: this script
re-assembles it under the cleaned, participant-level partition used by every other stage, so that
the architecture comparison sees exactly the same images in exactly the same roles
(outputs/08_architecture/PROTOCOL.md §2).

Images are symlinked, never copied. Category ids are carried over unchanged from the source export
(1 = diseti, 2 = dudak) together with the supercategory row, so that a class id in a prediction
means the same thing as in our YOLO runs. File names are the sanitised ``yolo_name`` used
everywhere else, so a prediction can be traced back to a uid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.io.coco import load_annotations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/rfdetr_dataset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    out = resolve(cfg, args.out)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    man = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    kept = man[man["keep"]].copy()
    idx = pd.read_csv(ds / "yolo_index.csv").set_index("uid")
    kept["yolo_name"] = kept["uid"].map(idx["yolo_name"])
    missing = kept[kept["yolo_name"].isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} kept images are not in yolo_index.csv: {list(missing['uid'])[:5]}")

    cats: Any = None
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"images": 0, "annotations": 0})
    for split in ("train", "valid", "test"):
        part = kept[kept["split"] == split]
        images, annotations = [], []
        next_img, next_ann = 1, 1
        cache: Dict[tuple, Dict[str, Any]] = {}
        for r in part.itertuples(index=False):
            key = (r.group, r.orig_split)
            if key not in cache:
                cache[key] = load_annotations(coco_root, r.group, r.orig_split, cfg["coco"])
            src = cache[key]
            if cats is None:
                cats = src["categories"]
            im = next((i for i in src["images"] if i["file_name"] == r.file_name), None)
            if im is None:
                raise SystemExit(f"{r.uid}: {r.file_name} not found in {coco_root / r.group / r.orig_split}")
            img_id = next_img
            next_img += 1
            images.append({"id": img_id, "file_name": r.yolo_name, "width": int(im["width"]), "height": int(im["height"]),
                           "uid": r.uid, "image": r.image})   # extra metadata; COCO readers ignore unknown keys
            for a in src["annotations"]:
                if int(a["image_id"]) != int(im["id"]):
                    continue
                annotations.append({**a, "id": next_ann, "image_id": img_id})
                next_ann += 1
            if not args.dry_run:
                link = out / split / r.yolo_name
                link.parent.mkdir(parents=True, exist_ok=True)
                target = (coco_root / r.group / r.orig_split / r.file_name).resolve()
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(target, link)
        counts[split] = {"images": len(images), "annotations": len(annotations)}
        if not args.dry_run:
            (out / split).mkdir(parents=True, exist_ok=True)
            (out / split / cfg["coco"]["annotation_file"]).write_text(
                json.dumps({"info": {"description": f"gummy_smile_v4 {split} split, participant level"},
                            "licenses": [], "categories": cats, "images": images, "annotations": annotations}), encoding="utf-8")
    total = sum(c["images"] for c in counts.values())
    print(f"[rfdetr-dataset] {'DRY RUN — ' if args.dry_run else ''}{out}")
    for split, c in counts.items():
        print(f"  {split:6s} images {c['images']:5d}  annotations {c['annotations']:6d}")
    print(f"  total images {total} (must equal the kept dataset: {len(kept)})")
    if total != len(kept):
        raise SystemExit("image count does not match the manifest")
    if cats:
        print(f"  categories: {[(c['id'], c['name']) for c in cats]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
