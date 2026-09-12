"""Read-only inventory of the v3 COCO export (Roboflow layout: group/split/_annotations.coco.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from gsv4.io.naming import coco_stem, normalize_coco_name


def load_annotations(coco_root: Path, group: str, split: str, coco_cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(coco_root) / group / split / coco_cfg["annotation_file"]
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def coco_inventory(coco_root: Path, coco_cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per image over all groups and splits.

    ``image`` is the Roboflow-free stem (the id used in every input CSV); it is unique
    within a group but not across groups (iPhone numbering repeats), so ``uid`` =
    ``group/stem`` is the identity used everywhere downstream.
    """
    coco_root = Path(coco_root)
    gid = int(coco_cfg["category_ids"]["gingiva"])
    lid = int(coco_cfg["category_ids"]["lip"])
    frame = coco_cfg["reference_frame"]
    tol = int(frame.get("tolerance_px", 2))
    records = []
    for group in coco_cfg["groups"]:
        for split in coco_cfg["splits"]:
            d = load_annotations(coco_root, group, split, coco_cfg)
            counts: Dict[int, Dict[int, int]] = {}
            for a in d["annotations"]:
                c = counts.setdefault(int(a["image_id"]), {gid: 0, lid: 0})
                c[int(a["category_id"])] = c.get(int(a["category_id"]), 0) + 1
            for im in d["images"]:
                fn = im["file_name"]
                name = normalize_coco_name(fn)
                w, h = int(im["width"]), int(im["height"])
                cnt = counts.get(int(im["id"]), {})
                records.append({
                    "uid": f"{group}/{coco_stem(fn)}", "image": coco_stem(fn), "file_name": fn, "group": group, "split": split,
                    "coco_image_id": int(im["id"]), "width": w, "height": h,
                    "n_gingiva": int(cnt.get(gid, 0)), "n_lip": int(cnt.get(lid, 0)),
                    "key": name.key, "base": name.base, "dot": name.dot, "age_prefix": name.age_prefix,
                    "path": f"{group}/{split}/{fn}",   # relative to paths.coco_root
                    "frame_ok": abs(w - int(frame["width"])) <= tol and abs(h - int(frame["height"])) <= tol,
                })
    inv = pd.DataFrame(records)
    if not inv["uid"].is_unique:
        dup = inv.loc[inv["uid"].duplicated(keep=False), "uid"].unique().tolist()
        raise ValueError(f"COCO image stems are not unique within a group: {dup[:10]}")
    return inv
