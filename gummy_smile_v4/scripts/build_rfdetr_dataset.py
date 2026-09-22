#!/usr/bin/env python
"""Build the COCO dataset RF-DETR expects, from our participant-level partition.

    python scripts/build_rfdetr_dataset.py [--variant main] [--dry-run]
    python scripts/build_rfdetr_dataset.py --variant fold0        # a cross-validation fold
    python scripts/build_rfdetr_dataset.py --variant lc25         # a learning-curve subset
    python scripts/build_rfdetr_dataset.py --verify-only --out data/rfdetr_dataset
    .venv-rfdetr/bin/python scripts/build_rfdetr_dataset.py --verify-only    # in the RF-DETR env

RF-DETR reads a directory with ``train/``, ``valid/`` and ``test/`` subdirectories, each holding the
images and one ``_annotations.coco.json`` (the Roboflow export layout). Our source export is split by
smile-line group and by the ORIGINAL splits, so it cannot be used directly: this script re-assembles
it under the cleaned, participant-level partition used by every other stage, so that the architecture
comparison sees the same images in the same roles (outputs/08_architecture/PROTOCOL.md §2).

Images are symlinked, never copied.

**Types are converted explicitly, never copied through.** The v3 export writes the width and height
of every ``bbox`` as formatted strings (``[655, 525, '1300.0000', '170.0000']``) in all 5,256 of its
annotations, and ``area`` and the polygon coordinates as a mix of int and float. Our own pipeline
never reads ``bbox`` (it rasterises ``segmentation``), which is why this went unnoticed until
RF-DETR's loader did ``torch.as_tensor(bbox, dtype=torch.float32)`` on it and raised
``TypeError: must be real number, not str``. Every numeric field written here is converted to the
type COCO specifies, and the result is validated before and after writing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")

# Which images each split holds, per dataset variant. "manifest" means the split column of the
# dataset manifest; anything else names a list file under the YOLO dataset's lists/ directory, so
# that RF-DETR trains on exactly the images the YOLO runs did (outputs/09_final_rfdetr/PLAN.md §2).
VARIANTS = {
    "main": {"train": "manifest:train", "valid": "manifest:valid", "test": "manifest:test"},
    **{f"fold{k}": {"train": f"lists/fold{k}_train.txt", "valid": f"lists/fold{k}_valid.txt",
                    "test": f"lists/fold{k}_heldout.txt"} for k in range(5)},
    **{f"lc{n}": {"train": f"lists/lc{n}_train.txt", "valid": "lists/main_valid.txt",
                  "test": "lists/main_test.txt"} for n in (25, 50, 75)},
}


def members(source: str, kept, lists_root: Path) -> list:
    """The uids of one split of one variant.

    ``manifest:<split>`` reads the manifest's own split column; a path reads a YOLO list file and
    resolves each line's basename back to a uid, so the two dataset formats hold the same images.
    """
    if source.startswith("manifest:"):
        want = source.split(":", 1)[1]
        return list(kept.loc[kept["split"] == want, "uid"])
    path = lists_root / source
    if not path.exists():
        raise SystemExit(f"list not found: {path} (run gsv4.train.prepare_yolo_dataset first)")
    by_name = dict(zip(kept["yolo_name"], kept["uid"]))
    uids, unknown = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = Path(line.strip()).name
        (uids.append(by_name[name]) if name in by_name else unknown.append(name))
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} image(s) are not in the kept manifest: {unknown[:5]}")
    return uids


# ----------------------------------------------------------------- conversion (stdlib only)
def to_float(value: Any, where: str) -> float:
    """A COCO numeric field as float. Accepts the formatted strings the v3 export contains."""
    if isinstance(value, bool):
        raise TypeError(f"{where}: bool is not a number ({value!r})")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{where}: cannot convert {value!r} ({type(value).__name__}) to float") from exc


def to_int(value: Any, where: str) -> int:
    f = to_float(value, where)
    if f != int(f):
        raise ValueError(f"{where}: {value!r} is not an integer")
    return int(f)


def convert_annotation(a: Dict[str, Any], ann_id: int, image_id: int, where: str) -> Dict[str, Any]:
    """One annotation with every field at the type COCO specifies. Nothing is copied through."""
    bbox = a.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{where}: bbox must have 4 values, got {bbox!r}")
    seg_in = a.get("segmentation")
    if not isinstance(seg_in, list) or not seg_in:
        raise ValueError(f"{where}: segmentation is empty or not a list of polygons ({type(seg_in).__name__})")
    segmentation: List[List[float]] = []
    for j, poly in enumerate(seg_in):
        if not isinstance(poly, (list, tuple)) or len(poly) < 6 or len(poly) % 2:
            raise ValueError(f"{where}: polygon {j} needs an even number of at least 6 coordinates, got {len(poly) if hasattr(poly, '__len__') else poly!r}")
        segmentation.append([to_float(v, f"{where}.segmentation[{j}][{i}]") for i, v in enumerate(poly)])
    out = {
        "id": int(ann_id),
        "image_id": int(image_id),
        "category_id": to_int(a.get("category_id"), f"{where}.category_id"),
        "bbox": [to_float(v, f"{where}.bbox[{i}]") for i, v in enumerate(bbox)],
        "area": to_float(a.get("area", 0.0), f"{where}.area"),
        "segmentation": segmentation,
        "iscrowd": to_int(a.get("iscrowd", 0), f"{where}.iscrowd"),
    }
    if out["bbox"][2] <= 0 or out["bbox"][3] <= 0:
        raise ValueError(f"{where}: bbox has a non-positive width or height: {out['bbox']}")
    return out


def convert_image(im: Dict[str, Any], image_id: int, file_name: str, extra: Dict[str, Any], where: str) -> Dict[str, Any]:
    return {"id": int(image_id), "file_name": str(file_name),
            "width": to_int(im.get("width"), f"{where}.width"), "height": to_int(im.get("height"), f"{where}.height"), **extra}


def convert_categories(cats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": to_int(c["id"], f"category[{i}].id"), "name": str(c["name"]),
             "supercategory": str(c.get("supercategory", "none"))} for i, c in enumerate(cats)]


# ----------------------------------------------------------------- validation (stdlib only)
def validate_split(images: Sequence[Dict[str, Any]], annotations: Sequence[Dict[str, Any]],
                   categories: Sequence[Dict[str, Any]], split: str) -> List[str]:
    """Every problem found, as a sentence. An empty list means the split is well formed."""
    problems: List[str] = []
    ids = {int(i["id"]) for i in images}
    if len(ids) != len(images):
        problems.append(f"{split}: image ids are not unique")
    cat_ids = {int(c["id"]) for c in categories}
    ann_ids = set()
    for a in annotations:
        w = f"{split}: annotation {a.get('id')}"
        if int(a["id"]) in ann_ids:
            problems.append(f"{w}: duplicate annotation id")
        ann_ids.add(int(a["id"]))
        if int(a["image_id"]) not in ids:
            problems.append(f"{w}: image_id {a['image_id']} is not in this split")
        if int(a["category_id"]) not in cat_ids:
            problems.append(f"{w}: category_id {a['category_id']} is not in the category list")
        if len(a["bbox"]) != 4 or not all(isinstance(v, float) for v in a["bbox"]):
            problems.append(f"{w}: bbox is not four floats: {a['bbox']!r}")
        if not isinstance(a["area"], float):
            problems.append(f"{w}: area is {type(a['area']).__name__}, not float")
        if not isinstance(a["iscrowd"], int) or not isinstance(a["category_id"], int) or not isinstance(a["image_id"], int):
            problems.append(f"{w}: iscrowd, category_id and image_id must be int")
        if not a["segmentation"] or not all(p and all(isinstance(v, float) for v in p) for p in a["segmentation"]):
            problems.append(f"{w}: segmentation must be non-empty polygons of floats")
    without = ids - {int(a["image_id"]) for a in annotations}
    if without:
        problems.append(f"{split}: {len(without)} image(s) carry no annotation")
    return problems


def check_categories(categories: Sequence[Dict[str, Any]], expected: Dict[str, int]) -> Dict[str, Any]:
    """Confirm that the configured class ids survived, and describe the zeroth dummy class.

    A Roboflow export carries a supercategory row at id 0 that is never used by an annotation.
    RF-DETR builds its own contiguous label space from the category list, so a shifted or missing
    id would silently relabel the classes; the names are therefore checked, not assumed.
    """
    by_id = {int(c["id"]): str(c["name"]) for c in categories}
    problems = [f"category id {cid} should be {name!r} but is {by_id.get(cid)!r}"
                for name, cid in expected.items() if by_id.get(cid) != name]
    return {"categories": by_id, "problems": problems,
            "dummy_class_0": by_id.get(0), "expected": expected}


# ----------------------------------------------------------------- verification of what was written
def verify_written(out: Path, n_samples: int = 5) -> Dict[str, Any]:
    """Re-open each written split the way RF-DETR will.

    ``pycocotools.COCO`` proves the file parses as COCO; the tensor step reproduces exactly the
    operation that failed before (rfdetr/datasets/coco.py: ``torch.as_tensor(box_values,
    dtype=torch.float32)``), so a regression of the string-bbox kind is caught here and not on the
    workstation. Both checks are skipped, loudly, when the package is not installed in the
    interpreter running this script.
    """
    report: Dict[str, Any] = {}
    try:
        from pycocotools.coco import COCO
    except ImportError:
        report["pycocotools"] = "not installed in this interpreter — run --verify-only with .venv-rfdetr/bin/python"
        COCO = None  # type: ignore[assignment]
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
        report["torch"] = "not installed in this interpreter — the tensor check was skipped"
    for split in SPLITS:
        f = out / split / ANNOTATION_FILE
        if not f.exists():
            report[split] = "MISSING"
            continue
        entry: Dict[str, Any] = {}
        if COCO is not None:
            import contextlib
            import io as _io

            with contextlib.redirect_stdout(_io.StringIO()):
                coco = COCO(str(f))
            img_ids = sorted(coco.imgs)[:n_samples]
            entry["pycocotools"] = f"ok, {len(coco.imgs)} images, {len(coco.anns)} annotations"
            if torch is not None:
                for iid in img_ids:
                    anno = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid)) if not a.get("iscrowd", 0)]
                    boxes = torch.as_tensor([a["bbox"] for a in anno], dtype=torch.float32).reshape(-1, 4)
                    boxes[:, 2:] += boxes[:, :2]
                entry["tensor_check"] = f"ok on the first {len(img_ids)} images (rfdetr/datasets/coco.py bbox step)"
        report[split] = entry or "checks skipped"
    try:
        import rfdetr.datasets.coco as _rf  # noqa: F401

        report["rfdetr_dataset_module"] = "importable"
    except ImportError:
        report["rfdetr_dataset_module"] = "rfdetr not installed in this interpreter"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default="main", choices=sorted(VARIANTS), help="which dataset configuration to build")
    ap.add_argument("--out", default=None, help="default data/rfdetr_dataset[_<variant>]")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="re-open an existing dataset and check it; no gsv4 imports")
    ap.add_argument("--verify-with", default=None, metavar="PYTHON",
                    help="after writing, run --verify-only again with this interpreter (e.g. .venv-rfdetr/bin/python)")
    args = ap.parse_args()
    default_out = "data/rfdetr_dataset" + ("" if args.variant == "main" else f"_{args.variant}")
    out_arg = args.out or default_out
    out = Path(out_arg) if Path(out_arg).is_absolute() else ROOT / out_arg

    if args.verify_only:
        report = verify_written(out)
        print(json.dumps(report, indent=1))
        bad = [k for k, v in report.items() if v == "MISSING"]
        return 1 if bad else 0

    sys.path.insert(0, str(ROOT))
    import pandas as pd

    from gsv4.config import load_config, resolve
    from gsv4.io.coco import load_annotations

    cfg = load_config(args.config)
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    man = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    kept = man[man["keep"]].copy()
    idx = pd.read_csv(ds / "yolo_index.csv").set_index("uid")
    kept["yolo_name"] = kept["uid"].map(idx["yolo_name"])
    missing = kept[kept["yolo_name"].isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} kept images are not in yolo_index.csv: {list(missing['uid'])[:5]}")

    categories: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"images": 0, "annotations": 0})
    problems: List[str] = []
    converted_strings = 0
    spec = VARIANTS[args.variant]
    kept_by_uid = kept.set_index("uid", drop=False)
    for split in SPLITS:
        uids = members(spec[split], kept, ds / "lists" / ".." )
        part = kept_by_uid.loc[uids]
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        next_img, next_ann = 1, 1
        cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in part.itertuples(index=False):
            key = (r.group, r.orig_split)
            if key not in cache:
                cache[key] = load_annotations(coco_root, r.group, r.orig_split, cfg["coco"])
            src = cache[key]
            if not categories:
                categories = convert_categories(src["categories"])
            im = next((i for i in src["images"] if i["file_name"] == r.file_name), None)
            if im is None:
                raise SystemExit(f"{r.uid}: {r.file_name} not found in {coco_root / r.group / r.orig_split}")
            img_id = next_img
            next_img += 1
            images.append(convert_image(im, img_id, r.yolo_name, {"uid": r.uid, "image": r.image}, f"{split}:{r.uid}"))
            for a in src["annotations"]:
                if int(a["image_id"]) != int(im["id"]):
                    continue
                converted_strings += sum(1 for v in a.get("bbox", []) if isinstance(v, str))
                annotations.append(convert_annotation(a, next_ann, img_id, f"{split}:{r.uid}:ann{a.get('id')}"))
                next_ann += 1
            if not args.dry_run:
                link = out / split / r.yolo_name
                link.parent.mkdir(parents=True, exist_ok=True)
                target = (coco_root / r.group / r.orig_split / r.file_name).resolve()
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(target, link)
        problems += validate_split(images, annotations, categories, split)
        counts[split] = {"images": len(images), "annotations": len(annotations)}
        if not args.dry_run:
            (out / split).mkdir(parents=True, exist_ok=True)
            (out / split / ANNOTATION_FILE).write_text(json.dumps(
                {"info": {"description": f"gummy_smile_v4 {split} split, participant level"},
                 "licenses": [], "categories": categories, "images": images, "annotations": annotations}), encoding="utf-8")

    cat_check = check_categories(categories, {cfg["class_names"]["gingiva"]: int(cfg["coco"]["category_ids"]["gingiva"]),
                                              cfg["class_names"]["lip"]: int(cfg["coco"]["category_ids"]["lip"])})
    problems += cat_check["problems"]
    total = sum(c["images"] for c in counts.values())
    print(f"[rfdetr-dataset] {'DRY RUN — ' if args.dry_run else ''}variant {args.variant} -> {out}")
    for split, c in counts.items():
        print(f"  {split:6s} images {c['images']:5d}  annotations {c['annotations']:6d}")
    print(f"  total images {total} (kept dataset: {len(kept)})")
    if args.variant == "main" and total != len(kept):
        problems.append(f"image count {total} does not match the manifest ({len(kept)})")
    print(f"  categories: {cat_check['categories']}; unused class 0 in the source export: {cat_check['dummy_class_0']!r}")
    print(f"  bbox values converted from string to float: {converted_strings} (the v3 export writes width and height as formatted strings)")
    if problems:
        for p in problems[:20]:
            print(f"  PROBLEM: {p}")
        raise SystemExit(f"{len(problems)} problem(s) found; nothing may be trained on this dataset")
    print("  validation: OK (bbox four floats, area float, ids int, polygons non-empty floats, image_ids resolve)")
    if args.dry_run:
        return 0

    print("[rfdetr-dataset] verifying what was written")
    print(json.dumps(verify_written(out), indent=1))
    if args.verify_with:
        print(f"[rfdetr-dataset] verifying again with {args.verify_with}")
        rc = subprocess.run([args.verify_with, str(Path(__file__).resolve()), "--verify-only", "--out", str(out)]).returncode
        if rc != 0:
            raise SystemExit(f"verification with {args.verify_with} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
