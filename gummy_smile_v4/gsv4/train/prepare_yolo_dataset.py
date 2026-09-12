"""Manifest + splits -> Ultralytics segmentation dataset (spec: task §Aşama 5).

* Images are **symlinked** (never copied) from the v3 COCO export into
  ``data/yolo_dataset/images/all/<group>__<stem>.<ext>``; labels are written next to
  them under ``labels/all/``. COCO polygons become YOLO-seg lines
  ``<cls> x1 y1 x2 y2 ...`` (normalised), class 0 = gingiva (COCO 1), 1 = lip (COCO 2).
* Every configuration is a pair of image-list files under ``lists/`` plus a
  ``data_<name>.yaml``: ``main`` (train/valid/test), ``lc25/50/75`` (nested, stratified
  by smile-line group; 100 % = main), ``fold0..4`` (train = all kept images minus the
  held-out measured-high fold and minus its same-patient twins; valid = main valid minus
  the held-out fold; ``heldout`` = the fold itself, predicted out-of-fold).
* Augmentation is Ultralytics' own, applied by the trainer to the training list only;
  validation and test lists are never augmented and no Roboflow duplication is used.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from gsv4.config import resolve
from gsv4.io.coco import load_annotations

YOLO_CLASSES = {"gingiva": 0, "lip": 1}


def coco_to_yolo_lines(annotations: Sequence[Dict[str, Any]], image_id: int, width: int, height: int, cat_to_cls: Dict[int, int]) -> Tuple[List[str], int]:
    """YOLO-seg label lines for one image; returns (lines, n_polygons_dropped)."""
    lines, dropped = [], 0
    for a in annotations:
        if int(a["image_id"]) != int(image_id):
            continue
        cls = cat_to_cls.get(int(a["category_id"]))
        if cls is None:
            continue
        for poly in a["segmentation"]:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                dropped += 1
                continue
            pts[:, 0] = np.clip(pts[:, 0] / width, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / height, 0, 1)
            lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    return lines, dropped


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def yolo_file_name(group: str, image: str, file_name: str) -> str:
    """``high__IMG_7307_jpg.jpg``; spaces/parentheses become ``_`` (list files are one path
    per line, but tools split on whitespace). The mapping back to the CSV id is in
    ``yolo_index.csv``."""
    return f"{group}__{_SAFE.sub('_', image)}{Path(file_name).suffix.lower()}"


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def nested_fractions(images: pd.DataFrame, fractions: Sequence[float], seed: int) -> Dict[float, List[str]]:
    """Nested subsets of the training images, stratified by group, largest-remainder rounding."""
    rng = np.random.default_rng(seed)
    order: Dict[str, List[str]] = {}
    for g, part in images.groupby("group"):
        ids = list(part["yolo_name"])
        rng.shuffle(ids)
        order[g] = ids
    out: Dict[float, List[str]] = {}
    for f in fractions:
        sel: List[str] = []
        for g, ids in order.items():
            sel.extend(ids[: int(round(f * len(ids)))])
        out[f] = sorted(sel)
    return out


def build_dataset(cfg: Dict[str, Any], out_dir: Path | None = None) -> Dict[str, Any]:
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    out_dir = Path(out_dir) if out_dir else resolve(cfg, cfg["paths"]["yolo_dataset"])
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    splits = json.loads(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "splits.json").read_text())
    kept = manifest[manifest["keep"]].copy()
    kept["yolo_name"] = [yolo_file_name(g, i, f) for g, i, f in zip(kept["group"], kept["image"], kept["file_name"])]
    kept["split"] = kept["uid"].map(splits["images"])
    kept["cv_fold"] = kept["uid"].map(splits["cv_folds"]["assignments"])
    cat_to_cls = {int(cfg["coco"]["category_ids"]["gingiva"]): YOLO_CLASSES["gingiva"], int(cfg["coco"]["category_ids"]["lip"]): YOLO_CLASSES["lip"]}

    img_dir, lbl_dir, list_dir = out_dir / "images" / "all", out_dir / "labels" / "all", out_dir / "lists"
    for d in (img_dir, lbl_dir, list_dir):
        d.mkdir(parents=True, exist_ok=True)
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_lines, n_dropped, n_empty = 0, 0, 0
    for _, r in kept.iterrows():
        key = (r["group"], r["orig_split"])
        if key not in cache:
            cache[key] = load_annotations(coco_root, r["group"], r["orig_split"], cfg["coco"])
        ann = cache[key]
        im = next(i for i in ann["images"] if i["file_name"] == r["file_name"])
        lines, dropped = coco_to_yolo_lines(ann["annotations"], int(im["id"]), int(im["width"]), int(im["height"]), cat_to_cls)
        n_lines += len(lines)
        n_dropped += dropped
        n_empty += int(not lines)
        _symlink(coco_root / r["path"], img_dir / r["yolo_name"])
        (lbl_dir / (Path(r["yolo_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_list(name: str, names: Sequence[str]) -> str:
        p = list_dir / f"{name}.txt"
        p.write_text("\n".join(f"images/all/{n}" for n in sorted(names)) + "\n", encoding="utf-8")
        return f"lists/{name}.txt"

    def write_yaml(name: str, train: str, val: str, test: str | None = None) -> Path:
        d: Dict[str, Any] = {"path": str(out_dir.resolve()), "train": train, "val": val,
                             "names": {YOLO_CLASSES["gingiva"]: cfg["class_names"]["gingiva"], YOLO_CLASSES["lip"]: cfg["class_names"]["lip"]}}
        if test:
            d["test"] = test
        p = out_dir / f"data_{name}.yaml"
        p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return p

    by_split = {s: list(kept.loc[kept["split"] == s, "yolo_name"]) for s in ("train", "valid", "test")}
    configs: Dict[str, Dict[str, Any]] = {}
    tr, va, te = write_list("main_train", by_split["train"]), write_list("main_valid", by_split["valid"]), write_list("main_test", by_split["test"])
    write_yaml("main", tr, va, te)
    configs["main"] = {"train": len(by_split["train"]), "valid": len(by_split["valid"]), "test": len(by_split["test"])}

    fractions = [f for f in cfg["yolo"]["learning_curve_fractions"] if f < 1.0]
    nested = nested_fractions(kept[kept["split"] == "train"], fractions, int(cfg["seed"]))
    for f, names in nested.items():
        tag = f"lc{int(round(f * 100))}"
        write_yaml(tag, write_list(f"{tag}_train", names), va, te)
        configs[tag] = {"train": len(names), "valid": len(by_split["valid"]), "by_group": kept[kept["yolo_name"].isin(names)].groupby("group").size().to_dict()}

    measured = kept[(kept["group"] == "high") & (kept["has_reference_measurement"]) & kept["cv_fold"].notna()]
    for k in range(int(cfg["dataset"]["cv_folds"])):
        held = measured[measured["cv_fold"] == k]
        held_patients = set(held["patient_id"])
        train_k = kept[~kept["yolo_name"].isin(held["yolo_name"]) & ~kept["patient_id"].isin(held_patients) & (kept["split"] != "valid")]
        valid_k = kept[(kept["split"] == "valid") & ~kept["yolo_name"].isin(held["yolo_name"]) & ~kept["patient_id"].isin(held_patients)]
        tag = f"fold{k}"
        write_yaml(tag, write_list(f"{tag}_train", train_k["yolo_name"]), write_list(f"{tag}_valid", valid_k["yolo_name"]))
        write_list(f"{tag}_heldout", held["yolo_name"])
        configs[tag] = {"train": len(train_k), "valid": len(valid_k), "heldout": len(held)}

    kept[["uid", "image", "group", "split", "cv_fold", "has_reference_measurement", "yolo_name", "file_name", "path"]].to_csv(out_dir / "yolo_index.csv", index=False)
    summary = {"out_dir": str(out_dir), "n_images": int(len(kept)), "n_label_lines": n_lines, "n_polygons_dropped": n_dropped,
               "n_images_without_labels": n_empty, "configs": configs, "classes": {v: cfg["class_names"][k] for k, v in YOLO_CLASSES.items()}}
    (out_dir / "DATASET_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def label_check_figure(cfg: Dict[str, Any], out_png: Path, n_high: int = 3, n_normal: int = 2) -> Path:
    """Draw the written YOLO polygons back onto the images (gingiva red, lip blue)."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    idx = pd.read_csv(ds / "yolo_index.csv")
    sel = pd.concat([idx[idx.group == "high"].sort_values("uid").head(n_high), idx[idx.group == "normal"].sort_values("uid").head(n_normal)])
    fig, axes = plt.subplots(len(sel), 1, figsize=(10, 2.6 * len(sel)))
    for ax, (_, r) in zip(axes, sel.iterrows()):
        img = cv2.cvtColor(cv2.imread(str(ds / "images" / "all" / r["yolo_name"])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        over = img.copy()
        n = {0: 0, 1: 0}
        for line in (ds / "labels" / "all" / (Path(r["yolo_name"]).stem + ".txt")).read_text().splitlines():
            parts = line.split()
            cls, pts = int(parts[0]), np.asarray(parts[1:], dtype=float).reshape(-1, 2) * [w, h]
            n[cls] += 1
            cv2.fillPoly(over, [pts.astype(np.int32)], (255, 0, 0) if cls == 0 else (0, 60, 255))
        blend = (0.55 * img + 0.45 * over).astype(np.uint8)
        ys = np.nonzero((over != img).any(axis=2).any(axis=1))[0]
        y0, y1 = (max(0, ys.min() - 40), min(h, ys.max() + 40)) if len(ys) else (0, h)
        ax.imshow(blend[y0:y1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r['group']} | {r['image']} | YOLO polygons: gingiva {n[0]}, lip {n[1]} | split {r['split']}", fontsize=9)
    fig.suptitle("COCO -> YOLO-seg label check (labels re-read from the written .txt files)", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def main() -> int:
    import argparse

    from gsv4.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    summary = build_dataset(cfg)
    print(json.dumps(summary, indent=1))
    pred_dir = resolve(cfg, cfg["paths"]["predictions"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# YOLO dataset summary", "", f"Images (symlinks): {summary['n_images']}; label lines {summary['n_label_lines']}; polygons dropped (< 3 points): {summary['n_polygons_dropped']}; images without labels: {summary['n_images_without_labels']}.", "",
             "| configuration | train | valid | test / held-out |", "|---|---|---|---|"]
    for name, c in summary["configs"].items():
        lines.append(f"| {name} | {c['train']} | {c['valid']} | {c.get('test', c.get('heldout', ''))} |")
    (pred_dir / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_figure:
        label_check_figure(cfg, pred_dir / "yolo_label_check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
