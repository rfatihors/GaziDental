#!/usr/bin/env python
"""Are the predicted class masks really the classes they claim to be?

    python scripts/check_mask_classes.py --masks outputs/08_architecture/masks/rfdetr-seg-large_s42

Cross-matches every predicted class against every annotated class: the IoU of the predicted gingiva
with the annotated gingiva AND with the annotated lip, and the same for the predicted lip. A
predictor whose label space was read wrongly scores higher against the other class, which is the
signature this script exists to catch — the first architecture-comparison run stored the lip band as
the gingiva mask and produced a 6 mm gingival edge bias while its own mAP looked healthy.

Writes a table and, with --figure, a side-by-side overlay (annotation vs prediction, gingiva red and
lip blue) so the answer can also be seen rather than only computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.boundary import mask_iou  # noqa: E402
from gsv4.io.coco import load_annotations  # noqa: E402
from gsv4.masks.extract import from_coco, from_png  # noqa: E402
from gsv4.report.figures import PLOT_DPI  # noqa: E402


def crop_box(masks: List[np.ndarray], margin: float = 0.25) -> tuple:
    """Bounding box of everything drawn, with a margin, so the band fills the panel.

    The same box is used for the annotation and the prediction of one image, otherwise a shifted
    mask would look aligned because each panel had been cropped to its own content.
    """
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        if m is not None:
            union |= m
    if not union.any():
        return 0, union.shape[0], 0, union.shape[1]
    rows, cols = np.where(union)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    dr, dc = int((r1 - r0) * margin) + 5, int((c1 - c0) * margin) + 5
    return max(0, r0 - dr), min(union.shape[0], r1 + dr + 1), max(0, c0 - dc), min(union.shape[1], c1 + dc + 1)


def overlay(ax, gingiva: np.ndarray, lip: np.ndarray | None, title: str, box: tuple) -> None:
    r0, r1, c0, c1 = box
    g = gingiva[r0:r1, c0:c1]
    rgb = np.ones((*g.shape, 3), dtype=float)
    if lip is not None:
        rgb[lip[r0:r1, c0:c1]] = (0.20, 0.40, 0.85)
    rgb[g] = (0.85, 0.15, 0.15)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--masks", required=True, help="directory of <image>_gingiva.png / _lip.png")
    ap.add_argument("--n", type=int, default=6, help="images to check (0 = all that are present)")
    ap.add_argument("--figure", default=None, help="write an overlay PNG here")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mask_dir = Path(args.masks)
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    man = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    man = man[man["keep"]].set_index("image")

    stems = sorted(p.name[: -len("_gingiva.png")] for p in mask_dir.glob("*_gingiva.png"))
    if not stems:
        raise SystemExit(f"no masks in {mask_dir}")
    if args.n:
        stems = stems[: args.n]
    rows: List[Dict[str, Any]] = []
    panels = []
    cache: Dict[Any, Any] = {}
    for stem in stems:
        if stem not in man.index:
            continue
        r = man.loc[stem]
        key = (r["group"], r["orig_split"])
        if key not in cache:
            cache[key] = load_annotations(coco_root, key[0], key[1], cfg["coco"])
        ann = cache[key]
        im = next(i for i in ann["images"] if i["file_name"] == r["file_name"])
        shape = (int(im["height"]), int(im["width"]))
        gt = from_coco(ann["annotations"], int(im["id"]), shape, cfg["coco"]["category_ids"])
        g = mask_dir / f"{stem}_gingiva.png"
        lp = mask_dir / f"{stem}_lip.png"
        pm = from_png(g, lp if lp.exists() else None, expected_shape=shape)
        empty = np.zeros(shape, dtype=bool)
        gt_lip = gt.lip if gt.lip is not None else empty
        pm_lip = pm.lip if pm.lip is not None else empty
        rows.append({"image": stem, "lip_png": lp.exists(),
                     "pred_gingiva_vs_gt_gingiva": mask_iou(pm.gingiva, gt.gingiva),
                     "pred_gingiva_vs_gt_lip": mask_iou(pm.gingiva, gt_lip),
                     "pred_lip_vs_gt_lip": mask_iou(pm_lip, gt_lip),
                     "pred_lip_vs_gt_gingiva": mask_iou(pm_lip, gt.gingiva)})
        if len(panels) < 4:
            panels.append((stem, gt, pm))

    t = pd.DataFrame(rows)
    swapped = t["pred_gingiva_vs_gt_lip"] > t["pred_gingiva_vs_gt_gingiva"]
    print(t.round(3).to_string(index=False))
    print()
    print(f"images checked: {len(t)}; without a lip PNG: {int((~t['lip_png']).sum())}")
    print(f"predicted gingiva matches the annotated LIP better on {int(swapped.sum())} of {len(t)} images")
    verdict = ("CLASS MISMATCH: the predicted gingiva is the annotated lip. The label space was read wrongly; "
               "the masks must be produced again, they cannot be relabelled on disk (the gingiva instance was "
               "never written)." if swapped.all() and len(t) else
               "classes look correct: each predicted class matches its own annotated class best" if not swapped.any() else
               f"MIXED: {int(swapped.sum())} of {len(t)} images match the other class better; inspect individually")
    print(verdict)

    if args.figure:
        fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 3.4), squeeze=False)
        for j, (stem, gt, pm) in enumerate(panels):
            box = crop_box([gt.gingiva, gt.lip, pm.gingiva, pm.lip])
            overlay(axes[0][j], gt.gingiva, gt.lip, f"{stem}\nannotation", box)
            overlay(axes[1][j], pm.gingiva, pm.lip, "prediction" + ("" if pm.lip is not None else "\n(no lip mask)"), box)
        fig.suptitle(f"Class check: {mask_dir.name} — gingiva red, lip blue", fontsize=10)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=PLOT_DPI); plt.close(fig)
        print(f"figure -> {args.figure}")
    return 0 if len(t) and not swapped.any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
