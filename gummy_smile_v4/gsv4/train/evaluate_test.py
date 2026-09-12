#!/usr/bin/env python
"""Evaluate the final model on the fixed test set (reported once).

    python -m gsv4.train.evaluate_test [--dry-run]

Outputs (outputs/05_predictions/): test_metrics.json (per-class box/mask P, R, F1, mAP@50,
mAP@50–95), confusion_matrix.png, boundary_error.csv (boundary IoU, upper/lower gingiva
edge distance errors per test image against the ground-truth masks), and test-set
class masks under test/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from gsv4.config import load_config, resolve
from gsv4.eval.boundary import boundary_report
from gsv4.io.coco import load_annotations
from gsv4.masks.extract import from_coco, from_png
from gsv4.train.common import is_done, mark_done, per_class_metrics, run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None, help="default runs/final/weights/best.pt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    pred = resolve(cfg, cfg["paths"]["predictions"])
    best = Path(args.weights) if args.weights else run_dir(cfg, "final") / "weights" / "best.pt"
    test_list = ds / "lists" / "main_test.txt"
    if args.dry_run:
        n = len(test_list.read_text().splitlines()) if test_list.exists() else "MISSING"
        print(f"[eval] DRY RUN — weights {best} (exists: {best.exists()}); test images: {n}; outputs -> {pred}")
        return 0
    if is_done(cfg, "eval"):
        print("[eval] already DONE")
        return 0
    if not best.exists():
        raise SystemExit(f"weights not found: {best}")
    from ultralytics import YOLO

    from gsv4.train.cv_predict import predict_list

    model = YOLO(str(best))
    m = model.val(data=str(ds / "data_main.yaml"), split="test", imgsz=int(cfg["yolo"]["imgsz"]), conf=float(cfg["yolo"]["conf"]),
                  iou=float(cfg["yolo"]["iou"]), max_det=int(cfg["yolo"]["max_det"]), plots=True, verbose=True,
                  project=str(run_dir(cfg, "eval")), name="test", exist_ok=True)
    names = {int(k): v for k, v in m.names.items()}
    metrics = {"weights": str(best), "split": "test", "n_images": len(test_list.read_text().splitlines()), "per_class": per_class_metrics(m, names),
               "speed_ms": getattr(m, "speed", None)}
    pred.mkdir(parents=True, exist_ok=True)
    cm = run_dir(cfg, "eval") / "test" / "confusion_matrix.png"
    if cm.exists():
        shutil.copy(cm, pred / "confusion_matrix.png")
    try:
        metrics["confusion_matrix"] = m.confusion_matrix.matrix.tolist()
    except Exception:  # noqa: BLE001
        pass

    # test-set masks from the final model + boundary errors against GT
    out_dir = pred / "test"
    df = predict_list(cfg, best, test_list, out_dir, "test", {"weights": str(best)})
    df.to_csv(out_dir / "test_predictions.csv", index=False)
    idx = pd.read_csv(ds / "yolo_index.csv").set_index("image")
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    cache = {}
    rows = []
    for _, r in df.iterrows():
        info = idx.loc[r["image"]]
        key = (info["group"], Path(info["path"]).parts[1])
        if key not in cache:
            cache[key] = load_annotations(coco_root, key[0], key[1], cfg["coco"])
        ann = cache[key]
        im = next(i for i in ann["images"] if i["file_name"] == info["file_name"])
        gt = from_coco(ann["annotations"], int(im["id"]), (int(im["height"]), int(im["width"])), cfg["coco"]["category_ids"])
        pm = from_png(out_dir / f"{r['image']}_gingiva.png", out_dir / f"{r['image']}_lip.png", expected_shape=(int(im["height"]), int(im["width"])))
        rec = {"image": r["image"], "group": info["group"], **{f"gingiva_{k}": v for k, v in boundary_report(pm.gingiva, gt.gingiva).items()}}
        if gt.lip is not None and pm.lip is not None:
            rec.update({f"lip_{k}": v for k, v in boundary_report(pm.lip, gt.lip).items() if k in ("mask_iou", "boundary_iou")})
        rows.append(rec)
    b = pd.DataFrame(rows)
    b.to_csv(pred / "boundary_error.csv", index=False)
    metrics["boundary_summary"] = {c: {"mean": float(b[c].mean()), "median": float(b[c].median())} for c in b.columns if c.startswith(("gingiva_", "lip_")) and b[c].dtype != object}
    metrics["boundary_by_group"] = {g: {c: float(part[c].mean()) for c in ("gingiva_mask_iou", "gingiva_boundary_iou", "gingiva_top_edge_mae_px", "gingiva_bottom_edge_mae_px")} for g, part in b.groupby("group")}
    (pred / "test_metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    mark_done(cfg, "eval")
    print(json.dumps(metrics["per_class"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
