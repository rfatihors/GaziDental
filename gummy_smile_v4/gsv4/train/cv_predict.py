#!/usr/bin/env python
"""5-fold out-of-fold prediction for the measured high-smile-line images.

    python -m gsv4.train.cv_predict --fold 0 [--dry-run]

Trains the fold model (train = everything except the held-out fold and its same-patient
twins; valid = main valid minus the fold) and predicts **only its own held-out fold**
with ``retina_masks=True``; class-separated binary PNGs go to
``outputs/05_predictions/oof/<image>_gingiva.png`` / ``_lip.png`` and a row per image is
appended to ``oof_predictions.csv``. Fold models never touch the test set; test-set
predictions come from the final model only (evaluate_test.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from gsv4.config import load_config, resolve
from gsv4.masks.extract import from_yolo, save_masks
from gsv4.train.common import is_done, mark_done, train_model


def stem_lookup(cfg) -> dict:
    """yolo file name -> CSV image id, from yolo_index.csv (names were sanitised)."""
    idx = pd.read_csv(resolve(cfg, cfg["paths"]["yolo_dataset"]) / "yolo_index.csv")
    return dict(zip(idx["yolo_name"], idx["image"]))


def predict_list(cfg, weights: Path, list_file: Path, out_dir: Path, tag: str, extra: dict) -> pd.DataFrame:
    from ultralytics import YOLO

    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    names = [n for n in list_file.read_text().splitlines() if n.strip()]
    paths = [str(ds / n) for n in names]
    lookup = stem_lookup(cfg)
    model = YOLO(str(weights))
    y = cfg["yolo"]
    rows = []
    for n, res in zip(names, model.predict(source=paths, imgsz=int(y["imgsz"]), conf=float(y["conf"]), iou=float(y["iou"]),
                                            max_det=int(y["max_det"]), retina_masks=bool(y["retina_masks"]), stream=True, verbose=False)):
        h, w = res.orig_shape[:2]
        masks = from_yolo(res, cfg["class_names"], (h, w))
        stem = lookup[Path(n).name]
        save_masks(masks, out_dir, stem)
        confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None and len(res.boxes) else []
        rows.append({"image": stem, "yolo_name": n, "n_gingiva": masks.n_gingiva_instances, "n_lip": masks.n_lip_instances,
                     "max_conf": max(confs) if confs else None, "mask_source": masks.source, "width": w, "height": h, **extra})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    name = f"fold{args.fold}"
    held = ds / "lists" / f"{name}_heldout.txt"
    if args.dry_run:
        n = len(held.read_text().splitlines()) if held.exists() else "MISSING"
        print(f"[{name}] DRY RUN — held-out images to predict: {n}; masks -> {resolve(cfg, cfg['paths']['predictions']) / 'oof'}")
    best = train_model(cfg, ds / f"data_{name}.yaml", name, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    step = f"{name}_predict"
    if is_done(cfg, step):
        print(f"[{step}] already DONE")
        return 0
    out_dir = resolve(cfg, cfg["paths"]["predictions"]) / "oof"
    df = predict_list(cfg, best, held, out_dir, name, {"fold": args.fold, "weights": str(best)})
    csv = out_dir / "oof_predictions.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        df = pd.concat([old[old["fold"] != args.fold], df], ignore_index=True)
    df.sort_values(["fold", "image"]).to_csv(csv, index=False)
    mark_done(cfg, step, {"n_predicted": int(len(df[df["fold"] == args.fold]))})
    print(f"[{step}] predicted {int((df['fold'] == args.fold).sum())} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
