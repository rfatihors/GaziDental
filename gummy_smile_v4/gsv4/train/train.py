#!/usr/bin/env python
"""Train the final YOLOv11x-seg model on the main split (or any data yaml).

    python -m gsv4.train.train --name final [--data data/yolo_dataset/data_main.yaml] [--dry-run]

Smoke test (5 minutes on the GPU, before the overnight run):

    python -m gsv4.train.train --name smoke --epochs 1 --fraction 0.1 --predict-check 5

``--fraction`` writes ``data_smoke.yaml`` with a random 10 % of the training list and 10 %
of the validation list; ``--predict-check N`` then predicts N validation images with
``retina_masks=True`` through the same code path as cv_predict and writes
``outputs/05_predictions/smoke/smoke_predictions.csv`` (``mask_source`` must read
``yolo:masks.data``). Everything under ``smoke`` is git-ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from gsv4.config import load_config, resolve
from gsv4.train.common import run_dir, train_model


def write_fraction_yaml(cfg, ds: Path, name: str, fraction: float, seed: int) -> Path:
    """Random ``fraction`` of the main train and valid lists -> data_<name>.yaml (smoke tests)."""
    rng = np.random.default_rng(seed)
    d = yaml.safe_load((ds / "data_main.yaml").read_text())
    out = dict(d)
    for key in ("train", "val"):
        names = [n for n in (ds / d[key]).read_text().splitlines() if n.strip()]
        rng.shuffle(names)
        keep = sorted(names[: max(4, int(round(fraction * len(names))))])
        lst = ds / "lists" / f"{name}_{key}.txt"
        lst.write_text("\n".join(keep) + "\n", encoding="utf-8")
        out[key] = f"lists/{name}_{key}.txt"
    out.pop("test", None)
    p = ds / f"data_{name}.yaml"
    p.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="final")
    ap.add_argument("--data", default=None, help="data yaml (default data_main.yaml of the prepared dataset)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override yolo.train.epochs (smoke test)")
    ap.add_argument("--fraction", type=float, default=None, help="random fraction of train/valid lists (smoke test)")
    ap.add_argument("--predict-check", type=int, default=0, help="after training, predict N validation images with retina_masks and report mask_source")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    data = Path(args.data) if args.data else ds / "data_main.yaml"
    if args.fraction is not None:
        data = write_fraction_yaml(cfg, ds, args.name, args.fraction, int(cfg["seed"]))
    overrides = {"epochs": args.epochs} if args.epochs else None
    best = train_model(cfg, data, args.name, dry_run=args.dry_run, overrides=overrides)
    if args.predict_check and not args.dry_run and best is not None:
        from gsv4.train.cv_predict import predict_list

        d = yaml.safe_load(data.read_text())
        val_list = ds / d["val"]
        sample = ds / "lists" / f"{args.name}_predict_check.txt"
        sample.write_text("\n".join([n for n in val_list.read_text().splitlines() if n.strip()][: args.predict_check]) + "\n", encoding="utf-8")
        out_dir = resolve(cfg, cfg["paths"]["predictions"]) / args.name
        df = predict_list(cfg, best, sample, out_dir, args.name, {"weights": str(best)})
        df.to_csv(out_dir / f"{args.name}_predictions.csv", index=False)
        src = df["mask_source"].value_counts().to_dict()
        ok = all(k == "yolo:masks.data" for k in src)
        print(f"[{args.name}] predict-check: {len(df)} images, mask_source = {src} -> {'OK (retina masks at original resolution)' if ok else 'CHECK: masks not at original resolution'}")
        print(f"[{args.name}] masks written to {out_dir}; DONE marker at {run_dir(cfg, args.name) / 'DONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
