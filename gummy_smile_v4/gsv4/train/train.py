#!/usr/bin/env python
"""Train the final YOLOv11x-seg model on the main split (or any data yaml).

    python -m gsv4.train.train --name final [--data data/yolo_dataset/data_main.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gsv4.config import load_config, resolve
from gsv4.train.common import train_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="final")
    ap.add_argument("--data", default=None, help="data yaml (default data_main.yaml of the prepared dataset)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = Path(args.data) if args.data else resolve(cfg, cfg["paths"]["yolo_dataset"]) / "data_main.yaml"
    train_model(cfg, data, args.name, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
