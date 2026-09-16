#!/usr/bin/env python
"""RF-DETR-Seg: probe, train and predict — runs in .venv-rfdetr, not in the training venv.

    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --probe          # 2 epochs, timing only
    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed 42        # full run

Reads the COCO dataset built by scripts/build_rfdetr_dataset.py, trains RF-DETR-Seg at its
published defaults with the shared budget of the protocol, then predicts the high-smile-line test
images and writes, in exactly the format the YOLO runs produce:

  outputs/08_architecture/masks/<tag>/<image>_gingiva.png and _lip.png
  outputs/08_architecture/predictions/<tag>.csv   (uid, image, yolo_name, instance counts)

The measurement and the comparison then come from the ordinary path:

    python scripts/run_architecture_comparison.py --measure-only

so RF-DETR is measured by the same code as every other architecture. Only the mask production
differs, which is the point of the comparison.

The probe reports the seconds per epoch and the peak GPU memory, and writes them next to the
environment report, so that the cost of the full run is known before it is committed to.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Shared budget (PROTOCOL.md §2); everything architecture-internal stays at the published default.
EPOCHS = 100
RESOLUTION = 624          # multiple of 24 closest to the 640 the YOLO runs use
VARIANT = "RFDETRSegLarge"
CONF = 0.25               # the pipeline's operating point, as for the YOLO masks


def load_split(ds: Path, split: str) -> dict:
    return json.loads((ds / split / "_annotations.coco.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/rfdetr_dataset")
    ap.add_argument("--out", default="outputs/08_architecture")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--probe", action="store_true", help="2 epochs; report seconds per epoch and peak memory, then stop")
    args = ap.parse_args()
    ds, out = ROOT / args.dataset, ROOT / args.out
    if not (ds / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"dataset not found: {ds} — run scripts/build_rfdetr_dataset.py first")
    tag = f"rfdetr-seg-large_s{args.seed}"
    epochs = 2 if args.probe else args.epochs
    run_dir = ROOT / "runs" / "arch" / (tag + ("_probe" if args.probe else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from rfdetr import __dict__ as rf

    if args.variant not in rf:
        raise SystemExit(f"unknown variant {args.variant}; available: {[n for n in rf if n.startswith('RFDETRSeg')]}")
    model = rf[args.variant]()
    print(f"[rfdetr] {args.variant}, resolution {args.resolution}, {epochs} epochs, seed {args.seed}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train(dataset_dir=str(ds), epochs=epochs, resolution=args.resolution, output_dir=str(run_dir))
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    record = {"variant": args.variant, "resolution": args.resolution, "epochs": epochs, "seed": args.seed,
              "seconds_total": round(elapsed, 1), "seconds_per_epoch": round(elapsed / max(1, epochs), 1),
              "peak_gpu_gb": round(peak_gb, 2), "probe": bool(args.probe),
              "projected_full_run_hours": round(elapsed / max(1, epochs) * args.epochs / 3600, 2)}
    (out / ("rfdetr_probe.json" if args.probe else f"rfdetr_train_{tag}.json")).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    if args.probe:
        print("[rfdetr] probe finished. Compare projected_full_run_hours with the budget before the full run.")
        return 0

    # ---- predict the high-smile-line test images at the operating point
    import cv2  # noqa: F401  (required by gsv4.masks.extract)
    import pandas as pd

    from gsv4.masks.extract import from_detections, save_masks

    test = load_split(ds, "test")
    id_to_name = {int(c["id"]): str(c["name"]) for c in test["categories"]}
    class_names = {"gingiva": "diseti", "lip": "dudak"}
    wanted = {r.uid for r in pd.read_csv(out / "arch_test_high_uids.csv").itertuples()} if (out / "arch_test_high_uids.csv").exists() else None
    mask_dir = out / "masks" / tag
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in test["images"]:
        if wanted is not None and im.get("uid") not in wanted:
            continue
        path = ds / "test" / im["file_name"]
        det = model.predict(str(path), threshold=CONF)
        cm = from_detections(det, id_to_name, class_names, (int(im["height"]), int(im["width"])), source="rfdetr:masks")
        save_masks(cm, mask_dir, im["image"])
        conf = getattr(det, "confidence", None)
        rows.append({"uid": im.get("uid"), "image": im["image"], "yolo_name": im["file_name"],
                     "n_gingiva": cm.n_gingiva_instances, "n_lip": cm.n_lip_instances,
                     "max_conf": float(np.max(conf)) if conf is not None and len(conf) else None,
                     "mask_source": cm.source, "width": int(im["width"]), "height": int(im["height"]),
                     "model": "rfdetr-seg-large", "seed": args.seed})
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "predictions" / f"{tag}.csv", index=False)
    print(f"[rfdetr] {len(rows)} images predicted -> {mask_dir}")
    print("[rfdetr] now run, in the training venv: python scripts/run_architecture_comparison.py --measure-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
