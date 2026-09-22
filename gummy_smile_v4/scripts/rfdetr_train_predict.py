#!/usr/bin/env python
"""RF-DETR-Seg: probe, train and predict — runs in .venv-rfdetr, not in the training venv.

    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --probe            # 2 epochs, timing only
    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed 42          # train, then predict
    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --seed 42 --predict-only
    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --variant fold0    # one cross-validation fold
    .venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --variant lc25 --no-predict

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
PATIENCE = 20             # same as the YOLO runs; see the early-stopping note below
RESOLUTION = 624          # multiple of 24 closest to the 640 the YOLO runs use
VARIANT = "RFDETRSegLarge"
CONF = 0.25               # the pipeline's operating point, as for the YOLO masks
BEST_CHECKPOINT = "checkpoint_best_total.pth"   # the winner of regular vs EMA, written by BestModelCallback

# Where each dataset variant's predictions belong (outputs/09_final_rfdetr/PLAN.md). A fold predicts
# only its own held-out images and they accumulate into one out-of-fold set; the main variant
# predicts the fixed test set; a learning-curve subset predicts nothing and is evaluated on valid.
VARIANT_OUTPUT = {
    "main": {"masks": "outputs/05_predictions/test_rfdetr", "csv": "test_predictions.csv", "evaluate": "test"},
    **{f"fold{k}": {"masks": "outputs/05_predictions/oof_rfdetr", "csv": "oof_predictions.csv",
                    "evaluate": "test", "fold": k} for k in range(5)},
    **{f"lc{n}": {"masks": None, "csv": None, "evaluate": "val"} for n in (25, 50, 75)},
}

# Class ids in an RF-DETR prediction are indices into the MODEL's own class list, never the category
# ids of the dataset: `_class_id_to_name = dict(enumerate(model_class_names))` in rfdetr/detr.py.
# The model's list comes from the dataset's categories after `filter_parent_categories` drops the
# unannotated Roboflow grouping category, and the survivors are renumbered contiguously
# (rfdetr/datasets/coco.py: `cat2label = {id: label for label, category in enumerate(kept)}`).
# Our export is 0 = dudak-diseti (grouping, never annotated), 1 = diseti, 2 = dudak, so the trained
# model emits 0 for gingiva and 1 for lip. Reading those ids through the dataset's own table maps
# gingiva onto the grouping name (dropped) and lip onto gingiva, which is exactly the fault that
# produced a 6 mm edge bias in the first comparison run. The model's own names are the only correct
# source, and the two label spaces are cross-checked below before a single mask is written.

# Early stopping (PROTOCOL.md §2 holds the rule identical across architectures; amendment of
# 17 Sep 2026 records what "identical" can and cannot mean here).
#
# rfdetr supports it through TrainConfig: early_stopping, early_stopping_patience,
# early_stopping_min_delta, early_stopping_use_ema. For a segmentation model the monitored key is
# ``val/segm_mAP_50_95`` (and ``val/ema_segm_mAP_50_95``), selected by best_model_metric="map".
# With use_ema left at its default the callback watches max(regular, EMA); when EMA is on and
# eval_base_model is off, the regular key already mirrors the EMA score, so this is the EMA metric.
#
# Ultralytics stops on its own fitness, which for a segmentation model is
# SegmentMetrics.fitness() = mask mAP@50-95 + box mAP@50-95 (Metric.fitness weights [0, 0, 0, 1]),
# with no minimum improvement. So: same patience, same "no improvement for N validation epochs"
# rule, and a monitored quantity that is mask mAP@50-95 in both, plus a box term in the YOLO case.
# min_delta is therefore set to 0.0 to match ultralytics, which requires no minimum improvement.
EARLY_STOPPING = {"early_stopping": True, "early_stopping_patience": PATIENCE, "early_stopping_min_delta": 0.0}


def load_split(ds: Path, split: str) -> dict:
    return json.loads((ds / split / "_annotations.coco.json").read_text(encoding="utf-8"))


def expected_label_space(ds: Path) -> list:
    """The class names the model should carry, derived the way rfdetr derives them.

    Uses rfdetr's own filter when it can be imported, so this check cannot drift from the library;
    falls back to the documented rule (drop a category that is named as another category's
    supercategory and carries no annotation) and says which path it took.
    """
    train = load_split(ds, "train")
    annotated = {int(a["category_id"]) for a in train["annotations"]}
    try:
        from rfdetr.datasets.coco import filter_parent_categories

        kept = filter_parent_categories(list(train["categories"]), annotated)
        return [str(c["name"]) for c in kept]
    except ImportError:
        parents = {str(c.get("supercategory")) for c in train["categories"]}
        kept = [c for c in sorted(train["categories"], key=lambda c: int(c["id"]))
                if not (str(c["name"]) in parents and int(c["id"]) not in annotated)]
        print("[rfdetr] note: rfdetr.datasets.coco.filter_parent_categories not importable; used the documented rule")
        return [str(c["name"]) for c in kept]


def check_label_space(model_names: list, ds: Path, class_names: dict) -> dict:
    """Refuse to predict unless the model's label space is the one this dataset implies.

    Returns the id -> name map to hand the adapter. Raises when the model's names differ from the
    dataset's filtered categories, or when a configured role is missing from them.
    """
    expected = expected_label_space(ds)
    if list(model_names) != expected:
        raise SystemExit(
            f"the model's class list {list(model_names)} is not the one this dataset implies {expected}. "
            "Predicting would assign the wrong class to every mask; check the dataset the checkpoint was trained on.")
    missing = [n for n in class_names.values() if n not in model_names]
    if missing:
        raise SystemExit(f"the model's class list {list(model_names)} does not contain {missing}")
    id_to_name = dict(enumerate(model_names))
    print(f"[rfdetr] label space: {id_to_name} (from the model; the dataset's category ids are NOT used)")
    return id_to_name


def predict(model, ds: Path, out_masks: Path, csv_path: Path, tag: str, seed: int,
            model_label: str = "rfdetr-seg-large", uid_filter: Path | None = None, fold: int | None = None) -> None:
    """Masks and a prediction table for the high-smile-line test images, in the YOLO runs' format.

    The class ids come from the model's own label space, checked against the dataset's
    (``check_label_space``), and the adapter runs in strict mode: an instance whose id maps to no
    role aborts instead of being dropped, because a dropped instance is how the first run's
    label-space fault stayed invisible.
    """
    import cv2  # noqa: F401  (required by gsv4.masks.extract)
    import pandas as pd

    from gsv4.masks.extract import from_detections, save_masks

    class_names = {"gingiva": "diseti", "lip": "dudak"}
    id_to_name = check_label_space(model.class_names, ds, class_names)
    test = load_split(ds, "test")
    wanted = None
    if uid_filter is not None and uid_filter.exists():
        wanted = {r.uid for r in pd.read_csv(uid_filter).itertuples()}
    mask_dir = Path(out_masks)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in test["images"]:
        if wanted is not None and im.get("uid") not in wanted:
            continue
        det = model.predict(str(ds / "test" / im["file_name"]), threshold=CONF)
        cm = from_detections(det, id_to_name, class_names, (int(im["height"]), int(im["width"])),
                             source="rfdetr:masks", strict=True)
        save_masks(cm, mask_dir, im["image"])
        conf = getattr(det, "confidence", None)
        rows.append({"uid": im.get("uid"), "image": im["image"], "yolo_name": im["file_name"],
                     "n_gingiva": cm.n_gingiva_instances, "n_lip": cm.n_lip_instances,
                     "n_ignored": cm.n_ignored_instances,
                     "max_conf": float(np.max(conf)) if conf is not None and len(conf) else None,
                     "mask_source": cm.source, "width": int(im["width"]), "height": int(im["height"]),
                     "model": model_label, "seed": seed, **({"fold": fold} if fold is not None else {})})
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fold is not None and csv_path.exists():
        # the out-of-fold set accumulates: replace this fold's rows, keep the others
        old_rows = pd.read_csv(csv_path)
        if "fold" in old_rows.columns:
            df = pd.concat([old_rows[old_rows["fold"] != fold], df], ignore_index=True).sort_values(["fold", "image"])
    df.to_csv(csv_path, index=False)
    empty_lip = int((pd.DataFrame(rows)["n_lip"] == 0).sum()) if rows else 0
    print(f"[rfdetr] {len(rows)} images predicted -> {mask_dir}; table {csv_path} ({len(df)} rows)")
    r = pd.DataFrame(rows)
    print(f"[rfdetr] instances per image: gingiva {r['n_gingiva'].mean():.2f}, lip {r['n_lip'].mean():.2f}; "
          f"images without a lip mask: {empty_lip}")
    if empty_lip:
        print("[rfdetr] WARNING: an image with no lip instance is unusual for this dataset; check the masks before measuring")


def evaluate_split(model, ds: Path, spec: dict, args) -> dict:
    """RF-DETR's own COCO evaluation of the split this variant is judged on (PLAN.md §4)."""
    split = spec.get("evaluate")
    if not split or args.probe:
        return {}
    print(f"[rfdetr] COCO evaluation on the {split} split")
    m = model.evaluate(split=split, dataset_dir=str(ds), resolution=args.resolution)
    return {str(k): float(v) for k, v in dict(m).items() if isinstance(v, (int, float))}


def run_prediction(model, ds: Path, out: Path, spec: dict, tag: str, seed: int, model_label: str, args) -> None:
    """Predict the images this variant is responsible for, if any."""
    if args.no_predict or not spec.get("masks"):
        print(f"[rfdetr] no prediction for variant {args.variant} (it contributes a metric, not masks)")
        return
    masks = ROOT / spec["masks"]
    csv_path = masks / spec["csv"]
    uid_filter = Path(args.uid_filter) if args.uid_filter else (out / "arch_test_high_uids.csv" if args.variant == "main" and (out / "arch_test_high_uids.csv").exists() else None)
    predict(model, ds, masks, csv_path, tag, seed, model_label, uid_filter=uid_filter, fold=spec.get("fold"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="main", choices=sorted(VARIANT_OUTPUT),
                    help="dataset configuration: the fixed split, a cross-validation fold, or a learning-curve subset")
    ap.add_argument("--dataset", default=None, help="default data/rfdetr_dataset[_<variant>]")
    ap.add_argument("--no-predict", action="store_true", help="train and evaluate only (learning-curve subsets)")
    ap.add_argument("--uid-filter", default=None, help="CSV with a uid column; predict only those images")
    ap.add_argument("--out", default="outputs/08_architecture")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--model-variant", default=VARIANT, help="RF-DETR segmentation variant class")
    ap.add_argument("--patience", type=int, default=PATIENCE, help="early-stopping patience (0 disables it)")
    ap.add_argument("--probe", action="store_true", help="2 epochs; report seconds per epoch and peak memory, then stop")
    ap.add_argument("--predict-only", action="store_true",
                    help="skip training and predict from the run's best checkpoint; use this to redo the masks "
                         "without retraining (the weights are unaffected by a label-space fault)")
    args = ap.parse_args()
    spec = VARIANT_OUTPUT[args.variant]
    ds = ROOT / (args.dataset or ("data/rfdetr_dataset" + ("" if args.variant == "main" else f"_{args.variant}")))
    out = ROOT / args.out
    if not (ds / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"dataset not found: {ds} — run scripts/build_rfdetr_dataset.py --variant {args.variant} first")
    # a resolution other than the protocol's is a control (PROTOCOL_ADDENDUM_resolution.md); the
    # label carries it so the aggregation keeps it out of the comparison
    model_label = "rfdetr-seg-large" if args.resolution == RESOLUTION else f"rfdetr-seg-large@{args.resolution}"
    tag = f"{model_label}_s{args.seed}" if args.variant == "main" else f"{args.variant}_s{args.seed}"
    epochs = 2 if args.probe else args.epochs
    run_root = "arch" if args.variant == "main" else "rfdetr"
    run_dir = ROOT / "runs" / run_root / (tag + ("_probe" if args.probe else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from rfdetr import __dict__ as rf

    if args.predict_only:
        ckpt = run_dir / BEST_CHECKPOINT
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint to predict from: {ckpt}")
        print(f"[rfdetr] loading {ckpt} (training skipped)")
        model = rf["from_checkpoint"](str(ckpt), trust_checkpoint=True)   # our own file, from our own run
        record = {"variant": args.variant, "seed": args.seed, "predict_only": True, "checkpoint": str(ckpt)}
        run_prediction(model, ds, out, spec, tag, args.seed, model_label, args)
        (out / f"rfdetr_predict_{tag}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return 0

    if args.model_variant not in rf:
        raise SystemExit(f"unknown model variant {args.model_variant}; available: {[n for n in rf if n.startswith('RFDETRSeg')]}")
    model = rf[args.model_variant]()
    print(f"[rfdetr] {args.model_variant} on dataset variant {args.variant}, resolution {args.resolution}, {epochs} epochs, seed {args.seed}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # the probe measures the cost of an epoch, so it must not stop early
    stopping = {} if args.probe or args.patience <= 0 else {**EARLY_STOPPING, "early_stopping_patience": args.patience}
    print(f"[rfdetr] early stopping: {stopping or 'disabled (probe)'}")
    t0 = time.time()
    model.train(dataset_dir=str(ds), epochs=epochs, resolution=args.resolution, output_dir=str(run_dir), **stopping)
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    record = {"model_variant": args.model_variant, "variant": args.variant, "model_label": model_label, "resolution": args.resolution, "epochs_budget": epochs, "seed": args.seed,
              "early_stopping": stopping or None, "monitored_metric": "val/segm_mAP_50_95 (max of regular and EMA)",
              "seconds_total": round(elapsed, 1), "seconds_per_epoch": round(elapsed / max(1, epochs), 1),
              "peak_gpu_gb": round(peak_gb, 2), "probe": bool(args.probe),
              "projected_full_run_hours": round(elapsed / max(1, epochs) * args.epochs / 3600, 2)}
    for line in (run_dir / "log.txt"), (run_dir / "results.json"):   # whatever the trainer left, for the epoch count
        if line.exists():
            record["trainer_log"] = str(line)
            break
    print(json.dumps(record, indent=1))
    if args.probe:
        (out / "rfdetr_probe.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print("[rfdetr] probe finished. Compare projected_full_run_hours with the budget before the full run.")
        return 0

    metrics = evaluate_split(model, ds, spec, args)
    if metrics:
        record["coco_metrics"] = metrics
        (out / f"rfdetr_metrics_{tag}.json").write_text(json.dumps({"variant": args.variant, "seed": args.seed,
                                                                    "split": spec["evaluate"], **metrics}, indent=1, default=str), encoding="utf-8")
    run_prediction(model, ds, out, spec, tag, args.seed, model_label, args)
    (out / f"rfdetr_train_{tag}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("[rfdetr] next, in the training venv: python scripts/run_architecture_comparison.py --measure-only "
          "(comparison) or python scripts/run_oracle.py --masks <mask dir> (Stage 6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
