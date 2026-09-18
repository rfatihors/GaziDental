#!/usr/bin/env python
"""5-fold out-of-fold prediction for the measured high-smile-line images.

    python -m gsv4.train.cv_predict --fold 0 [--batch 8] [--dry-run]

Trains the fold model (train = everything except the held-out fold and its same-patient
twins; valid = main valid minus the fold) and predicts **only its own held-out fold**
with ``retina_masks=True``; class-separated binary PNGs go to
``outputs/05_predictions/oof/<image>_gingiva.png`` / ``_lip.png`` and a row per image is
appended to ``oof_predictions.csv``. Fold models never touch the test set; test-set
predictions come from the final model only (evaluate_test.py).

Prediction is chunked (``yolo.predict_batch`` in config.yaml, ``--batch`` overrides): each
chunk is passed to ``model.predict(stream=True)``, every result is written to disk as soon
as it arrives and the chunk's GPU memory is released before the next one. ``retina_masks``
keeps one float mask per instance at the *original* image resolution, so a whole test set
in one call is several GB — that is what ran out of CUDA memory on the workstation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pandas as pd

from gsv4.config import load_config, resolve
from gsv4.masks.extract import from_yolo, save_masks
from gsv4.train.common import is_done, mark_done, release_cuda, train_model

DEFAULT_PREDICT_BATCH = 8


def stem_lookup(cfg) -> dict:
    """yolo file name -> CSV image id, from yolo_index.csv (names were sanitised)."""
    idx = pd.read_csv(resolve(cfg, cfg["paths"]["yolo_dataset"]) / "yolo_index.csv")
    return dict(zip(idx["yolo_name"], idx["image"]))


def predict_batch_size(cfg: Dict[str, Any], override: Optional[int] = None) -> int:
    """Images per prediction chunk: CLI override, else ``yolo.predict_batch``, else 8."""
    value = override if override is not None else cfg.get("yolo", {}).get("predict_batch", DEFAULT_PREDICT_BATCH)
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"predict_batch must be a positive integer, got {value!r}") from exc
    if n < 1:
        raise ValueError(f"predict_batch must be a positive integer, got {n}")
    return n


def chunked(items: Sequence[Any], size: int) -> Iterator[List[Any]]:
    """Consecutive slices of ``items`` with at most ``size`` elements (order preserved)."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


def is_cuda_oom(exc: BaseException) -> bool:
    """True for torch's CUDA out-of-memory error (class or message, torch not required)."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def oom_message(tag: str, chunk_index: int, n_chunks: int, batch: int, imgsz: int) -> str:
    return (f"[{tag}] CUDA out of memory while predicting chunk {chunk_index}/{n_chunks} (predict_batch={batch}, imgsz={imgsz}). "
            f"predict_batch değerini düşürün: lower `yolo.predict_batch` in configs/config.yaml (or pass --batch, e.g. --batch 2); "
            f"1 is always safe. Already written masks are kept; re-run the step to continue. "
            f"Also make sure PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is exported (scripts/train_all.sh does).")


def _load_model(weights: Path):
    from ultralytics import YOLO  # lazy: requirements-train.txt

    return YOLO(str(weights))


def predict_list(cfg, weights: Path, list_file: Path, out_dir: Path, tag: str, extra: dict, batch: Optional[int] = None,
                 model: Any = None) -> pd.DataFrame:
    """Predict every image in ``list_file`` with its own model instance, chunk by chunk.

    Masks are saved per result (``save_masks``) and only the CSV row is retained; after
    each chunk the results are dropped and the CUDA cache is emptied. On a CUDA OOM the
    step stops with an actionable message (``predict_batch``); the model is released in
    every case. ``model`` may be injected (tests) — otherwise the weights are loaded here.
    """
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    names = [n for n in list_file.read_text().splitlines() if n.strip()]
    lookup = stem_lookup(cfg)
    y = cfg["yolo"]
    size = predict_batch_size(cfg, batch)
    imgsz = int(y["imgsz"])
    kwargs = dict(imgsz=imgsz, conf=float(y["conf"]), iou=float(y["iou"]), max_det=int(y["max_det"]),
                  retina_masks=bool(y["retina_masks"]), stream=True, verbose=False)
    chunks = list(chunked(names, size))
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_csv = out_dir / f"{tag}_predictions.partial.csv"
    rows: List[Dict[str, Any]] = []
    model = _load_model(weights) if model is None else model
    print(f"[{tag}] predicting {len(names)} images in {len(chunks)} chunk(s) of <= {size} (predict_batch)")
    try:
        for ci, chunk in enumerate(chunks, start=1):
            paths = [str(ds / n) for n in chunk]
            try:
                results = model.predict(source=paths, batch=size, **kwargs)
                for n, res in zip(chunk, results):
                    h, w = res.orig_shape[:2]
                    masks = from_yolo(res, cfg["class_names"], (h, w))
                    stem = lookup[Path(n).name]
                    save_masks(masks, out_dir, stem)  # to disk immediately; the result object is dropped below
                    confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None and len(res.boxes) else []
                    rows.append({"image": stem, "yolo_name": n, "n_gingiva": masks.n_gingiva_instances, "n_lip": masks.n_lip_instances,
                                 "n_ignored": masks.n_ignored_instances,
                                 "max_conf": max(confs) if confs else None, "mask_source": masks.source, "width": w, "height": h, **extra})
                    del res, masks
            except Exception as exc:  # noqa: BLE001
                if is_cuda_oom(exc):
                    raise SystemExit(oom_message(tag, ci, len(chunks), size, imgsz)) from exc
                raise
            finally:
                results = None  # noqa: F841 — drop the generator/predictor references before freeing the cache
                release_cuda()
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            print(f"[{tag}] chunk {ci}/{len(chunks)} done ({len(rows)}/{len(names)} images)")
    finally:
        del model
        release_cuda()
    df = pd.DataFrame(rows)
    if partial_csv.exists():
        partial_csv.unlink()
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--batch", type=int, default=None, help="images per prediction chunk (default: yolo.predict_batch in config.yaml, 8)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    name = f"fold{args.fold}"
    held = ds / "lists" / f"{name}_heldout.txt"
    size = predict_batch_size(cfg, args.batch)
    if args.dry_run:
        n = len(held.read_text().splitlines()) if held.exists() else "MISSING"
        print(f"[{name}] DRY RUN — held-out images to predict: {n}; predict_batch {size}; masks -> {resolve(cfg, cfg['paths']['predictions']) / 'oof'}")
    best = train_model(cfg, ds / f"data_{name}.yaml", name, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    step = f"{name}_predict"
    if is_done(cfg, step):
        print(f"[{step}] already DONE")
        return 0
    out_dir = resolve(cfg, cfg["paths"]["predictions"]) / "oof"
    df = predict_list(cfg, best, held, out_dir, name, {"fold": args.fold, "weights": str(best)}, batch=size)
    csv = out_dir / "oof_predictions.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        df = pd.concat([old[old["fold"] != args.fold], df], ignore_index=True)
    df.sort_values(["fold", "image"]).to_csv(csv, index=False)
    mark_done(cfg, step, {"n_predicted": int(len(df[df["fold"] == args.fold])), "predict_batch": size})
    print(f"[{step}] predicted {int((df['fold'] == args.fold).sum())} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
