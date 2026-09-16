#!/usr/bin/env python
"""Evaluate the final model on the fixed test set (reported once).

    python -m gsv4.train.evaluate_test [--batch 8] [--metrics-only] [--repredict] [--dry-run]

Three stages, in this order, each restartable on its own:

1. validation (``model.val``) twice -> per-class metrics at the standard evaluation settings
   (conf 0.001, iou 0.7, max_det 300; the reported mAP) and at the pipeline's operating point
   (configs/config.yaml; its precision, recall, F1 and confusion matrix), written to
   test_metrics.json immediately so a later crash never costs the validation pass;
2. prediction -> class masks under ``test/`` plus ``test/test_predictions.csv`` (chunked, see
   cv_predict.py; an existing complete table is reused unless ``--repredict``);
3. boundary errors and the summary tables -> boundary_error.csv, the ``boundary_*`` keys of
   test_metrics.json, and only then ``runs/eval/DONE``.

``--metrics-only`` runs stage 3 alone against the predictions already on disk, so a failure
there never forces the 192 test images to be predicted again.

Validation and prediction use two separate model instances: the validation model is
released (del + gc + torch.cuda.empty_cache) before the test-set masks are predicted.

Outputs (outputs/05_predictions/): test_metrics.json (per-class box/mask P, R, F1, mAP@50,
mAP@50–95, boundary summary), confusion_matrix.png, boundary_error.csv (boundary IoU,
upper/lower gingiva edge distance errors per test image against the ground-truth masks),
and test-set class masks under test/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from gsv4.config import load_config, resolve
from gsv4.eval.boundary import boundary_report
from gsv4.io.coco import load_annotations
from gsv4.masks.extract import from_coco, from_png
from gsv4.train.common import is_done, mark_done, per_class_metrics, release_cuda, run_dir

# Columns reported per group in test_metrics.json["boundary_by_group"].
GROUP_COLUMNS = ("gingiva_mask_iou", "gingiva_boundary_iou", "gingiva_top_edge_mae_px", "gingiva_bottom_edge_mae_px")

# Average precision is only meaningful when the precision-recall curve is traced to its end, so the
# reported mAP is computed at the evaluation convention (a confidence floor near zero, NMS IoU 0.7,
# 300 detections). The pipeline itself runs at the operating point in configs/config.yaml
# (conf 0.25, iou 0.5, max_det 20): that is what produces the masks, and its precision, recall, F1
# and confusion matrix are reported separately. Reporting mAP at conf 0.25 truncates the curve and
# understates average precision; it is also not comparable with a COCO-style evaluation of another
# architecture (outputs/08_architecture/PROTOCOL.md §4).
STANDARD_EVAL = {"conf": 0.001, "iou": 0.7, "max_det": 300}


def eval_settings(cfg: Dict[str, Any], standard: bool) -> Dict[str, Any]:
    """The val() thresholds for the reported mAP (standard) or for the pipeline's operating point."""
    y = cfg["yolo"]
    s = dict(STANDARD_EVAL) if standard else {"conf": float(y["conf"]), "iou": float(y["iou"]), "max_det": int(y["max_det"])}
    return {**s, "imgsz": int(y["imgsz"]), "kind": "standard" if standard else "operating_point"}


def reported_metrics(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The per-class block to report and the settings it was computed at.

    Prefers the standard-settings block. A metrics file written before this distinction existed
    has only ``per_class``, computed at the operating point; it is returned with the settings
    marked ``operating_point_legacy`` so that every table says which convention it is showing.
    """
    if "per_class_standard" in metrics:
        return metrics["per_class_standard"], metrics.get("eval_settings_standard", {"kind": "standard"})
    return metrics.get("per_class", {}), metrics.get("eval_settings", {"kind": "operating_point_legacy"})


def manifest_identity(cfg: Dict[str, Any]) -> pd.DataFrame:
    """``uid`` -> group, COCO source split and COCO file name, from the dataset manifest.

    The manifest is the only place the *source* split (``orig_split``) is recorded, and it
    is what addresses the COCO export (``coco_root/<group>/<orig_split>/``). Deriving either
    from a path string is fragile — it depends on the path layout and silently returns the
    wrong component if that layout changes — so both are read as columns here.
    """
    path = resolve(cfg, cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"
    man = pd.read_csv(path, dtype={"uid": str, "group": str, "orig_split": str, "file_name": str})
    missing = [c for c in ("uid", "group", "orig_split", "file_name") if c not in man.columns]
    if missing:
        raise SystemExit(f"{path}: manifest is missing column(s) {missing}")
    if not man["uid"].is_unique:
        dup = man.loc[man["uid"].duplicated(keep=False), "uid"].unique().tolist()
        raise SystemExit(f"{path}: uid is not unique: {dup[:10]}")
    return man.set_index("uid")[["group", "orig_split", "file_name"]]


def uid_by_yolo_name(cfg: Dict[str, Any]) -> Dict[str, str]:
    """YOLO file name (basename) -> ``uid``, from yolo_index.csv.

    ``image`` (the Roboflow-free stem) is deliberately not usable as a key: it repeats
    across groups (iPhone numbering), so ``yolo_index.set_index("image").loc[stem]`` returns
    a *DataFrame* for those stems and an arbitrary group's row for the rest.
    """
    path = resolve(cfg, cfg["paths"]["yolo_dataset"]) / "yolo_index.csv"
    idx = pd.read_csv(path, dtype={"uid": str, "yolo_name": str})
    if not idx["yolo_name"].is_unique:
        dup = idx.loc[idx["yolo_name"].duplicated(keep=False), "yolo_name"].unique().tolist()
        raise SystemExit(f"{path}: yolo_name is not unique: {dup[:10]}")
    return dict(zip(idx["yolo_name"], idx["uid"]))


def resolve_identity(cfg: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Add scalar ``uid``, ``group``, ``orig_split`` and ``file_name`` columns to predictions.

    ``uid`` comes from the prediction table when present, else from ``basename(yolo_name)``
    through yolo_index.csv; group and source split then come from the manifest. Unresolvable
    rows and colliding mask stems abort the step instead of being silently mismatched.
    """
    for col in ("image", "yolo_name"):
        if col not in df.columns:
            raise SystemExit(f"prediction table is missing the '{col}' column: {list(df.columns)}")
    out = df.copy().reset_index(drop=True)
    if "uid" in out.columns and out["uid"].notna().all():
        out["uid"] = out["uid"].astype(str)
    else:
        lookup = uid_by_yolo_name(cfg)
        names = [Path(str(n)).name for n in out["yolo_name"]]
        unknown = sorted({n for n in names if n not in lookup})
        if unknown:
            raise SystemExit(f"{len(unknown)} predicted image(s) are not in yolo_index.csv: {unknown[:5]}")
        out["uid"] = [lookup[n] for n in names]
    if not out["uid"].is_unique:
        dup = out.loc[out["uid"].duplicated(keep=False), "uid"].unique().tolist()
        raise SystemExit(f"prediction table has duplicate uid(s): {dup[:10]}")
    # masks are stored as <image>_gingiva.png, so a repeated stem would mean overwritten masks
    if not out["image"].is_unique:
        dup = out.loc[out["image"].duplicated(keep=False), "image"].unique().tolist()
        raise SystemExit(f"prediction table has colliding mask stems (image): {dup[:10]}")
    man = manifest_identity(cfg)
    unknown = sorted(set(out["uid"]) - set(man.index))
    if unknown:
        raise SystemExit(f"{len(unknown)} predicted uid(s) are not in the manifest: {unknown[:5]}")
    meta = man.loc[out["uid"]].reset_index(drop=True)
    for col in ("group", "orig_split", "file_name"):
        out[col] = meta[col]
    return out


def boundary_table(cfg: Dict[str, Any], df: pd.DataFrame, mask_dir: Path) -> pd.DataFrame:
    """Per-image boundary errors of the saved masks against the COCO ground truth.

    ``df`` is a prediction table as written by cv_predict.predict_list; ``mask_dir`` holds the
    ``<image>_gingiva.png`` / ``_lip.png`` files of that run. Row access is explicit and
    scalar throughout: identity is resolved once (``resolve_identity``) and the frame is then
    walked with ``itertuples``.
    """
    rows: List[Dict[str, Any]] = []
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    mask_dir = Path(mask_dir)
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in resolve_identity(cfg, df).itertuples(index=False):
        key = (str(r.group), str(r.orig_split))
        if key not in cache:
            cache[key] = load_annotations(coco_root, key[0], key[1], cfg["coco"])
        ann = cache[key]
        im = next((i for i in ann["images"] if i["file_name"] == r.file_name), None)
        if im is None:
            raise SystemExit(f"{r.uid}: file_name {r.file_name!r} not found in {coco_root / key[0] / key[1]}")
        shape = (int(im["height"]), int(im["width"]))
        gt = from_coco(ann["annotations"], int(im["id"]), shape, cfg["coco"]["category_ids"])
        pm = from_png(mask_dir / f"{r.image}_gingiva.png", mask_dir / f"{r.image}_lip.png", expected_shape=shape)
        rec: Dict[str, Any] = {"uid": r.uid, "image": r.image, "group": r.group,
                               **{f"gingiva_{k}": v for k, v in boundary_report(pm.gingiva, gt.gingiva).items()}}
        if gt.lip is not None and pm.lip is not None:
            rec.update({f"lip_{k}": v for k, v in boundary_report(pm.lip, gt.lip).items() if k in ("mask_iou", "boundary_iou")})
        rows.append(rec)
    return pd.DataFrame(rows)


def boundary_summary(b: pd.DataFrame) -> Dict[str, Any]:
    """``boundary_summary`` / ``boundary_by_group`` blocks of test_metrics.json."""
    numeric = [c for c in b.columns if c.startswith(("gingiva_", "lip_")) and b[c].dtype != object]
    return {
        "boundary_n_images": int(len(b)),
        "boundary_summary": {c: {"mean": float(b[c].mean()), "median": float(b[c].median())} for c in numeric},
        "boundary_by_group": {str(g): {c: float(part[c].mean()) for c in GROUP_COLUMNS if c in part.columns}
                              for g, part in b.groupby("group")},
    }


def load_metrics(pred: Path) -> Dict[str, Any]:
    f = Path(pred) / "test_metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def write_metrics(pred: Path, metrics: Dict[str, Any]) -> Path:
    f = Path(pred) / "test_metrics.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    return f


def finalize(cfg: Dict[str, Any], pred: Path, df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3: boundary artefacts, then the summary, then DONE — strictly in that order.

    DONE is written last and only when the validation metrics are present, so an
    interrupted or metrics-only run never marks the step complete; the already written
    predictions stay on disk and ``--metrics-only`` picks up from here.
    """
    b = boundary_table(cfg, df, Path(pred) / "test")
    Path(pred).mkdir(parents=True, exist_ok=True)
    b.to_csv(Path(pred) / "boundary_error.csv", index=False)
    metrics.update(boundary_summary(b))
    write_metrics(pred, metrics)
    if "per_class" in metrics:
        mark_done(cfg, "eval", {"n_images": int(len(b))})
    else:
        print("[eval] boundary artefacts written, but test_metrics.json has no per_class block "
              "(validation has not run) — DONE not written; re-run without --metrics-only")
    return metrics


def load_predictions(pred: Path, test_list: Path) -> pd.DataFrame:
    """The prediction table of a previous run, checked against the test list."""
    csv = Path(pred) / "test" / "test_predictions.csv"
    if not csv.exists():
        raise SystemExit(f"no predictions to score: {csv} is missing — run the step without --metrics-only")
    df = pd.read_csv(csv)
    expected = len([n for n in test_list.read_text().splitlines() if n.strip()]) if test_list.exists() else None
    if expected is not None and len(df) != expected:
        raise SystemExit(f"{csv}: {len(df)} rows but the test list has {expected} images — re-run with --repredict")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None, help="default runs/final/weights/best.pt")
    ap.add_argument("--batch", type=int, default=None, help="images per prediction chunk (default: yolo.predict_batch in config.yaml, 8)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="skip validation and prediction; recompute boundary_error.csv and the summary from test/test_predictions.csv")
    ap.add_argument("--repredict", action="store_true", help="predict again even if test/test_predictions.csv is already complete")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    pred = resolve(cfg, cfg["paths"]["predictions"])
    best = Path(args.weights) if args.weights else run_dir(cfg, "final") / "weights" / "best.pt"
    test_list = ds / "lists" / "main_test.txt"
    from gsv4.train.cv_predict import predict_batch_size, predict_list

    if args.dry_run:
        n = len(test_list.read_text().splitlines()) if test_list.exists() else "MISSING"
        have = (pred / "test" / "test_predictions.csv").exists()
        print(f"[eval] DRY RUN — weights {best} (exists: {best.exists()}); test images: {n}; predict_batch "
              f"{predict_batch_size(cfg, args.batch)}; existing predictions: {have}; metrics_only={args.metrics_only}; outputs -> {pred}")
        return 0

    if args.metrics_only:
        df = load_predictions(pred, test_list)
        metrics = finalize(cfg, pred, df, load_metrics(pred))
        print(json.dumps(metrics["boundary_by_group"], indent=1))
        return 0

    if is_done(cfg, "eval"):
        print("[eval] already DONE")
        return 0
    if not best.exists():
        raise SystemExit(f"weights not found: {best}")
    from ultralytics import YOLO

    model = YOLO(str(best))  # validation instance only; a fresh one is loaded for prediction
    metrics: Dict[str, Any] = {"weights": str(best), "split": "test", "n_images": len(test_list.read_text().splitlines())}
    pred.mkdir(parents=True, exist_ok=True)
    for standard in (True, False):
        st = eval_settings(cfg, standard)
        print(f"[eval] validation at the {st['kind']} settings: conf {st['conf']}, iou {st['iou']}, max_det {st['max_det']}")
        m = model.val(data=str(ds / "data_main.yaml"), split="test", imgsz=st["imgsz"], conf=st["conf"], iou=st["iou"],
                      max_det=st["max_det"], plots=not standard, verbose=True, project=str(run_dir(cfg, "eval")),
                      name="test" if not standard else "test_standard", exist_ok=True)
        names = {int(k): v for k, v in m.names.items()}
        suffix = "_standard" if standard else "_operating_point"
        metrics[f"per_class{suffix}"] = per_class_metrics(m, names)
        metrics[f"eval_settings{suffix}"] = st
        metrics[f"speed_ms{suffix}"] = getattr(m, "speed", None)
        if not standard:
            # the confusion matrix and its false positive / false negative counts describe the
            # configuration the pipeline runs at; at conf 0.001 they would count noise
            cmp_ = run_dir(cfg, "eval") / "test" / "confusion_matrix.png"
            if cmp_.exists():
                shutil.copy(cmp_, pred / "confusion_matrix.png")
            try:
                metrics["confusion_matrix"] = m.confusion_matrix.matrix.tolist()
            except Exception:  # noqa: BLE001
                pass
        del m
        release_cuda()
    # `per_class` stays the reported block (standard settings) so that readers of the old key are
    # not silently handed operating-point numbers
    metrics["per_class"] = metrics["per_class_standard"]
    metrics["eval_settings"] = metrics["eval_settings_standard"]
    metrics["speed_ms"] = metrics["speed_ms_operating_point"]
    write_metrics(pred, metrics)  # before prediction: a crash below must not cost the validation pass

    # everything needed from validation is now plain Python; free the validation model
    # and its cached CUDA blocks before the (separate) prediction model is loaded
    del m, model
    release_cuda()

    # test-set masks from the final model (reused when a complete table is already on disk)
    out_dir = pred / "test"
    reuse = None
    if not args.repredict:
        try:
            reuse = load_predictions(pred, test_list)
        except SystemExit as exc:
            print(f"[eval] predicting from scratch ({exc})")
    if reuse is not None:
        print(f"[eval] reusing {len(reuse)} predictions from {out_dir / 'test_predictions.csv'} (--repredict to redo)")
        df = reuse
    else:
        df = predict_list(cfg, best, test_list, out_dir, "test", {"weights": str(best)}, batch=args.batch)
        df.to_csv(out_dir / "test_predictions.csv", index=False)

    metrics = finalize(cfg, pred, df, metrics)
    print(json.dumps(metrics["per_class"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
