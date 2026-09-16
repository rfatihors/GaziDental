"""Boundary/summary stage of evaluate_test.py against the *real* artefact format: a
prediction table as cv_predict.predict_list writes it, a dataset manifest, a yolo_index in
which the image stem repeats across groups, a Roboflow-style COCO export and saved mask
PNGs on disk.

Regression: ``image`` (the Roboflow-free stem) is not unique across groups, so indexing
yolo_index by it returned a DataFrame and ``Path(info["path"]).parts[1]`` raised
``TypeError: expected str, bytes or os.PathLike object, not Series``. Identity is ``uid``,
and group / COCO source split come from manifest columns, never from a path string.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsv4.masks.extract import ClassMasks, from_coco, save_masks
from gsv4.train.evaluate_test import (
    boundary_summary,
    boundary_table,
    finalize,
    load_predictions,
    resolve_identity,
)

CATEGORY_IDS = {"gingiva": 1, "lip": 2}
SHAPE = (60, 80)  # (height, width)


def _rect_poly(x0: int, x1: int, top: int, thickness: int) -> list:
    """COCO polygon that rasterises to exactly rows ``top..top+thickness-1``, cols ``x0..x1-1``."""
    xr, yb = x1 - 1, top + thickness - 1
    return [float(x0), float(top), float(xr), float(top), float(xr), float(yb), float(x0), float(yb)]


def _coco(images: list) -> dict:
    """Roboflow-shaped annotation file: ``images`` is [(file_name, gingiva_geom, lip_geom)]."""
    out = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "diseti"}, {"id": 2, "name": "dudak"}]}
    aid = 1
    for i, (file_name, ging, lip) in enumerate(images, start=1):
        out["images"].append({"id": i, "file_name": file_name, "height": SHAPE[0], "width": SHAPE[1]})
        for cid, geom in ((1, ging), (2, lip)):
            if geom is None:
                continue
            out["annotations"].append({"id": aid, "image_id": i, "category_id": cid, "segmentation": [_rect_poly(*geom)]})
            aid += 1
    return out


def _shift_down(mask: np.ndarray, dy: int) -> np.ndarray:
    out = np.zeros_like(mask)
    out[dy:] = mask[:-dy]
    return out


@pytest.fixture()
def project(tmp_path):
    """A whole project layout: manifest, yolo_index, COCO export, predictions and masks.

    ``IMG_10`` deliberately exists in both ``high`` (source split ``train``) and ``normal``
    (source split ``valid``) with *different* geometry, and only the ``high`` one is
    predicted — so resolving identity by stem, or the source split by path position, gives a
    visibly wrong ground truth.
    """
    root = tmp_path
    coco_root, ds, pred = root / "coco", root / "yolo", root / "outputs" / "05_predictions"
    (root / "manifest").mkdir(parents=True)
    ds.mkdir(parents=True)

    files = {
        "high/IMG_10": ("high", "train", "IMG_10.rf.aaa.jpg", (10, 70, 20, 6), (10, 70, 8, 5)),
        "normal/IMG_10": ("normal", "valid", "IMG_10.rf.bbb.jpg", (40, 60, 45, 4), None),
        "high/IMG_20": ("high", "test", "IMG_20.rf.ccc.jpg", (5, 75, 30, 8), (5, 75, 15, 4)),
    }
    for group, split in {(g, s) for g, s, *_ in files.values()}:
        d = coco_root / group / split
        d.mkdir(parents=True)
        rows = [(fn, ging, lip) for g, s, fn, ging, lip in files.values() if (g, s) == (group, split)]
        (d / "_annotations.coco.json").write_text(json.dumps(_coco(rows)), encoding="utf-8")

    pd.DataFrame([{"uid": uid, "image": uid.split("/")[1], "group": g, "orig_split": s, "file_name": fn,
                   "path": f"{g}/{s}/{fn}", "keep": True, "split": "test"}
                  for uid, (g, s, fn, _, _) in files.items()]).to_csv(root / "manifest" / "dataset_manifest.csv", index=False)
    pd.DataFrame([{"uid": uid, "image": uid.split("/")[1], "group": g, "split": "test",
                   "yolo_name": f"{g}__{uid.split('/')[1]}.jpg", "file_name": fn, "path": f"{g}/{s}/{fn}"}
                  for uid, (g, s, fn, _, _) in files.items()]).to_csv(ds / "yolo_index.csv", index=False)

    cfg = {"_root": root, "paths": {"coco_root": str(coco_root), "yolo_dataset": str(ds),
                                    "manifest_dir": str(root / "manifest"), "predictions": str(pred), "runs": str(root / "runs")},
           "coco": {"annotation_file": "_annotations.coco.json", "category_ids": CATEGORY_IDS}}

    # masks on disk: high/IMG_10 predicted perfectly, high/IMG_20 shifted 3 px down
    out_dir = pred / "test"
    for uid, dy in (("high/IMG_10", 0), ("high/IMG_20", 3)):
        g, s, fn, _, _ = files[uid]
        ann = json.loads((coco_root / g / s / "_annotations.coco.json").read_text(encoding="utf-8"))
        im = next(i for i in ann["images"] if i["file_name"] == fn)
        gt = from_coco(ann["annotations"], int(im["id"]), SHAPE, CATEGORY_IDS)
        g_mask = gt.gingiva if dy == 0 else _shift_down(gt.gingiva, dy)
        l_mask = None if gt.lip is None else (gt.lip if dy == 0 else _shift_down(gt.lip, dy))
        save_masks(ClassMasks(g_mask, l_mask, 1, int(l_mask is not None)), out_dir, uid.split("/")[1])

    rows = []
    for uid in ("high/IMG_10", "high/IMG_20"):
        g, s, fn, _, _ = files[uid]
        stem = uid.split("/")[1]
        rows.append({"image": stem, "yolo_name": str(ds / "images" / "all" / f"{g}__{stem}.jpg"), "n_gingiva": 1,
                     "n_lip": 1, "max_conf": 0.9, "mask_source": "yolo:masks.data", "width": SHAPE[1], "height": SHAPE[0],
                     "weights": str(root / "runs" / "final" / "weights" / "best.pt")})
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_predictions.csv", index=False)
    (ds / "lists").mkdir()
    (ds / "lists" / "main_test.txt").write_text("\n".join(r["yolo_name"] for r in rows) + "\n")
    return cfg, pred, df


def test_resolve_identity_gives_scalars_and_the_right_group(project):
    """The regression: a stem shared by two groups must still resolve to one scalar row."""
    cfg, _, df = project
    ident = resolve_identity(cfg, df)
    assert list(ident["uid"]) == ["high/IMG_10", "high/IMG_20"]
    assert list(ident["group"]) == ["high", "high"]
    assert list(ident["orig_split"]) == ["train", "test"]          # from the manifest column
    assert list(ident["file_name"]) == ["IMG_10.rf.aaa.jpg", "IMG_20.rf.ccc.jpg"]
    for col in ("uid", "group", "orig_split", "file_name"):
        assert all(isinstance(v, str) for v in ident[col]), col   # never a Series / DataFrame slice
    # the old stem-indexed lookup is exactly what broke
    idx = pd.read_csv(Path(cfg["paths"]["yolo_dataset"]) / "yolo_index.csv").set_index("image")
    assert isinstance(idx.loc["IMG_10"], pd.DataFrame)


def test_resolve_identity_is_immune_to_the_path_layout(project):
    """Group and source split are manifest columns, so a different path spelling changes nothing."""
    cfg, _, df = project
    man_path = Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"
    man = pd.read_csv(man_path)
    man["path"] = "some/other/prefix/" + man["file_name"]          # parts[1] would now be "other"
    man.to_csv(man_path, index=False)
    ident = resolve_identity(cfg, df)
    assert list(ident["orig_split"]) == ["train", "test"]
    assert list(ident["group"]) == ["high", "high"]


def test_boundary_table_and_summary_on_the_real_prediction_format(project):
    cfg, pred, df = project
    b = boundary_table(cfg, df, pred / "test")
    assert list(b["uid"]) == ["high/IMG_10", "high/IMG_20"]
    assert list(b["group"]) == ["high", "high"]
    r10 = b.set_index("uid").loc["high/IMG_10"]
    assert r10["gingiva_mask_iou"] == pytest.approx(1.0)           # right group -> perfect match
    assert r10["gingiva_boundary_iou"] == pytest.approx(1.0)
    assert r10["gingiva_top_edge_mae_px"] == pytest.approx(0.0)
    assert r10["lip_mask_iou"] == pytest.approx(1.0)
    r20 = b.set_index("uid").loc["high/IMG_20"]
    assert r20["gingiva_top_edge_mae_px"] == pytest.approx(3.0)    # the 3 px shift, recovered
    assert r20["gingiva_bottom_edge_mae_px"] == pytest.approx(3.0)
    assert 0.0 < r20["gingiva_mask_iou"] < 1.0

    s = boundary_summary(b)
    assert s["boundary_n_images"] == 2
    assert s["boundary_summary"]["gingiva_top_edge_mae_px"]["mean"] == pytest.approx(1.5)
    assert set(s["boundary_by_group"]) == {"high"}
    assert s["boundary_by_group"]["high"]["gingiva_mask_iou"] == pytest.approx(b["gingiva_mask_iou"].mean())
    json.dumps(s)                                                  # must be serialisable as written


def test_boundary_table_rejects_ambiguous_or_unknown_rows(project):
    cfg, pred, df = project
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate uid"):
        boundary_table(cfg, dup, pred / "test")
    unknown = df.copy()
    unknown.loc[0, "yolo_name"] = str(Path(unknown.loc[0, "yolo_name"]).parent / "high__IMG_99.jpg")
    with pytest.raises(SystemExit, match="not in yolo_index"):
        boundary_table(cfg, unknown, pred / "test")
    collide = df.copy()
    collide.loc[1, "image"] = "IMG_10"                             # would overwrite the other image's masks
    with pytest.raises(SystemExit, match="colliding mask stems"):
        boundary_table(cfg, collide, pred / "test")


def test_finalize_writes_done_only_after_the_boundary_artefacts(project):
    cfg, pred, df = project
    done = Path(cfg["paths"]["runs"]) / "eval" / "DONE"
    metrics = finalize(cfg, pred, df, {"split": "test", "per_class": {"all": {"seg_map50": 0.9}}})
    assert (pred / "boundary_error.csv").exists()
    assert metrics["boundary_n_images"] == 2 and "boundary_by_group" in metrics
    saved = json.loads((pred / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["per_class"]["all"]["seg_map50"] == 0.9 and saved["boundary_n_images"] == 2
    assert done.exists()


def test_finalize_does_not_write_done_when_boundary_fails(project):
    """A crash in stage 3 must leave the step not DONE (and the predictions on disk)."""
    cfg, pred, df = project
    (pred / "test" / "IMG_20_gingiva.png").unlink()
    with pytest.raises(FileNotFoundError):
        finalize(cfg, pred, df, {"per_class": {}})
    assert not (Path(cfg["paths"]["runs"]) / "eval" / "DONE").exists()
    assert (pred / "test" / "test_predictions.csv").exists()


def test_finalize_without_validation_metrics_does_not_write_done(project):
    """--metrics-only on a run whose validation pass was lost: artefacts yes, DONE no."""
    cfg, pred, df = project
    finalize(cfg, pred, df, {})
    assert (pred / "boundary_error.csv").exists()
    assert not (Path(cfg["paths"]["runs"]) / "eval" / "DONE").exists()


def test_load_predictions_requires_a_complete_table(project):
    cfg, pred, df = project
    test_list = Path(cfg["paths"]["yolo_dataset"]) / "lists" / "main_test.txt"
    assert len(load_predictions(pred, test_list)) == 2
    df.iloc[:1].to_csv(pred / "test" / "test_predictions.csv", index=False)
    with pytest.raises(SystemExit, match="--repredict"):
        load_predictions(pred, test_list)
    (pred / "test" / "test_predictions.csv").unlink()
    with pytest.raises(SystemExit, match="no predictions to score"):
        load_predictions(pred, test_list)
