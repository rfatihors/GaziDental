#!/usr/bin/env python
"""Architecture comparison — runs on the training workstation (outputs/08_architecture/PROTOCOL.md).

    python scripts/run_architecture_comparison.py --dry-run
    python scripts/run_architecture_comparison.py                      # yolo11x-seg, yolo26x-seg
    python scripts/run_architecture_comparison.py --aggregate-only     # rebuild the tables

For every architecture and seed, in order and restartable (a DONE marker per run):

1. train on the same participant-level partition with the SAME shared budget (epochs, patience,
   image size, batch, seed) and PUBLISHED DEFAULTS for everything architecture-internal — no
   optimiser, learning rate or augmentation override, so no architecture is tuned in favour of
   another (PROTOCOL.md §2);
2. evaluate once on the fixed test set at both settings (standard mAP, and the pipeline's
   operating point);
3. predict the high-smile-line test images with the operating-point settings and write the masks;
4. measure gingival display with the method and scale fixed in configs/config.yaml, and compute
   the gingival edge errors against the annotated masks.

Then aggregate: per seed, per model (mean ± SD), the paired difference between architectures with
a bootstrap interval over the shared images, and the pre-registered decision.

RF-DETR is added by ``scripts/rfdetr_probe.py`` once its install is verified; its per-image rows
land in the same table and the aggregation needs no change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.architecture import (  # noqa: E402
    bias_scatter_table, decision, edge_table, error_correlation, integrity_check, mask_pixel_mm, model_table,
    paired_difference, per_image_errors, resolution_verdict, seed_averaged, seed_table,
)
from gsv4.eval.oracle import combo_name, measure_images  # noqa: E402
from gsv4.eval.prediction import md_table  # noqa: E402
from gsv4.train.common import per_class_metrics  # noqa: E402
from gsv4.train.evaluate_test import boundary_table, eval_settings  # noqa: E402

# Published-default architectures. Only the shared budget below is imposed on them.
MODELS = {"yolo11x-seg": "yolo11x-seg.pt", "yolo26x-seg": "yolo26x-seg.pt"}
SEEDS = (42, 43, 44)
# Shared budget (PROTOCOL.md §2). Nothing else is passed to train(), so the optimiser, the learning
# rate, the schedule and the augmentation policy are each architecture's own published defaults.
SHARED = {"epochs": 100, "patience": 20, "imgsz": 640, "batch": 16, "deterministic": True}
BASELINE = "yolo11x-seg"
PROTOCOL_IMGSZ = 640
# Resolution controls (PROTOCOL_ADDENDUM_resolution.md). A run at any other input size is labelled
# "<model>@<imgsz>" and is a sensitivity analysis, never a member of the comparison: §2 of the
# protocol held the input resolution fixed, and these runs deliberately break that.
CONTROL_A = "yolo11x-seg@1024"           # finer vertical mask grid than RF-DETR at 624
CONTROL_B = "rfdetr-seg-large@432"       # vertical mask grid matched to YOLOv11x at 640
RFDETR_PROTOCOL = "rfdetr-seg-large"


def label_for(model: str, imgsz: int) -> str:
    return model if imgsz == PROTOCOL_IMGSZ else f"{model}@{imgsz}"


def is_control(label: str) -> bool:
    return "@" in label


def run_dir_for(cfg, label: str, seed: int) -> Path:
    return resolve(cfg, cfg["paths"]["runs"]) / "arch" / f"{label}_s{seed}"


def train_one(cfg, model: str, seed: int, data_yaml: Path, dry_run: bool, imgsz: int = PROTOCOL_IMGSZ) -> Path:
    """Train one architecture at its published defaults; returns best.pt.

    ``imgsz`` other than the protocol's 640 makes this a resolution control, labelled accordingly.
    """
    label = label_for(model, imgsz)
    d = run_dir_for(cfg, label, seed)
    best = d / "weights" / "best.pt"
    args = {"data": str(data_yaml), "seed": seed, "project": str(d.parent), "name": d.name,
            "exist_ok": True, "plots": False, "verbose": True, **{**SHARED, "imgsz": imgsz}}
    if dry_run:
        print(f"[{label} s{seed}] DRY RUN — weights {MODELS[model]}; args {json.dumps(args)}")
        return best
    if (d / "DONE").exists():
        print(f"[{label} s{seed}] already DONE")
        return best
    from ultralytics import YOLO

    last = d / "weights" / "last.pt"
    if last.exists():
        print(f"[{label} s{seed}] resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(MODELS[model]).train(**args)
    if not best.exists():
        raise SystemExit(f"[{label} s{seed}] training ended without weights/best.pt")
    (d / "DONE").write_text(json.dumps({"args": args, "published_defaults": True, "control": is_control(label)}, indent=1), encoding="utf-8")
    return best


def evaluate_one(cfg, label: str, seed: int, best: Path, ds: Path, out_dir: Path, batch: int, imgsz: int = PROTOCOL_IMGSZ) -> pd.DataFrame:
    """Test-set metrics at both settings, masks for the high test images, measurement and edges."""
    from ultralytics import YOLO

    from gsv4.train.cv_predict import predict_list
    from gsv4.train.common import release_cuda

    tag = f"{label}_s{seed}"
    m_rows: List[Dict[str, Any]] = []
    y = YOLO(str(best))
    for standard in (True, False):
        st = eval_settings(cfg, standard)
        r = y.val(data=str(ds / "data_main.yaml"), split="test", imgsz=imgsz, conf=st["conf"], iou=st["iou"],
                  max_det=st["max_det"], plots=False, verbose=False, project=str(out_dir / "val"), name=f"{tag}_{st['kind']}", exist_ok=True)
        names = {int(k): v for k, v in r.names.items()}
        for cname, vals in per_class_metrics(r, names).items():
            m_rows.append({"model": label, "seed": seed, "settings": st["kind"], "conf": st["conf"], "max_det": st["max_det"], "imgsz": imgsz, "class": cname, **vals})
        del r
        release_cuda()
    del y
    release_cuda()
    pd.DataFrame(m_rows).to_csv(out_dir / "segmentation_metrics" / f"{tag}.csv", index=False)

    # masks of the high-smile-line test images, at the operating point (what the pipeline uses)
    high_list = ds / "lists" / "arch_test_high.txt"
    mask_dir = out_dir / "masks" / tag
    df = predict_list(cfg, best, high_list, mask_dir, tag, {"model": label, "seed": seed}, batch=batch)
    df.to_csv(out_dir / "predictions" / f"{tag}.csv", index=False)
    return df


def measure_one(cfg, label: str, seed: int, rows: pd.DataFrame, pred_df: pd.DataFrame, out_dir: Path, combo: str, k: float) -> pd.DataFrame:
    """Millimetre measurement and gingival edge errors for one run."""
    tag = f"{label}_s{seed}"
    mask_dir = out_dir / "masks" / tag
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    meas = measure_images(rows, cfg, coco_root, mask_source=str(mask_dir)).set_index("uid")
    b = boundary_table(cfg, pred_df, mask_dir).set_index("uid")
    per = rows.set_index("uid").copy()
    per["model"], per["seed"] = label, seed
    per["selected_mm"] = meas[f"{combo}_px"] / k
    per["qc_flags"] = meas["qc_flags"].fillna("")
    per["n_gingiva_instances"] = meas["n_gingiva_instances"]
    per["n_lip_instances"] = meas["n_lip_instances"]
    # instances the extractor dropped for having no role; recorded by the predictor, carried here so
    # that integrity_check sees it without opening a second file
    if "n_ignored" in pred_df.columns and "uid" in pred_df.columns:
        per["n_ignored"] = pred_df.set_index("uid")["n_ignored"].reindex(per.index)
    elif "n_ignored" in pred_df.columns and "image" in pred_df.columns:
        per["n_ignored"] = per["image"].map(pred_df.set_index("image")["n_ignored"])
    else:
        per["n_ignored"] = 0
    for c in ("gingiva_mask_iou", "gingiva_boundary_iou", "gingiva_top_edge_mae_px", "gingiva_top_edge_bias_px",
              "gingiva_bottom_edge_mae_px", "gingiva_bottom_edge_bias_px", "lip_mask_iou"):
        per[c] = b[c] if c in b.columns else float("nan")
    return per.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of " + ", ".join(MODELS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--batch", type=int, default=None, help="prediction chunk size (cv_predict)")
    ap.add_argument("--imgsz", type=int, default=PROTOCOL_IMGSZ,
                    help=f"input size; anything other than {PROTOCOL_IMGSZ} is a resolution control "
                         "(PROTOCOL_ADDENDUM_resolution.md) and is labelled <model>@<imgsz>")
    ap.add_argument("--aggregate-only", action="store_true", help="rebuild the tables from the per-image rows on disk")
    ap.add_argument("--measure-only", action="store_true",
                    help="skip training; measure every predictions/<tag>.csv that has no per_image/<tag>.csv yet "
                         "(this is how RF-DETR masks, produced in their own virtualenv, enter the same tables)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    out_dir = outputs / "08_architecture"
    for sub in ("masks", "predictions", "segmentation_metrics", "val", "per_image"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    mcfg = cfg["measurement"]
    combo = combo_name(mcfg["method"]["regioning"], mcfg["method"]["estimator"], bool(mcfg["method"]["anchored"]))
    k = float(mcfg["px_per_mm"])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # the comparison set: the high-smile-line test images with a clinical reference measurement
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    rows = manifest[manifest["keep"] & manifest["has_reference_measurement"] & (manifest["split"] == "test")][
        ["uid", "image", "patient_id", "group", "width", "height", "frame_ok", "orig_split", "file_name"]].reset_index(drop=True)
    rows["ref_mm"] = rows["uid"].map(manifest.set_index("uid")["reference_mean_mm"])
    idx = pd.read_csv(ds / "yolo_index.csv").set_index("uid")
    high_list = ds / "lists" / "arch_test_high.txt"
    if not (args.aggregate_only or args.measure_only):
        high_list.parent.mkdir(parents=True, exist_ok=True)
        high_list.write_text("\n".join(f"images/all/{idx.loc[u, 'yolo_name']}" for u in rows["uid"]) + "\n", encoding="utf-8")
        # the same set as a uid list, for predictors that read the COCO dataset instead of this list
        rows[["uid", "image"]].to_csv(out_dir / "arch_test_high_uids.csv", index=False)
    print(f"[arch] comparison set: {len(rows)} high-smile-line test images with a clinical reference")

    if args.dry_run:
        for model in models:
            for seed in seeds:
                train_one(cfg, model, seed, ds / "data_main.yaml", dry_run=True, imgsz=args.imgsz)
        print(f"[arch] DRY RUN — {len(models) * len(seeds)} runs; masks -> {out_dir / 'masks'}; list {high_list} ({len(rows)} images)")
        return 0

    if args.measure_only:
        for pf in sorted((out_dir / "predictions").glob("*.csv")):
            tag = pf.stem
            per_path = out_dir / "per_image" / f"{tag}.csv"
            if per_path.exists():
                continue
            label, _, seed_s = tag.rpartition("_s")
            if not seed_s.isdigit():
                raise SystemExit(f"cannot read a model and seed from {pf.name}; expected <model>_s<seed>.csv")
            measure_one(cfg, label, int(seed_s), rows, pd.read_csv(pf), out_dir, combo, k).to_csv(per_path, index=False)
            print(f"[{tag}] measured -> {per_path}")
        models, seeds = [], []

    for model in models:
        if model not in MODELS:
            raise SystemExit(f"unknown model {model}; known: {', '.join(MODELS)}")
        for seed in seeds:
            label = label_for(model, args.imgsz)
            tag = f"{label}_s{seed}"
            per_path = out_dir / "per_image" / f"{tag}.csv"
            if per_path.exists():
                print(f"[{tag}] per-image table already on disk, skipping")
                continue
            best = train_one(cfg, model, seed, ds / "data_main.yaml", dry_run=False, imgsz=args.imgsz)
            pred_df = evaluate_one(cfg, label, seed, best, ds, out_dir, args.batch, imgsz=args.imgsz)
            measure_one(cfg, label, seed, rows, pred_df, out_dir, combo, k).to_csv(per_path, index=False)
            print(f"[{tag}] done -> {per_path}")

    # ---------------------------------------------------------------- aggregate
    parts = sorted((out_dir / "per_image").glob("*.csv"))
    if not parts:
        raise SystemExit("no per-image tables to aggregate")
    long = per_image_errors(pd.concat([pd.read_csv(p) for p in parts], ignore_index=True))
    long.to_csv(out_dir / "per_image_all.csv", index=False)
    integrity = integrity_check(long)
    integrity.to_csv(out_dir / "integrity_check.csv", index=False)
    suspect = integrity[integrity["suspect"]]
    for r in suspect.itertuples():
        print(f"[arch] SUSPECT {r.model}: {r.problems}")
    seeds_t = seed_table(long)
    models_t = model_table(seeds_t)
    edges = edge_table(long, k)
    seeds_t.to_csv(out_dir / "by_seed.csv", index=False)
    models_t.to_csv(out_dir / "by_model.csv", index=False)
    edges.to_csv(out_dir / "edges_by_seed.csv", index=False)
    avg = seed_averaged(long)
    present = sorted(long["model"].unique())
    members = [m for m in present if not is_control(m)]
    controls = [m for m in present if is_control(m)]
    comparisons = []
    base = BASELINE if BASELINE in members else (members or present)[0]
    for other in [m for m in members if m != base]:
        c = paired_difference(avg[avg.model == base], avg[avg.model == other])
        c.update({"model_a": base, "model_b": other})
        comparisons.append(c)
    dec = decision(comparisons, base)
    comp_t = pd.DataFrame(comparisons)
    if len(comp_t):
        comp_t.to_csv(out_dir / "paired_comparisons.csv", index=False)

    # ---- resolution controls (PROTOCOL_ADDENDUM_resolution.md); never merged into the comparison
    def pair(a: str, b: str):
        if a not in present or b not in present:
            return {"n": 0, "model_a": a, "model_b": b}
        c = paired_difference(avg[avg.model == a], avg[avg.model == b])
        c.update({"model_a": a, "model_b": b})
        return c

    ctl_b = pair(BASELINE, CONTROL_B)                 # YOLOv11x at 640 against RF-DETR at 432
    ctl_a = pair(CONTROL_A, RFDETR_PROTOCOL)          # YOLOv11x at 1024 against RF-DETR at 624
    mae_by_config = {m: float(np.abs(avg[avg.model == m]["selected_mm"].to_numpy(float)
                                     - avg[avg.model == m]["ref_mm"].to_numpy(float)).mean()) for m in present}
    verdict = resolution_verdict(ctl_b, ctl_a, mae_by_config)
    ctl_t = pd.DataFrame([c for c in (ctl_b, ctl_a) if c.get("n")])
    if len(ctl_t):
        ctl_t.to_csv(out_dir / "resolution_controls.csv", index=False)
    geom = pd.DataFrame([{"configuration": name, **mask_pixel_mm(size, k, letterbox=lb)}
                         for name, size, lb in (("yolo11x-seg @ 640", 640, True), (CONTROL_A, 1024, True),
                                                (RFDETR_PROTOCOL + " @ 624", 624, False), (CONTROL_B, 432, False))])
    geom.to_csv(out_dir / "mask_pixel_geometry.csv", index=False)
    bias_scatter = bias_scatter_table(long)
    bias_scatter.to_csv(out_dir / "bias_scatter.csv", index=False)
    corr = error_correlation(long)
    corr.to_csv(out_dir / "error_correlation.csv")
    sm = sorted((out_dir / "segmentation_metrics").glob("*.csv"))
    seg = pd.concat([pd.read_csv(p) for p in sm], ignore_index=True) if sm else pd.DataFrame()
    if len(seg):
        seg.to_csv(out_dir / "segmentation_metrics_all.csv", index=False)
    std_seg = seg[(seg["settings"] == "standard") & (seg["class"] != "all")] if len(seg) else pd.DataFrame()

    banner = "" if not len(suspect) else (
        "> **These results are not final.** The integrity check below flags "
        + ", ".join(f"`{r.model}` ({r.problems})" for r in suspect.itertuples())
        + ". A run flagged here is not measuring what it claims to; its rows must be withdrawn, the masks produced again "
          "and the tables rebuilt before anything is reported. See `scripts/check_mask_classes.py` and "
          "README_TRAINING.md §6.5.\n\n")
    def _stage6_row(sub: str, label: str) -> str:
        """One line per final model, from its own Stage-6 error decomposition (nothing re-measured here)."""
        d = resolve(cfg, Path(cfg["paths"]["outputs"]) / sub / "error_decomposition.csv")
        if not d.exists():
            return f"| {label} | — | — | — | — | pending: `{sub}/error_decomposition.csv` |"
        e = pd.read_csv(d)
        acc = resolve(cfg, Path(cfg["paths"]["outputs"]) / sub / "measurement_accuracy.csv")
        model = "?"
        if acc.exists():
            a = pd.read_csv(acc)
            names = sorted({str(x) for x in a.get("model", pd.Series(dtype=str)).dropna().unique() if "COCO annotations" not in str(x)})
            names = [n for n in names if not any(o != n and o.startswith(n) for o in names)]
            model = ", ".join(names) or "?"
        return (f"| {label} | {model} | {e['e_total'].abs().mean():.2f} | {e['e_meas'].abs().mean():.2f} | "
                f"{e['e_total'].abs().mean() - e['e_meas'].abs().mean():+.2f} | "
                f"{e['e_seg'].abs().mean():.2f} (bias {e['e_seg'].mean():+.2f}) |")

    stage6_md = ("| Stage 6 run | model | pipeline MAE, mm | geometry part (GT masks), mm | what the model adds, mm | segmentation error component, mm |\n"
                 "|---|---|---|---|---|---|\n"
                 + _stage6_row("09_final_rfdetr", "current final model") + "\n"
                 + _stage6_row("06_prediction", "previous final model") + "\n\n"
                 "The geometry part is the same measurement on the annotated masks (Stage 3) and is a property of the method, "
                 "not of the model; the segmentation part is what the predicted masks add to it. A model whose segmentation part "
                 "is near zero measures the gingival display as well as the annotation allows.")

    md = f"""# Architecture comparison — results

{banner}
Protocol: `PROTOCOL.md`, written and committed before any run. Comparison set: {len(rows)} high-smile-line
test images with a clinical reference measurement. Measurement method **{combo}** at **{k:.2f} px/mm**, both fixed
in configs/config.yaml and unchanged here. Every architecture ran at its published defaults with the shared budget
{json.dumps(SHARED)} and seeds {list(seeds_t['seed'].unique())}.

## Primary outcome: millimetre error against the clinical reference

Per seed:

{md_table(seeds_t)}

Per model (mean ± SD over seeds):

{md_table(models_t)}

## Paired difference (seed-averaged, bootstrap over the shared images)

Positive `diff_mae_mm` means the first model has the larger error, i.e. the second is better.

{md_table(comp_t) if len(comp_t) else '(only one architecture present)'}

Comparison members: {', '.join(members) or 'none'}. Resolution controls, reported separately: {', '.join(controls) or 'none'}.

## Gingival edge errors, mm at the global scale

{md_table(edges)}

## Is the difference a shift or is it scatter?

A lead in mean absolute error can be a constant offset, which calibration removes, or scatter, which
it does not. The seed-averaged per-image error is split below; `removable_by_calibration_mm` is the
most any post-hoc bias correction could take off that model's MAE.

{md_table(bias_scatter)}

Correlation of the per-image error between architectures (seed-averaged):

{md_table(corr.reset_index())}

## Secondary: per-class segmentation metrics on the fixed test set, standard settings

{md_table(std_seg) if len(std_seg) else '(pending)'}

Models evaluated here: {', '.join(sorted(std_seg['model'].unique())) if len(std_seg) else 'none'}. A
predictor outside the Ultralytics framework is not evaluated through `model.val()` and therefore has
no row; its own trainer's mask AP is reported in its run record instead, and the two are not
interchangeable.

## Resolution controls (PROTOCOL_ADDENDUM_resolution.md)

Sensitivity analyses, not members of the comparison: §2 held the input resolution fixed and these
runs deliberately break that. Vertical mask-pixel size is the one that matters, because the measured
quantity is a vertical thickness.

{md_table(geom)}

{md_table(ctl_t) if len(ctl_t) else '(no control has run yet)'}

Pre-registered reading: control B {'has not run' if not ctl_b.get('n') else ('points to resolution' if verdict['resolution_explains_b'] else 'points to architecture')},
control A {'has not run' if not ctl_a.get('n') else ('points to resolution' if verdict['resolution_explains_a'] else 'points to architecture')}.

**Verdict (rule {verdict['rule']}): {verdict['outcome']}.**

## Pre-registered decision (PROTOCOL.md §8)

Rule: {dec['rule']}.

**Outcome: {dec['outcome']}.**

## Integrity check

A run that produced no lip mask anywhere, dropped instances for having no role, or whose gingival
edge error is entirely systematic is flagged here and must not be reported until it is repeated.

{md_table(integrity)}

## What the difference means for the measurement (Stage 6 of each final model)

The comparison above is 29 test images. The clinically meaningful statement is what each architecture
adds to the measurement error on the full out-of-fold set, against the same method and scale on the
ground-truth masks — i.e. how much of the pipeline error is the segmentation's.

{stage6_md}

## Declared limits

* The architectures do not share a native mask resolution; part of any difference may be mask-head
  resolution rather than the ability to find the gingival margin (PROTOCOL.md §6).
* Class agreement and kappa are deliberately not reported per architecture: at n = {len(rows)} the interval
  is too wide to separate architectures (PROTOCOL.md §3).
* The rows above are published-default runs. The final model of the manuscript is the tuned
  YOLOv11x-seg of the main study and is not a member of this comparison.
"""
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    print(md.split("## Pre-registered decision")[1][:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
