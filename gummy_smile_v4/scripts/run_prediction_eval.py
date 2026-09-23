#!/usr/bin/env python
"""Stage 6 — accuracy of the full pipeline on predicted masks (after the workstation outputs
came back with ``git pull``).

    python scripts/run_prediction_eval.py [--out 06_prediction] [--n-boot 2000]

The measurement uses the method and global scale **fixed in configs/config.yaml** (chosen in
Stage 3 on ground-truth masks; nothing is re-selected or re-fitted here). Three image sets:

  (a) all 145 reference images, 5-fold out-of-fold masks (fold models)  — PRIMARY
  (b) the 29 test-set high images with the final model's masks           — secondary
  (c) the whole 192-image fixed test set (segmentation quality only)

Outputs (outputs/06_prediction/): oof_check.md, per_image_results.csv (a; read by Stage 7 and
by run_expert_analysis --model-table), per_image_results_test.csv (b), measurement_accuracy.csv/.md,
class_agreement.md, error_decomposition.csv/.md, boundary_error_oof.csv, boundary_by_set.csv/.md,
tooth_level.csv, mixed_models.md, figures, prediction_summary.md, OZET.md, SAPMALAR.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.expert import tooth_mixed_models  # noqa: E402
from gsv4.eval.oracle import COMBOS, combo_name, measure_images  # noqa: E402
from gsv4.eval.prediction import (  # noqa: E402
    EVALUATORS, agreement_metrics, boundary_sets_table, check_oof, error_decomposition, fallback_rate, label_confusion,
    mask_provenance, md_table, measurement_table, rel_to, seg_error_vs_boundary, tooth_long, tooth_table,
)
from gsv4.io.forms import TEETH  # noqa: E402
from gsv4.measure.calibration import bottom_edge_offset_from_config, offset_mm_at  # noqa: E402
from gsv4.measure.qc import QCFlag  # noqa: E402
from gsv4.report.figures import PLOT_DPI, scatter_and_bland_altman  # noqa: E402
from gsv4.rules.thresholds import label_for_mm  # noqa: E402
from gsv4.train.evaluate_test import boundary_table  # noqa: E402

# Okabe–Ito (colour-blind safe), fixed order: (a) OOF, (b) test high final, low, normal
SET_COLOURS = {"a": "#0072B2", "b": "#D55E00", "low": "#009E73", "normal": "#CC79A7"}
MEAS_COLS = ["set", "masks", "masks_dir", "model", "correction", "n", "n_segmentation_failure", "images_with_zeroed_columns", "zeroed_columns_frac_mean", "mae", "rmse", "median_abs_err", "r", "icc2_1", "icc2_1_ci_low", "icc2_1_ci_high",
             "ba_bias", "ba_bias_ci_low", "ba_bias_ci_high", "ba_loa_low", "ba_loa_high", "ba_prop_slope", "ba_prop_p",
             "threshold_agreement", "threshold_kappa_linear", "threshold_kappa_linear_ci_low", "threshold_kappa_linear_ci_high", "within_0_5_mm", "within_1_mm"]


def build_per_image(rows: pd.DataFrame, meas: pd.DataFrame, stage3: pd.DataFrame, combo: str, k: float, k_alt: dict, masks: str,
                    meas_corrected: pd.DataFrame = None, bottom_edge_offset_px: float = 0.0) -> pd.DataFrame:
    """Stage-3-compatible per-image table (fixed method and scale) plus GT-mask columns."""
    df = rows.merge(meas.drop(columns=["image", "group", "width", "height"]), on="uid", how="left")
    s3 = stage3.set_index("uid")
    base = df[["uid", "image", "patient_id", "group", "split", "cv_fold", "width", "height", "frame_ok"]].rename(columns={"split": "main_split", "cv_fold": "fold"})
    per: dict = {c: base[c] for c in base.columns}
    per["masks"] = masks
    per["split"] = per["uid"].map(s3["split"])                       # Stage 3 dev / holdout (scale fitted on dev, GT masks)
    for c in ["ref_mm"] + [f"ref_mm_{i}" for i in range(1, 7)] + ["ref_label", "has_dash_zero", "has_ambiguous_100_999"]:
        per[c] = per["uid"].map(s3[c])
    for reg, est, anch in COMBOS:
        name = combo_name(reg, est, anch)
        per[f"{name}_px"] = df[f"{name}_px"]
        for i in range(1, 7):
            per[f"{name}_region_{i}_px"] = df[f"{name}_region_{i}_px"]
    per["selected_method"] = combo
    per["selected_px_per_mm"] = k
    per["selected_mm"] = df[f"{combo}_px"] / k
    per["selected_label"] = per["selected_mm"].map(label_for_mm)
    for i in range(1, 7):
        per[f"selected_region_{i}_mm"] = df[f"{combo}_region_{i}_px"] / k
    # secondary: MASK-LEVEL correction (config bottom_edge_offset_px) — the lower gingiva edge moved up
    # before the profile is built; every estimator re-measured on the corrected mask
    if meas_corrected is not None:
        dc = rows.merge(meas_corrected.drop(columns=["image", "group", "width", "height"]), on="uid", how="left")
        assert (dc["uid"].to_numpy() == df["uid"].to_numpy()).all()
        for reg, est, anch in COMBOS:
            name = combo_name(reg, est, anch)
            per[f"{name}_px_corrected"] = dc[f"{name}_px"]
            for i in range(1, 7):
                per[f"{name}_region_{i}_px_corrected"] = dc[f"{name}_region_{i}_px"]
        per["selected_mm_corrected"] = dc[f"{combo}_px"] / k
        for i in range(1, 7):
            per[f"selected_region_{i}_mm_corrected"] = dc[f"{combo}_region_{i}_px"] / k
        per["n_columns_zeroed"] = dc["n_columns_zeroed"].fillna(0).astype(int)
        per["columns_zeroed_frac"] = dc["columns_zeroed_frac"].fillna(0.0)
        per["qc_flags_corrected"] = dc["qc_flags"].fillna("")
        per["n_zeniths_found_corrected"] = dc["n_zeniths_found"]
    else:
        per["selected_mm_corrected"] = per["selected_mm"]
        for i in range(1, 7):
            per[f"selected_region_{i}_mm_corrected"] = per[f"selected_region_{i}_mm"]
        per["n_columns_zeroed"] = 0
        per["columns_zeroed_frac"] = 0.0
    per["selected_label_corrected"] = per["selected_mm_corrected"].map(label_for_mm)
    per["bottom_edge_offset_px"] = float(bottom_edge_offset_px)
    for name, ka in k_alt.items():
        per[f"{name}_mm"] = df[f"{name}_px"] / ka
    per["gap_median_px"] = df["gap_median_px"]
    per["n_gingiva_instances"] = df["n_gingiva_instances"]
    per["n_lip_instances"] = df["n_lip_instances"]
    per["n_zeniths_found"] = df["n_zeniths_found"]
    per["qc_flags"] = df["qc_flags"].fillna("")
    per["alignment_uncertain"] = per["qc_flags"].str.contains(QCFlag.ZENITH_DETECTION_FAILED.value)
    per["mask_missing"] = df["mask_missing"].fillna(True).astype(bool)
    per["empty_prediction"] = df["n_gingiva_instances"].fillna(0).astype(int) == 0
    # same method and scale on the ground-truth masks (Stage 3 table)
    per["gt_mm"] = per["uid"].map(s3[f"{combo}_px"]) / k
    per["gt_label"] = per["gt_mm"].map(label_for_mm)
    for i in range(1, 7):
        per[f"gt_mm_{i}"] = per["uid"].map(s3[f"selected_region_{i}_mm"])
    per["gt_alignment_uncertain"] = per["uid"].map(s3["alignment_uncertain"]).astype(bool)
    return pd.DataFrame(per)


def seg_metrics_md(metrics: dict, model: str, mask_dir: str, evaluator: str, source: str) -> str:
    """The final model's own detection/segmentation metrics, from its own framework's evaluator.

    Ultralytics ``val()`` and RF-DETR's COCO evaluation agree in definition but not in matching
    detail, so the two never share a table (PLAN.md 4 of outputs/09_final_rfdetr): the evaluator is
    named in the heading and the cross-architecture comparison is left to the framework-independent
    boundary and IoU metrics.
    """
    head = (f"# Detection / segmentation metrics of the final model\n\n"
            f"Model: **{model}**; masks `{mask_dir}`; evaluator: **{evaluator}**; source: "
            f"`{source}`.\n\nThese numbers come from one framework's evaluator and are not "
            f"comparable, row by row, with another framework's mAP; architecture comparisons in this project use the "
            f"framework-independent boundary and IoU metrics of `boundary_by_set.md`.\n")
    if not metrics:
        return head + ("\n**Not available.** No metrics file for this model was found, and no other model's file is "
                       "substituted for it. Run that model's own evaluation and re-run this script.\n")
    if "per_class" in metrics:            # Ultralytics val() layout
        rows = [{"class": c, **{kk: vv for kk, vv in v.items()}} for c, v in metrics["per_class"].items()]
        body = md_table(pd.DataFrame(rows))
        extra = f"\n\nWeights: `{metrics.get('weights', '?')}`; split `{metrics.get('split', '?')}`, {metrics.get('n_images', '?')} images."
    else:                                  # a flat metric block (RF-DETR's COCO evaluation)
        flat = {kk: vv for kk, vv in metrics.items() if isinstance(vv, (int, float))}
        body = md_table(pd.DataFrame({"metric": list(flat), "value": list(flat.values())}), "{:.4f}")
        extra = f"\n\nSplit: `{metrics.get('split', '?')}`; variant `{metrics.get('variant', '?')}`, seed {metrics.get('seed', '?')}."
    return head + "\n" + body + extra + "\n"


def mixed_md(title: str, models: list) -> str:
    out = [f"### {title}"]
    for m in models:
        if "error" in m:
            out.append(f"- `{m['formula']}`: failed ({m['error']})")
            continue
        fe = m["fixed_effects"]
        lines = [f"`{m['formula']}` — {m['estimator']}; n = {m['n_obs']} teeth in {m['n_groups']} patients; "
                 f"var(patient) {m['var_patient']:.3f}, var(residual) {m['var_resid']:.3f}, ICC(patient) {m['icc_patient']:.2f}"
                 + (f"; note: {m['note']}" if m.get("note") else "")]
        for name, r in fe.iterrows():
            lines.append(f"  - {name}: {r['estimate']:+.3f} mm [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], p = {r['p']:.3g}")
        out.append("- " + "\n".join(lines))
    return "\n".join(out)


def boundary_figure(sets: dict, k: float, out_png: Path) -> None:
    """Three panels (mask IoU, top-edge MAE mm, bottom-edge bias mm) × the image sets; dots + median."""
    panels = [("gingiva_mask_iou", "Gingiva mask IoU", 1.0, "IoU"), ("gingiva_top_edge_mae_px", "Upper gingiva edge, MAE", k, "mm"),
              ("gingiva_bottom_edge_bias_px", "Lower gingiva edge, bias (pred − GT)", k, "mm")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    names = list(sets.keys())
    for ax, (col, title, div, unit) in zip(axes, panels):
        for j, name in enumerate(names):
            v = (sets[name][1][col].dropna() / div).to_numpy(float)
            rng = np.random.default_rng(j)
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=9, alpha=0.45, color=sets[name][0], lw=0)
            if len(v):
                ax.hlines(np.median(v), j - 0.3, j + 0.3, color="0.15", lw=2)
                ax.text(j, 1.0, f"n={len(v)}", ha="center", va="bottom", fontsize=7, color="0.3", transform=ax.get_xaxis_transform())
        if unit == "mm" and "bias" in col:
            ax.axhline(0, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7.5)
        ax.set_title(title, fontsize=9, pad=14); ax.set_ylabel(unit, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Segmentation quality against the ground-truth masks (bar = median)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_DPI); plt.close(fig)


def decomposition_figure(dec: pd.DataFrame, b: pd.DataFrame, k: float, out_png: Path) -> None:
    j = dec
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    bins = np.linspace(-3, 3, 31)
    for c, lab, col in (("e_total", "pipeline − reference", SET_COLOURS["a"]), ("e_seg", "pipeline − GT-mask measurement", SET_COLOURS["b"]), ("e_meas", "GT-mask measurement − reference", "0.45")):
        v = j[c].dropna()
        ax.hist(v, bins=bins, histtype="step", lw=1.8, color=col, label=f"{lab} (bias {v.mean():+.2f}, MAE {v.abs().mean():.2f} mm)")
    ax.axvline(0, color="0.5", lw=0.8, ls="--"); ax.set_xlabel("error, mm"); ax.set_ylabel("images"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Error decomposition, OOF masks (n = %d)" % j["e_total"].notna().sum(), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    x = j["gingiva_bottom_edge_bias_px"] / k; y = j["e_seg"]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=14, alpha=0.7, color=SET_COLOURS["a"], lw=0)
    if ok.sum() > 2:
        sl, ic = np.polyfit(x[ok], y[ok], 1); xs = np.linspace(x[ok].min(), x[ok].max(), 10)
        ax.plot(xs, ic + sl * xs, color="0.2", lw=1.2, label=f"slope {sl:.2f} mm/mm, r = {np.corrcoef(x[ok], y[ok])[0, 1]:.2f}")
        ax.legend(fontsize=8, loc="upper left")
    ax.axhline(0, color="0.5", lw=0.8, ls="--"); ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("lower gingiva edge bias (pred − GT), mm"); ax.set_ylabel("segmentation-induced measurement error, mm")
    ax.set_title("Where the segmentation error comes from", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_DPI); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="06_prediction")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--oof-masks", default=None, help="default outputs/05_predictions/oof; use oof_rfdetr for Stage 6 with RF-DETR")
    ap.add_argument("--test-masks", default=None, help="default outputs/05_predictions/test")
    ap.add_argument("--exclude-uids", default=None,
                    help="CSV with a uid column to leave out; the pre-registered sensitivity analysis of "
                         "outputs/09_final_rfdetr/PLAN.md 5 passes the 29 images used to choose the architecture")
    ap.add_argument("--seg-metrics", default=None,
                    help="JSON with the final model's own detection/segmentation metrics; default: the file that "
                         "belongs to the predictor of --test-masks (YOLO: outputs/05_predictions/test_metrics.json, "
                         "RF-DETR: outputs/08_architecture/rfdetr_metrics_<model>_s<seed>.json)")
    ap.add_argument("--seg-evaluator", default=None, help="name of the evaluator that produced --seg-metrics (required with it)")
    ap.add_argument("--reference-boundary", default=None,
                    help="boundary_error.csv of an EARLIER final model, reported as clearly labelled reference rows "
                         "only; default: outputs/05_predictions/boundary_error.csv (YOLOv11x) when the masks are not its own")
    args = ap.parse_args()
    if bool(args.seg_metrics) != bool(args.seg_evaluator):
        raise SystemExit("--seg-metrics and --seg-evaluator go together: a metric without its evaluator is not reportable")
    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    out_dir = outputs / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = resolve(cfg, cfg["paths"]["predictions"])
    oof_dir = Path(args.oof_masks) if args.oof_masks else pred_dir / "oof"
    test_dir = Path(args.test_masks) if args.test_masks else pred_dir / "test"
    oracle_dir = outputs / "03_oracle"
    if not oof_dir.is_absolute():
        oof_dir = resolve(cfg, oof_dir)
    if not test_dir.is_absolute():
        test_dir = resolve(cfg, test_dir)
    print(f"[stage6] out-of-fold masks {oof_dir}; test masks {test_dir}")
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    seed = int(cfg["seed"])
    mcfg = cfg["measurement"]
    combo = combo_name(mcfg["method"]["regioning"], mcfg["method"]["estimator"], bool(mcfg["method"]["anchored"]))
    k = float(mcfg["px_per_mm"])
    off = bottom_edge_offset_from_config(cfg)
    off_mm = offset_mm_at(k, off)
    corr_tag = f"lower gingiva edge {off:+.0f} px at mask level ({off_mm:+.2f} mm at {k:.2f} px/mm), secondary"
    deviations: list = []

    # ---- inputs
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    ref = manifest[manifest["keep"] & manifest["has_reference_measurement"]].copy()
    stage3 = pd.read_csv(oracle_dir / "per_image_results.csv")
    est3 = pd.read_csv(oracle_dir / "estimator_comparison.csv").set_index("combo")
    k3 = float(stage3["selected_px_per_mm"].iloc[0])
    if stage3["selected_method"].iloc[0] != combo or abs(k3 - k) / k > 1e-3:
        raise SystemExit(f"config method/scale ({combo}, {k}) differ from Stage 3 ({stage3['selected_method'].iloc[0]}, {k3}) — re-run Stage 3 with --write-config")
    alt = combo_name("A", mcfg["method"]["estimator"], bool(mcfg["method"]["anchored"]))
    k_alt = {alt: float(est3.loc[alt, "px_per_mm_dev"])} if alt in est3.index and alt != combo else {}
    oof = pd.read_csv(oof_dir / "oof_predictions.csv")
    test_pred = pd.read_csv(test_dir / "test_predictions.csv")

    # ---- provenance: which model wrote which masks, and whose evaluator may be quoted for it.
    # Everything below is labelled with these; anything ambiguous stopped the run inside mask_provenance.
    prov_oof = mask_provenance(oof, oof_dir, cfg, "out-of-fold masks", single_model=False, root=ROOT)
    prov_test = mask_provenance(test_pred, test_dir, cfg, "final-model test masks", single_model=True, root=ROOT)
    rel_oof, rel_test = prov_oof["dir"], prov_test["dir"]
    if prov_oof["family"] != prov_test["family"]:
        raise SystemExit(f"out-of-fold masks come from {prov_oof['label']} but the test masks from {prov_test['label']} — "
                         "one Stage-6 report describes one segmentation model")
    print(f"[stage6] out-of-fold masks: {prov_oof['text']}\n[stage6] final-model masks: {prov_test['text']}")
    if args.seg_metrics:
        seg_path, seg_evaluator = Path(args.seg_metrics), args.seg_evaluator
        if not seg_path.is_absolute():
            seg_path = resolve(cfg, seg_path)
    elif prov_test["family"] == "yolo":
        seg_path, seg_evaluator = pred_dir / "test_metrics.json", prov_test["evaluator"]
    else:                                   # RF-DETR reports its own COCO evaluation (PLAN.md 4)
        seeds = sorted({int(x) for x in test_pred["seed"].dropna()}) if "seed" in test_pred.columns else []
        # scripts/rfdetr_train_predict.py writes outputs/08_architecture/rfdetr_metrics_<model>_s<seed>.json
        seg_path = (outputs / "08_architecture" / f"rfdetr_metrics_{prov_test['runs'][0]}_s{seeds[0]}.json") if seeds else None
        seg_evaluator = prov_test["evaluator"]
    seg_metrics = json.loads(seg_path.read_text()) if seg_path and seg_path.exists() else {}
    # An earlier final model's numbers may be shown as reference rows, never as this model's own.
    ref_b_path = Path(args.reference_boundary) if args.reference_boundary else pred_dir / "boundary_error.csv"
    if not ref_b_path.is_absolute():
        ref_b_path = resolve(cfg, ref_b_path)
    ref_label = "YOLOv11x (previous final model)"
    ref_b = pd.read_csv(ref_b_path) if (ref_b_path.exists() and prov_test["family"] != "yolo") else None
    ref_oof_b_path = outputs / "06_prediction" / "boundary_error_oof.csv"
    ref_oof_b = pd.read_csv(ref_oof_b_path) if (ref_oof_b_path.exists() and out_dir != outputs / "06_prediction" and prov_oof["family"] != "yolo") else None

    # ---- 1. OOF integrity
    # both predictors run at conf 0.25, but only the YOLO one takes it from this config
    conf_txt = f"conf {cfg['yolo']['conf']}" if prov_oof["family"] == "yolo" else "the predictor's confidence threshold"
    chk = check_oof(oof, ref[["image", "uid", "cv_fold"]], oof_dir)
    chk_md = [f"# OOF prediction check ({rel_oof})", "",
              f"- model: **{prov_oof['label']}**", f"- masks: `{rel_oof}`",
              f"- rows: {chk['n_rows']} (reference images: {chk['n_reference']})", f"- mask_source: {chk['mask_source']}",
              f"- images per fold: {chk['folds']}", f"- gingiva PNG missing: {chk['n_missing_png']}",
              f"- empty gingiva prediction (no instance above {conf_txt}): {chk['empty_gingiva_prediction'] or 'none'}",
              f"- result: **{'OK' if chk['ok'] else 'PROBLEMS'}**"] + [f"  - {p}" for p in chk["problems"]]
    (out_dir / "oof_check.md").write_text("\n".join(chk_md) + "\n", encoding="utf-8")
    print("\n".join(chk_md))
    if not chk["ok"]:
        raise SystemExit("OOF prediction table is inconsistent — see oof_check.md")
    for img in chk["empty_gingiva_prediction"]:
        deviations.append(f"`{img}`: **segmentation failure** — the fold model predicted no gingiva instance above {conf_txt} (empty mask). Reported as its own category (`n_segmentation_failure`, 1 of 145 = 0.7 %), not as a 0 mm measurement: the image has no mm value and is excluded from the mm and label metrics (n = 144), flagged `empty_prediction` in per_image_results.csv.")

    # ---- 2. measurement with the fixed method and scale: (a) OOF masks, (b) test high with the final model
    rows_cols = ["uid", "image", "patient_id", "group", "split", "cv_fold", "width", "height", "frame_ok", "orig_split", "file_name"]
    rows = ref[rows_cols].reset_index(drop=True)
    print(f"[stage6] measuring {len(rows)} OOF masks with {combo} at {k:.4f} px/mm …")
    meas_oof = measure_images(rows, cfg, coco_root, mask_source=str(oof_dir))
    meas_oof_c = measure_images(rows, cfg, coco_root, mask_source=str(oof_dir), bottom_edge_offset_px=off) if off else None
    per = build_per_image(rows, meas_oof, stage3, combo, k, k_alt, "oof", meas_oof_c, off)
    per.to_csv(out_dir / "per_image_results.csv", index=False)
    rows_t = rows[rows["split"] == "test"].reset_index(drop=True)
    print(f"[stage6] measuring {len(rows_t)} test-set masks of the final model …")
    meas_t = measure_images(rows_t, cfg, coco_root, mask_source=str(test_dir))
    meas_t_c = measure_images(rows_t, cfg, coco_root, mask_source=str(test_dir), bottom_edge_offset_px=off) if off else None
    per_t = build_per_image(rows_t, meas_t, stage3, combo, k, k_alt, "final", meas_t_c, off)
    per_t.to_csv(out_dir / "per_image_results_test.csv", index=False)
    n_missing_t = int(per_t["mask_missing"].sum())
    if n_missing_t:
        deviations.append(f"{n_missing_t} test high images have no final-model mask PNG in outputs/05_predictions/test.")

    # ---- 3. measurement accuracy tables
    T = pd.Series(True, index=per.index)
    excluded = set()
    if args.exclude_uids:
        ex = Path(args.exclude_uids)
        if not ex.is_absolute():
            ex = resolve(cfg, ex)
        excluded = set(pd.read_csv(ex)["uid"])
        print(f"[stage6] pre-registered sensitivity analysis: {len(excluded)} uid(s) excluded from the secondary set")
    not_selection = ~per["uid"].isin(excluded)
    sets_a = {
        "(a) OOF masks, all reference images [PRIMARY]": T,
        "(a) OOF, Stage-3 holdout images only (scale never fitted on these)": per["split"] == "holdout",
        "(a) OOF, Stage-3 dev images only": per["split"] == "dev",
        "(a) OOF, main-split train": per["main_split"] == "train",
        "(a) OOF, main-split valid": per["main_split"] == "valid",
        "(b') OOF masks on the test high images (fold models)": per["main_split"] == "test",
        "(a) sensitivity: without dash-zero reference rows": ~per["has_dash_zero"].astype(bool),
        "(a) sensitivity: only 2698x1799 (±2 px) frames": per["frame_ok"].astype(bool),
        "(a) sensitivity: without zenith-fallback images": ~per["alignment_uncertain"],
        "(a) sensitivity: without empty predictions": ~per["empty_prediction"],
    }
    if excluded:
        # PLAN.md 5: the architecture was chosen partly on the 29 test images, so the same analysis is
        # pre-registered on the images that played no part in that choice
        sets_a[f"(a) PRE-REGISTERED SENSITIVITY: excluding the {len(excluded)} images used to choose the architecture"] = not_selection
    def both(per_df: pd.DataFrame, sets: dict, masks_label: str) -> pd.DataFrame:
        """Every set twice: uncorrected (PRIMARY) first, then corrected (secondary)."""
        u = measurement_table(per_df, sets, n_boot=args.n_boot, seed=seed)
        u.insert(1, "masks", masks_label); u.insert(2, "correction", "none, PRIMARY"); u["images_with_zeroed_columns"] = 0; u["zeroed_columns_frac_mean"] = 0.0
        c = measurement_table(per_df, sets, value_col="selected_mm_corrected", n_boot=args.n_boot, seed=seed)
        c.insert(1, "masks", masks_label); c.insert(2, "correction", corr_tag)
        c["set"] = c["set"] + " — corrected"
        c["images_with_zeroed_columns"] = [int((per_df.loc[m.reindex(per_df.index).fillna(False).astype(bool), "n_columns_zeroed"] > 0).sum()) for m in sets.values()]
        c["zeroed_columns_frac_mean"] = [float(per_df.loc[m.reindex(per_df.index).fillna(False).astype(bool), "columns_zeroed_frac"].mean()) for m in sets.values()]
        rows = []
        for i in range(len(u)):
            rows.append(u.iloc[i]); rows.append(c.iloc[i])
        return pd.DataFrame(rows).reset_index(drop=True)

    label_oof = f"OOF fold models ({prov_oof['short']})"
    label_fin = f"final model ({prov_test['short']})"
    acc = both(per, sets_a, label_oof)
    acc_b = both(per_t, {"(b) final model masks, test high images [secondary set]": pd.Series(True, index=per_t.index)}, label_fin)
    acc_gt = measurement_table(per, {"GT masks, all reference images (Stage 3, same method and scale)": T, "GT masks, Stage-3 holdout": per["split"] == "holdout"},
                               value_col="gt_mm", n_boot=args.n_boot, seed=seed)
    acc_gt.insert(1, "masks", "ground truth"); acc_gt.insert(2, "correction", "n/a (GT masks)")
    parts = [acc, acc_b, acc_gt]
    if k_alt:
        acc_alt = measurement_table(per, {f"(a) OOF, {alt} at its Stage-3 scale {k_alt[alt]:.2f} px/mm (fallback sensitivity)": T}, value_col=f"{alt}_mm", n_boot=args.n_boot, seed=seed)
        acc_alt.insert(1, "masks", label_oof); acc_alt.insert(2, "correction", "none")
        parts.append(acc_alt)
    acc_all = pd.concat(parts, ignore_index=True)
    # provenance on every row of the table, so no number travels without its model
    src_of = {label_oof: (prov_oof["label"], prov_oof["dir"]), label_fin: (prov_test["label"], prov_test["dir"]),
              "ground truth": ("COCO annotations (no model)", rel_to(coco_root, ROOT))}
    acc_all["model"] = [src_of[m][0] for m in acc_all["masks"]]
    acc_all["masks_dir"] = [src_of[m][1] for m in acc_all["masks"]]
    acc_all = acc_all[[c for c in MEAS_COLS if c in acc_all.columns]]
    acc_all.to_csv(out_dir / "measurement_accuracy.csv", index=False)
    prim, prim_c = acc_all.iloc[0], acc_all.iloc[1]
    sec, sec_c = acc_b.iloc[0], acc_b.iloc[1]
    gt_all = acc_gt.iloc[0]
    fb = fallback_rate(per)
    fb_gt = fallback_rate(stage3)

    # ---- 4. threshold-class agreement
    ok_a = per["selected_mm"].notna() & per["ref_mm"].notna()
    ct_a = label_confusion(per.loc[ok_a, "ref_label"], per.loc[ok_a, "selected_label"])
    ok_b = per_t["selected_mm"].notna() & per_t["ref_mm"].notna()
    ct_b = label_confusion(per_t.loc[ok_b, "ref_label"], per_t.loc[ok_b, "selected_label"])
    ct_gt = label_confusion(per.loc[ok_a, "ref_label"], per.loc[ok_a, "gt_label"])
    ct_a_c = label_confusion(per.loc[ok_a, "ref_label"], per.loc[ok_a, "selected_label_corrected"])
    ct_b_c = label_confusion(per_t.loc[ok_b, "ref_label"], per_t.loc[ok_b, "selected_label_corrected"])
    class_md = f"""# Threshold-class agreement (Table 1 labels; a measurement check, not a clinical validation)

Rows = clinical reference label, columns = pipeline label. Overlapping bands give combined labels (E1-E2, E2-E3).

## (a) OOF masks, all reference images (n = {int(ok_a.sum())}) — primary
observed agreement {100 * prim['threshold_agreement']:.1f} %, linear-weighted κ {prim['threshold_kappa_linear']:.3f} [{prim['threshold_kappa_linear_ci_low']:.3f}, {prim['threshold_kappa_linear_ci_high']:.3f}]

{md_table(ct_a.reset_index())}

### (a) corrected ({corr_tag})
observed agreement {100 * prim_c['threshold_agreement']:.1f} %, linear-weighted κ {prim_c['threshold_kappa_linear']:.3f} [{prim_c['threshold_kappa_linear_ci_low']:.3f}, {prim_c['threshold_kappa_linear_ci_high']:.3f}]; images with zeroed columns: {int(prim_c['images_with_zeroed_columns'])}

{md_table(ct_a_c.reset_index())}

## (b) final model, test high images (n = {int(ok_b.sum())}) — secondary set
observed agreement {100 * sec['threshold_agreement']:.1f} %, linear-weighted κ {sec['threshold_kappa_linear']:.3f} [{sec['threshold_kappa_linear_ci_low']:.3f}, {sec['threshold_kappa_linear_ci_high']:.3f}]

{md_table(ct_b.reset_index())}

### (b) corrected ({corr_tag})
observed agreement {100 * sec_c['threshold_agreement']:.1f} %, linear-weighted κ {sec_c['threshold_kappa_linear']:.3f} [{sec_c['threshold_kappa_linear_ci_low']:.3f}, {sec_c['threshold_kappa_linear_ci_high']:.3f}]; images with zeroed columns: {int(sec_c['images_with_zeroed_columns'])}

{md_table(ct_b_c.reset_index())}

## GT masks, all reference images (Stage 3 geometry only)
observed agreement {100 * gt_all['threshold_agreement']:.1f} %, linear-weighted κ {gt_all['threshold_kappa_linear']:.3f} [{gt_all['threshold_kappa_linear_ci_low']:.3f}, {gt_all['threshold_kappa_linear_ci_high']:.3f}]

{md_table(ct_gt.reset_index())}
"""
    (out_dir / "class_agreement.md").write_text(class_md, encoding="utf-8")

    # ---- 5. boundary / segmentation quality in the three sets
    # Both tables are computed HERE from the masks that were actually measured above, never read from
    # a CSV another model's run left behind; the provenance of every row is written into the table.
    print(f"[stage6] boundary errors of the OOF masks ({prov_oof['label']}) against the COCO ground truth …")
    b_oof = boundary_table(cfg, oof, oof_dir)
    b_oof.to_csv(out_dir / "boundary_error_oof.csv", index=False)
    print(f"[stage6] boundary errors of the {len(test_pred)} test masks ({prov_test['label']}) …")
    b_test = boundary_table(cfg, test_pred, test_dir)
    b_test.to_csv(out_dir / "boundary_error_test.csv", index=False)
    test_high_uids = set(rows_t["uid"])
    if not test_high_uids <= set(b_test["uid"]):
        missing = sorted(test_high_uids - set(b_test["uid"]))
        raise SystemExit(f"{test_dir}/test_predictions.csv does not cover every test high image ({len(missing)} missing: {missing[:5]})")
    b_sets = {
        "(a) OOF, 145 reference high": b_oof,
        "(b) test high, final model": b_test[b_test["group"] == "high"],
        "(b') test high, OOF masks": b_oof[b_oof["uid"].isin(test_high_uids)],
        "(c) test all": b_test,
        "(c) test low": b_test[b_test["group"] == "low"],
        "(c) test normal": b_test[b_test["group"] == "normal"],
    }
    set_source = {name: (prov_oof if name.startswith(("(a)", "(b')")) else prov_test) for name in b_sets}
    n_own_sets = len(b_sets)
    if ref_b is not None:                       # earlier final model, reported side by side and labelled as such
        b_sets[f"[reference] {ref_label}, test high"] = ref_b[ref_b["group"] == "high"]
        b_sets[f"[reference] {ref_label}, test all"] = ref_b
        for nm in (f"[reference] {ref_label}, test high", f"[reference] {ref_label}, test all"):
            set_source[nm] = {"label": ref_label, "dir": rel_to(ref_b_path, ROOT), "evaluator": EVALUATORS["yolo"]}
    if ref_oof_b is not None:
        nm = f"[reference] {ref_label}, OOF 145 reference high"
        b_sets[nm] = ref_oof_b
        set_source[nm] = {"label": ref_label, "dir": rel_to(ref_oof_b_path, ROOT), "evaluator": EVALUATORS["yolo"]}
    bt = boundary_sets_table(b_sets, k)
    bt.insert(1, "model", [set_source[n]["label"] for n in b_sets])
    bt.insert(2, "masks_dir", [set_source[n]["dir"] for n in b_sets])
    bt.to_csv(out_dir / "boundary_by_set.csv", index=False)
    show_b = bt[["set", "model", "n", "n_gt_gingiva", "n_both", "n_missed", "n_spurious", "n_neither", "gingiva_mask_iou_n", "gingiva_mask_iou_mean", "gingiva_mask_iou_median",
                 "gingiva_boundary_iou_mean", "gingiva_top_edge_mae_mm_mean", "gingiva_top_edge_mae_mm_median", "gingiva_top_edge_bias_mm_mean",
                 "gingiva_bottom_edge_mae_mm_mean", "gingiva_bottom_edge_mae_mm_median", "gingiva_bottom_edge_bias_mm_mean", "gingiva_thickness_mae_mm_mean",
                 "gingiva_columns_missed_frac_mean", "gingiva_columns_spurious_frac_mean", "gingiva_n_columns_gt_median", "lip_mask_iou_mean", "lip_mask_iou_median"]]
    bt_px = bt[["set", "model", "gingiva_top_edge_mae_px_mean", "gingiva_top_edge_median_px_mean" if "gingiva_top_edge_median_px_mean" in bt.columns else "gingiva_top_edge_mae_px_median",
                "gingiva_top_edge_bias_px_mean", "gingiva_bottom_edge_mae_px_mean", "gingiva_bottom_edge_bias_px_mean", "gingiva_thickness_mae_px_mean"]]
    ba, bb, bc = bt.iloc[0], bt.iloc[1], bt.iloc[3]
    b_low, b_norm = bt.iloc[4], bt.iloc[5]
    ref_note = ""
    if ref_b is not None or ref_oof_b is not None:
        ref_note = (f"\n\nRows marked **[reference]** are the {ref_label}, computed on ITS OWN masks in an earlier run "
                    f"(`{rel_to(ref_b_path if ref_b is not None else ref_oof_b_path, ROOT)}`) and reported here only so the two models can be read side by side. "
                    f"They are not this model's numbers and were not recomputed here.")
    boundary_md = f"""# Boundary error and segmentation quality in three image sets

**Model: {prov_test['label']}.** Out-of-fold masks `{rel_oof}` ({prov_oof['label']}); final-model masks `{rel_test}`. Every table below carries the model and the mask directory of each row; both boundary tables were computed here from those masks, not read from another run's CSV.{ref_note}

Same function as on the workstation (`gsv4.eval.boundary.boundary_report`, boundary IoU with 5 px dilation; edge errors column-wise over columns where both masks have gingiva). Pixel values converted at the global scale {k:.2f} px/mm. `n_*` columns: `n_gt_gingiva` images whose ground truth contains gingiva; `n_both` both masks non-empty (IoU and edge errors defined); `n_missed` GT gingiva but empty prediction; `n_spurious` prediction without GT gingiva; `n_neither` both empty (correct absence — IoU undefined, not zero). Statistics are computed over the images where they are defined (`*_n`).

{md_table(show_b)}

Pixel units of the edge metrics:

{md_table(bt_px, '{:.2f}')}

## Reading the three sets

* **(a) is the primary segmentation result** for the measurement task: 145 high-smile-line images with a clinical reference, every mask predicted by a fold model that never saw the image or its same-patient twin. Gingiva mask IoU {ba['gingiva_mask_iou_mean']:.3f} (median {ba['gingiva_mask_iou_median']:.3f}); upper edge MAE {ba['gingiva_top_edge_mae_mm_mean']:.2f} mm, lower edge MAE {ba['gingiva_bottom_edge_mae_mm_mean']:.2f} mm with a systematic lower-edge bias of {ba['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm (predicted gingiva extends further down than the annotation); lip IoU {ba['lip_mask_iou_mean']:.3f}.
* **(b)** the final model on the 29 test high images: IoU {bb['gingiva_mask_iou_mean']:.3f}, upper edge MAE {bb['gingiva_top_edge_mae_mm_mean']:.2f} mm, lower edge bias {bb['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm — the same picture as (a) on an independent model and the fixed test split; (b') shows the fold models on the same 29 images (IoU {bt.iloc[2]['gingiva_mask_iou_mean']:.3f}).
* **(c)** the whole test set (n = {int(bc['n'])}) has a lower mean gingiva IoU ({bc['gingiva_mask_iou_mean']:.3f}) **by construction, not because the model is worse there**: in low and normal smile lines the gingiva is thin or not visible at all. In the test low subset the annotated gingiva spans a median of {b_low['gingiva_n_columns_gt_median']:.0f} image columns (vs {ba['gingiva_n_columns_gt_median']:.0f} in the high set) and {int(b_low['n_neither']) + int(b_low['n_missed']) + int(b_low['n_spurious'])} of {int(b_low['n'])} images have an empty GT or predicted gingiva; a few pixels of edge disagreement on a sliver one or two pixels tall drive IoU towards 0 even though the edge errors themselves are *smaller* than in the high set (upper edge MAE {b_low['gingiva_top_edge_mae_mm_mean']:.2f} mm low, {b_norm['gingiva_top_edge_mae_mm_mean']:.2f} mm normal). Presence agreement in (c): both {int(bc['n_both'])}, correct absence {int(bc['n_neither'])}, missed {int(bc['n_missed'])}, spurious {int(bc['n_spurious'])}. The pipeline is specified for the high smile line only (visible gingiva → mm; otherwise NO_VISIBLE_GINGIVA), so (c) is reported for completeness of the segmentation evaluation and is not the basis of any measurement claim.
"""
    (out_dir / "boundary_by_set.md").write_text(boundary_md, encoding="utf-8")
    (out_dir / "segmentation_metrics.md").write_text(
        seg_metrics_md(seg_metrics, prov_test["label"], rel_test, seg_evaluator, rel_to(seg_path, ROOT)), encoding="utf-8")

    # ---- 6. error decomposition (a)
    dec = error_decomposition(per)
    dec_per = dec["per_image"].copy()
    dec_per.insert(0, "uid", per["uid"])
    dec_per = dec_per.set_index("uid").join(b_oof.set_index("uid")[["gingiva_mask_iou", "gingiva_boundary_iou", "gingiva_top_edge_bias_px", "gingiva_bottom_edge_bias_px",
                                                                     "gingiva_thickness_mae_px", "gingiva_columns_missed_frac", "gingiva_columns_spurious_frac"]])
    dec_per.reset_index().to_csv(out_dir / "error_decomposition.csv", index=False)
    ds = dec["summary"]
    corr = seg_error_vs_boundary(dec["per_image"].set_index(per["uid"]), b_oof.set_index("uid"), k)
    dec_md = f"""# Error decomposition (a: OOF masks, n = {ds['n']})

total = pipeline − reference = (pipeline − GT-mask measurement) + (GT-mask measurement − reference), with the same method ({combo}) and scale ({k:.2f} px/mm) on both mask sources. The first term is what the segmentation adds; the second is the geometry/scale error already characterised in Stage 3.

| component | bias, mm | MAE, mm | SD, mm | RMSE, mm |
|---|---|---|---|---|
| total (pipeline − reference) | {ds['e_total_bias']:+.3f} | {ds['e_total_mae']:.3f} | {ds['e_total_sd']:.3f} | {ds['e_total_rmse']:.3f} |
| segmentation (pipeline − GT-mask) | {ds['e_seg_bias']:+.3f} | {ds['e_seg_mae']:.3f} | {ds['e_seg_sd']:.3f} | {ds['e_seg_rmse']:.3f} |
| geometry (GT-mask − reference) | {ds['e_meas_bias']:+.3f} | {ds['e_meas_mae']:.3f} | {ds['e_meas_sd']:.3f} | {ds['e_meas_rmse']:.3f} |

Variance of the total error: {100 * ds['share_seg']:.0f} % segmentation, {100 * ds['share_meas']:.0f} % geometry, {100 * ds['share_cov']:+.0f} % covariance (r between the two components {ds['r_seg_meas']:+.2f}).

## Segmentation-induced error vs boundary metrics of the same image
Pixel metrics converted to mm; slope in mm of measurement error per mm (or per unit) of the metric.

{md_table(corr)}
"""
    (out_dir / "error_decomposition.md").write_text(dec_md, encoding="utf-8")

    # ---- 7. tooth level: mixed models (patient = same-patient cluster)
    long_ref = tooth_long(per, TEETH, "ref_mm_", group_col="patient_id")
    long_gt = tooth_long(per, TEETH, "gt_mm_", group_col="patient_id")
    tt = tooth_table(long_ref); tt.insert(0, "comparison", "OOF region i vs reference tooth i")
    tg = tooth_table(long_gt); tg.insert(0, "comparison", "OOF region i vs GT-mask region i")
    pd.concat([tt, tg], ignore_index=True).to_csv(out_dir / "tooth_level.csv", index=False)
    mm_ref = tooth_mixed_models(long_ref)
    mm_gt = tooth_mixed_models(long_gt)
    per_c = per.copy()
    for i in range(1, 7):
        per_c[f"selected_region_{i}_mm"] = per[f"selected_region_{i}_mm_corrected"]
    long_ref_c = tooth_long(per_c, TEETH, "ref_mm_", group_col="patient_id")
    tc = tooth_table(long_ref_c); tc.insert(0, "comparison", "OOF region i (corrected, secondary) vs reference tooth i")
    pd.concat([tt, tg, tc], ignore_index=True).to_csv(out_dir / "tooth_level.csv", index=False)
    mm_ref_c = tooth_mixed_models(long_ref_c)
    mixed_txt = f"""# Tooth-level analysis with linear mixed models (a: OOF masks)

Six teeth per image are not independent: `diff ~ … + (1 | patient)` (REML; `patient` = same-patient cluster from the manifest, {per['patient_id'].nunique()} clusters for {len(per)} images). Region i (left-to-right, {combo}) is compared with reference tooth i (FDI {TEETH}); alignment is approximate and images where the zenith regioning fell back to equal splits are flagged `alignment_uncertain` ({int(per['alignment_uncertain'].sum())} images).

## Per tooth
{md_table(pd.concat([tt, tg, tc], ignore_index=True))}

{mixed_md("Pipeline (OOF, uncorrected, PRIMARY) − clinical reference", mm_ref)}

{mixed_md(f"Pipeline (OOF, corrected: {corr_tag}) − clinical reference", mm_ref_c)}

{mixed_md("Pipeline (OOF) − GT-mask measurement (segmentation part only)", mm_gt)}
"""
    (out_dir / "mixed_models.md").write_text(mixed_txt, encoding="utf-8")

    # ---- 8. figures
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    scatter_and_bland_altman(per, fig_dir / "measurement_oof.png", f"(a) Full pipeline, out-of-fold masks vs clinical reference — {combo}, {k:.2f} px/mm", subset_col=None)
    scatter_and_bland_altman(per_t, fig_dir / "measurement_test_high.png", f"(b) Final model, test high images vs clinical reference — {combo}, {k:.2f} px/mm", subset_col=None)
    scatter_and_bland_altman(per, fig_dir / "measurement_oof_corrected.png", f"(a) corrected, secondary: offset_px {off:+.0f} — out-of-fold masks vs clinical reference", value_col="selected_mm_corrected", subset_col=None)
    scatter_and_bland_altman(per_t, fig_dir / "measurement_test_high_corrected.png", f"(b) corrected, secondary: offset_px {off:+.0f} — final model, test high images", value_col="selected_mm_corrected", subset_col=None)
    boundary_figure({"(a) OOF high\nn=145": (SET_COLOURS["a"], b_oof), "(b) test high\nfinal model": (SET_COLOURS["b"], b_sets["(b) test high, final model"]),
                     "(c) test low": (SET_COLOURS["low"], b_sets["(c) test low"]), "(c) test normal": (SET_COLOURS["normal"], b_sets["(c) test normal"])}, k, fig_dir / "boundary_by_set.png")
    decomposition_figure(dec_per, b_oof.set_index("uid"), k, fig_dir / "error_decomposition.png")

    # ---- 9. deviations
    # No metrics file is ever borrowed from another model: the table stays empty and says what to run.
    where = rel_to(seg_path, ROOT) if seg_path else "no metrics file could be named for this model"
    if not seg_metrics:
        how = ("re-run `python -m gsv4.train.evaluate_test` on the workstation (validation runs, the predictions are reused)"
               if prov_test["family"] == "yolo" else
               "the final model is reused from the architecture comparison with `--predict-only`, which does not evaluate; run "
               "`.venv-rfdetr/bin/python scripts/rfdetr_train_predict.py --variant main --seed 42 --predict-only --evaluate` on the workstation")
        deviations.append(f"No detection/segmentation metrics for {prov_test['label']} (PLAN.md 4): `{where}` is missing, and another model's "
                          f"file is NOT used in its place — `segmentation_metrics.md` stays empty and the Stage-7 table stays pending. To fill it, {how}, then re-run this script.")
    elif prov_test["family"] == "yolo" and "per_class" not in seg_metrics:
        deviations.append(f"`{where}` has no `per_class` block (box/mask mAP, P, R, F1 of the final model on the test set): the workstation eval ran the boundary stage only (`--metrics-only`) and `runs/eval/DONE` was not written. Re-run `python -m gsv4.train.evaluate_test` on the workstation (validation runs, predictions are reused) and pull; the detection metrics table of Stage 7 stays pending until then.")
    nan_iou = int(b_test["gingiva_mask_iou"].isna().sum()); nan_edge = int(b_test["gingiva_top_edge_mae_px"].isna().sum())
    deviations.append(f"Test-set boundary table: gingiva IoU undefined on {nan_iou} images (both masks empty) and edge errors undefined on {nan_edge} images (no column with gingiva in both masks) — all low/normal; reported as presence categories, not as zeros.")
    deviations.append(f"Mask-level correction ({off:+.0f} px): columns whose bottom run was shorter than the shift became 0 (never negative) in "
                      f"{int((per['n_columns_zeroed'] > 0).sum())} of {len(per)} OOF images (mean fraction of gingiva columns {100 * per['columns_zeroed_frac'].mean():.2f} %, max {100 * per['columns_zeroed_frac'].max():.1f} %) "
                      f"and {int((per_t['n_columns_zeroed'] > 0).sum())} of {len(per_t)} test images (mean {100 * per_t['columns_zeroed_frac'].mean():.2f} %). "
                      f"Zenith fallback on the corrected masks: {int(per['qc_flags_corrected'].str.contains(QCFlag.ZENITH_DETECTION_FAILED.value).sum())} images (uncorrected {int(per['alignment_uncertain'].sum())}).")
    if fb["reevaluate"]:
        deviations.append(f"Zenith-regioning fallback on {fb['n_fallback']}/{fb['n']} OOF images ({100 * fb['frac']:.0f} %) exceeds the pre-registered 30 % — {alt} row in measurement_accuracy.csv is the re-evaluation.")
    (out_dir / "SAPMALAR.md").write_text("# Aşama 6 — sapmalar ve notlar\n\n" + "\n".join(f"- {d}" for d in deviations) + "\n", encoding="utf-8")

    # ---- 10. summaries
    hold, hold_c = acc_all.iloc[2], acc_all.iloc[3]
    bprime = acc_all[acc_all["set"].str.startswith("(b')") & (acc_all["correction"] == "none, PRIMARY")].iloc[0]
    fmt_row = lambda r: (f"n = {int(r['n'])}: MAE {r['mae']:.3f} mm, RMSE {r['rmse']:.3f} mm, r {r['r']:.3f}, ICC(2,1) {r['icc2_1']:.3f} [{r['icc2_1_ci_low']:.3f}, {r['icc2_1_ci_high']:.3f}], "
                         f"bias {r['ba_bias']:+.3f} mm [{r['ba_bias_ci_low']:+.3f}, {r['ba_bias_ci_high']:+.3f}], LoA {r['ba_loa_low']:.2f} to {r['ba_loa_high']:.2f} mm, "
                         f"proportional-bias slope {r['ba_prop_slope']:+.3f} (p = {r['ba_prop_p']:.3f}); within 0.5 mm {100 * r['within_0_5_mm']:.0f} %, within 1 mm {100 * r['within_1_mm']:.0f} %; "
                         f"label agreement {100 * r['threshold_agreement']:.1f} %, linear-weighted κ {r['threshold_kappa_linear']:.3f} [{r['threshold_kappa_linear_ci_low']:.3f}, {r['threshold_kappa_linear_ci_high']:.3f}]")
    show = acc_all[["set", "model", "correction", "n", "mae", "rmse", "r", "icc2_1", "icc2_1_ci_low", "icc2_1_ci_high", "ba_bias", "ba_loa_low", "ba_loa_high", "ba_prop_slope", "ba_prop_p", "threshold_agreement", "threshold_kappa_linear"]]
    summary = f"""# Stage 6 — full pipeline on predicted masks

Method **{combo}**, global scale **{k:.2f} px/mm**, both fixed in `configs/config.yaml` from Stage 3 (ground-truth masks, dev subset). Nothing was re-selected or re-fitted on predicted masks. **Uncorrected values are the PRIMARY result.** The post-hoc mask-level correction of config.yaml (`measurement.bottom_edge_offset_px = {off:+.0f}`: the lower gingiva edge of the predicted mask is moved up by {abs(off):.0f} px in every column before the thickness profile is built, {off_mm:+.2f} mm at the global scale; estimated on the Stage-3 dev subset, `offset_correction.md` / `offset_checks.md`) gives the SECONDARY, corrected values; every table carries both, primary first. Bootstrap CIs of κ: {args.n_boot} resamples, seed {seed}.

## Provenance of every number below
| what | model | masks | metrics evaluator |
|---|---|---|---|
| (a) out-of-fold masks, {chk['n_rows']} reference images | {prov_oof['label']} | `{rel_oof}` (`oof_check.md`) | — (mm accuracy is measured here, not by a framework) |
| (b) final-model masks, test high images | {prov_test['label']} | `{rel_test}` | {seg_evaluator} → `segmentation_metrics.md` |
| ground-truth masks (Stage 3 comparison) | COCO annotations, no model | `{rel_to(coco_root, ROOT)}` | — |
{f"| [reference] rows in `boundary_by_set.md` | {ref_label} | `{rel_to(ref_b_path if ref_b is not None else ref_oof_b_path, ROOT)}` | {EVALUATORS['yolo']} |" if (ref_b is not None or ref_oof_b is not None) else ""}

Boundary and IoU tables are computed here from exactly these masks. An earlier final model's figures appear only in rows marked **[reference]**, never merged into this model's rows, and detection metrics are taken only from the evaluator of the model that produced the masks ({seg_evaluator}) — the two frameworks' mAP definitions are not interchangeable (PLAN.md 4).

## Primary result — (a) all {int(prim['n'])} reference images, out-of-fold masks
{fmt_row(prim)}.
**Segmentation failure: n = {int(prim['n_segmentation_failure'])} of {len(per)} ({100 * prim['n_segmentation_failure'] / len(per):.1f} %)** — {', '.join(chk['empty_gingiva_prediction']) or 'none'}: no gingiva instance predicted, so no measurement exists; excluded from the mm and label metrics above and reported as a separate failure category (a deployed system must flag such images for manual review rather than output a value).
Stage-3 holdout images only (the scale was never fitted on them): {fmt_row(hold)}.

### Secondary — same set, corrected ({corr_tag})
{fmt_row(prim_c)}. Images with columns zeroed by the shift: {int(prim_c['images_with_zeroed_columns'])} (mean {100 * prim_c['zeroed_columns_frac_mean']:.2f} % of gingiva columns).
Stage-3 holdout only, corrected: {fmt_row(hold_c)}.
Same method and scale on the ground-truth masks (Stage 3): MAE {gt_all['mae']:.3f} mm, r {gt_all['r']:.3f}, ICC {gt_all['icc2_1']:.3f}, bias {gt_all['ba_bias']:+.3f} mm → the segmentation adds {prim['mae'] - gt_all['mae']:+.3f} mm MAE and {prim['ba_bias'] - gt_all['ba_bias']:+.3f} mm bias (see `error_decomposition.md`).

## Secondary — (b) final model, {int(sec['n'])} test-set high images
{fmt_row(sec)}.
The fold models on the same images (b'): MAE {bprime['mae']:.3f} mm, bias {bprime['ba_bias']:+.3f} mm, ICC {bprime['icc2_1']:.3f}.
Corrected ({corr_tag}): {fmt_row(sec_c)}. Images with zeroed columns: {int(sec_c['images_with_zeroed_columns'])}.

## Error decomposition (a)
Segmentation part: bias {ds['e_seg_bias']:+.3f} mm, MAE {ds['e_seg_mae']:.3f} mm; geometry part (Stage 3): bias {ds['e_meas_bias']:+.3f} mm, MAE {ds['e_meas_mae']:.3f} mm; variance shares {100 * ds['share_seg']:.0f} % / {100 * ds['share_meas']:.0f} % / covariance {100 * ds['share_cov']:+.0f} %. The segmentation error follows the lower gingiva edge: bias {ba['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm (predicted gingiva extends below the annotated edge), upper edge bias {ba['gingiva_top_edge_bias_mm_mean']:+.2f} mm — `error_decomposition.md`, figure `figures/error_decomposition.png`.

## Segmentation quality (three sets; `boundary_by_set.md`) — {prov_test['label']}, masks `{rel_test}` / `{rel_oof}`
| set | n | gingiva IoU mean (median) | upper edge MAE, mm | lower edge MAE, mm | lower edge bias, mm | lip IoU |
|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | {int(ba['n'])} | {ba['gingiva_mask_iou_mean']:.3f} ({ba['gingiva_mask_iou_median']:.3f}) | {ba['gingiva_top_edge_mae_mm_mean']:.2f} | {ba['gingiva_bottom_edge_mae_mm_mean']:.2f} | {ba['gingiva_bottom_edge_bias_mm_mean']:+.2f} | {ba['lip_mask_iou_mean']:.3f} |
| (b) test high, final model | {int(bb['n'])} | {bb['gingiva_mask_iou_mean']:.3f} ({bb['gingiva_mask_iou_median']:.3f}) | {bb['gingiva_top_edge_mae_mm_mean']:.2f} | {bb['gingiva_bottom_edge_mae_mm_mean']:.2f} | {bb['gingiva_bottom_edge_bias_mm_mean']:+.2f} | {bb['lip_mask_iou_mean']:.3f} |
| (c) test all | {int(bc['n'])} | {bc['gingiva_mask_iou_mean']:.3f} ({bc['gingiva_mask_iou_median']:.3f}), defined on {int(bc['gingiva_mask_iou_n'])} | {bc['gingiva_top_edge_mae_mm_mean']:.2f} | {bc['gingiva_bottom_edge_mae_mm_mean']:.2f} | {bc['gingiva_bottom_edge_bias_mm_mean']:+.2f} | {bc['lip_mask_iou_mean']:.3f} |
| (c) test low | {int(b_low['n'])} | {b_low['gingiva_mask_iou_mean']:.3f} ({b_low['gingiva_mask_iou_median']:.3f}) | {b_low['gingiva_top_edge_mae_mm_mean']:.2f} | {b_low['gingiva_bottom_edge_mae_mm_mean']:.2f} | {b_low['gingiva_bottom_edge_bias_mm_mean']:+.2f} | {b_low['lip_mask_iou_mean']:.3f} |
| (c) test normal | {int(b_norm['n'])} | {b_norm['gingiva_mask_iou_mean']:.3f} ({b_norm['gingiva_mask_iou_median']:.3f}) | {b_norm['gingiva_top_edge_mae_mm_mean']:.2f} | {b_norm['gingiva_bottom_edge_mae_mm_mean']:.2f} | {b_norm['gingiva_bottom_edge_bias_mm_mean']:+.2f} | {b_norm['lip_mask_iou_mean']:.3f} |

Low/normal IoU is low by construction (thin or absent gingiva: median annotated width {b_low['gingiva_n_columns_gt_median']:.0f} columns in low vs {ba['gingiva_n_columns_gt_median']:.0f} in high; undefined IoU on {int(b_low['n_neither']) + int(b_norm['n_neither'])} images with no gingiva in either mask), while the edge errors there are no larger than in the high set. The pipeline is specified for the high smile line; (a) is the segmentation result that matters for the measurement.

## Fallback transparency
Zenith regioning ({combo}) fell back to equal splits on {fb['n_fallback']} of {fb['n']} OOF images ({100 * fb['frac']:.0f} %; GT masks in Stage 3: {fb_gt['n_fallback']} of {fb_gt['n']}, {100 * fb_gt['frac']:.0f} %). Pre-registered rule: re-evaluate against {alt} above 30 % → {'TRIGGERED' if fb['reevaluate'] else 'not triggered'}; the {alt} row is in the table below either way.

## All rows (`measurement_accuracy.csv`)
{md_table(show)}

## Tooth level
`mixed_models.md`, `tooth_level.csv` — random intercept per patient; intercept of pipeline − reference: {mm_ref[0]['fixed_effects'].loc['Intercept', 'estimate']:+.3f} mm [{mm_ref[0]['fixed_effects'].loc['Intercept', 'ci_low']:+.3f}, {mm_ref[0]['fixed_effects'].loc['Intercept', 'ci_high']:+.3f}] ({mm_ref[0]['estimator']}); segmentation part alone: {mm_gt[0]['fixed_effects'].loc['Intercept', 'estimate']:+.3f} mm [{mm_gt[0]['fixed_effects'].loc['Intercept', 'ci_low']:+.3f}, {mm_gt[0]['fixed_effects'].loc['Intercept', 'ci_high']:+.3f}].

## Deviations
See `SAPMALAR.md`.
"""
    (out_dir / "prediction_summary.md").write_text(summary, encoding="utf-8")
    ozet = f"""# Aşama 6 — Türkçe özet (tahmin maskeleri üzerinde doğruluk)

- **Model ve kaynak:** OOF maskeleri {prov_oof['label']} → `{rel_oof}`; final model {prov_test['label']} → `{rel_test}`. Sınır/IoU tabloları bu maskelerden burada hesaplandı; tespit/segmentasyon metrikleri yalnız bu modelin kendi değerlendiricisinden ({seg_evaluator}) alınır, başka bir modelin dosyası kullanılmaz.
- OOF kontrolü: {chk['n_rows']}/{chk['n_reference']} görüntü, hepsi `{list(chk['mask_source'])[0]}`, fold başına {list(chk['folds'].values())}; **segmentasyon başarısızlığı n = {len(chk['empty_gingiva_prediction'])}** ({', '.join(chk['empty_gingiva_prediction']) or 'yok'}: dişeti örneği yok → mm değeri yok; ayrı kategori olarak raporlanır, mm/sınıf metriklerine girmez).
- Yöntem ve ölçek sabit (Aşama 3, GT maske): **{combo}**, **{k:.2f} px/mm**; tahmin maskelerinde yeniden seçim/yeniden uydurma yapılmadı.
- **Birincil (a) — 145 ölçümlü high, OOF maske (ölçülen n = {int(prim['n'])}):** MAE {prim['mae']:.2f} mm, RMSE {prim['rmse']:.2f}, r {prim['r']:.3f}, ICC(2,1) {prim['icc2_1']:.3f} [{prim['icc2_1_ci_low']:.3f}, {prim['icc2_1_ci_high']:.3f}]; sapma {prim['ba_bias']:+.2f} mm, LoA {prim['ba_loa_low']:.2f}…{prim['ba_loa_high']:.2f}; sınıf uyumu {100 * prim['threshold_agreement']:.0f} %, doğrusal ağırlıklı κ {prim['threshold_kappa_linear']:.2f} [{prim['threshold_kappa_linear_ci_low']:.2f}, {prim['threshold_kappa_linear_ci_high']:.2f}].
  - Aynı yöntem GT maskede (Aşama 3): MAE {gt_all['mae']:.2f}, sapma {gt_all['ba_bias']:+.2f} → segmentasyonun eklediği: MAE {prim['mae'] - gt_all['mae']:+.2f} mm, sapma {ds['e_seg_bias']:+.2f} mm (alt dişeti kenarı GT'den {ba['gingiva_bottom_edge_bias_mm_mean']:.2f} mm aşağıda çiziliyor; üst kenar {ba['gingiva_top_edge_bias_mm_mean']:+.2f} mm).
  - Aşama 3 holdout'u (ölçek hiç görmedi, n = {int(hold['n'])}): MAE {hold['mae']:.2f}, ICC {hold['icc2_1']:.3f}.
  - **İkincil, düzeltilmiş (maske düzeyi, config `bottom_edge_offset_px` = {off:+.0f} px = {off_mm:+.2f} mm):** MAE {prim_c['mae']:.2f} mm, ICC {prim_c['icc2_1']:.3f}, sapma {prim_c['ba_bias']:+.2f}, sınıf uyumu {100 * prim_c['threshold_agreement']:.0f} %, κ {prim_c['threshold_kappa_linear']:.2f} [{prim_c['threshold_kappa_linear_ci_low']:.2f}, {prim_c['threshold_kappa_linear_ci_high']:.2f}]; holdout MAE {hold_c['mae']:.2f}; kaydırmayla sıfırlanan sütunu olan görüntü {int(prim_c['images_with_zeroed_columns'])}.
- **İkincil (b) — test high {int(sec['n'])} görüntü, final model:** MAE {sec['mae']:.2f} mm, r {sec['r']:.3f}, ICC {sec['icc2_1']:.3f}, sapma {sec['ba_bias']:+.2f}; κ {sec['threshold_kappa_linear']:.2f} [{sec['threshold_kappa_linear_ci_low']:.2f}, {sec['threshold_kappa_linear_ci_high']:.2f}] (n küçük, GA geniş). Düzeltilmiş: MAE {sec_c['mae']:.2f}, sapma {sec_c['ba_bias']:+.2f}, κ {sec_c['threshold_kappa_linear']:.2f}.
- **Segmentasyon kalitesi:** (a) dişeti IoU {ba['gingiva_mask_iou_mean']:.3f}, üst kenar MAE {ba['gingiva_top_edge_mae_mm_mean']:.2f} mm, alt kenar MAE {ba['gingiva_bottom_edge_mae_mm_mean']:.2f} mm (sapma {ba['gingiva_bottom_edge_bias_mm_mean']:+.2f}), dudak IoU {ba['lip_mask_iou_mean']:.3f}; (b) IoU {bb['gingiva_mask_iou_mean']:.3f}; (c) tüm test {int(bc['n'])} görüntü IoU {bc['gingiva_mask_iou_mean']:.3f} — low {b_low['gingiva_mask_iou_mean']:.3f}, normal {b_norm['gingiva_mask_iou_mean']:.3f}. Low/normal'da düşük IoU beklenen bir durum: dişeti ince ya da görünmüyor (low'da GT dişeti genişliği medyan {b_low['gingiva_n_columns_gt_median']:.0f} sütun, high'da {ba['gingiva_n_columns_gt_median']:.0f}); kenar hataları high'dan büyük değil. Birincil sonuç (a) üzerinden verilir.
- Zenith bölgeleme (C) geri düşme oranı OOF'ta {100 * fb['frac']:.0f} % (GT'de {100 * fb_gt['frac']:.0f} %); %30 kuralı {'TETİKLENDİ' if fb['reevaluate'] else 'tetiklenmedi'}; {alt} satırı tabloda.
- Diş düzeyi karma model (hasta rastgele kesişim): sapma {mm_ref[0]['fixed_effects'].loc['Intercept', 'estimate']:+.2f} mm [{mm_ref[0]['fixed_effects'].loc['Intercept', 'ci_low']:+.2f}, {mm_ref[0]['fixed_effects'].loc['Intercept', 'ci_high']:+.2f}].
- Sapmalar: `SAPMALAR.md` ({len(deviations)} madde{'' if seg_metrics else f'; {prov_test["label"]} için değerlendirici çıktısı yok — `segmentation_metrics.md` boş'}).
"""
    (out_dir / "OZET.md").write_text(ozet, encoding="utf-8")
    print(ozet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
