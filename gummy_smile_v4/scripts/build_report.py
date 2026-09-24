#!/usr/bin/env python
"""Stage 7 — manuscript figures, tables and the revision map (outputs/07_report/).

Runs on whatever is available: items that need workstation predictions (Stage 5/6) or real
expert forms (Stage 4) are written as PENDING placeholders and filled in by re-running the
same command once the inputs exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import input_path, load_config, resolve  # noqa: E402
from gsv4.report import figures as F  # noqa: E402
from gsv4.report import tables as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage6", default="09_final_rfdetr",
                    help="Stage-6 output sub-directory the manuscript is built from (default: the RF-DETR final model; "
                         "pass 06_prediction for the YOLOv11x results, which stay available as the previous final model)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg, Path(cfg["paths"]["outputs"]) / "07_report")
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    oracle_dir, pred_dir, expert_dir = outputs / "03_oracle", outputs / "05_predictions", outputs / "04_expert"
    prediction_eval_dir = outputs / args.stage6
    prev_dir = outputs / "06_prediction"          # YOLOv11x: the previous final model, reference only
    # Which model the manuscript numbers belong to — recorded by Stage 6 in every table it writes.
    acc6 = prediction_eval_dir / "measurement_accuracy.csv"
    model_label = "pending (Stage 6 has not run for this directory)"
    if acc6.exists():
        m6 = pd.read_csv(acc6)
        names = sorted({str(x) for x in m6.get("model", pd.Series(dtype=str)).dropna().unique() if "COCO annotations" not in str(x)})
        # the fold models and the final model carry the same architecture label, one of them extended
        names = [n for n in names if not any(o != n and o.startswith(n) for o in names)]
        model_label = ", ".join(names) if names else "not recorded by Stage 6 — re-run scripts/run_prediction_eval.py"
    elif args.stage6 != "06_prediction":
        raise SystemExit(f"{acc6} does not exist: Stage 6 has not been run for {args.stage6}. Run scripts/run_prediction_eval.py "
                         f"--out {args.stage6} (with its --oof-masks/--test-masks), or build the report from another directory with --stage6.")
    print(f"[stage7] manuscript numbers from {prediction_eval_dir} — {model_label}")
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    splits = json.loads(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "splits.json").read_text())
    pairs = pd.read_csv(input_path(cfg, "same_patient_pairs_csv"), encoding="utf-8-sig")
    status = []

    # ---------------- figures
    status.append({"item": "Figure: system block diagram (Reviewer 1)",
                   **F.block_diagram(fig_dir / "pipeline_block_diagram.md", fig_dir / "pipeline_block_diagram.png",
                                     model="\n".join(model_label.split(", ")[:2]) if acc6.exists() else "Instance segmentation model")})
    status.append({"item": "Figure: segmentation examples GT vs prediction (Reviewer 1)",
                   **F.segmentation_examples(cfg, manifest, [pred_dir / "test", pred_dir / "oof"], fig_dir / "segmentation_examples.png")})
    per = oracle_dir / "per_image_results.csv"
    if per.exists():
        status.append({"item": "Figure: measurement vs clinical reference, GT masks (replaces Figure 6)",
                       **F.scatter_and_bland_altman(pd.read_csv(per), fig_dir / "measurement_gt_masks.png", "Corrected measurement pipeline on ground-truth masks vs clinical reference")})
    else:
        F.placeholder(fig_dir / "measurement_gt_masks.png", "Measurement on GT masks", "outputs/03_oracle/per_image_results.csv")
        status.append({"item": "Figure: measurement vs clinical reference, GT masks", "status": "pending", "needs": str(per), "path": str(fig_dir / "measurement_gt_masks.png")})
    per6 = prediction_eval_dir / "per_image_results.csv"
    if per6.exists():
        p6 = pd.read_csv(per6)
        status.append({"item": "Figure: measurement vs clinical reference, predicted masks (OOF)",
                       **F.scatter_and_bland_altman(p6, fig_dir / "measurement_predicted_masks.png", f"Full pipeline (out-of-fold masks) vs clinical reference — {model_label}", subset_col=None)})
        if "selected_mm_corrected" in p6.columns and int(p6["bottom_edge_offset_px"].iloc[0]):
            status.append({"item": "Figure: measurement vs clinical reference, predicted masks (OOF), corrected (secondary)",
                           **F.scatter_and_bland_altman(p6, fig_dir / "measurement_predicted_masks_corrected.png",
                                                        f"Full pipeline, out-of-fold masks, corrected at mask level (lower gingiva edge {int(p6['bottom_edge_offset_px'].iloc[0]):+d} px) — secondary", value_col="selected_mm_corrected", subset_col=None)})
        else:   # no post-hoc calibration for this model: a stale corrected figure would contradict the tables
            (fig_dir / "measurement_predicted_masks_corrected.png").unlink(missing_ok=True)
    else:
        F.placeholder(fig_dir / "measurement_predicted_masks.png", "Measurement on predicted masks (OOF)", "Stage 6: scripts/run_prediction_eval.py")
        status.append({"item": "Figure: measurement vs clinical reference, predicted masks (OOF)", "status": "pending", "needs": "Stage 6 outputs", "path": str(fig_dir / "measurement_predicted_masks.png")})
    # the learning curve and the boundary figure belong to the model the manuscript reports
    lc_png = prediction_eval_dir / "learning_curve.png"
    lc_src = lc_png if lc_png.exists() else (pred_dir / "learning_curve.png" if args.stage6 == "06_prediction" else lc_png)
    status.append({"item": "Figure: learning curve (Reviewers 2 & 4, Supplementary S1)",
                   **F.copy_or_placeholder(lc_src, fig_dir / "learning_curve.png", f"Learning curve — {model_label}",
                                           f"{lc_png} (scripts/build_rfdetr_learning_curve.py on the workstation metrics)")})
    b_test_csv = prediction_eval_dir / "boundary_error_test.csv"
    b_src = b_test_csv if b_test_csv.exists() else (pred_dir / "boundary_error.csv" if args.stage6 == "06_prediction" else b_test_csv)
    status.append({"item": "Figure: boundary error, upper/lower gingiva edge (Reviewer 2 #10)", **F.boundary_error_figure(b_src, fig_dir / "boundary_error.png")})
    status.append({"item": "Figure: GT overlay examples (Stage 2)", **F.copy_or_placeholder(outputs / "02_measure" / "gt_overlay_examples.png", fig_dir / "gt_overlay_examples.png", "GT overlays", "outputs/02_measure")})

    # ---------------- tables
    def emit(name: str, df, st: dict, label: str):
        st = dict(st)
        if df is not None:
            df.to_csv(tab_dir / f"{name}.csv", index=False)
            (tab_dir / f"{name}.md").write_text(f"# {label}\n\n" + T.md_table(df) + "\n\n" + "\n".join(f"- {k}: {v}" for k, v in st.items() if k not in ("status",)) + "\n", encoding="utf-8")
            st["path"] = str(tab_dir / f"{name}.md")
        else:
            (tab_dir / f"{name}.md").write_text(f"# {label}\n\nPENDING — needs: {st.get('needs')}\n", encoding="utf-8")
            st["path"] = str(tab_dir / f"{name}.md")
        status.append({"item": f"Table: {label}", **st})

    df, st = T.dataset_counts(manifest, splits, pairs); emit("dataset_counts", df, st, "Dataset before/after cleaning and per split (Reviewers 2 #5/#6, 4)")
    df, st = T.demographics(manifest); emit("demographics", df, st, "Demographic coverage (Reviewer 3)")
    df, st = T.measurement_accuracy(oracle_dir, prediction_eval_dir); emit("measurement_accuracy", df, st, "Millimetre accuracy vs clinical reference (Reviewers 2, 4; Figure 6 replacement)")
    mcfg = cfg["measurement"]["method"]
    sel_combo = f"{mcfg['regioning']}_{mcfg['estimator']}" + ("_lipanchored" if mcfg.get("anchored") else "")
    df, st = T.estimator_sensitivity(oracle_dir, sel_combo, float(cfg["measurement"]["px_per_mm"]),
                                     predicted_oracle_dir=prediction_eval_dir / "oracle")
    emit("estimator_sensitivity", df, st, "Measurement-method sensitivity: every regioning x estimator combination (Appendix)")
    df, st = T.segmentation_metrics(pred_dir, prediction_eval_dir)
    # the two evaluators do not produce the same table: Ultralytics gives a row per class, a COCO
    # evaluation gives metric/value rows pooled over the classes. The heading says which one this is.
    emit("segmentation_metrics_test", df, st, "Segmentation metrics on the fixed test set"
         + (" (per class)" if df is not None and "class" in df.columns else " (pooled over the classes by this model's own evaluator)"))
    # the points come from the same directory as the figure above: the curve of the model this report
    # reports, never the previous final model's curve standing in for it (PLAN.md 4, Amendment 2)
    lc_dir = pred_dir if (args.stage6 == "06_prediction" and not (prediction_eval_dir / "learning_curve.csv").exists()) else prediction_eval_dir
    lc_producer = ("gsv4.train.learning_curve --collect on the workstation" if lc_dir == pred_dir else
                   f"scripts/build_rfdetr_learning_curve.py --out {args.stage6} on the workstation metrics")
    df, st = T.learning_curve(lc_dir, lc_producer); emit("learning_curve", df, st, "Learning curve points (Supplementary S1)")
    df, st = T.expert_agreement(expert_dir); emit("expert_agreement", df, st, "Model vs expert agreement (Reviewer 4: clinical validity)")
    intra = oracle_dir / "intra_observer.md"
    status.append({"item": "Table: intra-observer reliability of the reference", "status": "done" if intra.exists() else "pending", "path": str(intra)})

    # ---------------- revision map
    def find(prefix):
        return next((s for s in status if s["item"].startswith(prefix)), {"status": "pending", "path": "?"})

    def cell(prefix):
        s = find(prefix)
        return f"{'✅' if s['status'] == 'done' else '⏳'} `{Path(s.get('path', '?')).relative_to(out) if s.get('path', '?').startswith(str(out)) else s.get('path', '?')}`"

    ms = T.measurement_accuracy(oracle_dir, prediction_eval_dir)[1]
    est_st = find("Table: Measurement-method sensitivity")
    # R4-8: the controlled architecture comparison. Read from its own result files, so the map cannot
    # say "pending" for a run that has finished, nor report an outcome the tables do not carry.
    arch_dir = outputs / "08_architecture"
    arch_pair, arch_res = arch_dir / "paired_comparisons.csv", arch_dir / "resolution_controls.csv"
    arch_cell = ("⏳ pre-registered in `08_architecture/PROTOCOL.md`; the comparison has not produced "
                 "`paired_comparisons.csv` yet")
    if arch_pair.exists():
        ap_ = pd.read_csv(arch_pair)
        base = str(ap_["model_a"].mode().iloc[0])
        lead = ap_.sort_values("diff_mae_mm", ascending=False).iloc[0]
        ties = ap_[~((ap_["diff_mae_mm"] > 0.15) & ap_["excludes_zero"])]
        tie_txt = "; ".join(f"{r['model_b']} vs {r['model_a']} {r['diff_mae_mm']:+.3f} mm [{r['ci_low']:.3f}, {r['ci_high']:.3f}] (contains zero)"
                            for _, r in ties.iterrows())
        ctrl = ""
        if arch_res.exists():
            rc = pd.read_csv(arch_res)
            ctrl = ("; both pre-registered resolution controls kept the lead ("
                    + "; ".join(f"{r['diff_mae_mm']:.3f} mm [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" for _, r in rc.iterrows()) + ")")
        arch_cell = (f"✅ Repeated under a protocol committed before any run: same images, same participant-level partition, same fixed "
                     f"test set, same budget and early stopping, three seeds each, every architecture at its published defaults. "
                     f"Primary outcome mm MAE against the clinical reference on {int(lead['n'])} paired images. "
                     f"{tie_txt} — not distinguishable. {lead['model_b']} led {base} by {lead['diff_mae_mm']:.3f} mm "
                     f"[{lead['ci_low']:.3f}, {lead['ci_high']:.3f}]{ctrl}, which triggered the pre-registered 0.15 mm rule, so "
                     f"**the final model changed to {lead['model_b']}** and Stage 6 was repeated with it. A class-mapping fault in the "
                     f"first RF-DETR run is recorded in `PROTOCOL.md` Amendment 2; the predictions were regenerated from the saved "
                     f"checkpoints and every run is screened by `integrity_check.csv`")
    r2_10 = "Boundary IoU and upper (lip-side) / lower (gingival margin) edge distance errors on the test set"
    bset = prediction_eval_dir / "boundary_by_set.csv"
    if bset.exists():
        b6 = pd.read_csv(bset).set_index("set")
        a6 = b6.loc["(a) OOF, 145 reference high"]
        prov6 = prediction_eval_dir / "segmentation_metrics.json"
        block = json.loads(prov6.read_text()) if prov6.exists() else {}
        tm = block.get("metrics", {}) if block.get("available") else {}
        pc = tm.get("per_class", {})
        off_px = int(cfg["measurement"].get("bottom_edge_offset_px", 0))
        gap = (f"The mAP gap between lip (seg mAP@50 {pc['dudak']['seg_map50']:.2f}) and gingiva ({pc['diseti']['seg_map50']:.2f}) is decomposed at the boundary: "
               if {"dudak", "diseti"} <= set(pc) else
               f"The segmentation quality of the gingiva class is decomposed at the boundary ({block.get('evaluator', 'own evaluator')}; per-class mAP in `tables/segmentation_metrics_test.md`): ")
        r2_10 = (gap + 
                 f"on the 145 reference images (OOF) the upper, lip-side gingiva edge is accurate (MAE {a6['gingiva_top_edge_mae_mm_mean']:.2f} mm, bias {a6['gingiva_top_edge_bias_mm_mean']:+.2f} mm) while the lower, festooned gingival margin is placed systematically too low "
                 f"(MAE {a6['gingiva_bottom_edge_mae_mm_mean']:.2f} mm, bias {a6['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm; gingiva mask IoU {a6['gingiva_mask_iou_mean']:.2f}, lip IoU {a6['lip_mask_iou_mean']:.2f}). "
                 + (f"The gap is therefore not model incapacity but a constant over-inclusion of the thin lower margin — the same shift in every fold and in the final model — which the pipeline reports as such and corrects post hoc as a secondary result "
                    f"(mask-level correction: lower gingiva edge moved up {abs(off_px)} px before measurement, `06_prediction/offset_correction.md`, `offset_checks.md`). "
                    if off_px else
                    f"The gingiva class is a thin structure whose mAP is dominated by a few pixels of edge disagreement, while the edge that the measurement uses is accurate; "
                    f"this model needs no post-hoc correction of the lower margin (bias {a6['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm), unlike the previous final model, whose "
                    f"+0.66 mm over-inclusion and its calibration stay in the appendix (`06_prediction/offset_correction.md`). ")
                 + f"Low/normal smile lines lower the pooled test IoU further because their annotated gingiva is thin or absent (`{args.stage6}/boundary_by_set.md`)")
    rev = f"""# REVIZYON_OZETI — reviewer items and the outputs that answer them

Legend: ✅ available now, ⏳ pending (what is needed is written in `report_status.md`). Paths are relative to `outputs/07_report/` unless absolute.

**Final model of this revision: {model_label}.** Every measurement number, figure and segmentation metric below comes from
`outputs/{args.stage6}/`, which records the mask directory and the evaluator of each table. The YOLOv11x results
(`outputs/06_prediction/`, `outputs/05_predictions/`) remain in the repository as the **previous final model**: they are cited
as such in the architecture appendix and are never merged into the rows above.

| reviewer item | answer in the revision | output |
|---|---|---|
| R1 — no system diagram | Block diagram of the corrected pipeline (Mermaid + PNG) | {cell('Figure: system block diagram')} |
| R1 — show segmentation outputs | Six images, GT vs predicted class masks side by side | {cell('Figure: segmentation examples')} |
| R2 #5 / R4 — how 1,315 images became 2,235 / 3,403 | Instances ≠ images (papillae annotated separately: 3,938 gingiva + 1,318 lip instances) and the Roboflow 2× augmentation of the train split (2,814/923 → 5,628/1,846); exact Roboflow version metadata still to be exported by the user | `tables/dataset_counts.md`, audit §5.3 |
| R2 #6 / R4 (CLAIM) — data leakage | Content-based check found 55 same-patient pairs (22 cross-split, 15 touching the original test set); one image per patient kept, patient-level re-split with a fixed test set, model retrained | {cell('Table: Dataset before/after')} (pairs: {T.dataset_counts(manifest, splits, pairs)[1]['same_patient_pairs_detected']}) |
| R2 #10 — boundary accuracy (lip vs gingiva mAP gap) | {r2_10} | {cell('Figure: boundary error')}; `{prediction_eval_dir / 'boundary_by_set.md'}`, `{prediction_eval_dir / 'error_decomposition.md'}` |
| R2 / R4 — G*Power calculation inappropriate | Calculation removed; learning curve for data adequacy + precision-based justification for the agreement analysis | {cell('Figure: learning curve')}; precision note: {ms.get('precision_note', 'pending')} |
| R3 — demographics / sex distribution | Age/sex coverage per group on the cleaned dataset ("n with a record") | {cell('Table: Demographic coverage')} |
| R4 — calibration (px/mm) | Reference: per-image 1 mm probe interval in ImageJ on 2698×1799 copies; pipeline: single global scale fitted on the dev subset ({ms.get('selected_method', '?')}), per-image scale variability and expert-entered scales reported | `{oracle_dir / 'scale_estimation.md'}`, `{expert_dir / 'scale_agreement.md'}` (expert part ⏳ real forms) |
| R4 — circular E/T validation | Blinded three-expert evaluation (majority reference, linear-weighted κ primary, OOF predictions primary set); E4 not validated (no case) | {cell('Table: Model vs expert agreement')} |
| R4 — overlap with the J Dent 2026 cohort | Name matching: 149 of 216 high-smile-line images have a reference measurement from the earlier study's measurement file; clinical confirmation pending | `{outputs / '01_data' / 'parse_report.md'}` |
| R4 — external validity claim | Removed; single-centre, single-device limitation stated | manuscript text (Guc_analizi §2) |
| R4 #8 — architecture comparison not fair (different dataset versions) | {arch_cell} | `{outputs / '08_architecture' / 'RESULTS.md'}`, `{outputs / '08_architecture' / 'PROTOCOL.md'}`, `{outputs / '08_architecture' / 'PROTOCOL_ADDENDUM_resolution.md'}`, `{outputs / '08_architecture' / 'FINDINGS.md'}` |
| R4 / R2 — is the measurement method itself a free choice? | All {est_st.get('n_combinations', '?')} regioning x estimator combinations with their dev MAE, holdout MAE, ICC(2,1) and Bland-Altman bias. The method ({est_st.get('selected', '?')}) and the scale ({est_st.get('selected_px_per_mm', '?')} px/mm) were selected **once**, on ground-truth masks on the development subset, and were never re-selected or re-fitted on predicted masks; the table shows how little the result depends on that choice. Selection rule: {est_st.get('selection_rule', 'see oracle_summary.md')} Regioning fallback: {est_st.get('regioning_fallback', '?')} on ground-truth masks and {est_st.get('regioning_fallback_predicted_masks', '?')} on the out-of-fold predicted masks (pre-registered re-evaluation threshold {est_st.get('fallback_reeval_threshold', '30 %')}) | {cell('Table: Measurement-method sensitivity')} |
| Figure 6 inconsistency | Explained as a corrected software error (lip contour top-edge deviation, pixels reported as mm; v3 vs v4 on the same masks) and replaced by the corrected measurement vs reference figure | `{outputs / '02_measure' / 'v3_vs_v4.md'}`, {cell('Figure: measurement vs clinical reference, GT masks')}, {cell('Figure: measurement vs clinical reference, predicted')} |
| Intra-observer reliability (20 images) | ICC(2,1) 0.995 tooth level / 0.998 image level, SD 0.17 mm | {cell('Table: intra-observer')} |
| Segmentation performance, clean split | Per-class box/mask mAP@50, mAP@50–95, P, R, F1 and confusion matrix on the fixed test set | {cell('Table: Segmentation metrics')} |
"""
    (out / "REVIZYON_OZETI.md").write_text(rev, encoding="utf-8")
    st_df = pd.DataFrame(status)
    (out / "report_status.md").write_text("# Report build status\n\n" + T.md_table(st_df[["item", "status"] + [c for c in ("needs", "source", "path") if c in st_df.columns]]) + "\n", encoding="utf-8")
    n_done, n_pend = int((st_df["status"] == "done").sum()), int((st_df["status"] == "pending").sum())
    (out / "OZET.md").write_text(f"""# Aşama 7 — Türkçe özet

- `scripts/build_report.py` mevcut verilerle {n_done} öge üretti, {n_pend} öge bekliyor (`report_status.md`: ne gerektiği yazıyor). Veri gelince aynı komut yeniden çalıştırılır.
- Hazır:
{chr(10).join('  - ' + i for i in st_df.loc[st_df["status"] == "done", "item"])}
- Bekleyen:
{(chr(10).join('  - ' + i + ' — ' + str(n) for i, n in zip(st_df.loc[st_df["status"] == "pending", "item"], st_df.loc[st_df["status"] == "pending", "needs"] if "needs" in st_df.columns else st_df.loc[st_df["status"] == "pending", "item"])) or "  - yok")}
- `REVIZYON_OZETI.md`: hakem maddesi ↔ çıktı eşlemesi (✅ / ⏳).
- Hakem cevabı taslağı: `RESPONSE_TO_REVIEWERS.md`, durum tablosu `REBUTTAL_DURUM.md`, Türkçe özet `REBUTTAL_OZET.md` (üretim: `scripts/build_rebuttal.py`).
""", encoding="utf-8")
    # portable reports: strip the absolute project root from every written markdown/csv
    root = str(cfg["_root"]) + "/"
    for f in list(out.rglob("*.md")) + list(out.rglob("*.csv")):
        txt = f.read_text(encoding="utf-8")
        if root in txt:
            f.write_text(txt.replace(root, ""), encoding="utf-8")
    print(T.md_table(st_df[["item", "status"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
