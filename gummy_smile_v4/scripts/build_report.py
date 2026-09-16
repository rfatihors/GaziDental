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
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg, Path(cfg["paths"]["outputs"]) / "07_report")
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    oracle_dir, pred_dir, expert_dir, prediction_eval_dir = outputs / "03_oracle", outputs / "05_predictions", outputs / "04_expert", outputs / "06_prediction"
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    splits = json.loads(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "splits.json").read_text())
    pairs = pd.read_csv(input_path(cfg, "same_patient_pairs_csv"), encoding="utf-8-sig")
    status = []

    # ---------------- figures
    status.append({"item": "Figure: system block diagram (Reviewer 1)", **F.block_diagram(fig_dir / "pipeline_block_diagram.md", fig_dir / "pipeline_block_diagram.png")})
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
                       **F.scatter_and_bland_altman(p6, fig_dir / "measurement_predicted_masks.png", "Full pipeline (predicted masks, out-of-fold) vs clinical reference — uncorrected (primary)", subset_col=None)})
        if "selected_mm_corrected" in p6.columns:
            status.append({"item": "Figure: measurement vs clinical reference, predicted masks (OOF), corrected (secondary)",
                           **F.scatter_and_bland_altman(p6, fig_dir / "measurement_predicted_masks_corrected.png",
                                                        f"Full pipeline, out-of-fold masks, corrected by the post-hoc pixel offset ({int(p6['offset_px'].iloc[0]):+d} px) — secondary", value_col="selected_mm_corrected", subset_col=None)})
    else:
        F.placeholder(fig_dir / "measurement_predicted_masks.png", "Measurement on predicted masks (OOF)", "Stage 6: scripts/run_prediction_eval.py")
        status.append({"item": "Figure: measurement vs clinical reference, predicted masks (OOF)", "status": "pending", "needs": "Stage 6 outputs", "path": str(fig_dir / "measurement_predicted_masks.png")})
    status.append({"item": "Figure: learning curve (Reviewers 2 & 4, Supplementary S1)",
                   **F.copy_or_placeholder(pred_dir / "learning_curve.png", fig_dir / "learning_curve.png", "Learning curve", "outputs/05_predictions/learning_curve.png (workstation)")})
    status.append({"item": "Figure: boundary error, upper/lower gingiva edge (Reviewer 2 #10)", **F.boundary_error_figure(pred_dir / "boundary_error.csv", fig_dir / "boundary_error.png")})
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
    df, st = T.segmentation_metrics(pred_dir); emit("segmentation_metrics_test", df, st, "Segmentation metrics on the fixed test set (per class)")
    df, st = T.learning_curve(pred_dir); emit("learning_curve", df, st, "Learning curve points (Supplementary S1)")
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
    r2_10 = "Boundary IoU and upper (lip-side) / lower (gingival margin) edge distance errors on the test set"
    bset = prediction_eval_dir / "boundary_by_set.csv"
    if bset.exists():
        b6 = pd.read_csv(bset).set_index("set")
        a6 = b6.loc["(a) OOF, 145 reference high"]
        tm = json.loads((pred_dir / "test_metrics.json").read_text()) if (pred_dir / "test_metrics.json").exists() else {}
        pc = tm.get("per_class", {})
        off_px = cfg["measurement"].get("offset_px", 0)
        r2_10 = (f"The mAP gap between lip (seg mAP@50 {pc.get('dudak', {}).get('seg_map50', float('nan')):.2f}) and gingiva ({pc.get('diseti', {}).get('seg_map50', float('nan')):.2f}) is decomposed at the boundary: "
                 f"on the 145 reference images (OOF) the upper, lip-side gingiva edge is accurate (MAE {a6['gingiva_top_edge_mae_mm_mean']:.2f} mm, bias {a6['gingiva_top_edge_bias_mm_mean']:+.2f} mm) while the lower, festooned gingival margin is placed systematically too low "
                 f"(MAE {a6['gingiva_bottom_edge_mae_mm_mean']:.2f} mm, bias {a6['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm; gingiva mask IoU {a6['gingiva_mask_iou_mean']:.2f}, lip IoU {a6['lip_mask_iou_mean']:.2f}). "
                 f"The gap is therefore not model incapacity but a constant over-inclusion of the thin lower margin — the same shift in every fold and in the final model — which the pipeline reports as such and corrects post hoc as a secondary result "
                 f"(pixel offset {off_px:+d} px, `06_prediction/offset_correction.md`, `offset_checks.md`). Low/normal smile lines lower the pooled test IoU further because their annotated gingiva is thin or absent (`06_prediction/boundary_by_set.md`)")
    rev = f"""# REVIZYON_OZETI — reviewer items and the outputs that answer them

Legend: ✅ available now, ⏳ pending (what is needed is written in `report_status.md`). Paths are relative to `outputs/07_report/` unless absolute.

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
- Hazır: blok diyagramı, GT overlay, GT-maske ölçüm figürü (Figure 6 yerine), veri seti sayıları, demografi, mm doğruluğu (GT satırları), gözlemci içi.
- Bekleyen: segmentasyon örnekleri, öğrenme eğrisi, sınır hatası, test metrikleri (iş istasyonu); tahmin maskesi doğruluğu (Aşama 6); uzman uyumu (gerçek formlar).
- `REVIZYON_OZETI.md`: hakem maddesi ↔ çıktı eşlemesi (✅ / ⏳).
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
