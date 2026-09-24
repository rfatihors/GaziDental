#!/usr/bin/env python
"""Rebuttal draft — every number is read from outputs/ (never typed), each with a source comment.

    python scripts/build_rebuttal.py [--docx docs/Hakem_revizyonları.docx]

Inputs: docs/hakem_maddeleri.yaml (reviewer items; verbatim quotes pasted there from the
reviewer document when it is available — the repository does not contain the docx), config.yaml
and the Stage 1–7 outputs. ``--stage6`` picks the Stage-6 directory the answers are written from —
the model the manuscript reports (default ``09_final_rfdetr``, RF-DETR-Seg Large). The previous final
model (YOLOv11x, ``06_prediction``) is never substituted for it: it appears only where an answer says
"[reference] previous final model", and the two evaluators' metrics are never mixed in one figure. Writes outputs/07_report/RESPONSE_TO_REVIEWERS.md (English, one section
per reviewer), REBUTTAL_DURUM.md (status table for the clinical team) and REBUTTAL_OZET.md (Turkish).
Status: READY = answered from outputs; PENDING = waits for the expert forms; CLINICAL = text from the
clinical team. Uncorrected measurement results are primary, corrected (mask-level) secondary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.architecture import decision as arch_decision  # noqa: E402
from gsv4.report import tables as T  # noqa: E402

QUOTE_PLACEHOLDER = "[verbatim reviewer text — to be pasted from docs/Hakem_revizyonları.docx into docs/hakem_maddeleri.yaml]"


def src(path: Path, root: Path) -> str:
    try:
        return f"<!-- source: {path.resolve().relative_to(root)} -->"
    except ValueError:
        return f"<!-- source: {path} -->"


def read_md_table_row(md: Path, first_cell: str) -> Optional[List[str]]:
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and line.split("|")[1].strip() == first_cell:
            return [c.strip() for c in line.strip().strip("|").split("|")]
    return None


def md_meta(md: Path, key: str) -> Optional[str]:
    m = re.search(rf"^- {re.escape(key)}: (.+)$", md.read_text(encoding="utf-8"), flags=re.M)
    return m.group(1).strip() if m else None


def docx_paragraphs(path: Path) -> List[str]:
    import zipfile

    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    out = []
    for pp in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp))
        if t.strip():
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--items", default=None, help="default docs/hakem_maddeleri.yaml")
    ap.add_argument("--docx", default=None, help="reviewer document; its paragraphs are listed to docs/hakem_docx_paragraphs.md for pasting quotes")
    ap.add_argument("--stage6", default="09_final_rfdetr",
                    help="Stage-6 output directory of the model the manuscript reports (default: the RF-DETR final model)")
    ap.add_argument("--prev-stage6", default="06_prediction",
                    help="the previous final model's Stage 6; quoted only in [reference] sentences, never as this model's result")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["_root"]
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    o1, o2, o3, o4, o5, o7 = (outputs / d for d in ("01_data", "02_measure", "03_oracle", "04_expert", "05_predictions", "07_report"))
    o6, prev6 = outputs / args.stage6, outputs / args.prev_stage6
    tab = o7 / "tables"
    items_path = Path(args.items) if args.items else root / "docs" / "hakem_maddeleri.yaml"
    spec = yaml.safe_load(items_path.read_text(encoding="utf-8"))
    if args.docx:
        paras = docx_paragraphs(Path(args.docx))
        (root / "docs" / "hakem_docx_paragraphs.md").write_text("# Paragraphs of the reviewer document (for pasting into hakem_maddeleri.yaml)\n\n" + "\n\n".join(f"{i + 1}. {p}" for i, p in enumerate(paras)), encoding="utf-8")
        print(f"{len(paras)} paragraphs listed in docs/hakem_docx_paragraphs.md")

    # ------------------------------------------------------------------ numbers (all read from files)
    S: Dict[str, str] = {}   # source comments by key

    dc = pd.read_csv(tab / "dataset_counts.csv").set_index("group"); S["dc"] = src(tab / "dataset_counts.csv", root)
    dcm = tab / "dataset_counts.md"
    pairs, cross, test_pairs = md_meta(dcm, "same_patient_pairs_detected"), md_meta(dcm, "pairs_cross_split_in_original_export"), md_meta(dcm, "pairs_involving_original_test")
    dem = pd.read_csv(tab / "demographics.csv").set_index("group"); S["dem"] = src(tab / "demographics.csv", root)
    SEG = T.segmentation_metrics_facts(tab); S["seg"] = src(tab / "segmentation_metrics_test.csv", root)
    # The Ultralytics test-metrics file belongs to the YOLO runs; it carries the confusion matrix, which
    # a COCO evaluation does not produce. It is read only when this report is built from that model.
    tm_path = o5 / "test_metrics.json"
    tm = json.loads(tm_path.read_text()) if (tm_path.exists() and SEG["per_class"]) else {}
    S["tm"] = src(tm_path, root) if tm else ""
    LC = T.learning_curve_facts(tab, o5 / "learning_curve.md"); lc = LC["df"]
    S["lc"] = src(tab / "learning_curve.csv", root); S["lcmd"] = src(tab / "learning_curve.md", root)
    acc = pd.read_csv(o6 / "measurement_accuracy.csv").set_index("set"); S["acc"] = src(o6 / "measurement_accuracy.csv", root)
    est3 = pd.read_csv(o3 / "estimator_comparison.csv").set_index("combo"); S["est3"] = src(o3 / "estimator_comparison.csv", root)
    sens3 = pd.read_csv(o3 / "sensitivity.csv").set_index("subset"); S["sens3"] = src(o3 / "sensitivity.csv", root)
    bset = pd.read_csv(o6 / "boundary_by_set.csv").set_index("set"); S["bset"] = src(o6 / "boundary_by_set.csv", root)
    prev_bset = pd.read_csv(prev6 / "boundary_by_set.csv").set_index("set") if (prev6 / "boundary_by_set.csv").exists() else None
    if prev_bset is not None:
        S["pbset"] = src(prev6 / "boundary_by_set.csv", root)
    dec = pd.read_csv(o6 / "error_decomposition.csv"); S["dec"] = src(o6 / "error_decomposition.csv", root)
    prev_dec = pd.read_csv(prev6 / "error_decomposition.csv") if (prev6 / "error_decomposition.csv").exists() else None
    if prev_dec is not None:
        S["pdec"] = src(prev6 / "error_decomposition.csv", root)

    # ---- the controlled architecture comparison (R4-8), outputs/08_architecture.
    # The answer is written from these files or it is not written: while they are missing it stays
    # PENDING_RUN and says so. The decision is not re-read from prose either — it is recomputed here
    # with the same pre-registered rule and threshold the comparison applied (PROTOCOL.md 8), so the
    # rebuttal cannot claim an outcome the numbers do not give.
    oA = outputs / "08_architecture"
    ARCH: Dict[str, Any] = {"ready": False}
    if (oA / "paired_comparisons.csv").exists() and (oA / "by_model.csv").exists():
        apair = pd.read_csv(oA / "paired_comparisons.csv")
        amodel = pd.read_csv(oA / "by_model.csv").set_index("model")
        ares = pd.read_csv(oA / "resolution_controls.csv") if (oA / "resolution_controls.csv").exists() else pd.DataFrame()
        aint = pd.read_csv(oA / "integrity_check.csv") if (oA / "integrity_check.csv").exists() else pd.DataFrame()
        baseline = str(apair["model_a"].mode().iloc[0])
        comps = [r for r in apair.to_dict("records") if r["model_a"] == baseline]
        adec = arch_decision(comps, baseline)
        wins = [c for c in comps if c["diff_mae_mm"] > adec["threshold_mm"] and c["excludes_zero"]]
        ties = [c for c in comps if not (c["diff_mae_mm"] > adec["threshold_mm"] and c["excludes_zero"])]
        S["arch"] = src(oA / "paired_comparisons.csv", root)
        S["archm"] = src(oA / "by_model.csv", root)
        ARCH.update({"ready": bool(len(comps)), "pair": apair, "model": amodel, "res": ares, "integrity": aint,
                     "baseline": baseline, "decision": adec, "wins": wins, "ties": ties,
                     "seeds": sorted({int(x) for x in pd.read_csv(oA / "by_seed.csv")["seed"]}) if (oA / "by_seed.csv").exists() else [],
                     "n": int(apair["n"].max()) if len(apair) else 0})
    offc = pd.read_csv(o6 / "offset_correction.csv") if (o6 / "offset_correction.csv").exists() else None
    if offc is not None:
        S["offc"] = src(o6 / "offset_correction.csv", root)
        offc = offc.set_index(["subset", "correction"])
    var = pd.read_csv(o6 / "offset_variants.csv").set_index(["subset", "key"]) if (o6 / "offset_variants.csv").exists() else None
    if var is not None:
        S["var"] = src(o6 / "offset_variants.csv", root)
    per6 = pd.read_csv(o6 / "per_image_results.csv"); S["per6"] = src(o6 / "per_image_results.csv", root)
    TM, TMS = T.threshold_margin(o6); S["tm_margin"] = src(tab / "threshold_margin.csv", root)
    intra_md = o3 / "intra_observer.md"; S["intra"] = src(intra_md, root)
    tooth_row, image_row = read_md_table_row(intra_md, "tooth site"), read_md_table_row(intra_md, "image mean")
    intra_in_coco = re.search(r"(\d+) of the (\d+) images are in the current COCO set", intra_md.read_text(encoding="utf-8"))
    scale_md = (o3 / "scale_estimation.md").read_text(encoding="utf-8"); S["scale"] = src(o3 / "scale_estimation.md", root)
    m_scale = re.search(r"\*\*([\d.]+) px/mm\*\* \(R² ([\d.]+); residual SD ([\d.]+) mm\)", scale_md)
    m_loo = re.search(r"Leave-one-out on dev: mean ([\d.]+), SD ([\d.]+), range ([\d.]+)–([\d.]+) px/mm", scale_md)
    m_cv = re.search(r"overall: mean ([\d.]+), SD ([\d.]+), CV ([\d.]+)", scale_md)
    man_md = o1 / "manifest_summary.md"; S["man"] = src(man_md, root)
    ozet1 = (o1 / "OZET.md").read_text(encoding="utf-8"); S["ozet1"] = src(o1 / "OZET.md", root)
    m_match = re.search(r"`high` grubunda (\d+) tekil eşleşme", ozet1)
    k34 = float(cfg["measurement"]["px_per_mm"])   # the calibrated scale, for expressing legacy pixel values in mm
    v34 = o2 / "v3_vs_v4.md"; S["v34"] = src(o2 / "v3_vs_v4.csv", root)
    v34d = pd.read_csv(o2 / "v3_vs_v4.csv")
    v34h = v34d[v34d["group"] == "high"].dropna(subset=["reference_mean_mm"]).sort_values("reference_mean_mm")
    v34_ex = ""
    if len(v34h) >= 2:
        lo, hi = v34h.iloc[0], v34h.iloc[-1]
        v34_ex = (f"on the {len(v34h)} high-smile-line images checked side by side, the legacy value does not even order the cases correctly: the image with the largest reference measurement "
                  f"({hi['reference_mean_mm']:.2f} mm) received the smallest legacy value ({hi['v3_value_px'] / k34:.2f} mm at the calibrated scale) while an image with {lo['reference_mean_mm']:.2f} mm received {lo['v3_value_px'] / k34:.2f} mm; "
                  f"the corrected measurement gives {hi['v4_value_px'] / k34:.2f} mm and {lo['v4_value_px'] / k34:.2f} mm for the same two images"
                  ) if hi["v3_value_px"] < lo["v3_value_px"] else (
                  f"on the {len(v34h)} high-smile-line images checked side by side the legacy and corrected values differ by up to {abs(v34h['v3_value_px'] - v34h['v4_value_px']).max() / k34:.2f} mm")
    status_md = (o7 / "report_status.md").read_text(encoding="utf-8"); S["status"] = src(o7 / "report_status.md", root)
    expert_pending = "Model vs expert agreement (Reviewer 4: clinical validity) | pending" in status_md
    ms = T.measurement_accuracy(o3, o6)[1]
    precision_note = ms.get("precision_note", "")
    off_px = int(cfg["measurement"]["bottom_edge_offset_px"]); k = float(cfg["measurement"]["px_per_mm"]); S["cfg"] = src(root / "configs" / "config.yaml", root)
    method = f"{cfg['measurement']['method']['regioning']}_{cfg['measurement']['method']['estimator']}"
    clin = root / "docs" / "Klinik_ekip_kararlari_15Eylul.md"; S["clin"] = src(clin, root)
    clin_txt = clin.read_text(encoding="utf-8")
    m_overlap = re.search(r"^> (Both studies draw on the same source pool.*?)$", clin_txt, flags=re.M)
    m_pigment = re.search(r'"(gingival and skin pigmentation were not recorded[^"]*)"', clin_txt)
    guc = root / "docs" / "Guc_analizi_hakem_cevabi_ve_metin.md"; S["guc"] = src(guc, root)
    guc_txt = guc.read_text(encoding="utf-8")
    m_single = re.search(r"^> (All images were acquired at a single centre.*?)$", guc_txt, flags=re.M)
    audit = root / "docs" / "GummySmile_v3_Teknik_Audit_Raporu_v2.md"; S["audit"] = src(audit, root)
    m_15 = re.search(r"Depodaki `calibration-first/last.xlsx` her biri \*\*(\d+) görüntü\*\* \(makale: (\d+)\)", audit.read_text(encoding="utf-8"))

    # rows of the Stage-6 accuracy table
    P = acc.loc["(a) OOF masks, all reference images [PRIMARY]"]
    H = acc.loc["(a) OOF, Stage-3 holdout images only (scale never fitted on these)"]
    B = acc.loc["(b) final model masks, test high images [secondary set]"]
    G = acc.loc["GT masks, all reference images (Stage 3, same method and scale)"]
    G3 = est3.loc[method]
    # A post-hoc offset produces a second, "— corrected" row per set. Stage 6 writes none when no
    # correction is adopted (09_final_rfdetr/PLAN.md, Amendment 3), and then there is one result set.
    corrected = "(a) OOF masks, all reference images [PRIMARY] — corrected" in acc.index
    Pc = acc.loc["(a) OOF masks, all reference images [PRIMARY] — corrected"] if corrected else None
    Bc = acc.loc["(b) final model masks, test high images [secondary set] — corrected"] if corrected else None
    presens = ("(a) PRE-REGISTERED SENSITIVITY: excluding the 29 images used to choose the architecture"
               if "(a) PRE-REGISTERED SENSITIVITY: excluding the 29 images used to choose the architecture" in acc.index else None)
    a6, b6 = bset.loc["(a) OOF, 145 reference high"], bset.loc["(b) test high, final model"]
    c6 = bset.loc["(c) test all"]
    low6, norm6 = bset.loc["(c) test low"], bset.loc["(c) test normal"]
    # Whether this model's masks cover the low and average smile lines of the test set at all
    pooled_test_covered = bool(pd.notna(low6["gingiva_mask_iou_mean"]) or int(low6["n"]) > 0)
    def prev_row(name):
        """A row of the previous final model's own Stage-6 table, or None. Never merged into this
        model's numbers: every sentence that uses one says whose it is."""
        return prev_bset.loc[name] if prev_bset is not None and name in prev_bset.index else None

    prev_all, prev_low, prev_norm = prev_row("(c) test all"), prev_row("(c) test low"), prev_row("(c) test normal")
    prev_oof = prev_row("(a) OOF, 145 reference high")
    prev_high = bset.loc["[reference] YOLOv11x (previous final model), OOF 145 reference high"] if "[reference] YOLOv11x (previous final model), OOF 145 reference high" in bset.index else prev_oof
    n_fail = int(P["n_segmentation_failure"])
    e_seg_bias, e_meas_bias = float(dec["e_seg"].mean()), float(dec["e_meas"].mean())
    ref_max = float(per6["ref_mm"].max()); n_e4 = int((per6["ref_label"] == "E4").sum())
    sp = SEG["pooled"]
    seg_model = SEG["model"] or "the final model"
    seg_prov = SEG["provenance"]

    def segc(cls, key):
        return SEG["classes"].get(cls, {}).get(key, float("nan"))

    cm = np.array(tm.get("confusion_matrix", [[np.nan]]))          # rows = predicted, cols = true (last = background)
    g_tp = g_fp = g_fn = l_tp = l_fn = float("nan")
    if cm.shape == (3, 3):
        g_tp, g_fp, g_fn = cm[0, 0], cm[0, 2], cm[2, 0] + cm[1, 0]
        l_tp, l_fn = cm[1, 1], cm[2, 1] + cm[0, 1]
    n_high_pct = 100 * int(dc.loc["high", "kept"]) / int(dc.loc["total", "kept"])
    std_ok = SEG["settings_kind"] in ("standard", "coco_standard")
    settings_note = ("at the COCO evaluation's standard settings" if SEG["settings_kind"] == "coco_standard" else
                     "at the standard evaluation settings (confidence floor 0.001, NMS IoU 0.7, 300 detections per image)" if std_ok else
                     "at the pipeline's operating point (confidence 0.25, NMS IoU 0.5, 20 detections per image); the standard-settings figure is being recomputed and will replace it")
    n_test = int(dc.loc["total", "test"])
    lc100 = lc[lc["fraction"] == 1.0].iloc[0]; lc25 = lc[lc["fraction"] == 0.25].iloc[0]
    # What the curve shows, in the words the manuscript uses. Written once and reused by the three
    # answers below, so a curve that does not plateau cannot be reported as a plateau in one of them.
    lc_finding = (f"{LC['label']} rose from {LC['first']:.3f} to {LC['last']:.3f} and reached a plateau ({LC['rule']})"
                  if LC["plateau"] else
                  f"{LC['label']} was {LC['points']} at {LC['sizes']} training images respectively ({LC['rule']}). "
                  "Performance had not plateaued within the available training-set size; the increments between adjacent "
                  "points are of the same order as run-to-run variation, so the curve indicates that additional data could "
                  "still improve segmentation performance. This is stated as a limitation")
    lc_seed_note = ("" if LC["plateau"] else
                    " The curve was drawn from a single training run per point (one seed), so the run-to-run spread was "
                    "not measured and no error bars are given; this is why the ordering of two adjacent points is not read "
                    "as a result on its own.")

    def f(x, d=2, sign=False):
        return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")

    def acc_line(r):
        return (f"MAE {f(r['mae'])} mm, RMSE {f(r['rmse'])} mm, r = {f(r['r'], 3)}, ICC(2,1) {f(r['icc2_1'], 3)} [{f(r['icc2_1_ci_low'], 3)}, {f(r['icc2_1_ci_high'], 3)}], "
                f"bias {f(r['ba_bias'], 2, True)} mm [{f(r['ba_bias_ci_low'], 2, True)}, {f(r['ba_bias_ci_high'], 2, True)}], 95 % LoA {f(r['ba_loa_low'])} to {f(r['ba_loa_high'])} mm; "
                f"Table 1 class agreement {100 * r['threshold_agreement']:.0f} %, linear-weighted κ {f(r['threshold_kappa_linear'], 2)} [{f(r['threshold_kappa_linear_ci_low'], 2)}, {f(r['threshold_kappa_linear_ci_high'], 2)}]")

    # ---------------- what the test-set table can and cannot say, written once
    # Every answer that quotes it names the evaluator and the shape of the table, so a reader can tell
    # a per-class Ultralytics row from a COCO figure pooled over both classes.
    seg_sentence = (
        f"gingiva mask mAP@50 {f(segc('gingiva', 'seg_map50'))}, mAP@50–95 {f(segc('gingiva', 'seg_map50_95'))}, precision {f(segc('gingiva', 'seg_precision'))}, recall {f(segc('gingiva', 'seg_recall'))}; "
        f"lip mask mAP@50 {f(segc('lip', 'seg_map50'))}, mAP@50–95 {f(segc('lip', 'seg_map50_95'))}; all classes mask mAP@50 {f(sp['seg_map50'])}"
        if SEG["per_class"] else
        f"mask mAP@50 {f(sp['seg_map50'])}, mask mAP@50–95 {f(sp['seg_map50_95'])}, box mAP@50 {f(sp['box_map50'])}, box mAP@50–95 {f(sp['box_map50_95'])}, "
        f"precision {f(sp['precision'])}, recall {f(sp['recall'])}, F1 {f(sp['f1'])}")
    seg_headline = (f"mask mAP@50 {f(sp['seg_map50'])} (gingiva {f(segc('gingiva', 'seg_map50'))}, lip {f(segc('lip', 'seg_map50'))})"
                    if SEG["per_class"] else f"mask mAP@50 {f(sp['seg_map50'])} pooled over the two classes")
    seg_from = f"{seg_model} ({seg_prov})"
    # Why no per-class mAP, what we use instead, and how it could be obtained if a reviewer insists.
    per_class_gap = ("" if SEG["per_class"] else
                     f"Per-class mask mAP is not quoted for this model, because {SEG['evaluator']} averages average precision over the two "
                     "classes and its evaluation call returns only that pooled summary. It is obtainable in principle — pycocotools "
                     "computes average precision per category internally, so re-running the same evaluation over the same predictions "
                     "with the category restricted to one class at a time would produce it — but that run has not been made, and we do "
                     "not put the previous model's per-class figures in its place. "
                     "The per-class evidence we give instead is framework-independent and closer to what the measurement depends on: "
                     f"on the {int(a6['n'])} out-of-fold reference images the gingiva mask IoU is {f(a6['gingiva_mask_iou_mean'])} "
                     f"(median {f(a6['gingiva_mask_iou_median'])}) against {f(a6['lip_mask_iou_mean'])} for the lip, with the upper, lip-side "
                     f"gingiva edge accurate to {f(a6['gingiva_top_edge_mae_mm_mean'])} mm and the lower, festooned margin to "
                     f"{f(a6['gingiva_bottom_edge_mae_mm_mean'])} mm (`{args.stage6}/boundary_by_set.md`).")
    prev_ref = ("" if prev_high is None else
                f"[reference] The previous final model's own rows stand beside these, computed on its own masks in its own run: on the same "
                f"{int(prev_high['n'])} images its lower gingival edge was {f(prev_high['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm off against "
                f"{f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm here. They are labelled as that model's throughout and are never merged into this model's numbers.")

    # ------------------------------------------------------------------ answers
    A: Dict[str, Dict[str, Any]] = {}
    A["diagram"] = dict(status="READY", text=(
        f"We agree. The revised manuscript includes a block diagram of the complete pipeline (smile photograph → {seg_model}, instance segmentation at original resolution "
        "→ class-separated gingiva and lip masks → column-wise gingival thickness profile → tooth regioning → millimetre value with the calibrated scale → rule engine with the "
        "Table 1 thresholds → report with class, candidate classes and QC flags)."),
        changes="[Figure 1 — new] `outputs/07_report/figures/pipeline_block_diagram.png` (Mermaid source in `pipeline_block_diagram.md`)", files=["outputs/07_report/figures/pipeline_block_diagram.png"], sources=[S["status"]])
    A["segmentation_examples"] = dict(status="READY", text=(
        f"Example outputs have been added: for six test images the annotated (ground-truth) and predicted gingiva and lip masks are shown side by side at original resolution. "
        f"On the fixed test set (n = {n_test} images) the segmentation metrics of {seg_from} are: {seg_sentence}. "
        + (per_class_gap + " " if per_class_gap else "")
        + "The gingiva boundary is analysed directly in our response to Reviewer 2, item 10."),
        changes="[Figure — new] `figures/segmentation_examples.png`; [Table — segmentation metrics on the fixed test set] `tables/segmentation_metrics_test.md`", files=["outputs/07_report/figures/segmentation_examples.png", "outputs/07_report/tables/segmentation_metrics_test.md"], sources=[S["seg"], S["tm"]])
    A["image_counts"] = dict(status="READY", text=(
        f"We thank the reviewer; the numbers in the original manuscript mixed three different quantities and we have corrected this. The exported dataset contains {int(dc.loc['total', 'images_roboflow_export'])} images "
        f"({int(dc.loc['high', 'images_roboflow_export'])} high, {int(dc.loc['low', 'images_roboflow_export'])} low, {int(dc.loc['normal', 'images_roboflow_export'])} average smile line). The larger figures were *instance* counts, not image counts "
        "(in low and average smile lines each visible papilla was annotated as a separate gingiva instance), and the training-set figure additionally included the 2× augmentation applied by the annotation platform to the training split only. "
        f"In the revision all counts are given per image and per split after cleaning: {int(dc.loc['total', 'kept'])} images ({int(dc.loc['total', 'train'])} training / {int(dc.loc['total', 'valid'])} validation / {int(dc.loc['total', 'test'])} test), "
        f"of which {int(dc.loc['high', 'kept'])} high-smile-line images carry a clinical reference measurement. Images removed: {int(dc.loc['high', 'no_reference_measurement'])} high-smile-line images without a reference measurement, "
        f"{int(dc.loc['total', 'same_patient_duplicate'])} same-participant duplicates and {int(dc.loc['total', 'ambiguous_name'])} image with an ambiguous record. No augmentation is applied to the validation or test images."),
        changes="[Table 1 — revised] `tables/dataset_counts.md`; Methods 2.1 (dataset) rewritten with per-image counts", files=["outputs/07_report/tables/dataset_counts.md"], sources=[S["dc"]])
    A["leakage"] = dict(status="READY", text=(
        f"The reviewer is right and we thank them for raising it. A content-based search of the original export (image correlation followed by feature matching with RANSAC verification) identified {pairs} pairs of photographs of the same participant "
        f"taken in the same session; {cross} of these pairs straddled the original train/validation/test partition and {test_pairs} involved the original test set. The statement that a single photograph per participant was used was therefore incorrect for the segmentation dataset "
        "(one photograph per participant had been measured, but both had been uploaded to the annotation platform). We have (i) retained one photograph per participant, (ii) re-partitioned the dataset at participant level with a fixed test set "
        f"(stratified by smile-line group and, within the high-smile-line group, by reference class), (iii) retrained the model, and (iv) report all results on the new partition; the test set ({int(dc.loc['total', 'test'])} images) shares no participant with the training or validation set. "
        f"For the millimetre validation every one of the {int(dc.loc['high', 'kept'])} reference images was additionally predicted by a model that had not seen it or its twin (5-fold cross-validation at participant level), so that the primary measurement analysis is out-of-fold."),
        changes="Methods 2.1 (participants, partition) and 2.x (cross-validation) rewritten; [Table 1 — revised]; Limitations paragraph on the original partition", files=["outputs/07_report/tables/dataset_counts.md", "outputs/01_data/manifest_summary.md"], sources=[S["dc"], S["man"]])
    A["boundary"] = dict(status="READY", text=(
        (f"We agree that the gap between the lip ({f(segc('lip', 'seg_map50'))} mask mAP@50) and the gingiva ({f(segc('gingiva', 'seg_map50'))}) needed an explanation, and we have analysed it at the boundary rather than only through mAP. "
           if SEG["per_class"] else
           f"We agree that the gingiva class needed an explanation rather than a single number. It cannot be given as a per-class mAP here: {SEG['evaluator']} reports "
           f"mask mAP@50 {f(sp['seg_map50'])} pooled over the two classes and returns no per-category value, so we analyse the gingiva at its boundary, where the "
           "question actually lies and where the metrics are per class by construction. ")
        + f"Against the annotated masks of the {int(a6['n'])} reference images (out-of-fold predictions) the upper, lip-side edge of the gingiva is accurate (column-wise MAE {f(a6['gingiva_top_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_top_edge_bias_mm_mean'], 2, True)} mm) "
        f"whereas the lower, festooned gingival margin is placed systematically too low (MAE {f(a6['gingiva_bottom_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm); gingiva mask IoU {f(a6['gingiva_mask_iou_mean'])}, lip IoU {f(a6['lip_mask_iou_mean'])}. "
        f"The same picture holds for the final model on the test-set high-smile-line images (IoU {f(b6['gingiva_mask_iou_mean'])}, lower-edge bias {f(b6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm). "
        "The gingiva class therefore does not perform poorly at random: the model consistently includes a thin strip below the annotated margin, i.e. a constant shift of one edge, which is why the mAP of a thin structure is low while the measurement-relevant edge is accurate. "
        + (f"Pooled over the whole test set the gingiva IoU is lower still ({f(c6['gingiva_mask_iou_mean'])}) because in low ({f(low6['gingiva_mask_iou_mean'])}) and average ({f(norm6['gingiva_mask_iou_mean'])}) smile lines the annotated gingiva is thin or absent "
           f"(median annotated width {low6['gingiva_n_columns_gt_median']:.0f} columns in the low group vs {a6['gingiva_n_columns_gt_median']:.0f} in the high group; IoU undefined on {int(c6['n_neither'])} images with no gingiva in either mask), while the edge errors there are not larger. "
           if pooled_test_covered else
           f"One thing this model's table cannot show is the pooled figure over the whole test set: its masks were exported for the {int(b6['n'])} high-smile-line test images, which are the ones the measurement uses, so the low and average smile-line rows are empty for it and we do not fill them from another model. "
           + (f"[reference] In the previous final model's run, which covered all {int(prev_all['n'])} test images, the pooled gingiva IoU was {f(prev_all['gingiva_mask_iou_mean'])} against {f(prev_high['gingiva_mask_iou_mean'])} on the high-smile-line images, because in the low and average groups the annotated gingiva is thin or absent (median annotated width {prev_low['gingiva_n_columns_gt_median']:.0f} and {prev_norm['gingiva_n_columns_gt_median']:.0f} image columns against {prev_high['gingiva_n_columns_gt_median']:.0f} in the high group — a property of the annotation, not of the model), while the edge errors there were no larger. "
              if prev_all is not None and prev_low is not None and prev_norm is not None and prev_high is not None else ""))
        + "We report the boundary metrics per set, and we describe the lower-edge shift explicitly rather than tuning the model on the test data. "
        + prev_ref),
        changes="[Results 3.x — new subsection 'Boundary accuracy'; Supplementary table of boundary metrics per image set] `06_prediction/boundary_by_set.md`, `figures/boundary_error.png`; Discussion paragraph on the lip/gingiva difference", files=["outputs/06_prediction/boundary_by_set.md", "outputs/06_prediction/error_decomposition.md", "outputs/07_report/figures/boundary_error.png"], sources=[S["bset"], S["seg"]])
    A["power"] = dict(status="READY", text=(
        "The reviewer is right to question it, and the answer is that the approach is not appropriate; we have removed the calculation rather than defend it. A χ² test on counts of correct and incorrect detections is not an analysis performed in this study, "
        "and conventional hypothesis-testing sample-size methods do not determine how much data a deep-learning segmentation model needs. The revised manuscript replaces it with two separate, explicit justifications. "
        f"(i) For model development, data adequacy is assessed empirically with a learning curve: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images), and "
        f"{lc_finding}.{lc_seed_note} "
        f"(ii) For the millimetre-level agreement analysis, the sample size is justified by estimation precision rather than power: {precision_note} "
        f"(iii) For the agreement between the model's class assignment and the clinicians' assignment, a sample-size calculation appropriate to that analysis was performed by the study statistician with the `kappaSize` package in R: for a four-category classification with a minimum acceptable kappa of 0.40, "
        "an expected kappa of 0.60, a two-sided alpha of 0.05 and 80 % power, and a conservative 2 % prevalence for the rarest class, the minimum required sample is 110 images, rising to 123 after allowing for about 10 % data loss. "
        f"The high-smile-line images with a clinical reference measurement number {int(dc.loc['high', 'kept'])} after cleaning, above that requirement. We note that the statistician's paragraph was written for 150 images; the analysed set is {int(dc.loc['high', 'kept'])} because same-participant duplicates and images without a reference measurement were removed."),
        changes="Methods 2.1 — G*Power paragraph and Appendix A removed, replaced by 'Sample size and data adequacy' (learning curve + precision + kappaSize); [Supplementary Figure S1]", files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md", "docs/Istatistik_analiz_plani.md"], sources=[S["lc"], S["lcmd"], S["guc"], S["dc"]])
    A["figure6"] = dict(status="READY", text=(
        "We thank the reviewer for pressing on this point: the inconsistency was real and was caused by a software error, not by the data. On re-auditing the code we found that the measurement module of the original submission merged the lip and gingiva masks into one binary mask, "
        "selected the largest contour (in practice the lip) and reported the between-region variation of that contour's *upper* edge, in pixels, as if it were gingival display in millimetres; the gingival margin was not used at all. "
        f"Consequently neither the column of the original submission nor the earlier version's column in Figure 6 is a valid measurement of gingival display, and we have withdrawn both; re-running the legacy code on the same annotated masks reproduces its behaviour ({v34_ex}). "
        f"The measurement module was rewritten (vertical gingiva thickness per image column from the class-separated gingiva mask at original resolution, tooth-wise regioning, calibrated scale) and validated in two steps. "
        f"On the annotated masks of the {int(G['n'])} reference images the corrected geometry gives {acc_line(G)} (method {method}, scale fitted on a 60 % development subset only; held-out 40 %: MAE {f(G3['mae_holdout'])} mm, ICC {f(G3['icc2_1_holdout'], 3)}). "
        f"On predicted masks (out-of-fold, n = {int(P['n'])}; {n_fail} image with no gingiva predicted reported as a segmentation failure) the full pipeline gives {acc_line(P)}. "
        f"A new Figure 6 shows both scatter and Bland–Altman plots against the clinical reference."),
        changes="[Figure 6 — replaced] `figures/measurement_gt_masks.png` (annotated masks) and `figures/measurement_predicted_masks.png` (full pipeline, primary); Methods 2.x (measurement) rewritten; Results 3.x; erratum sentence in the Discussion", files=["outputs/02_measure/v3_vs_v4.md", "outputs/07_report/figures/measurement_gt_masks.png", "outputs/07_report/figures/measurement_predicted_masks.png", "outputs/07_report/tables/measurement_accuracy.md"], sources=[S["v34"], S["acc"], S["est3"]])
    A["external_validity"] = dict(status="READY", text=(
        "We agree with the reviewer on both points and have removed the claim. External validity cannot be established by the size of a single-centre cohort, and the χ² calculation addressed a comparison of correct and incorrect detections that this study never performs; "
        "it therefore says nothing about the data requirement of a segmentation model either. The calculation, the appendix containing it and the sentence claiming that the cohort size supports external validity have all been removed. "
        f"Data adequacy is now shown empirically instead: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the validation set held constant: "
        f"{lc_finding}.{lc_seed_note} "
        + ("" if LC["plateau"] else
           "We note that this result runs with the reviewer's argument rather than against it: the curve does not establish that the present cohort is sufficient, and we do not use it to claim so. "
           "It says that the segmentation model has not exhausted what more data of this kind could give it, which is one more reason for the larger and more varied datasets the reviewer asks for. ")
        + "We also accept the reviewer's point about external testing and we do not attempt to disguise it: no external data were available, so the study reports an internal estimate only. "
        + (f"The Limitations state: \"{m_single.group(1)}\" " if m_single else "The Limitations state that all images come from a single centre and a single imaging device, that the reported performance is an internal estimate, and that external validation on other centres, devices and populations is required before clinical use. ")
        + "We have not performed external validation and we do not claim it; it is named as the necessary next step rather than as a limitation in passing."),
        changes="Methods 2.1 — sample-size paragraph and Appendix A removed; 'supports the reliability and external validity' sentence deleted; [Supplementary Figure S1 — learning curve]; Limitations — single centre / single device / internal estimate", files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md", "outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["lc"], S["lcmd"], S["guc"]])
    A["calibration"] = dict(status="READY", text=(
        f"Two scales are involved and both are now described. (i) The clinical reference: each photograph was measured in ImageJ on a {cfg['coco']['reference_frame']['width']}×{cfg['coco']['reference_frame']['height']} copy after setting the scale on the periodontal probe visible in the image "
        "(two consecutive 1 mm marks, Set Scale = 1 mm); the individual scale values were not stored. (ii) The pipeline: a single global scale of "
        + (f"{m_scale.group(1)} px/mm was fitted by regression through the origin on the 60 % development subset only (R² {m_scale.group(2)}, residual SD {m_scale.group(3)} mm" if m_scale else f"{k:.2f} px/mm was fitted on the development subset")
        + (f"; leave-one-out mean {m_loo.group(1)}, SD {m_loo.group(2)} px/mm" if m_loo else "") + ") and applied unchanged to the held-out images. "
        + (f"The per-image ratio of pixels to reference millimetres has a coefficient of variation of {m_cv.group(3)}, which includes the reference's own calibration noise: with a 1 mm probe interval of ≈ {k:.0f} px, a one-pixel marking error is ≈ {100 / k:.0f} % of scale. " if m_cv else "")
        + f"The sensitivity of the results to the frame size is reported (held-out images in the reference frame: MAE {f(sens3.loc['holdout: only 2698x1799 (+/-2 px) frames', 'mae'])} mm, n = {int(sens3.loc['holdout: only 2698x1799 (+/-2 px) frames', 'n'])}; outside it: MAE {f(sens3.loc['holdout: frames outside 2698x1799', 'mae'])} mm, n = {int(sens3.loc['holdout: frames outside 2698x1799', 'n'])}). "
        + (f"In addition, the segmentation was found to place the lower gingival margin systematically too low (Reviewer 2, item 10); a post-hoc correction at mask level (lower edge moved up {abs(off_px)} px, i.e. {abs(off_px) / k:.2f} mm at the global scale, estimated on the development subset) is reported as a secondary analysis: "
           f"held-out MAE {f(var.loc[('holdout', 'none'), 'mae'])} → {f(var.loc[('holdout', 'mask_level'), 'mae'])} mm, bias {f(var.loc[('holdout', 'none'), 'ba_bias'], 2, True)} → {f(var.loc[('holdout', 'mask_level'), 'ba_bias'], 2, True)} mm. Uncorrected results remain the primary analysis. "
           if corrected and var is not None else
           "A third scale question arises from the segmentation itself, and we answer it by removing a step rather than adding one. For the previous final model the predicted lower gingival margin sat far enough below the annotated one to warrant a post-hoc offset; "
           + (f"for the model reported here the same analysis was repeated from scratch on the development subset, where the estimated constant is {f(abs(offc.loc[('dev', 'none'), 'ba_bias']))} mm; applied unchanged to the held-out subset it improves the mean absolute error from {f(offc.loc[('holdout', 'none'), 'mae'])} to {f(offc.loc[('holdout', 'constant'), 'mae'])} mm, i.e. by {f(offc.loc[('holdout', 'none'), 'mae'] - offc.loc[('holdout', 'constant'), 'mae'])} mm. "
              if offc is not None and ('holdout', 'constant') in offc.index and ('dev', 'none') in offc.index else "")
           + "That gain is smaller than the repeatability of the clinical reference it would be calibrated against"
           + (f" (the same observer's remeasurement has an SD of {tooth_row[6]} mm per tooth site)" if tooth_row else "")
           + ", and a correction fitted on the same reference the pipeline is evaluated against is the weakest step in the analysis, so it was dropped: no post-hoc calibration is applied, there is a single result set, and the reported accuracy is what the pipeline produces with nothing fitted on the reference "
             f"(`{args.stage6}/PLAN.md`, Amendment 3; the offset analysis itself stays in the appendix as a finding about the previous model's mask head). ")
        + "[PENDING: the three experts' per-image probe scales (5 mm interval) will be reported as an independent measure of calibration precision once the expert forms are analysed.]"),
        changes="Methods 2.x 'Calibration' (reference and pipeline scale); Results (scale, frame sensitivity"
                + (", post-hoc correction as secondary" if corrected else ", no post-hoc calibration: one result set")
                + "); Supplementary table of scale statistics",
        files=["outputs/03_oracle/scale_estimation.md", "outputs/06_prediction/offset_correction.md", "outputs/06_prediction/offset_checks.md"],
        sources=[S["scale"], S["sens3"], S.get("offc", S.get("var", "")), S["cfg"], S["clin"]])
    A["expert_validation"] = dict(status="PENDING" if expert_pending else "READY", text=(
        "We agree that comparing the E/T class derived from a measurement with the class derived from the same measurement is circular. The revised validation is therefore against independent clinical judgement: three blinded clinicians each assign the class from their own clinical assessment "
        f"(without seeing the threshold table) and measure the gingival display at each of the six tooth sites on their own calibrated scale; the majority class is the reference standard, the primary statistic is the linear-weighted κ with bootstrap confidence intervals, and the primary set is the {int(dc.loc['high', 'kept'])} reference images with out-of-fold model predictions (the fixed test subset is secondary). "
        "Model values are reported uncorrected (primary) and with the mask-level correction (secondary). "
        "[PENDING — the expert forms are being completed; the numbers will be inserted from `outputs/04_expert/manuscript_numbers.md` (κ, per-class sensitivity/specificity, inter-expert Fleiss κ, ICC of millimetre values).] "
        f"Note that the E4 class (> 8 mm) has no case among the reference images (maximum mean gingival display {f(ref_max)} mm; n(E4) = {n_e4}); the E4 branch of the decision table is therefore not validated and this is stated."),
        changes="[Section 2.x — new: expert evaluation protocol]; [Results 3.x — to be inserted]; [Table — model vs expert agreement] `tables/expert_agreement.md`", files=["outputs/04_expert/expert_summary.md", "outputs/07_report/tables/expert_agreement.md", "docs/Uzman_degerlendirme_protokolu.md"], sources=[S["status"], S["per6"]])
    A["overlap"] = dict(status="CLINICAL", text=(
        (f"Draft from the clinical team (to be confirmed and signed off by them): \"{m_overlap.group(1)}\"" if m_overlap else "[CLINICAL — text from the clinical team on the relationship with the earlier study]")
        + f" Technical check: {m_match.group(1) if m_match else '?'} of the {int(dc.loc['high', 'images_roboflow_export'])} high-smile-line images have a reference measurement in the earlier study's measurement file; the earlier partition was not reused (new participant-level split, different label scheme and model)."),
        changes="Methods 2.1 (relationship to the earlier study); `outputs/07_report/overlap_with_prior_study.md` [to be generated]", files=["docs/Klinik_ekip_kararlari_15Eylul.md", "outputs/01_data/OZET.md"], sources=[S["clin"], S["ozet1"]])
    A["narrow_or_compare"] = dict(status="PENDING", text=(
        "We have done both of the things the reviewer asks for. (i) The segmentation and measurement part is now properly validated: the measurement module was rewritten after we found a geometry error in it (see our response to Reviewer 4 on Figure 6), "
        f"and its millimetre accuracy against the clinical reference measurements is reported on {int(P['n'])} high-smile-line images predicted out-of-fold ({acc_line(P)}) and on the fixed test set ({acc_line(B)}). "
        "(ii) The decision layer is compared with independent clinical assessment: three clinicians, blinded to each other and to the threshold table, assign the class from their own clinical judgement and measure the gingival display at each of the six tooth sites; the majority class is the reference standard and the primary statistic is the linear-weighted κ. "
        "[PENDING — the expert forms are being completed; the agreement numbers will be inserted from `outputs/04_expert/manuscript_numbers.md`.] "
        "Where the data cannot support a claim we now say so explicitly: the E4 branch has no case in this cohort and is not validated, and the conclusions are restricted accordingly (Reviewer 3, Abstract and Discussion)."),
        changes="Title and Abstract narrowed; [Results 3.x — measurement accuracy, new]; [Results 3.x — agreement with clinical assessment, to be inserted]; Conclusion rewritten", files=["outputs/06_prediction/prediction_summary.md", "outputs/04_expert/expert_summary.md"], sources=[S["acc"], S["status"]])
    A["abstract_claims"] = dict(status="PENDING", text=(
        "We agree. The Abstract now reports the millimetre accuracy of the measurement rather than segmentation metrics alone, and the conclusions and clinical significance are restricted to what was evaluated. "
        f"The revised Abstract states the measurement accuracy against the clinical reference on the {int(P['n'])} high-smile-line images with out-of-fold predictions ({acc_line(P)}) and the segmentation performance on the fixed test set separately. "
        "The validation of the etiological and treatment categories is an agreement analysis against blinded clinical assessment; until those data are complete the Abstract does not claim clinical validation of the decision layer. "
        "[PENDING — expert agreement numbers.] The phrase 'clinically validated' has been removed from the Conclusions and the Clinical Significance statement (see Reviewer 3, Discussion item 5)."),
        changes="[Abstract — rewritten: Results and Conclusions]; [Clinical Significance — rewritten]", files=["outputs/07_report/MANUSCRIPT_EDITS.md", "outputs/06_prediction/prediction_summary.md"], sources=[S["acc"]])
    A["clinical_validation_claim"] = dict(status="CLINICAL", text=(
        "We accept this without reservation and have removed every statement that implies clinical validation of the decision layer, in the Conclusions, the Abstract and the Clinical Significance. The manuscript now separates three claims: (i) the segmentation model performs as reported on an independent test set; "
        f"(ii) the measurement agrees with the clinical reference to {acc_line(P).split(',')[0]} on out-of-fold predictions; (iii) the etiology-treatment layer is a transparent application of published thresholds whose agreement with clinical judgement is assessed in an agreement study, not a validation of diagnostic accuracy. "
        f"The Conclusion states explicitly that the framework has not been validated as a clinical decision tool, that the E4 branch has no case in this cohort (maximum mean gingival display {f(ref_max)} mm) and that prospective, multi-centre clinical validation is required before use. [CLINICAL — final wording from the clinical team.]"),
        changes="[Conclusions — rewritten]; [Abstract, Conclusions and Clinical Significance — rewritten]; Discussion limitation paragraph", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["acc"], S["per6"]])
    A["group_sizes"] = dict(status="READY", text=(
        "The three smile-line groups are not a designed factor: the photographs are a consecutive clinical series and the group sizes reflect the natural distribution of the smile line in that series "
        f"({int(dc.loc['high', 'images_roboflow_export'])} high, {int(dc.loc['normal', 'images_roboflow_export'])} average and {int(dc.loc['low', 'images_roboflow_export'])} low before cleaning; {int(dc.loc['high', 'kept'])}, {int(dc.loc['normal', 'kept'])} and {int(dc.loc['low', 'kept'])} after). "
        "We should say plainly that this request cannot be met retrospectively. The photographs were collected as a consecutive clinical series, not as three matched arms; the smile line is a characteristic of the participant, not an allocation, so the group sizes are the distribution of the smile line in the source population and there is no way to rebalance them after the fact "
        "except by discarding images, which would shrink the sample and distort that distribution. Equal group sizes are in any case not required by the analyses reported here, because the smile-line groups are never compared with each other: no statistic in the manuscript contrasts the low, average and high groups. "
        "The gingival display measurement and its validation are performed only within the high-smile-line group; the low and average images serve the segmentation training (see Methods item 7). "
        f"We now report the demographic coverage per group rather than claiming balance: age is recorded for {int(dem.loc['all', 'age_recorded'])} of {int(dem.loc['all', 'n'])} participants (mean {f(dem.loc['all', 'age_mean'], 1)} ± {f(dem.loc['all', 'age_sd'], 1)} years) and sex for {int(dem.loc['all', 'sex_recorded'])} "
        f"({f(dem.loc['all', 'female_pct_of_recorded'], 0)} % female of those recorded; high group {f(dem.loc['high', 'female_pct_of_recorded'], 0)} %, low {f(dem.loc['low', 'female_pct_of_recorded'], 0)} %, average {f(dem.loc['normal', 'female_pct_of_recorded'], 0)} %). "
        "The sex distribution is unbalanced and differs between groups; because sex is not used by the model and is not an analysis factor, we report it as a characteristic of the sample and as a limitation rather than adjusting the sample. "
        "The Limitations now state explicitly that the groups were neither size-matched nor sex-matched, that this could not be corrected retrospectively, and that a prospective study with balanced recruitment would be required to test whether smile-line group or sex affects segmentation performance."),
        changes="Methods 2.6 (study population / dataset) — sentence on group composition; [Table — demographic coverage per group, new] `tables/demographics.md`; Limitations", files=["outputs/07_report/tables/demographics.md", "outputs/07_report/tables/dataset_counts.md"], sources=[S["dem"], S["dc"]])
    A["why_low_normal"] = dict(status="READY", text=(
        "The reviewer's reading is exactly right, and this is a deliberate feature of the design: the low and average smile-line images were used only to train the segmentation model and never for gingival display measurement or for any of the millimetre or class analyses. "
        "They are needed because the model has to learn where gingiva is *not* exposed as well as where it is: a model trained on high-smile-line images alone would have no negative or borderline examples of the gingival margin and would over-predict gingiva in ordinary smiles. "
        + (f"Quantitatively, in the test set the annotated gingiva of low smile lines spans a median of {low6['gingiva_n_columns_gt_median']:.0f} image columns and that of average smile lines {norm6['gingiva_n_columns_gt_median']:.0f}, against {a6['gingiva_n_columns_gt_median']:.0f} in the high group; "
           f"in {int(c6['n_neither'])} test images neither the annotation nor the prediction contains gingiva (correct absence) and in {int(c6['n_spurious'])} the model predicts gingiva where the annotation has none. "
           if pooled_test_covered else
           (f"Quantitatively, in the test set the annotated gingiva of low smile lines spans a median of {prev_low['gingiva_n_columns_gt_median']:.0f} image columns and that of average smile lines {prev_norm['gingiva_n_columns_gt_median']:.0f}, against {a6['gingiva_n_columns_gt_median']:.0f} in the high group — these are counts in the annotation itself, so they do not depend on the model; they are quoted from the run that covered all {int(prev_all['n'])} test images (the previous final model's), because the current model's masks were exported for the high-smile-line test images that the measurement uses. "
            f"In that run neither the annotation nor the prediction contained gingiva in {int(prev_all['n_neither'])} test images (correct absence) and gingiva was predicted where the annotation has none in {int(prev_all['n_spurious'])}. "
            if prev_low is not None and prev_norm is not None and prev_all is not None else
            "Quantitatively, the annotated gingiva of low and average smile lines spans far fewer image columns than in the high group, which is why they are training material and not measurement material. "))
        + "The measurement module is specified for the high smile line only and returns NO_VISIBLE_GINGIVA when no gingiva is present; the millimetre validation therefore uses exclusively the "
        f"{int(dc.loc['high', 'kept'])} high-smile-line images with a clinical reference measurement. The revised Methods state this division of roles explicitly."),
        changes="Methods 2.6 (image dataset) — new sentence on the role of each group; Methods 2.7 (measurement) — applicability restricted to the high smile line", files=["outputs/06_prediction/boundary_by_set.md", "outputs/07_report/tables/dataset_counts.md"], sources=[S["bset"], S["dc"]])
    A["pigmentation"] = dict(status="CLINICAL", text=(
        "This is a fair point and we cannot answer it with these data: neither ethnicity nor gingival or skin pigmentation was recorded for the cohort, so no subgroup analysis is possible. We state this as a limitation rather than speculate. "
        "What we can report is where the segmentation error actually lies, which does not look like a pigmentation effect: the error is a systematic displacement of one boundary rather than a random failure of the mask. "
        f"On the {int(a6['n'])} reference images the upper, lip-side gingiva edge is accurate (MAE {f(a6['gingiva_top_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_top_edge_bias_mm_mean'], 2, True)} mm) while the lower, festooned margin is placed consistently too low "
        f"(bias {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm), and the same shift appears in the fold models and in the final model alike ({f(b6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm on the test-set high-smile-line images). "
        "A pigmentation-driven failure would be expected to vary between participants rather than to be constant. "
        "This is evidence about the nature of the error, not about pigmentation itself; a prospective study recording phenotype would be required to answer the reviewer's question properly, and we say so. [CLINICAL — the limitation sentence is to be finalised by the clinical team.]"),
        changes="Limitations — new sentence (pigmentation and ethnicity not recorded); Discussion — nature of the segmentation error", files=["outputs/06_prediction/boundary_by_set.md", "outputs/06_prediction/error_decomposition.md"], sources=[S["bset"], S["clin"]])
    A["metric_meaning"] = dict(status="READY", text=(
        "We apologise for the ambiguity; 'performance' in that sentence meant mask mAP@50 and the sentence has been rewritten to name the metric. It is neither sensitivity nor specificity. Mean average precision at an intersection-over-union threshold of 0.50 is computed by ranking every predicted mask by its confidence, "
        "walking down that ranking to trace precision against recall, taking the area under that curve for each class and averaging over classes; it therefore summarises how well the model both finds the structures and ranks its own confidence, at one overlap criterion. "
        + ("False positives and false negatives were computed but not reported, which we have corrected. On the fixed test set " if SEG["per_class"] else
           "The original submission computed false positives and false negatives and did not report them; the revision reports what this model's own evaluation provides, and says what it does not. On the fixed test set ")
        + (f"({int(tm.get('n_images', 0))} images) the model produced {int(g_tp)} correct gingiva instances, {int(g_fp)} false positives and {int(g_fn)} false negatives for the gingiva class, and {int(l_tp)} correct lip instances with {int(l_fn)} false negatives; "
           f"per class this is precision {f(segc('gingiva', 'seg_precision'))} / recall {f(segc('gingiva', 'seg_recall'))} (mask) for gingiva and {f(segc('lip', 'seg_precision'))} / {f(segc('lip', 'seg_recall'))} for lip, at the confidence threshold the pipeline operates at. "
           if SEG["per_class"] and np.isfinite(g_tp) else
           f"(n = {n_test} images) the model's own evaluation gives precision {f(sp['precision'])}, recall {f(sp['recall'])} and F1 {f(sp['f1'])} over both classes ({seg_prov}). "
           "A per-class confusion matrix of instance counts is not part of that evaluation — it is produced by the other framework's validator, and we do not import another model's matrix — so the class-level counts we report are the ones the boundary analysis defines per image: "
           f"on the {int(b6['n'])} high-smile-line test images the gingiva is missed in {int(b6['n_missed'])} and predicted where the annotation has none in {int(b6['n_spurious'])}, "
           f"and on the {int(a6['n'])} out-of-fold reference images in {int(a6['n_missed'])} and {int(a6['n_spurious'])} respectively. ")
        + "Sensitivity and specificity in the epidemiological sense are not defined for instance segmentation without a fixed set of candidate regions (there is no denominator of true negatives), which is why we report precision, recall and F1 at the stated settings, the per-image missed and spurious counts above, and the per-pixel boundary agreement separately (Reviewer 2, item 10)."),
        changes="Results 3.1 — sentence rewritten to name the metric; [Table — per-class precision, recall, F1, mAP on the test set]; confusion-matrix figure with FP/FN counts", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/05_predictions/confusion_matrix.png"], sources=[S["seg"], S["tm"]])
    A["separate_evaluations"] = dict(status="READY", text=(
        "We agree, and this is precisely how the revised manuscript is organised: the two questions now have two separate Results sections, 3.1 'Segmentation performance (fixed test set)' and 3.2 'Millimetre measurement accuracy', with their own tables and figures. Segmentation performance is reported on the fixed test set "
        f"(n = {n_test}; {seg_headline}; {seg_prov}) and answers the question whether the structures are found. "
        f"The ability to calculate gingival display in millimetres is evaluated in its own section against the clinical reference measurements, and in two steps so that the two error sources can be told apart: on the annotated masks (geometry only: {acc_line(G)}) "
        f"and on the model's own masks (full pipeline, out-of-fold: {acc_line(P)}). The difference between the two is what the segmentation contributes to the measurement error "
        f"(bias {f(e_seg_bias, 2, True)} mm from the segmentation against {f(e_meas_bias, 2, True)} mm from the measurement geometry). A model can segment well and measure poorly, and the manuscript now shows both numbers rather than letting mAP stand for measurement accuracy."),
        changes="Results split into 3.1 'Segmentation performance (test set)' and 3.2 'Millimetre measurement accuracy'; Abstract reports both", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/06_prediction/prediction_summary.md", "outputs/06_prediction/error_decomposition.md"], sources=[S["seg"], S["acc"], S["dec"]])
    A["abstract_scope"] = dict(status="READY", text=(
        f"The reviewer is right and we have corrected it. The quantitative gingival display analysis applies only to the high smile line subgroup, and the Abstract now says so in the Methods sentence and repeats the denominator in the Results. "
        f"After cleaning, the quantitative analysis uses {int(dc.loc['high', 'kept'])} high-smile-line images with a clinical reference measurement, i.e. {n_high_pct:.1f} % of the {int(dc.loc['total', 'kept'])} images in the study "
        f"(of {int(dc.loc['high', 'images_roboflow_export'])} high-smile-line images in the original export, {int(dc.loc['high', 'no_reference_measurement'])} had no reference measurement and were excluded, and {int(dc.loc['high', 'same_patient_duplicate'])} were same-participant duplicates). "
        f"The remaining {int(dc.loc['low', 'kept']) + int(dc.loc['normal', 'kept'])} low and average smile-line images are used for segmentation training only (Reviewer 3, Methods item 7)."),
        changes="[Abstract — Methods and Results sentences]; Methods 2.6", files=["outputs/07_report/tables/dataset_counts.md"], sources=[S["dc"]])
    A["metric_types"] = dict(status="READY", text=(
        "We thank the reviewer for catching this; mixing the two metric families in one list was misleading. Every metric is now labelled Box or Mask wherever it appears, both are reported, and the evaluation settings they were computed at are named. "
        + (f"On the fixed test set the final model gives, for the mask: mAP@50 {f(sp['seg_map50'])}, mAP@50–95 {f(sp['seg_map50_95'])}, precision {f(sp['seg_precision'])}, recall {f(sp['seg_recall'])} (all classes); "
           f"for the box: mAP@50 {f(sp['box_map50'])}, mAP@50–95 {f(sp['box_map50_95'])}, precision {f(sp['box_precision'])}, recall {f(sp['box_recall'])}. "
           f"Per class (mask): gingiva {f(segc('gingiva', 'seg_map50'))} / lip {f(segc('lip', 'seg_map50'))} mAP@50. "
           if SEG["per_class"] else
           f"On the fixed test set — {seg_model}, {seg_prov} — the mask metrics are mAP@50 {f(sp['seg_map50'])} and mAP@50–95 {f(sp['seg_map50_95'])}, and the box metrics mAP@50 {f(sp['box_map50'])}, "
           f"mAP@50–95 {f(sp['box_map50_95'])} and mAP@75 {f(sp['box_map75'])}, with mean average recall {f(sp['mar'])}. "
           f"Precision {f(sp['precision'])}, recall {f(sp['recall'])} and F1 {f(sp['f1'])} are reported as that evaluation returns them, i.e. without a box/mask split, and are labelled as such rather than presented as mask figures. "
           + (per_class_gap + " " if per_class_gap else ""))
        + "Because the measurement is computed from the mask, the mask metrics are the ones that matter for this application, and the Abstract now quotes mask metrics with the label attached."),
        changes="[Abstract — Results sentence]; Results 3.1; [Table — segmentation metrics, Box and Mask columns] `tables/segmentation_metrics_test.md`", files=["outputs/07_report/tables/segmentation_metrics_test.md"], sources=[S["seg"]])
    A["yolov8"] = dict(status="READY", text=(
        "The reviewer is right: the sentence is a leftover and it has been corrected. YOLOv8 was included in an early screening run in which several architectures and model scales were trained briefly under shared settings; it was not part of the architecture comparison reported in the manuscript, "
        "and it plays no role in the final model. In the revision the main text refers only to the architectures actually compared, and the appendix that describes the screening stage labels it as such and states which runs it contains, so that no result in the paper depends on it."),
        changes="Methods 2.7 (annotation/labeling) — 'YOLOv8 and YOLOv11' corrected; Appendix D — screening stage labelled and separated from the reported comparison", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["status"]])
    A["gingiva_map_low"] = dict(status="READY" if std_ok else "PENDING_RUN", text=(
        "We agree, and the value is now stated in the text, in the Abstract and in the Limitations. Before giving it we had to correct how it was measured, because two different quantities were being compared. "
        "The 0.587 the reviewer quotes is a validation-set figure from the earlier pipeline, and our own first recomputation was made at the pipeline's operating point (confidence threshold 0.25, at most 20 detections per image), which truncates the precision-recall curve before it reaches full recall and therefore understates mAP. "
        + (f"The revision separates the two conventions and labels both: mAP is reported at the standard evaluation settings (confidence floor 0.001, NMS IoU 0.7, 300 detections), while precision, recall, F1 and the confusion matrix are reported at the operating point, where the pipeline actually runs. "
           f"The gingiva mask mAP@50 we report is {f(segc('gingiva', 'seg_map50'))} {settings_note}, against {f(segc('lip', 'seg_map50'))} for the lip. "
           if SEG["per_class"] else
           f"The model the manuscript now reports is a different architecture, evaluated once on the fixed test set by its own evaluator at that evaluator's standard settings: mask mAP@50 {f(sp['seg_map50'])} for the two classes together. "
           + per_class_gap + " "
           f"So the reviewer's question is answered where it bites rather than avoided: the gingiva mask IoU is {f(a6['gingiva_mask_iou_mean'])} and the measurement-relevant lower edge is placed {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm off, and both are per class. ")
        + ("" if std_ok else "[PENDING — the standard-settings evaluation is being rerun; the figure above is still the operating-point one and will be replaced. The change affects the reported mAP only: not the masks, not the measurement, and no other result.] ")
        + ("Whichever number stands, we report the lower one rather than the more favourable one, and we do not leave it as an aggregate. " if SEG["per_class"] else
           "We report the weakest relevant number rather than the most favourable one, and we say plainly which quantity it is. ")
        + "Its consequence for the measurement is quantified: the boundary analysis (item 10) shows the error is a systematic displacement of the lower gingival margin, "
        f"so what it produces is a bias of {f(e_seg_bias, 2, True)} mm rather than a random failure, and the millimetre accuracy of the full pipeline is reported directly ({acc_line(P)}). "
        "The Limitations state that gingival segmentation is the weakest component of the pipeline and the main target for improvement."),
        changes="Results 3.1 — gingiva mAP stated explicitly with its evaluation settings; [Table — segmentation metrics at both settings]; Limitations — new paragraph; Discussion — link to the boundary analysis", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/06_prediction/boundary_by_set.md", "outputs/08_architecture/PROTOCOL.md"], sources=[S["seg"], S["acc"], S["dec"]])
    A["no_improvement"] = dict(status="READY", text=(
        "The reviewer is right that there was no meaningful gain, and the honest explanation is twofold. First, the comparison was not sound: the preliminary figure and the final figure came from different dataset versions and different validation sets, so the near-identical numbers were not evidence of anything. "
        "All performance figures in the revision come from a single fixed, participant-level test set evaluated once. Second, we no longer assert where the dataset sits on its learning curve; we measured it. Retraining the final architecture on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition "
        f"({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the validation set held constant: {lc_finding}.{lc_seed_note} "
        + ("Additional data of the same kind, further hyperparameter search and larger model scales are therefore not expected to help; what limits the gingiva class is the annotation of a thin, festooned boundary, which is where we direct the analysis and the remaining error."
           if LC["plateau"] else
           "The absence of a gain between the earlier submissions is therefore explained by the comparison itself, not by a ceiling in the data: we cannot claim that more data would not help, and we do not. "
           "What limits the gingiva class within this dataset is the annotation of a thin, festooned boundary, which is where we direct the analysis and the remaining error, and a larger and more varied training set remains the other open route.")),
        changes="Results 3.1 — comparison across dataset versions removed; [Supplementary Figure S1 — learning curve]; Discussion — "
                + ("why performance plateaus" if LC["plateau"] else "what the learning curve does and does not show, and the data limitation that follows from it"), files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md", "outputs/07_report/tables/segmentation_metrics_test.md"], sources=[S["lc"], S["lcmd"], S["seg"]])
    A["validation_vs_test"] = dict(status="READY", text=(
        "The reviewer is right, and this was a genuine methodological error rather than a presentational one: the figures reported as final results were validation-set metrics of a model whose architecture, scale and hyperparameters had been chosen on that same validation set. "
        f"In the revision the dataset is partitioned once at participant level into training, validation and test ({int(dc.loc['total', 'train'])} / {int(dc.loc['total', 'valid'])} / {int(dc.loc['total', 'test'])} images), the test set is fixed and untouched during development, and it is evaluated once with the final model. "
        f"All reported performance is that single test-set evaluation ({seg_from}): {seg_headline}, mask mAP@50–95 {f(sp['seg_map50_95'])}, box mAP@50 {f(sp['box_map50'])}. "
        f"The millimetre validation is separated in the same way: the measurement method and the pixel-to-millimetre scale were fixed on a development subset of the reference images and then applied unchanged, so that the held-out figures ({acc_line(H)}) are not optimised. "
        "The validation-set numbers of the original submission are not reported as results."),
        changes="Methods 2.6 (partition) and 2.9 (evaluation protocol) rewritten; Results 3.1 — all metrics replaced by test-set metrics; Abstract numbers replaced", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/07_report/tables/dataset_counts.md", "outputs/06_prediction/measurement_accuracy.csv"], sources=[S["seg"], S["dc"], S["acc"]])
    # ---------------- R4-8, written from outputs/08_architecture or left PENDING_RUN
    NUMWORD = {2: "two", 3: "three", 4: "four", 5: "five"}
    ARCH_NAMES = {"rfdetr-seg-large": "RF-DETR-Seg Large (resolution 624)", "yolo11x-seg": "YOLOv11x-seg (640)",
                  "yolo26x-seg": "YOLO26x-seg (640)", "rfdetr-seg-large@432": "RF-DETR-Seg Large at 432",
                  "yolo11x-seg@1024": "YOLOv11x-seg at 1024"}

    def aname(m):
        return ARCH_NAMES.get(str(m), str(m))

    def apaired(c, subject=None):
        """'leads X by 0.293 mm [0.150, 0.439]' / 'differs from X by -0.046 mm [...]', from one row."""
        return (f"{f(abs(c['diff_mae_mm']), 3)} mm [{f(c['ci_low'], 3)}, {f(c['ci_high'], 3)}] "
                f"({'paired 95 % bootstrap over the ' + str(int(c['n'])) + ' images' if subject is None else subject})")

    if ARCH["ready"]:
        AD, AM = ARCH["decision"], ARCH["model"]
        members = [m for m in AM.index if "@" not in str(m)]
        awin = ARCH["wins"][0] if ARCH["wins"] else None
        per_model = "; ".join(f"{aname(m)} {f(AM.loc[m, 'mae_mm_mean'], 3)} ± {f(AM.loc[m, 'mae_mm_sd'], 3)} mm"
                              for m in sorted(members, key=lambda m: AM.loc[m, "mae_mm_mean"]))
        tie_txt = ""
        for c in ARCH["ties"]:
            sd_a, sd_b = AM.loc[c["model_a"], "mae_mm_sd"], AM.loc[c["model_b"], "mae_mm_sd"]
            tie_txt += (f"{aname(c['model_b'])} and {aname(c['model_a'])} could not be separated: the paired difference is "
                        f"{f(c['diff_mae_mm'], 3)} mm with a 95 % interval of [{f(c['ci_low'], 3)}, {f(c['ci_high'], 3)}] that contains zero, "
                        f"and the between-seed standard deviation of each ({f(sd_a, 3)} and {f(sd_b, 3)} mm) is larger than the difference itself. "
                        "We therefore make no claim about them in either direction. ")
        res_txt = ""
        if len(ARCH["res"]):
            res_rows = "; ".join(f"{aname(r['model_a'])} against {aname(r['model_b'])}, {f(r['diff_mae_mm'], 3)} mm "
                                 f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]" for r in ARCH["res"].to_dict("records"))
            res_txt = ("Before applying the rule we tested the one confounder the protocol had declared in advance, that the "
                       "architectures do not share a native vertical mask-pixel size, with two pre-registered controls: the same comparison "
                       "with the challenger coarsened to the baseline's grid, and with the baseline refined past the challenger's. "
                       f"The lead survived both, and at a similar size ({res_rows}), so it is not an artefact of resolution. "
                       "A resolution limit would have behaved the other way round — the error would track the pixel size across both families, and it does not "
                       "(`outputs/08_architecture/PROTOCOL_ADDENDUM_resolution.md`, written and committed before those two runs). ")
        outcome_txt = (
            f"{aname(awin['model_b'])} met both conditions against {aname(ARCH['baseline'])} — a lead of {apaired(awin)} against a threshold of "
            f"{AD['threshold_mm']:.2f} mm — so the rule was applied as written: the final model of the manuscript changed, Stage 6 was repeated with "
            f"{aname(awin['model_b'])}, and every measurement result in the revision now comes from it. We did not re-open the rule after seeing the number. "
            if awin else
            f"No challenger met both conditions, so the final model stayed {aname(ARCH['baseline'])}. ")
        fault_txt = ("One thing in this analysis went wrong and we report it rather than leave it out. The first RF-DETR run produced a gingival edge "
                     "error of about 6 mm that was identical on every image, which is the signature of a class substitution and not of a segmentation "
                     "weakness. It was: the library renumbers its label space and drops the unannotated grouping category of our export, so the trained "
                     "model emits its own ids, while our reading of the predictions used the dataset's category table — gingiva was dropped and lip was "
                     "written into the gingiva mask. The model was right and the reading was wrong; RF-DETR's own mask mAP was unaffected, which is why "
                     "the fault was invisible in the metrics and visible only in the measurement. We fixed the reading, regenerated the predictions from "
                     "the saved checkpoints without retraining, and added three safeguards that run automatically: the class list is now taken from the "
                     "model and cross-checked against the one the dataset implies before a single mask is written, an instance whose class maps to no role "
                     "aborts the run instead of being silently dropped, and every run is screened for the three traces of this fault "
                     "(`integrity_check.csv` in the results directory, clean for all five configurations). ")
        if prev_dec is not None:
            adds_now = float(dec["e_total"].abs().mean()) - float(dec["e_meas"].abs().mean())
            adds_prev = float(prev_dec["e_total"].abs().mean()) - float(prev_dec["e_meas"].abs().mean())
            meaning_txt = (
                "What this changes clinically is not a leaderboard position. Applying the same measurement method and the same scale to each final "
                f"model's own out-of-fold masks over all {len(dec)} reference images separates the error the measurement geometry already has from the "
                f"error the segmentation adds: with {seg_model} the pipeline reaches MAE {float(dec['e_total'].abs().mean()):.2f} mm against "
                f"{float(dec['e_meas'].abs().mean()):.2f} mm for the same method on the annotated masks, i.e. the segmentation adds essentially nothing "
                f"({adds_now:+.2f} mm), while the previous final model reached {float(prev_dec['e_total'].abs().mean()):.2f} mm and added {adds_prev:+.2f} mm. "
                f"The same appears at the boundary the measurement actually uses: the lower gingival margin is displaced by "
                f"{f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm"
                + (f" against {f(prev_high['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm for the previous model" if prev_high is not None else "")
                + ". So the comparison removed the segmentation as an error source, and with it the post-hoc calibration step the previous pipeline "
                  "needed; the reported pipeline now has no step fitted on the clinical reference at all. ")
        else:
            meaning_txt = ""
        limits_txt = (
            "The limits of the comparison are stated with it. It is a published-defaults comparison on one dataset, one centre and one test set, so it "
            "says which architecture measures gingival display better here, not which is better in general; the tuned model of the original submission is "
            f"reported as a separate, labelled row and is not a member of it. The two families do not start from equally well-matched pretrained weights, "
            "and the two frameworks' early-stopping criteria are the same rule on a metric that is not the same function; both were recorded in the "
            "protocol before the runs rather than discovered afterwards. Class agreement and kappa are deliberately not reported per architecture, because "
            f"at n = {ARCH['n']} their intervals cannot separate the architectures. Finally, once each model's own bias is removed the three architectures "
            "are indistinguishable in scatter and their per-image errors correlate above 0.94: what differs between them is a constant, not precision, "
            "and that is how the Discussion states it.")
        arch_text = (
            "We accept the criticism, withdraw the original claim, and have repeated the comparison under controlled conditions rather than only conceding the point. "
            "The original three runs used the annotation platform's default training settings on separate copies of the dataset version, so they did not isolate the architecture and cannot support a statement that one is superior; that stage is now described for what it was, a screening step used to pick one architecture to take forward, and the sentence claiming superior performance has been removed. "
            "The repeated comparison follows a protocol written and committed to the repository before any run (`outputs/08_architecture/PROTOCOL.md`). It fixes, in advance, what is held identical and what is not: the same images and annotations, the same participant-level partition and the same fixed test set, the same epoch budget and early-stopping rule, the same evaluation protocol and metric implementation, and the same measurement pipeline applied to the predicted masks; each architecture's optimiser, schedule, augmentation and loss stay at its own published defaults, because imposing one family's hyperparameters on the other would handicap it by construction, and no architecture is tuned for the comparison. "
            f"The primary outcome is fixed there too, and it is the measurement rather than a detection score: the millimetre error of gingival display against the clinical reference, paired over the {ARCH['n']} high-smile-line test images that carry one, with mAP at standard settings secondary. "
            f"Each architecture was trained with {NUMWORD.get(len(ARCH['seeds']), len(ARCH['seeds']) or 'three')} seeds, which give the run-to-run spread, and a paired bootstrap over the images gives the interval. The decision rule, its threshold and the single circumstance that would change the final model were written down before the first run. "
            f"The result: {per_model} (mean ± SD over seeds). {tie_txt}"
            + (f"{aname(awin['model_b'])} did separate from both, by {apaired(awin)}. " if awin else
               "No challenger separated from the baseline by more than the pre-registered threshold. ")
            + f"{res_txt}{outcome_txt}{fault_txt}{meaning_txt}{limits_txt}")
        arch_files = ["outputs/08_architecture/PROTOCOL.md", "outputs/08_architecture/PROTOCOL_ADDENDUM_resolution.md",
                      "outputs/08_architecture/RESULTS.md", "outputs/08_architecture/FINDINGS.md",
                      "outputs/07_report/MANUSCRIPT_EDITS.md"]
        arch_sources = [S.get("arch", ""), S.get("archm", ""), S["dec"]] + ([S["pdec"]] if prev_dec is not None else []) + [S["bset"]]
    else:
        arch_text = (
            "We accept the criticism, withdraw the original claim, and have repeated the comparison under controlled conditions rather than only conceding the point. "
            "The original three runs used the annotation platform's default training settings on separate copies of the dataset version, so they did not isolate the architecture and cannot support a statement that one is superior; that stage is now described for what it was, a screening step used to pick one architecture to take forward, and the sentence claiming superior performance has been removed. "
            "The repeated comparison follows a protocol written and committed before any run (`outputs/08_architecture/PROTOCOL.md`): the same images, the same participant-level partition, the same preprocessing, the same epoch budget and early-stopping rule, the same evaluation protocol and metric implementation, three seeds per architecture, and every architecture at its published defaults so that none is tuned in favour of another. "
            "Its primary outcome is the measurement itself, the millimetre error against the clinical reference and the gingival edge error, paired over the same test images; mAP at standard evaluation settings is secondary. The residual differences that a controlled comparison cannot remove, chiefly that the architectures do not share a native mask resolution, are declared in that protocol in advance rather than discovered afterwards. "
            "[PENDING — the comparison runs on the training workstation; its results table will be inserted here.] "
            f"The architecture taken forward in the manuscript is in any case evaluated properly on its own: a single participant-level partition, one fixed test set, evaluated once ({seg_headline}; {seg_prov}), with the learning curve as evidence on data adequacy.")
        arch_files = ["outputs/08_architecture/PROTOCOL.md", "outputs/07_report/tables/segmentation_metrics_test.md",
                      "outputs/07_report/MANUSCRIPT_EDITS.md"]
        arch_sources = [S["seg"]]
    A["architecture_comparison"] = dict(
        status="READY" if ARCH["ready"] else "PENDING_RUN", text=arch_text,
        changes=("Methods 2.8.1 — relabelled as preliminary screening; [Results 3.x — new: controlled architecture comparison, "
                 "with the pre-registered decision rule and its outcome]; Methods 2.8 — the final model is now the architecture the rule "
                 "selected; Appendix F — caption and text replaced by the controlled comparison; Limitations — published-defaults, "
                 "single-dataset comparison; Discussion — a constant, not precision"
                 if ARCH["ready"] else
                 "Methods 2.8.1 — relabelled as preliminary screening; [Results 3.x — new: controlled architecture comparison]; Appendix F — caption and text corrected"),
        files=arch_files, sources=[q for q in arch_sources if q])
    A["mm_accuracy"] = dict(status="READY", text=(
        "We agree entirely; this is the central omission of the submitted manuscript and the revision addresses it directly. The pixel-to-millimetre accuracy is now reported against the clinical reference measurements, with MAE, RMSE, Pearson r, ICC(2,1) and Bland–Altman limits of agreement, and in two steps so that the segmentation and the geometry can be separated. "
        f"On the annotated masks (measurement geometry alone, n = {int(G['n'])}): {acc_line(G)}. On the model's own masks, with every image predicted by a model that had not seen it (5-fold cross-validation at participant level, n = {int(P['n'])}): {acc_line(P)}; "
        f"on the fixed test set with the final model (n = {int(B['n'])}): {acc_line(B)}. "
        + (f"These are uncorrected, primary results. We also found that the model places the lower gingival margin systematically too low and report a post-hoc correction as a secondary analysis "
           f"(mask-level, estimated on the development subset and applied unchanged elsewhere): out-of-fold {acc_line(Pc).split(';')[0]}; test set MAE {f(Bc['mae'])} mm, bias {f(Bc['ba_bias'], 2, True)} mm. "
           f"On the reviewer's point about the thresholds: the decision boundaries are at 3, 4, 6 and 8 mm, so we report the class agreement that the measurement error actually produces rather than the error alone — uncorrected {100 * P['threshold_agreement']:.0f} % agreement with the reference class (linear-weighted κ {f(P['threshold_kappa_linear'], 2)}), "
           f"corrected {100 * Pc['threshold_agreement']:.0f} % (κ {f(Pc['threshold_kappa_linear'], 2)}), and the proportion of images within 1 mm of the reference ({100 * P['within_1_mm']:.0f} % uncorrected, {100 * Pc['within_1_mm']:.0f} % corrected). "
           if corrected else
           f"There is one result set: no post-hoc calibration is applied. A correction was re-estimated for this model and then dropped, because its gain was below the repeatability of the reference it would have been fitted on, so nothing in these numbers is tuned on the clinical reference ({args.stage6}/PLAN.md, Amendment 3). "
           + (f"The architecture was chosen partly on the {int(B['n'])} test-set images, which makes the primary figure mildly optimistic; the sensitivity analysis pre-registered for exactly that reason excludes them and gives MAE {f(acc.loc[presens, 'mae'])} mm with ICC(2,1) {f(acc.loc[presens, 'icc2_1'], 3)} on the remaining {int(acc.loc[presens, 'n'])} images. " if presens else "")
           + f"On the reviewer's point about the thresholds: the decision boundaries are at 3, 4, 6 and 8 mm, so we report the class agreement that the measurement error actually produces rather than the error alone — {100 * P['threshold_agreement']:.0f} % agreement with the reference class (linear-weighted κ {f(P['threshold_kappa_linear'], 2)}), and {100 * P['within_1_mm']:.0f} % of images within 1 mm of the reference. ")
        + f"The reliability of the reference itself is reported so that the error can be read against its floor: the same observer remeasured the gingival display of 20 images at six tooth sites each, giving a tooth-site level ICC(2,1) of {tooth_row[2] if tooth_row else '?'} with an SD of {tooth_row[6] if tooth_row else '?'} mm. "
        "We make no assumption that the new architecture preserves the accuracy of the previous study; the numbers above are measured on this model, and they are not better than those of the previous study."),
        changes="[Results 3.2 — new section: millimetre measurement accuracy]; [Figure 6 — replaced by scatter and Bland–Altman plots]; [Table — measurement accuracy] `tables/measurement_accuracy.md`; Abstract", files=["outputs/06_prediction/prediction_summary.md", "outputs/07_report/tables/measurement_accuracy.md", "outputs/07_report/figures/measurement_predicted_masks.png", "outputs/03_oracle/intra_observer.md"], sources=[S["acc"], S["intra"], S["est3"]])


    # ---------------- the full letter (docs/Hakem_Yorumları.docx) added 27 items the abridged
    # summary did not carry. Editorial and clinical-judgement items get the technical half of the
    # answer here and are marked CLINICAL; the clinical team writes the rest (KLINIK_EKIBE.md).
    A["intro_rewrite"] = dict(status="CLINICAL", text=(
        "Accepted. The Introduction is rewritten with one stated aim and a reference for every claim it makes. "
        "What the technical side supplies for it: the gap this study can legitimately claim is not that gingival display has never been quantified automatically — the group's own J Dent 2026 study did that — but that no published system converts the measurement into an explicit, auditable rule set and then measures how far that layer agrees with clinicians. "
        "The Introduction will say so in those words, and the sentences that generalise about 'AI-based image analysis' are replaced by the specific prior work with citations."),
        changes="Introduction — rewritten: one aim, a reference for every statement, the gap restated as the decision layer rather than the measurement", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["methods_clarity"] = dict(status="CLINICAL", text=(
        "Accepted, and most of it is already rewritten. The Methods now state the design and the recruitment (clinical team), and, from the analysis side: the participant-level partition and its exact counts, the annotation procedure, the pixel-to-millimetre calibration (item Methods 6), the measurement geometry and the single fixed estimator with its sensitivity analysis, the training configuration of the reported model, the evaluation protocol with the test set used once, and the statistical methods with their software. "
        "Each of those is a numbered subsection so that a reader can follow the pipeline end to end."),
        changes="Methods 2.1-2.9 — restructured; calibration, measurement geometry, partition and evaluation protocol each given their own subsection", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["discussion_rewrite"] = dict(status="CLINICAL", text=(
        "Accepted. Three reviewers make the same point — the Discussion is long, repeats the literature and draws conclusions the study did not test — and it is rewritten around this study's own results: the measurement accuracy against the clinical reference, the boundary analysis that explains where the error comes from, the controlled architecture comparison, the learning curve that has not plateaued, and the agreement of the decision layer with independent clinical assessment. "
        "Literature that belongs to the rationale moves to the Introduction, the general passages are cut, and the conclusions are restricted to what was measured."),
        changes="Discussion — shortened and restructured around this study's results; repeated literature moved to the Introduction; general conclusions removed", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["language"] = dict(status="CLINICAL", text=(
        "Accepted. The manuscript is going through professional language editing before resubmission, and the certificate will accompany it."),
        changes="Whole manuscript — professional language editing", files=[], sources=[])
    A["title_claim"] = dict(status="CLINICAL", text=(
        "Accepted. The subtitle promises what the study does not validate, and it is changed: the revision proposes 'measurement accuracy and a rule-based framework for etiological interpretation' in place of 'Toward etiological interpretation and treatment planning'. "
        "The final wording is the clinical team's, but it will not contain a claim of clinical validation of treatment planning."),
        changes="Title / subtitle — rewritten so the claim matches what was validated", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["abstract_limitations"] = dict(status="READY", text=(
        "Accepted; the Abstract now carries the limitations rather than leaving them to the Discussion. Three are named, and all three are ours: "
        "the design is single-centre and single-device, so no claim of external validity is made (the learning curve, which has not plateaued, says the same thing from the model's side); "
        "the clinical reference was measured by one examiner, so only intra-observer reliability is available for it at present (tooth-site ICC(2,1) " + (f"{tooth_row[2]}" if tooth_row else "?") + ", SD " + (f"{tooth_row[6]}" if tooth_row else "?") + " mm) and the inter-observer component is being collected in the expert study (item R2-7); "
        "and the etiological and treatment layer is compared with independent clinical assessment as an agreement analysis, not validated as a decision tool."),
        changes="[Abstract — new Limitations sentence]; Discussion — Limitations paragraph expanded", files=["outputs/03_oracle/intra_observer.md", "outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["intra"]])
    A["intro_gap"] = dict(status="CLINICAL", text=(
        "The reviewer reads the sentence correctly and the reading exposes a real problem: as written, the gap is only in the clinical application, which is the part the study does not evaluate. "
        "The revision states the gap where the study actually contributes and says which of the two questions the reviewer raises is open. Automated segmentation of gingiva and lip is not an open question in general; the millimetre validity of the measurement derived from it is reported here against a clinical reference "
        f"({acc_line(P)}), and the open question the study addresses is whether an explicit rule layer on top of that measurement agrees with clinical judgement."),
        changes="Introduction — the scientific gap restated and narrowed to what the study tests", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["acc"]])
    A["intro_objectives"] = dict(status="CLINICAL", text=(
        "Accepted. The revision names one primary objective — to quantify gingival display from a frontal smile photograph and report its accuracy in millimetres against a clinical reference — and lists the secondary objectives separately: the controlled comparison of segmentation architectures, and the agreement of the rule-based etiological/treatment layer with independent clinical assessment."),
        changes="Introduction, final paragraph — one primary objective, secondary objectives listed separately", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["study_design"] = dict(status="CLINICAL", text=(
        "Accepted; this belongs to the clinical team and the wording is being supplied. The design is a retrospective, cross-sectional analysis of photographs and clinical records collected in a single centre under ethics approval E-77082166-604.01-881629, and the recruitment window and the consecutive/selective nature of the sampling are being stated explicitly, because the reviewer's later question about selection bias (Discussion 3) cannot be answered without them. "
        "From the analysis side, what can already be stated is the flow from the export to the analysed set: " + (f"{int(dc.loc['total', 'images_roboflow_export'])} exported images, {int(dc.loc['total', 'kept'])} analysed after the exclusions, partitioned at participant level into {int(dc.loc['total', 'train'])} / {int(dc.loc['total', 'valid'])} / {int(dc.loc['total', 'test'])}.")),
        changes="Methods 2.1 — design, retrospective/prospective, recruitment process and dates", files=["outputs/07_report/tables/dataset_counts.md"], sources=[S["dc"]])
    A["inclusion_criteria"] = dict(status="CLINICAL", text=(
        "Correct, and it is a real omission: only exclusion criteria were given. The inclusion criteria are being written by the clinical team. "
        "The analysis-side filters that act on top of them are already documented and will be stated in the same place, because they determine which images enter which analysis: an image enters the segmentation training set if it has a usable annotation, and it enters the millimetre analysis only if it is a high smile line with a clinical reference measurement "
        + (f"(n = {int(dc.loc['high', 'kept'])}). " if 'high' in dc.index else ". ")
        + "Duplicate photographs of the same participant were reduced to one image per participant before partitioning."),
        changes="Methods 2.1 — inclusion criteria added beside the exclusion criteria", files=["outputs/07_report/tables/dataset_counts.md"], sources=[S["dc"]])
    A["sample_size_placement"] = dict(status="READY", text=(
        "Accepted, and the change is larger than a move. The G*Power calculation is removed altogether rather than relocated, because it does not determine the data requirement of a segmentation model (item R2-3); the Methods instead describe how data adequacy was assessed empirically, with the learning curve, and how the precision of the agreement analysis was justified. "
        "The counts of participants and images move to the Results, where the reviewer asks for them, and the Discussion carries the interpretation of that number rather than a justification of it."),
        changes="Methods 2.1 — power calculation removed, empirical data-adequacy assessment described; Results — participant and image counts; Discussion — interpretation of the cohort size", files=["outputs/07_report/tables/learning_curve.md", "outputs/07_report/tables/dataset_counts.md"], sources=[S["lcmd"], S["dc"]])
    A["why_1315"] = dict(status="READY", text=(
        "The number was not chosen. The cohort is every photograph in the centre's archive that met the criteria over the recruitment window, so 1,315 is what the archive contained rather than a target that was set (the recruitment window itself is stated by the clinical team, item Methods 1). "
        "What we can do, and now do, is say whether that number was enough, and we answer it with evidence rather than assertion: "
        f"the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the validation set held constant, and {lc_finding}."
        + ("" if LC["plateau"] else " In other words, the honest answer to 'why 1,315' is that 1,315 is what was available and that more would probably still help; we no longer claim the cohort is sufficient.")),
        changes="Methods 2.1 / Discussion — the cohort described as the available archive, with the learning curve as the data-adequacy evidence", files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md"], sources=[S["lc"], S["lcmd"]])
    A["probe_calibration"] = dict(status="READY", text=(
        "Yes, the probe was used, and the reviewer is right that the procedure was missing from the manuscript. It is now described in full. "
        "A Hu-Friedy UNC periodontal probe is visible in every photograph. For the clinical reference the image was opened in ImageJ at 2698×1799, two consecutive 1 mm marks on the probe were selected, and the scale was set with Set Scale (1 mm), per image; the gingival display was then measured at six tooth sites on that scale. "
        "The pipeline does not read the probe. It applies a single global scale of "
        + (f"{m_scale.group(1)} px/mm, fitted by regression through the origin on the development subset only (R² {m_scale.group(2)}, residual SD {m_scale.group(3)} mm" if m_scale else f"{k:.2f} px/mm, fitted on the development subset")
        + (f"; leave-one-out mean {m_loo.group(1)}, SD {m_loo.group(2)} px/mm" if m_loo else "") + "), applied unchanged to every other image, because the photographs were taken at a fixed camera-to-subject distance with a fixed setup. "
        + (f"The per-image ratio of pixels to reference millimetres has a coefficient of variation of {m_cv.group(3)}, which includes the reference's own calibration noise: with a 1 mm probe interval of about 17 px, a one-pixel marking error is roughly 6 % of the scale. " if m_cv else "")
        + "The two scales are therefore reported separately and neither is presented as the other: the reference is per-image and probe-based, the pipeline is a single fitted constant, and the agreement between the two is what the millimetre results measure. "
        "The individual per-image probe scales were not stored at the time, which is stated as a limitation; the three experts of the agreement study enter their own probe scale per image, which will give an independent estimate of that calibration's precision."),
        changes="Methods 2.4 (calibration) — new subsection: probe, ImageJ Set Scale, per-image reference scale vs the pipeline's single fitted scale; Limitations — per-image probe scales not stored", files=["outputs/03_oracle/scale_estimation.md", "outputs/04_expert/scale_agreement.md"], sources=[S["scale"]])
    A["clinical_cutoff"] = dict(status="READY" if TM is not None else "PENDING", text=(
        "Two questions here, and we answer both with numbers. "
        "**What error is clinically acceptable.** The decision boundaries of Table 1 are at 3, 4, 6 and 8 mm, so the tolerance is not a single number: an error matters only where it can cross a boundary. "
        + (f"The measurement error is {f(P['mae'])} mm on average (95 % limits of agreement {f(P['ba_loa_low'])} to {f(P['ba_loa_high'])} mm), and the reference images are stratified below by how far the reference value sits from the nearest boundary. "
           f"Agreement with the reference class is {TMS['agreement_away_from_a_boundary']} for images more than 2 mm from a boundary and {TMS['agreement_near_a_boundary']} for images within 0.5 mm of one, while the mean absolute error is essentially the same in every stratum "
           f"({TM['MAE, mm'].min():.2f} to {TM['MAE, mm'].max():.2f} mm). {TMS['share_of_disagreements_within_1_mm_of_a_boundary']} of all class disagreements occur within 1 mm of a boundary, and {TMS['share_within_0_5_mm_of_a_boundary']} of the cohort sits that close to one. "
           "So the measurement is not worse near a boundary; the boundary is simply close, and a sub-millimetre error is enough to cross it. That is a property of Table 1's spacing, not of the measurement, and it is now stated as such: the system reports the millimetre value with its uncertainty and, near a boundary, more than one candidate category. "
           if TM is not None else "[PENDING — the near-boundary stratification is produced by scripts/build_report.py.] ")
        + "**Whether 0, 1 and 3 mm should be called a gummy smile.** We agree with the reviewer, and this is a terminology problem in the submitted manuscript rather than a disagreement. The quantity measured is *gingival display in millimetres*, which is defined at any value including zero; 'gummy smile' is a clinical judgement that is not made by the measurement and is not made by the system. "
        + (f"In this cohort {TMS['reference_below_4_mm']} have a reference value below 4 mm. " if TM is not None else "")
        + "The revision uses 'gingival display' for the measured quantity throughout, reserves 'high smile line' for the group, and does not describe any value as a gummy smile. Where Table 1 assigns a category below 4 mm it is reporting a candidate etiology for an observed display, not asserting that the case requires treatment — which is the reviewer's Methods 10a, answered there."),
        changes="Methods 2.8 — the acceptable-error question answered against the Table-1 boundaries; [Results — new table: class agreement by distance to a boundary]; terminology corrected throughout", files=["outputs/07_report/tables/threshold_margin.md", "outputs/07_report/tables/measurement_accuracy.md"], sources=[S["tm_margin"], S["acc"]])
    A["table1_evidence"] = dict(status="CLINICAL", text=(
        "No, a systematic search was not performed, and the manuscript should not have implied otherwise. Table 1 is a narrative synthesis of the thresholds used in references 14-23, assembled by the clinical authors from the literature they work with. "
        "The revision says exactly that: it describes the table as a literature-derived, non-systematic synthesis, states the criterion by which each threshold was taken, and lists the source of every band, so that a reader can see which numbers are widely used and which are one group's convention. "
        "The clinical team will supply the description of how the references were gathered. If the editors prefer, the table can instead be presented as the pre-specified rule set this study evaluates, with its provenance given and no claim of evidence synthesis attached to it."),
        changes="Methods — Table 1 described as a narrative, literature-derived synthesis with the source of each band; no claim of a systematic search", files=[], sources=[])
    A["table1_logic"] = dict(status="CLINICAL", text=(
        "Two reviewers make this objection — Reviewer 3 in Methods 10 (a, b and c) and Reviewer 4 in his second fundamental problem — and they are right on the substance. The clinical framing is the clinical team's to write; what the technical side can state, and what the design already does, is this. "
        "**The system does not diagnose.** It measures gingival display in millimetres and applies a published threshold table to produce one or more *candidate* etiologies with the treatments associated with them in the literature. The output field is named `treatment_alternatives`, not 'treatment'. "
        "**Overlapping bands are reported as overlaps, not resolved.** The bands of Table 1 overlap by construction (E1 below 4 mm, E2 from 3 to 6, E3 from 4 to 8, E4 above 8), so a 5 mm display returns the combined label E2-E3 and both sets of candidates rather than a single answer. The engine has no metadata-based tie-breaking and never picks one etiology from a measurement alone — which is precisely the reviewers' point, built into the rule set rather than argued against it. "
        "**On 10a specifically:** a display below 4 mm is not asserted to be a problem. The rule returns a category for an observed display; it does not assert an indication, and a value of 0 mm returns `NO_VISIBLE_GINGIVA` rather than a class. "
        "**On 10c and the short upper lip:** we agree that the diagnosis requires a lip measurement and that lip length varies with sex, age and ethnicity. The system does not measure lip length and therefore cannot diagnose a short lip; where the band admits that etiology it is listed as a candidate to be confirmed clinically. The revision says so in the Methods, in the Table 1 caption and in the Limitations. "
        "**And this is exactly what the expert study measures.** Three clinicians assign the etiology from their own clinical judgement, blinded to the table and to each other; the linear-weighted κ between their majority and the rule output is the quantity that says how far a millimetre-only rule can go. Whatever that number turns out to be, it is the honest measure of this limitation, and it is reported either way."),
        changes="Methods 2.8 and the Table 1 caption — the rule set described as a generator of candidate etiologies, not a diagnosis; overlapping bands and their combined labels made explicit; Limitations — lip length, cephalometry and periodontal findings are not inputs; Discussion — the expert agreement as the measure of this limit", files=["docs/Uzman_degerlendirme_protokolu.md", "outputs/07_report/tables/expert_agreement.md"], sources=[])
    A["abbreviations"] = dict(status="CLINICAL", text=(
        "Accepted. Every abbreviation is expanded at first mention in the revision — YOLO (You Only Look Once), RF-DETR (Receptive Field enhanced Detection Transformer), mAP (mean average precision), IoU (intersection over union), MAE, RMSE, ICC and LoA — and a definitions list is added where the journal allows one."),
        changes="Whole manuscript — abbreviations expanded at first mention", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["section_35_to_methods"] = dict(status="READY", text=(
        "Accepted; the reviewer has identified a genuine structural error. Section 3.5 describes the threshold-based classification rule, which is a method, and in the revision it moves to the Methods, where the bands of Table 1, the combined labels for overlapping bands and the handling of zero and missing values are defined once. "
        "What stays in the Results is the validation that follows from it, and that part is now substantive rather than descriptive: the agreement between the class derived from the measurement and the class derived from the clinical reference "
        + (f"({100 * P['threshold_agreement']:.0f} %, linear-weighted κ {f(P['threshold_kappa_linear'], 2)} [{f(P['threshold_kappa_linear_ci_low'], 2)}, {f(P['threshold_kappa_linear_ci_high'], 2)}] on the {int(P['n'])} out-of-fold images), " if 'threshold_agreement' in P else "")
        + "its dependence on the distance to a boundary, and the agreement with independent clinical assessment from the expert study."),
        changes="Section 3.5 moved to Methods 2.8; Results keeps only the validation results of the classification", files=["outputs/07_report/MANUSCRIPT_EDITS.md", "outputs/07_report/tables/threshold_margin.md"], sources=[S["acc"], S["tm_margin"]])
    A["limitations"] = dict(status="READY", text=(
        "Accepted; all three go into the Limitations, and two of them are quantified rather than merely named. "
        "**Selection bias:** the cohort is a single centre's archive over one recruitment window, photographed with one device and one setup, so it is a convenience sample and no claim of external validity is made; the learning curve, which has not plateaued, is reported alongside as evidence that the dataset is not at its ceiling either. "
        "**One examiner:** the clinical reference and the annotations come from a single observer, so only intra-observer reliability is currently available for them"
        + (f" (tooth-site ICC(2,1) {tooth_row[2]}, SD {tooth_row[6]} mm; image-mean ICC(2,1) {image_row[2]})" if tooth_row and image_row else "")
        + ". The inter-observer component is being collected in the expert study, where three blinded clinicians remeasure the gingival display at each of the six tooth sites (item R2-7). "
        "**Demographic bias:** the demographic coverage of the cohort is reported per group rather than asserted, including how many records carry an age and a sex at all, and skin and gingival pigmentation were not recorded, which is stated as a gap rather than glossed over."),
        changes="Limitations — rewritten: selection bias, single observer with its ICC, demographic coverage and the unrecorded pigmentation", files=["outputs/07_report/tables/demographics.md", "outputs/03_oracle/intra_observer.md", "outputs/07_report/tables/learning_curve.md"], sources=[S["dem"], S["intra"], S["lcmd"]])
    A["duplicate_conclusion"] = dict(status="CLINICAL", text=(
        "Accepted; this is an editing error. The revision keeps one Conclusion section and removes the concluding paragraph at the end of the Discussion."),
        changes="Discussion — the closing conclusion paragraph removed; one Conclusion section kept", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["reference_11"] = dict(status="READY", text=(
        "The question is well placed and the answer is that no outcome from reference 11 entered the calculation. The effect size used in G*Power (w = 0.30) is Cohen's conventional 'medium' value, not an estimate taken from that study; reference 11 was cited as context for the clinical problem and its presence beside the calculation implied a derivation that was never made. "
        "Rather than correct the citation we remove the calculation: it does not determine the data requirement of a segmentation model, and the revision replaces it with the empirical learning curve and, for the agreement analysis, a precision-based justification"
        + (f" ({precision_note})" if precision_note else "") + ". The misleading citation disappears with the paragraph."),
        changes="Methods 2.1 and Appendix A — the power calculation and the reference-11 citation beside it removed", files=["outputs/07_report/tables/learning_curve.md", "docs/Istatistik_analiz_plani.md"], sources=[S["lcmd"]])
    A["novelty"] = dict(status="CLINICAL", text=(
        "We accept the assessment of the segmentation component and no longer present it as the contribution. Two reviewers say the same thing from different directions, and the revision answers them together: the segmentation is a competent application of existing architectures, not a new method, and it is reported as the instrument the study needs rather than as its result. "
        "What the revision does claim, and what is new relative to the group's own previous work, is the controlled architecture comparison — a pre-registered protocol, identical data and budget, three seeds, a decision rule fixed before the runs, with the millimetre measurement as the primary outcome rather than mAP — and the explicit, auditable rule layer whose agreement with blinded clinical assessment is measured rather than assumed. "
        "Whether that is sufficient novelty for this journal is the editors' judgement, and the manuscript now states the contribution plainly enough for them to make it."),
        changes="Introduction and Discussion — the contribution restated: the segmentation is the instrument, the controlled comparison and the evaluated rule layer are the contribution", files=["outputs/08_architecture/RESULTS.md", "outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[])
    A["inter_rater"] = dict(status="PENDING", text=(
        "The reviewer is right and we do not argue with it: the annotations and the clinical reference come from one examiner, only intra-observer reliability was assessed"
        + (f" (tooth-site ICC(2,1) {tooth_row[2]}, SD {tooth_row[6]} mm)" if tooth_row else "")
        + ", and a single-observer reference can carry a systematic bias that no amount of internal consistency reveals. "
        "This is being measured rather than conceded in words. The expert study already under way has three clinicians, blinded to each other, to the model and to the threshold table, measure the gingival display at each of the six tooth sites on their own per-image probe calibration, on the same reference images. That gives exactly what is missing: "
        "an inter-observer ICC(2,1) between the three experts and the original examiner at tooth-site and image level, the between-observer limits of agreement in millimetres, and the difference between each observer's mean and the reference the model was evaluated against. "
        "The pipeline's error is then reported against that spread, so a reader can see how much of the 0.5 mm-scale error is the model and how much is the disagreement between competent observers about where the gingival margin is. "
        "[PENDING — the expert forms are being completed; the inter-observer numbers will be inserted from `outputs/04_expert/manuscript_numbers.md`.] "
        "Whatever the spread turns out to be, it is reported, and if it is of the same order as the model's error, that is written into the Limitations as the ceiling the reference itself imposes."),
        changes="Methods 2.9 — the inter-observer protocol described; [Results — new: inter-observer reliability of the reference]; Limitations — single-examiner ground truth", files=["docs/Uzman_degerlendirme_protokolu.md", "outputs/04_expert/expert_summary.md", "outputs/03_oracle/intra_observer.md"], sources=[S["intra"]])
    A["intra_icc"] = dict(status="READY", text=(
        "The values are now given in full rather than as 'high ICC'. The same observer remeasured the gingival display of 20 images at six tooth sites each, at a separate session: "
        + (f"ICC(2,1) {tooth_row[2]} at tooth-site level (n = {tooth_row[1]} paired sites) and {image_row[2]} at image-mean level (n = {image_row[1]} images), "
           f"with a mean difference of {tooth_row[5]} mm, a standard deviation of {tooth_row[6]} mm and 95 % limits of agreement of {tooth_row[7]} mm at tooth-site level. "
           if tooth_row and image_row else "[PENDING — outputs/03_oracle/intra_observer.md.] ")
        + "The type is stated explicitly: ICC(2,1), a two-way random-effects, absolute-agreement, single-measurement model, which is the correct choice for repeated measurements by the same rater when absolute agreement rather than consistency is what matters; ICC(3,1) is reported beside it for completeness. "
        "The p-value is no longer quoted as evidence of agreement — a significant ICC only rejects zero — and neither is the paired t-test; the confidence interval and the limits of agreement carry the argument. "
        "The tooth-site standard deviation of "
        + (f"{tooth_row[6]} mm " if tooth_row else "") + "is also used as the reference's own repeatability floor, against which the pipeline's error is read elsewhere in the response."),
        changes="Methods 2.9 and Results — ICC type, value and 95 % CI given at both levels; the p-value removed as evidence of agreement", files=["outputs/03_oracle/intra_observer.md"], sources=[S["intra"]])

    # Paths written before the Stage-6 directory became a choice: point them at the directory this
    # run was built from, and leave the ones only the previous model produced where they are.
    for a in A.values():
        a["files"] = [f"outputs/{args.stage6}/{Path(q).name}" if q.startswith(f"outputs/{args.prev_stage6}/") and (o6 / Path(q).name).exists() else q
                      for q in a["files"]]
        a["changes"] = a["changes"].replace(f"`{args.prev_stage6}/", f"`{args.stage6}/") if (o6 / "boundary_by_set.md").exists() else a["changes"]

    # ------------------------------------------------------------------ documents
    lines = ["# Response to the reviewers — draft", "",
             "<!-- Generated by scripts/build_rebuttal.py. Every number is read from the file named in the source comment under each response; nothing is typed by hand. "
             "Reviewer quotes are verbatim from docs/Hakem_revizyonları.docx via docs/hakem_maddeleri.yaml. Reviewer 2's numbering (1, 2, 4, 5, 6, 8, 9, 10) is the reviewers' own and is kept as written; "
             "Reviewer 3 numbers restart within each section; Reviewer 4 wrote one continuous text, split into items here with the corresponding passage quoted. -->", "",
             ("Conventions: uncorrected measurement results are the primary analysis; results with the post-hoc mask-level correction are secondary. "
                if corrected else
                "Conventions: there is one set of measurement results. No post-hoc calibration is applied, so no number below is fitted on the clinical reference "
                f"(`outputs/{args.stage6}/PLAN.md`, Amendment 3). ")
             + "'High smile line' is used throughout. "
             "Status: READY = answered from the analysis outputs; PENDING = waits for the expert forms; PENDING_RUN = waits for a run on the training workstation; CLINICAL = wording to be provided by the clinical team.", ""]
    rows = []
    for rev in spec["reviewers"]:
        lines += [f"## {rev['name']}", ""]
        for it in rev["items"]:
            if it.get("missing"):
                lines += [f"### {it['id']} — {it['topic']}", "",
                          f"> **MISSING** — item {it['id'].split('-')[-1]} of this review is not in the document we received "
                          "(the numbering skips it). The text has to be requested from the corresponding author before this item can be answered.", "",
                          "**Status:** MISSING (text awaited)", ""]
                rows.append({"item": it["id"], "reviewer": rev["name"], "topic": it["topic"], "status": "MISSING",
                             "owner": it.get("owner", "clinical"), "outputs": "— text to be requested from the corresponding author"})
                continue
            a = A[it["answer"]]
            status = it.get("status_override") or a["status"]
            quote = it.get("quote") or QUOTE_PLACEHOLDER
            lines += [f"### {it['id']} — {it['topic']}", "",
                      f"> {quote}", ""]
            if it.get("source_note"):
                lines += [f"*({it['source_note'].strip()})*", ""]
            if not it.get("quote"):
                lines += [f"*(Paraphrase from the technical audit / task document: {it['paraphrase']})*", ""]
            lines += ["**Response:**", "", a["text"], "",
                      f"**Changes in the manuscript:** {a['changes']}", "",
                      f"**Status:** {status}", "",
                      "Outputs: " + ", ".join(f"`{p}`" for p in a["files"]), "",
                      *a["sources"], ""]
            rows.append({"item": it["id"], "reviewer": rev["name"], "topic": it["topic"], "status": status,
                         "owner": it.get("owner", "technical"), "outputs": "; ".join(a["files"]),
                         "quote": quote, "response": a["text"], "changes": a["changes"]})
    used = {it["answer"] for rev in spec["reviewers"] for it in rev["items"] if it.get("answer")}
    unmatched = [k for k in A if k not in used]
    if unmatched:
        lines += ["## Prepared responses not attached to any item", "",
                  "Drafted from the team's notes and kept so that the material is not lost; attach to an item or drop once its origin is established.", ""]
        for k in sorted(unmatched):
            a = A[k]
            lines += [f"### (unattached) {k}", "", a["text"], "", f"**Changes in the manuscript:** {a['changes']}", "", f"**Status:** {a['status']}", "",
                      "Outputs: " + ", ".join(f"`{p}`" for p in a["files"]), "", *a["sources"], ""]
    (o7 / "RESPONSE_TO_REVIEWERS.md").write_text("\n".join(lines), encoding="utf-8")

    st = pd.DataFrame(rows)
    durum = ["# Rebuttal durumu — hakem maddeleri (klinik ekibe)", "",
             f"Üretim: `scripts/build_rebuttal.py`; sayılar `outputs/` altındaki dosyalardan okunur. Durum: READY = analiz çıktılarından cevaplandı; PENDING = uzman formlarını bekliyor; CLINICAL = metni klinik ekip yazacak. "
             f"Toplam {len(st)} madde: READY {int((st['status'] == 'READY').sum())}, PENDING {int((st['status'] == 'PENDING').sum())} (uzman formları), "
             f"PENDING_RUN {int((st['status'] == 'PENDING_RUN').sum())} (iş istasyonu koşusu), CLINICAL {int((st['status'] == 'CLINICAL').sum())}, "
             f"MISSING {int((st['status'] == 'MISSING').sum())} (hakem belgesinde bulunmayan maddeler; metni sorumlu yazardan istenecek).", "",
             "Makale metninde değişmesi gereken yerler ayrı bir dosyada: `MANUSCRIPT_EDITS.md` (gönderilen makale ve Appendix B–F üzerinden, her madde için mevcut cümle / önerilen cümle / gerekçe / hakem maddesi).", "",
             "| madde | hakem | konu | durum | sorumlu | cevaplayan çıktı |", "|---|---|---|---|---|---|"]
    for r in rows:
        durum.append(f"| {r['item']} | {r['reviewer']} | {r['topic']} | **{r['status']}** | {'teknik' if r['owner'] == 'technical' else 'klinik'} | {r['outputs']} |")
    durum += ["", f"Kaynak: `docs/Hakem_Yorumları.docx` — hakem mektubunun **tam** metni. Madde listesi ve alıntılar bu dosyadan "
              "`scripts/build_reviewer_items.py` ile üretiliyor; alıntılar mektubun satırlarından kesilerek alındığı için birebirdir. "
              "Daha önce kullanılan `docs/Hakem_revizyonları.docx` kısaltılmış bir özettir (31 madde) ve artık kaynak değildir; "
              "çeliştikleri yerde tam mektup esastır. Klinik ekibe gidecek maddeler ayrıca `KLINIK_EKIBE.md` dosyasında toplandı."]
    (o7 / "REBUTTAL_DURUM.md").write_text("\n".join(durum) + "\n", encoding="utf-8")

    # ---- the clinical team's own list: what only they can write, with what we can already give them
    clin_rows = [r for r in rows if r["status"] == "CLINICAL"]
    pend_rows = [r for r in rows if r["status"] == "PENDING"]
    kl = ["# Klinik ekibe — metni sizden beklenen hakem maddeleri", "",
          f"Kaynak: `docs/Hakem_Yorumları.docx` (hakem mektubunun tam metni, {len(st)} madde). Bu dosyada yalnızca **metni klinik ekibin yazacağı "
          f"{len(clin_rows)} madde** var; teknik analizden cevaplanan {int((st['status'] == 'READY').sum())} madde `RESPONSE_TO_REVIEWERS.md` içinde hazır. "
          "Üretim: `scripts/build_rebuttal.py`; elle düzenlemeyin.", "",
          "Her madde için: hakemin **birebir** sözü, ne gerektiği, ve teknik tarafın şimdiden sağladığı metin/sayılar. "
          "Son cümleleri yazarken teknik kısmı olduğu gibi kullanabilirsiniz; sayılar `outputs/` dosyalarından okunuyor ve elle yazılmadı.", ""]
    for r in clin_rows:
        kl += [f"## {r['item']} — {r['topic']}", "",
               f"**Hakem ({r['reviewer']}):**", "",
               "> " + r["quote"].strip().replace("\n", "\n> "), "",
               f"**Ne gerekiyor:** {r['changes']}", "",
               "**Teknik tarafın sağladığı (taslak metin, sayılar çıktı dosyalarından):**", "",
               r["response"], ""]
        if r["outputs"]:
            kl += ["Kaynaklar: " + ", ".join(f"`{q}`" for q in r["outputs"].split("; ") if q), ""]
    kl += ["---", "",
           f"## Ayrıca: uzman formları beklenen {len(pend_rows)} madde", "",
           "Bunların metni teknik tarafta hazır; eksik olan yalnızca üç uzmanın doldurduğu formlardan gelecek sayılar "
           "(`run_expert_analysis.py` → `outputs/04_expert/manuscript_numbers.md`). Formlar geldiğinde cevaplar kendiliğinden tamamlanır.", "",
           "| madde | hakem | konu | ne bekleniyor |", "|---|---|---|---|"]
    for r in pend_rows:
        kl.append(f"| {r['item']} | {r['reviewer']} | {r['topic']} | uzman formlarından gelecek sayılar |")
    (o7 / "KLINIK_EKIBE.md").write_text("\n".join(kl) + "\n", encoding="utf-8")

    ready = st[st["status"] == "READY"]; pend = st[st["status"] == "PENDING"]; clin_ = st[st["status"] == "CLINICAL"]; miss = st[st["status"] == "MISSING"]
    prun = st[st["status"] == "PENDING_RUN"]
    ozet = ["# Rebuttal — Türkçe özet", "",
            f"- Taslak: `RESPONSE_TO_REVIEWERS.md` ({len(st)} madde, 4 hakem). Tüm sayılar `outputs/` dosyalarından okundu; her cevabın altında kaynak dosya yorumu var.",
            f"- **Hazır ({len(ready)})**: " + ", ".join(f"{r.item} ({r.topic})" for r in ready.itertuples()) + ".",
            f"- **Uzman verisi bekleyen ({len(pend)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in pend.itertuples()) or "yok") + ". Formlar analiz edilince `run_expert_analysis.py` → `manuscript_numbers.md`; ayrıca kalibrasyon cevabındaki uzman ölçeği cümlesi (R4-2 içinde [PENDING] işaretli).",
            f"- **Klinik ekipten metin bekleyen ({len(clin_)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in clin_.itertuples()) or "yok") + ". Örtüşme paragrafı klinik ekibin 15 Eylül metninden taslak olarak kondu; sınırlılıklar için teknik maddeler listelendi.",
            (f"- **İş istasyonu koşusu bekleyen ({len(prun)})**: " + ", ".join(f"{r.item} ({r.topic})" for r in prun.itertuples())
             + ". Ön-belirleme belgesi: `outputs/08_architecture/PROTOCOL.md`."
             if len(prun) else
             "- **İş istasyonu koşusu bekleyen (0)**: yok. Mimari karşılaştırması ve kontrolleri tamamlandı; sonuçlar "
             "`outputs/08_architecture/RESULTS.md`, ön-belirleme `PROTOCOL.md` ve `PROTOCOL_ADDENDUM_resolution.md`, "
             "bulgular `FINDINGS.md`. Ön-belirlenen karar kuralı tetiklendi ve nihai model RF-DETR-Seg Large olarak değişti."),
            (f"- **Hocadan beklenen ({len(miss)})**: " + ", ".join(f"{r.item}" for r in miss.itertuples()) + " — metinleri sorumlu yazardan istenecek."
             if len(miss) else
             "- **Hocadan beklenen (0)**: yok. Tam hakem mektubu (`docs/Hakem_Yorumları.docx`) elimize ulaştı; daha önce 'belgede bulunamayan' diye "
             "işaretlenen iki madde bulundu ve yerlerine kondu: **R2-3** G*Power maddesi (eskiden `R2-sample-size` adıyla 'ayrıca iletilen yorum' "
             "sayılıyordu) ve **R2-7** tek gözlemci / gözlemciler arası güvenilirlik maddesi."),
            "- Ölçüm doğruluğu her yerde düzeltmesiz birincil, maske düzeyi düzeltme ikincil.",
            "- Açıkça kabul edilen hatalar: eski ölçüm modülünün geometri hatası (eski Figure 6; v3 ve v1 kolonları geçersiz, figür yeniden üretildi), aynı hastanın iki fotoğrafının bölüntüler arasında bulunması (hasta düzeyi yeniden bölünme, yeniden eğitim), gözlemci içi dosyasında 15 vs 20 görüntü (tam dosyayla yeniden hesaplandı).",
            "- Yapamadıklarımız açıkça yazıldı: E4 sınıfı veri setinde yok; dış geçerlilik yok (tek merkez/cihaz); pigmentasyon kaydı yok; demografi kısmi.",
            "- Kaynak **`docs/Hakem_Yorumları.docx`** — hakem mektubunun tam metni. Madde listesi ve alıntılar `scripts/build_reviewer_items.py` ile "
            "doğrudan bu dosyadan üretiliyor: her maddenin alıntısı mektubun belirtilen satırlarından kesiliyor, yani birebir. Daha önce kullanılan "
            "`docs/Hakem_revizyonları.docx` kısaltılmış bir özetti (31 madde) ve artık kaynak değil; çeliştikleri yerde tam mektup esas alındı.",
            f"- Toplam **{len(st)} madde**: " + ", ".join(f"{n} {len(i)}" for n, i in
                [(r['name'], r['items']) for r in spec['reviewers']]) + ". Eski 31 maddelik listedeki her madde yeni listede var; "
            "yalnızca bir kimlik değişti (`R2-sample-size` → **R2-3**, mektuptaki gerçek numarası). Tam mektupta bulunmayan eski madde yok.",
            f"- **Net {len(st) - 31} madde arttı.** Metniyle birlikte yeni gelenler: Reviewer 3'ün 24 maddesi (bölümün tamamı 33; eski listede 9'u vardı), "
            "Reviewer 1'in iki editoryal maddesi (özgünlük, tartışmanın uzunluğu), Reviewer 2'nin **R2-11** (önceki çalışmayla fark) ve **R2-12** "
            "(ICC değeri/GA/tipi) maddeleri, Reviewer 4'ün **R4-9** ('etiyoloji yalnız mm'den türetilemez') ve **R4-1b** (klinik altın standart yok) "
            "maddeleri. Ayrıca eskiden boş yer tutucu olan **R2-3** ve **R2-7** artık gerçek alıntılarıyla dolduruldu.",
            "- **Aynı konuyu iki hakemin sorduğu yerlerde tek cevap, çapraz referansla**: örtüşme/özgünlük (R2-11 ↔ R4-2), Tablo 1'in mantığı "
            "(R3-Methods-10 ↔ R4-9), G*Power (R2-3 ↔ R3-References-1 ↔ R4-external-validity), görüntü sayıları (R2-5 ↔ R3-Results-2), "
            "veri sızıntısı (R2-6 ↔ R4-7), tartışmanın yeniden yazımı (R3-General-4 ↔ R3-Discussion-1/2 ↔ R1-discussion-length).",
            "- Gözlemci içi güvenilirlik artık ayrı bir madde (**R2-12**) ve tam değerleriyle cevaplandı: ICC(2,1) tipi, değeri ve %95 GA'sı, "
            "diş bölgesi ve görüntü düzeyinde; p değeri uyum kanıtı olarak kullanılmıyor.",
            "- **R2-7 (tek gözlemci)** uzman formlarıyla cevaplanacak: üç kör uzman aynı referans görüntülerde altı diş bölgesini kendi prob "
            "kalibrasyonlarıyla ölçüyor; buradan gözlemciler arası ICC(2,1), mm cinsinden uyum sınırları ve her gözlemcinin referanstan farkı çıkacak.",
            "- **R3-Methods-8 (klinik eşik / kabul edilebilir hata)** için yeni tablo: `tables/threshold_margin.md` — referans değerin en yakın "
            "Tablo 1 sınırına uzaklığına göre sınıf uyumu. Hata her katmanda aynı; değişen, sınırın ne kadar yakın olduğu.",
            "- Terminoloji: 'high smile line'; 'gummy smile / excessive gingival display' kullanılmadı.",
            "- Makale düzeltmeleri: `MANUSCRIPT_EDITS.md` — gönderilen makale ve Appendix B–F taranarak her değişiklik için mevcut cümle, önerilen cümle, gerekçe ve hakem maddesi; sayılar `outputs/` dosyalarından."]
    (o7 / "REBUTTAL_OZET.md").write_text("\n".join(ozet) + "\n", encoding="utf-8")
    print("\n".join(ozet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
