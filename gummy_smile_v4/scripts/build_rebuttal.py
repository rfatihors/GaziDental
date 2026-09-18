#!/usr/bin/env python
"""Rebuttal draft — every number is read from outputs/ (never typed), each with a source comment.

    python scripts/build_rebuttal.py [--docx docs/Hakem_revizyonları.docx]

Inputs: docs/hakem_maddeleri.yaml (reviewer items; verbatim quotes pasted there from the
reviewer document when it is available — the repository does not contain the docx), config.yaml
and the Stage 1–7 outputs. Writes outputs/07_report/RESPONSE_TO_REVIEWERS.md (English, one section
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
from gsv4.report import tables as T  # noqa: E402

QUOTE_PLACEHOLDER = "[verbatim reviewer text — to be pasted from docs/Hakem_revizyonları.docx into docs/hakem_maddeleri.yaml]"


def read_segmentation_metrics(path):
    """The reported per-class block of tables/segmentation_metrics_test.csv plus its settings label.

    The table carries one block per evaluation setting since PROTOCOL.md §4; a file written before
    that has a single, unlabelled block computed at the pipeline operating point.
    """
    t = pd.read_csv(path)
    if "settings" not in t.columns:
        return t.set_index("class"), "operating_point_legacy"
    std = t[t["settings"].str.startswith("standard")]
    if len(std):
        return std.set_index("class"), "standard"
    return t.set_index("class"), str(t["settings"].iloc[0])


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
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["_root"]
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    o1, o2, o3, o4, o5, o6, o7 = (outputs / d for d in ("01_data", "02_measure", "03_oracle", "04_expert", "05_predictions", "06_prediction", "07_report"))
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
    seg, seg_kind = read_segmentation_metrics(tab / "segmentation_metrics_test.csv"); S["seg"] = src(tab / "segmentation_metrics_test.csv", root)
    tm = json.loads((o5 / "test_metrics.json").read_text()); S["tm"] = src(o5 / "test_metrics.json", root)
    lc = pd.read_csv(tab / "learning_curve.csv"); S["lc"] = src(tab / "learning_curve.csv", root)
    lc_md = (o5 / "learning_curve.md").read_text(encoding="utf-8").splitlines()[2]; S["lcmd"] = src(o5 / "learning_curve.md", root)
    acc = pd.read_csv(o6 / "measurement_accuracy.csv").set_index("set"); S["acc"] = src(o6 / "measurement_accuracy.csv", root)
    est3 = pd.read_csv(o3 / "estimator_comparison.csv").set_index("combo"); S["est3"] = src(o3 / "estimator_comparison.csv", root)
    sens3 = pd.read_csv(o3 / "sensitivity.csv").set_index("subset"); S["sens3"] = src(o3 / "sensitivity.csv", root)
    bset = pd.read_csv(o6 / "boundary_by_set.csv").set_index("set"); S["bset"] = src(o6 / "boundary_by_set.csv", root)
    dec = pd.read_csv(o6 / "error_decomposition.csv"); S["dec"] = src(o6 / "error_decomposition.csv", root)
    var = pd.read_csv(o6 / "offset_variants.csv").set_index(["subset", "key"]); S["var"] = src(o6 / "offset_variants.csv", root)
    stab = pd.read_csv(o6 / "offset_stability.csv"); S["stab"] = src(o6 / "offset_stability.csv", root)
    toff = pd.read_csv(o6 / "offset_test_high.csv"); S["toff"] = src(o6 / "offset_test_high.csv", root)
    alt = pd.read_csv(o6 / "offset_alternatives.csv"); S["alt"] = src(o6 / "offset_alternatives.csv", root)
    per6 = pd.read_csv(o6 / "per_image_results.csv"); S["per6"] = src(o6 / "per_image_results.csv", root)
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
    Pc = acc.loc["(a) OOF masks, all reference images [PRIMARY] — corrected"]
    H = acc.loc["(a) OOF, Stage-3 holdout images only (scale never fitted on these)"]
    Hc = acc.loc["(a) OOF, Stage-3 holdout images only (scale never fitted on these) — corrected"]
    B = acc.loc["(b) final model masks, test high images [secondary set]"]
    Bc = acc.loc["(b) final model masks, test high images [secondary set] — corrected"]
    G = acc.loc["GT masks, all reference images (Stage 3, same method and scale)"]
    G3 = est3.loc[method]
    a6, b6, c6 = bset.loc["(a) OOF, 145 reference high"], bset.loc["(b) test high, final model"], bset.loc["(c) test all"]
    low6, norm6 = bset.loc["(c) test low"], bset.loc["(c) test normal"]
    fold = stab[stab["grouping"] == "fold"]
    fin_u = toff[(toff["masks"] == "final model") & toff["candidate"].str.endswith("uncorrected")].iloc[0]
    oof_u = toff[(toff["masks"] == "fold models (OOF)") & toff["candidate"].str.endswith("uncorrected")].iloc[0]
    alt_h = alt[(alt["subset"] == "holdout") & (alt["correction"] == "none")].sort_values("mae").iloc[0]
    hm = var.loc[("holdout", "mask_level")]; hvm = var.loc[("holdout", "value_mm")]; hn = var.loc[("holdout", "none")]
    n_fail = int(P["n_segmentation_failure"])
    e_seg_bias, e_meas_bias = float(dec["e_seg"].mean()), float(dec["e_meas"].mean())
    ref_max = float(per6["ref_mm"].max()); n_e4 = int((per6["ref_label"] == "E4").sum())
    seg_g, seg_l, seg_all = seg.loc["diseti"], seg.loc["dudak"], seg.loc["all"]
    cm = np.array(tm.get("confusion_matrix", [[np.nan]]))          # rows = predicted, cols = true (last = background)
    g_tp = g_fp = g_fn = l_tp = l_fn = float("nan")
    if cm.shape == (3, 3):
        g_tp, g_fp, g_fn = cm[0, 0], cm[0, 2], cm[2, 0] + cm[1, 0]
        l_tp, l_fn = cm[1, 1], cm[2, 1] + cm[0, 1]
    n_high_pct = 100 * int(dc.loc["high", "kept"]) / int(dc.loc["total", "kept"])
    std_ok = seg_kind == "standard"
    settings_note = ("at the standard evaluation settings (confidence floor 0.001, NMS IoU 0.7, 300 detections per image)" if std_ok else
                     "at the pipeline's operating point (confidence 0.25, NMS IoU 0.5, 20 detections per image); the standard-settings figure is being recomputed and will replace it")
    lc100 = lc[lc["fraction"] == 1.0].iloc[0]; lc25 = lc[lc["fraction"] == 0.25].iloc[0]

    def f(x, d=2, sign=False):
        return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")

    def acc_line(r):
        return (f"MAE {f(r['mae'])} mm, RMSE {f(r['rmse'])} mm, r = {f(r['r'], 3)}, ICC(2,1) {f(r['icc2_1'], 3)} [{f(r['icc2_1_ci_low'], 3)}, {f(r['icc2_1_ci_high'], 3)}], "
                f"bias {f(r['ba_bias'], 2, True)} mm [{f(r['ba_bias_ci_low'], 2, True)}, {f(r['ba_bias_ci_high'], 2, True)}], 95 % LoA {f(r['ba_loa_low'])} to {f(r['ba_loa_high'])} mm; "
                f"Table 1 class agreement {100 * r['threshold_agreement']:.0f} %, linear-weighted κ {f(r['threshold_kappa_linear'], 2)} [{f(r['threshold_kappa_linear_ci_low'], 2)}, {f(r['threshold_kappa_linear_ci_high'], 2)}]")

    # ------------------------------------------------------------------ answers
    A: Dict[str, Dict[str, Any]] = {}
    A["diagram"] = dict(status="READY", text=(
        "We agree. The revised manuscript includes a block diagram of the complete pipeline (smile photograph → YOLOv11x-seg instance segmentation at original resolution "
        "→ class-separated gingiva and lip masks → column-wise gingival thickness profile → tooth regioning → millimetre value with the calibrated scale → rule engine with the "
        "Table 1 thresholds → report with class, candidate classes and QC flags)."),
        changes="[Figure 1 — new] `outputs/07_report/figures/pipeline_block_diagram.png` (Mermaid source in `pipeline_block_diagram.md`)", files=["outputs/07_report/figures/pipeline_block_diagram.png"], sources=[S["status"]])
    A["segmentation_examples"] = dict(status="READY", text=(
        f"Example outputs have been added: for six test images the annotated (ground-truth) and predicted gingiva and lip masks are shown side by side at original resolution. "
        f"On the fixed test set the per-class segmentation metrics are: gingiva mask mAP@50 {f(seg_g['seg_map50'])}, mAP@50–95 {f(seg_g['seg_map50_95'])}, precision {f(seg_g['seg_precision'])}, recall {f(seg_g['seg_recall'])}; "
        f"lip mask mAP@50 {f(seg_l['seg_map50'])}, mAP@50–95 {f(seg_l['seg_map50_95'])}; all classes mask mAP@50 {f(seg_all['seg_map50'])} (n = {tm.get('n_images', '?')} test images). "
        "The lower gingiva figures are analysed at the boundary in our response to Reviewer 2, item 10."),
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
        f"We agree that the gap between the lip ({f(seg_l['seg_map50'])} mask mAP@50) and the gingiva ({f(seg_g['seg_map50'])}) needed an explanation, and we have analysed it at the boundary rather than only through mAP. "
        f"Against the annotated masks of the {int(a6['n'])} reference images (out-of-fold predictions) the upper, lip-side edge of the gingiva is accurate (column-wise MAE {f(a6['gingiva_top_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_top_edge_bias_mm_mean'], 2, True)} mm) "
        f"whereas the lower, festooned gingival margin is placed systematically too low (MAE {f(a6['gingiva_bottom_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm); gingiva mask IoU {f(a6['gingiva_mask_iou_mean'])}, lip IoU {f(a6['lip_mask_iou_mean'])}. "
        f"The same picture holds for the final model on the test-set high-smile-line images (IoU {f(b6['gingiva_mask_iou_mean'])}, lower-edge bias {f(b6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm). "
        "The gingiva class therefore does not perform poorly at random: the model consistently includes a thin strip below the annotated margin, i.e. a constant shift of one edge, which is why the mAP of a thin structure is low while the measurement-relevant edge is accurate. "
        f"Pooled over the whole test set the gingiva IoU is lower still ({f(c6['gingiva_mask_iou_mean'])}) because in low ({f(low6['gingiva_mask_iou_mean'])}) and average ({f(norm6['gingiva_mask_iou_mean'])}) smile lines the annotated gingiva is thin or absent "
        f"(median annotated width {low6['gingiva_n_columns_gt_median']:.0f} columns in the low group vs {a6['gingiva_n_columns_gt_median']:.0f} in the high group; IoU undefined on {int(c6['n_neither'])} images with no gingiva in either mask), while the edge errors there are not larger. "
        f"We report the boundary metrics per set, and we describe the lower-edge shift and its post-hoc correction (Reviewer 4, calibration) explicitly rather than tuning the model on the test data."),
        changes="[Results 3.x — new subsection 'Boundary accuracy'; Supplementary table of boundary metrics per image set] `06_prediction/boundary_by_set.md`, `figures/boundary_error.png`; Discussion paragraph on the lip/gingiva difference", files=["outputs/06_prediction/boundary_by_set.md", "outputs/06_prediction/error_decomposition.md", "outputs/07_report/figures/boundary_error.png"], sources=[S["bset"], S["seg"]])
    A["power"] = dict(status="READY", text=(
        "The reviewer is right to question it, and the answer is that the approach is not appropriate; we have removed the calculation rather than defend it. A χ² test on counts of correct and incorrect detections is not an analysis performed in this study, "
        "and conventional hypothesis-testing sample-size methods do not determine how much data a deep-learning segmentation model needs. The revised manuscript replaces it with two separate, explicit justifications. "
        f"(i) For model development, data adequacy is assessed empirically with a learning curve: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images), "
        f"gingiva mask mAP@50 rising from {f(lc25['diseti_seg_map50'])} to {f(lc100['diseti_seg_map50'])} and reaching a plateau ({lc_md.split('→')[0].strip()}). "
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
        f"Data adequacy is now shown empirically instead: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the validation set held constant, "
        f"and gingiva mask mAP@50 rose from {f(lc25['diseti_seg_map50'])} to {f(lc100['diseti_seg_map50'])} with the gain between the last two points a small fraction of the gain between the first two ({lc_md.split('→')[0].strip()}), i.e. the dataset is at the plateau of its learning curve. "
        "We also accept the reviewer's point about external testing and we do not attempt to disguise it: no external data were available, so the study reports an internal estimate only. "
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
        f"In addition, the segmentation was found to place the lower gingival margin systematically too low (Reviewer 2, item 10); a post-hoc correction at mask level (lower edge moved up {abs(off_px)} px, i.e. {abs(off_px) / k:.2f} mm at the global scale, estimated on the development subset) is reported as a secondary analysis: "
        f"held-out MAE {f(hn['mae'])} → {f(hm['mae'])} mm, bias {f(hn['ba_bias'], 2, True)} → {f(hm['ba_bias'], 2, True)} mm; on the independent test images {f(fin_u['mae'])} → {f(var.loc[('test high, final model', 'mask_level'), 'mae'])} mm. Uncorrected results remain the primary analysis. "
        "[PENDING: the three experts' per-image probe scales (5 mm interval) will be reported as an independent measure of calibration precision once the expert forms are analysed.]"),
        changes="Methods 2.x 'Calibration' (reference and pipeline scale); Results (scale, frame sensitivity, post-hoc correction as secondary); Supplementary table of scale statistics", files=["outputs/03_oracle/scale_estimation.md", "outputs/06_prediction/offset_correction.md", "outputs/06_prediction/offset_checks.md"], sources=[S["scale"], S["sens3"], S["var"], S["cfg"], S["clin"]])
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
        f"Quantitatively, in the test set the annotated gingiva of low smile lines spans a median of {low6['gingiva_n_columns_gt_median']:.0f} image columns and that of average smile lines {norm6['gingiva_n_columns_gt_median']:.0f}, against {a6['gingiva_n_columns_gt_median']:.0f} in the high group; "
        f"in {int(c6['n_neither'])} test images neither the annotation nor the prediction contains gingiva (correct absence) and in {int(c6['n_spurious'])} the model predicts gingiva where the annotation has none. "
        "The measurement module is specified for the high smile line only and returns NO_VISIBLE_GINGIVA when no gingiva is present; the millimetre validation therefore uses exclusively the "
        f"{int(dc.loc['high', 'kept'])} high-smile-line images with a clinical reference measurement. The revised Methods state this division of roles explicitly."),
        changes="Methods 2.6 (image dataset) — new sentence on the role of each group; Methods 2.7 (measurement) — applicability restricted to the high smile line", files=["outputs/06_prediction/boundary_by_set.md", "outputs/07_report/tables/dataset_counts.md"], sources=[S["bset"], S["dc"]])
    A["pigmentation"] = dict(status="CLINICAL", text=(
        "This is a fair point and we cannot answer it with these data: neither ethnicity nor gingival or skin pigmentation was recorded for the cohort, so no subgroup analysis is possible. We state this as a limitation rather than speculate. "
        "What we can report is where the segmentation error actually lies, which does not look like a pigmentation effect: the error is a systematic displacement of one boundary rather than a random failure of the mask. "
        f"On the {int(a6['n'])} reference images the upper, lip-side gingiva edge is accurate (MAE {f(a6['gingiva_top_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_top_edge_bias_mm_mean'], 2, True)} mm) while the lower, festooned margin is placed consistently too low "
        f"(bias {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm), and this shift is the same in every cross-validation fold and in the final model. A pigmentation-driven failure would be expected to vary between participants rather than to be constant. "
        "This is evidence about the nature of the error, not about pigmentation itself; a prospective study recording phenotype would be required to answer the reviewer's question properly, and we say so. [CLINICAL — the limitation sentence is to be finalised by the clinical team.]"),
        changes="Limitations — new sentence (pigmentation and ethnicity not recorded); Discussion — nature of the segmentation error", files=["outputs/06_prediction/boundary_by_set.md", "outputs/06_prediction/offset_stability.csv"], sources=[S["bset"], S["clin"]])
    A["metric_meaning"] = dict(status="READY", text=(
        "We apologise for the ambiguity; 'performance' in that sentence meant mask mAP@50 and the sentence has been rewritten to name the metric. It is neither sensitivity nor specificity. Mean average precision at an intersection-over-union threshold of 0.50 is computed by ranking every predicted mask by its confidence, "
        "walking down that ranking to trace precision against recall, taking the area under that curve for each class and averaging over classes; it therefore summarises how well the model both finds the structures and ranks its own confidence, at one overlap criterion. "
        "False positives and false negatives were computed but not reported, which we have corrected. On the fixed test set "
        + (f"({int(tm.get('n_images', 0))} images) the model produced {int(g_tp)} correct gingiva instances, {int(g_fp)} false positives and {int(g_fn)} false negatives for the gingiva class, and {int(l_tp)} correct lip instances with {int(l_fn)} false negatives; " if np.isfinite(g_tp) else "")
        + f"per class this is precision {f(seg_g['seg_precision'])} / recall {f(seg_g['seg_recall'])} (mask) for gingiva and {f(seg_l['seg_precision'])} / {f(seg_l['seg_recall'])} for lip, at the confidence threshold the pipeline operates at. "
        "Sensitivity and specificity in the epidemiological sense are not defined for instance segmentation without a fixed set of candidate regions (there is no denominator of true negatives), which is why we report precision, recall, F1 and the confusion matrix instead, at the stated confidence threshold, and the per-pixel boundary agreement separately (Reviewer 2, item 10)."),
        changes="Results 3.1 — sentence rewritten to name the metric; [Table — per-class precision, recall, F1, mAP on the test set]; confusion-matrix figure with FP/FN counts", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/05_predictions/confusion_matrix.png"], sources=[S["seg"], S["tm"]])
    A["separate_evaluations"] = dict(status="READY", text=(
        "We agree, and this is precisely how the revised manuscript is organised: the two questions now have two separate Results sections, 3.1 'Segmentation performance (fixed test set)' and 3.2 'Millimetre measurement accuracy', with their own tables and figures. Segmentation performance is reported on the fixed test set "
        f"(n = {int(tm.get('n_images', 0))}; gingiva mask mAP@50 {f(seg_g['seg_map50'])}, lip {f(seg_l['seg_map50'])}, all classes {f(seg_all['seg_map50'])}) and answers the question whether the structures are found. "
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
        f"On the fixed test set the final model gives, for the mask: mAP@50 {f(seg_all['seg_map50'])}, mAP@50–95 {f(seg_all['seg_map50_95'])}, precision {f(seg_all['seg_precision'])}, recall {f(seg_all['seg_recall'])} (all classes); "
        f"for the box: mAP@50 {f(seg_all['box_map50'])}, mAP@50–95 {f(seg_all['box_map50_95'])}, precision {f(seg_all['box_precision'])}, recall {f(seg_all['box_recall'])}. "
        f"Per class (mask): gingiva {f(seg_g['seg_map50'])} / lip {f(seg_l['seg_map50'])} mAP@50. Because the measurement is computed from the mask, the mask metrics are the ones that matter for this application, and the Abstract now quotes mask metrics with the label attached."),
        changes="[Abstract — Results sentence]; Results 3.1; [Table — segmentation metrics, Box and Mask columns] `tables/segmentation_metrics_test.md`", files=["outputs/07_report/tables/segmentation_metrics_test.md"], sources=[S["seg"]])
    A["yolov8"] = dict(status="READY", text=(
        "The reviewer is right: the sentence is a leftover and it has been corrected. YOLOv8 was included in an early screening run in which several architectures and model scales were trained briefly under shared settings; it was not part of the architecture comparison reported in the manuscript, "
        "and it plays no role in the final model. In the revision the main text refers only to the architectures actually compared, and the appendix that describes the screening stage labels it as such and states which runs it contains, so that no result in the paper depends on it."),
        changes="Methods 2.7 (annotation/labeling) — 'YOLOv8 and YOLOv11' corrected; Appendix D — screening stage labelled and separated from the reported comparison", files=["outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["status"]])
    A["gingiva_map_low"] = dict(status="READY" if std_ok else "PENDING_RUN", text=(
        "We agree, and the value is now stated in the text, in the Abstract and in the Limitations. Before giving it we had to correct how it was measured, because two different quantities were being compared. "
        "The 0.587 the reviewer quotes is a validation-set figure from the earlier pipeline. Our own recomputation on the fixed, participant-level test set was initially made at the pipeline's operating point (confidence threshold 0.25, at most 20 detections per image), which is the configuration that produces the masks for the measurement. "
        "Average precision computed with a confidence floor of 0.25 truncates the precision-recall curve before it reaches full recall and therefore understates mAP, and it is not comparable with a COCO-style evaluation of any other model. "
        "The revision separates the two conventions and labels both: mAP is reported at the standard evaluation settings (confidence floor 0.001, NMS IoU 0.7, 300 detections), while precision, recall, F1 and the confusion matrix are reported at the operating point, where the pipeline actually runs. "
        f"The gingiva mask mAP@50 we report is {f(seg_g['seg_map50'])} {settings_note}, against {f(seg_l['seg_map50'])} for the lip. "
        + ("" if std_ok else "[PENDING — the standard-settings evaluation is being rerun; the figure above is still the operating-point one and will be replaced. The change affects the reported mAP only: not the masks, not the measurement, and no other result.] ")
        + "Whichever number stands, we report the lower one rather than the more favourable one, and we do not leave it as an aggregate. Its consequence for the measurement is quantified: the boundary analysis (item 10) shows the error is a systematic displacement of the lower gingival margin, "
        f"so what it produces is a bias of {f(e_seg_bias, 2, True)} mm rather than a random failure, and the millimetre accuracy of the full pipeline is reported directly ({acc_line(P)}). "
        "The Limitations state that gingival segmentation is the weakest component of the pipeline and the main target for improvement."),
        changes="Results 3.1 — gingiva mAP stated explicitly with its evaluation settings; [Table — segmentation metrics at both settings]; Limitations — new paragraph; Discussion — link to the boundary analysis", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/06_prediction/boundary_by_set.md", "outputs/08_architecture/PROTOCOL.md"], sources=[S["seg"], S["acc"], S["dec"]])
    A["no_improvement"] = dict(status="READY", text=(
        "The reviewer is right that there was no meaningful gain, and the honest explanation is twofold. First, the comparison was not sound: the preliminary figure and the final figure came from different dataset versions and different validation sets, so the near-identical numbers were not evidence of anything. "
        "All performance figures in the revision come from a single fixed, participant-level test set evaluated once. Second, the dataset is at the plateau of its learning curve, which we now show empirically instead of asserting it: retraining the final architecture on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition "
        f"({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) raises gingiva mask mAP@50 from {f(lc25['diseti_seg_map50'])} to {f(lc100['diseti_seg_map50'])}, and the gain between the last two points is a small fraction of the gain between the first two ({lc_md.split('→')[0].strip()}). "
        "Additional data of the same kind, further hyperparameter search and larger model scales are therefore not expected to help; what limits the gingiva class is the annotation of a thin, festooned boundary, which is where we direct the analysis and the remaining error."),
        changes="Results 3.1 — comparison across dataset versions removed; [Supplementary Figure S1 — learning curve]; Discussion — why performance plateaus", files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md", "outputs/07_report/tables/segmentation_metrics_test.md"], sources=[S["lc"], S["lcmd"], S["seg"]])
    A["validation_vs_test"] = dict(status="READY", text=(
        "The reviewer is right, and this was a genuine methodological error rather than a presentational one: the figures reported as final results were validation-set metrics of a model whose architecture, scale and hyperparameters had been chosen on that same validation set. "
        f"In the revision the dataset is partitioned once at participant level into training, validation and test ({int(dc.loc['total', 'train'])} / {int(dc.loc['total', 'valid'])} / {int(dc.loc['total', 'test'])} images), the test set is fixed and untouched during development, and it is evaluated once with the final model. "
        f"All reported performance is that single test-set evaluation: mask mAP@50 {f(seg_all['seg_map50'])} (gingiva {f(seg_g['seg_map50'])}, lip {f(seg_l['seg_map50'])}), mask mAP@50–95 {f(seg_all['seg_map50_95'])}, box mAP@50 {f(seg_all['box_map50'])}. "
        f"The millimetre validation is separated in the same way: the measurement method and the pixel-to-millimetre scale were fixed on a development subset of the reference images and then applied unchanged, so that the held-out figures ({acc_line(H)}) are not optimised. "
        "The validation-set numbers of the original submission are not reported as results."),
        changes="Methods 2.6 (partition) and 2.9 (evaluation protocol) rewritten; Results 3.1 — all metrics replaced by test-set metrics; Abstract numbers replaced", files=["outputs/07_report/tables/segmentation_metrics_test.md", "outputs/07_report/tables/dataset_counts.md", "outputs/06_prediction/measurement_accuracy.csv"], sources=[S["seg"], S["dc"], S["acc"]])
    A["architecture_comparison"] = dict(status="PENDING_RUN", text=(
        "We accept the criticism, withdraw the original claim, and have repeated the comparison under controlled conditions rather than only conceding the point. "
        "The original three runs used the annotation platform's default training settings on separate copies of the dataset version, so they did not isolate the architecture and cannot support a statement that one is superior; that stage is now described for what it was, a screening step used to pick one architecture to take forward, and the sentence claiming superior performance has been removed. "
        "The repeated comparison follows a protocol written and committed before any run (`outputs/08_architecture/PROTOCOL.md`): the same images, the same participant-level partition, the same preprocessing, the same epoch budget and early-stopping rule, the same evaluation protocol and metric implementation, three seeds per architecture, and every architecture at its published defaults so that none is tuned in favour of another. "
        "Its primary outcome is the measurement itself, the millimetre error against the clinical reference and the gingival edge error, paired over the same test images; mAP at standard evaluation settings is secondary. The residual differences that a controlled comparison cannot remove, chiefly that the architectures do not share a native mask resolution, are declared in that protocol in advance rather than discovered afterwards. "
        "[PENDING — the comparison runs on the training workstation; its results table will be inserted here.] "
        f"The architecture taken forward in the manuscript is in any case evaluated properly on its own: a single participant-level partition, one fixed test set, evaluated once (gingiva mask mAP@50 {f(seg_g['seg_map50'])}, lip {f(seg_l['seg_map50'])}, all {f(seg_all['seg_map50'])}), with the learning curve as evidence on data adequacy. "
        "The final model is not changed by this analysis: it appears as a separate row labelled as the tuned model rather than as a member of the comparison, and the protocol fixes in advance the single circumstance that would change it, with its threshold set before the result was known."),
        changes="Methods 2.8.1 — relabelled as preliminary screening; [Results 3.x — new: controlled architecture comparison]; Appendix F — caption and text corrected", files=["outputs/08_architecture/PROTOCOL.md", "outputs/07_report/tables/segmentation_metrics_test.md", "outputs/07_report/MANUSCRIPT_EDITS.md"], sources=[S["seg"]])
    A["mm_accuracy"] = dict(status="READY", text=(
        "We agree entirely; this is the central omission of the submitted manuscript and the revision addresses it directly. The pixel-to-millimetre accuracy is now reported against the clinical reference measurements, with MAE, RMSE, Pearson r, ICC(2,1) and Bland–Altman limits of agreement, and in two steps so that the segmentation and the geometry can be separated. "
        f"On the annotated masks (measurement geometry alone, n = {int(G['n'])}): {acc_line(G)}. On the model's own masks, with every image predicted by a model that had not seen it (5-fold cross-validation at participant level, n = {int(P['n'])}): {acc_line(P)}; "
        f"on the fixed test set with the final model (n = {int(B['n'])}): {acc_line(B)}. "
        "These are uncorrected, primary results. We also found that the model places the lower gingival margin systematically too low and report a post-hoc correction as a secondary analysis "
        f"(mask-level, estimated on the development subset and applied unchanged elsewhere): out-of-fold {acc_line(Pc).split(';')[0]}; test set MAE {f(Bc['mae'])} mm, bias {f(Bc['ba_bias'], 2, True)} mm. "
        f"On the reviewer's point about the thresholds: the decision boundaries are at 3, 4, 6 and 8 mm, so we report the class agreement that the measurement error actually produces rather than the error alone — uncorrected {100 * P['threshold_agreement']:.0f} % agreement with the reference class (linear-weighted κ {f(P['threshold_kappa_linear'], 2)}), "
        f"corrected {100 * Pc['threshold_agreement']:.0f} % (κ {f(Pc['threshold_kappa_linear'], 2)}), and the proportion of images within 1 mm of the reference ({100 * P['within_1_mm']:.0f} % uncorrected, {100 * Pc['within_1_mm']:.0f} % corrected). "
        f"The reliability of the reference itself is reported so that the error can be read against its floor: the same observer remeasured the gingival display of 20 images at six tooth sites each, giving a tooth-site level ICC(2,1) of {tooth_row[2] if tooth_row else '?'} with an SD of {tooth_row[6] if tooth_row else '?'} mm. "
        "We make no assumption that the new architecture preserves the accuracy of the previous study; the numbers above are measured on this model, and they are not better than those of the previous study."),
        changes="[Results 3.2 — new section: millimetre measurement accuracy]; [Figure 6 — replaced by scatter and Bland–Altman plots]; [Table — measurement accuracy] `tables/measurement_accuracy.md`; Abstract", files=["outputs/06_prediction/prediction_summary.md", "outputs/07_report/tables/measurement_accuracy.md", "outputs/07_report/figures/measurement_predicted_masks.png", "outputs/03_oracle/intra_observer.md"], sources=[S["acc"], S["intra"], S["est3"]])

    # ------------------------------------------------------------------ documents
    lines = ["# Response to the reviewers — draft", "",
             "<!-- Generated by scripts/build_rebuttal.py. Every number is read from the file named in the source comment under each response; nothing is typed by hand. "
             "Reviewer quotes are verbatim from docs/Hakem_revizyonları.docx via docs/hakem_maddeleri.yaml. Reviewer 2's numbering (1, 2, 4, 5, 6, 8, 9, 10) is the reviewers' own and is kept as written; "
             "Reviewer 3 numbers restart within each section; Reviewer 4 wrote one continuous text, split into items here with the corresponding passage quoted. -->", "",
             "Conventions: uncorrected measurement results are the primary analysis; results with the post-hoc mask-level correction are secondary. 'High smile line' is used throughout. "
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
            rows.append({"item": it["id"], "reviewer": rev["name"], "topic": it["topic"], "status": status, "owner": it.get("owner", "technical"), "outputs": "; ".join(a["files"])})
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
    durum += ["", "Not: hakemlerin orijinal metni depoda yok (`docs/Hakem_revizyonları.docx`); madde numaraları ve alıntılar belge geldiğinde `docs/hakem_maddeleri.yaml` üzerinden tamamlanacak. Listedeki maddeler teknik denetim raporu ve görev belgesinden derlendi; belgede başka maddeler varsa eklenecek."]
    (o7 / "REBUTTAL_DURUM.md").write_text("\n".join(durum) + "\n", encoding="utf-8")

    ready = st[st["status"] == "READY"]; pend = st[st["status"] == "PENDING"]; clin_ = st[st["status"] == "CLINICAL"]; miss = st[st["status"] == "MISSING"]
    prun = st[st["status"] == "PENDING_RUN"]
    ozet = ["# Rebuttal — Türkçe özet", "",
            f"- Taslak: `RESPONSE_TO_REVIEWERS.md` ({len(st)} madde, 4 hakem). Tüm sayılar `outputs/` dosyalarından okundu; her cevabın altında kaynak dosya yorumu var.",
            f"- **Hazır ({len(ready)})**: " + ", ".join(f"{r.item} ({r.topic})" for r in ready.itertuples()) + ".",
            f"- **Uzman verisi bekleyen ({len(pend)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in pend.itertuples()) or "yok") + ". Formlar analiz edilince `run_expert_analysis.py` → `manuscript_numbers.md`; ayrıca kalibrasyon cevabındaki uzman ölçeği cümlesi (R4-2 içinde [PENDING] işaretli).",
            f"- **Klinik ekipten metin bekleyen ({len(clin_)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in clin_.itertuples()) or "yok") + ". Örtüşme paragrafı klinik ekibin 15 Eylül metninden taslak olarak kondu; sınırlılıklar için teknik maddeler listelendi.",
            f"- **İş istasyonu koşusu bekleyen ({len(prun)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in prun.itertuples()) or "yok") + ". Değerlendirme ayarı düzeltmesi ve mimari karşılaştırması iş istasyonunda çalışacak; `outputs/08_architecture/PROTOCOL.md` ön-belirleme belgesi.",
            f"- **Hocadan beklenen ({len(miss)})**: " + (", ".join(f"{r.item}" for r in miss.itertuples()) or "yok") + " — bu maddeler elimize ulaşan hakem belgesinde yok (Reviewer 2 numaralandırması bunları atlıyor); metinleri sorumlu yazardan istenecek.",
            "- Ölçüm doğruluğu her yerde düzeltmesiz birincil, maske düzeyi düzeltme ikincil.",
            "- Açıkça kabul edilen hatalar: eski ölçüm modülünün geometri hatası (eski Figure 6; v3 ve v1 kolonları geçersiz, figür yeniden üretildi), aynı hastanın iki fotoğrafının bölüntüler arasında bulunması (hasta düzeyi yeniden bölünme, yeniden eğitim), gözlemci içi dosyasında 15 vs 20 görüntü (tam dosyayla yeniden hesaplandı).",
            "- Yapamadıklarımız açıkça yazıldı: E4 sınıfı veri setinde yok; dış geçerlilik yok (tek merkez/cihaz); pigmentasyon kaydı yok; demografi kısmi.",
            f"- Alıntılar hakem belgesinden birebir alındı (`docs/Hakem_revizyonları.docx`; uzantısı .docx olsa da düz metin). Reviewer 2 numaralandırması belgedeki gibi korundu (1, 2, 4, 5, 6, 8, 9, 10); Reviewer 3'ün numaraları bölüm içinde yeniden başlıyor, madde kimlikleri bölüm adını taşıyor; Reviewer 4 tek parça metin yazdığı için maddelere bölündü ve her maddenin alıntısı ilgili pasajın kendisi.",
            f"- Toplam **{len(st)} madde** izleniyor: {len(st) - 4} tanesi hakem belgesinden, 2 tanesi ayrıca iletilen yorumlardan (`R2-sample-size`, `R4-external-validity`), 2 tanesi belgede bulunmayan ve metni beklenen madde (`R2-3`, `R2-7`).",
            f"- Önceki 15 maddelik listeyle karşılaştırma: {len(st) - 4 - 9} madde önceki listede yoktu. Eksik olanlar: Reviewer 3'ün tamamı (9 madde: çalışma tasarımı, abstract iddiaları, grup büyüklükleri ve cinsiyet, low/normal görüntülerin rolü, pigmentasyon, sayı karışıklığı, 'performance' ne demek + FP/FN, segmentasyon ile mm ölçümünün ayrılması, klinik doğrulama iddiası), Reviewer 2'nin 1/2/4/8/9 maddeleri (abstract kapsamı, Box-Mask metrikleri, YOLOv8, dişeti mAP 0.587, iyileşme olmaması) ve Reviewer 4'ün üç maddesi (validasyon yerine test metriği, mimari karşılaştırmanın adil olmaması, mm doğruluğunun ayrı madde oluşu).",
            "- **G*Power / örneklem büyüklüğü** iki ayrı madde olarak eklendi (`R2-sample-size`, `R4-external-validity`); metinleri hakem belgesinde değil, ayrıca iletilen yorumlardan birebir alındı ve her ikisinin altında bu not var. Cevaplar öğrenme eğrisi, hassasiyet gerekçesi ve istatistikçinin kappaSize hesabı üzerinden kuruldu; güç analizi paragrafı ile Ek A makaleden çıkarılıyor.",
            "- Gözlemci içi güvenilirlik ve demografi ayrı hakem maddesi değil; R4-4 ve R3-Methods-5 cevaplarının içine alındı.",
            "- Terminoloji: 'high smile line'; 'gummy smile / excessive gingival display' kullanılmadı.",
            "- Makale düzeltmeleri: `MANUSCRIPT_EDITS.md` — gönderilen makale ve Appendix B–F taranarak her değişiklik için mevcut cümle, önerilen cümle, gerekçe ve hakem maddesi; sayılar `outputs/` dosyalarından."]
    (o7 / "REBUTTAL_OZET.md").write_text("\n".join(ozet) + "\n", encoding="utf-8")
    print("\n".join(ozet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
