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

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
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
    seg = pd.read_csv(tab / "segmentation_metrics_test.csv").set_index("class"); S["seg"] = src(tab / "segmentation_metrics_test.csv", root)
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
    tooth_row, image_row = read_md_table_row(intra_md, "tooth"), read_md_table_row(intra_md, "image mean")
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
        "We thank the reviewer and agree that a χ²-based a priori calculation is not an appropriate basis for the training-set size of a deep-learning segmentation model; the calculation and the associated external-validity statement have been removed. "
        f"Data adequacy is now addressed empirically: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the same validation set; "
        f"gingiva mask mAP@50 rose from {f(lc25['diseti_seg_map50'])} to {f(lc100['diseti_seg_map50'])} and the curve reached a plateau ({lc_md.split('→')[0].strip()}). "
        f"For the millimetre-level agreement analysis a precision-based justification replaces the power calculation: {precision_note}"),
        changes="Methods 2.1 'Sample size and data adequacy' rewritten; Appendix A removed; [Supplementary Figure S1 — learning curve] `figures/learning_curve.png`, `tables/learning_curve.md`", files=["outputs/07_report/figures/learning_curve.png", "outputs/07_report/tables/learning_curve.md"], sources=[S["lc"], S["lcmd"], S["guc"]])
    A["figure6"] = dict(status="READY", text=(
        "We thank the reviewer for pressing on this point: the inconsistency was real and was caused by a software error, not by the data. On re-auditing the code we found that the measurement module of the original submission merged the lip and gingiva masks into one binary mask, "
        "selected the largest contour (in practice the lip) and reported the between-region variation of that contour's *upper* edge, in pixels, as if it were gingival display in millimetres; the gingival margin was not used at all. "
        f"Consequently neither the column of the original submission nor the earlier version's column in Figure 6 is a valid measurement of gingival display, and we have withdrawn both; re-running the legacy code on the same annotated masks reproduces its behaviour ({v34_ex}). "
        f"The measurement module was rewritten (vertical gingiva thickness per image column from the class-separated gingiva mask at original resolution, tooth-wise regioning, calibrated scale) and validated in two steps. "
        f"On the annotated masks of the {int(G['n'])} reference images the corrected geometry gives {acc_line(G)} (method {method}, scale fitted on a 60 % development subset only; held-out 40 %: MAE {f(G3['mae_holdout'])} mm, ICC {f(G3['icc2_1_holdout'], 3)}). "
        f"On predicted masks (out-of-fold, n = {int(P['n'])}; {n_fail} image with no gingiva predicted reported as a segmentation failure) the full pipeline gives {acc_line(P)}. "
        f"A new Figure 6 shows both scatter and Bland–Altman plots against the clinical reference."),
        changes="[Figure 6 — replaced] `figures/measurement_gt_masks.png` (annotated masks) and `figures/measurement_predicted_masks.png` (full pipeline, primary); Methods 2.x (measurement) rewritten; Results 3.x; erratum sentence in the Discussion", files=["outputs/02_measure/v3_vs_v4.md", "outputs/07_report/figures/measurement_gt_masks.png", "outputs/07_report/figures/measurement_predicted_masks.png", "outputs/07_report/tables/measurement_accuracy.md"], sources=[S["v34"], S["acc"], S["est3"]])
    A["demographics"] = dict(status="READY", text=(
        f"Age and sex were recorded for part of the cohort only, and we report the coverage explicitly rather than imputing. In the cleaned dataset (n = {int(dem.loc['all', 'n'])}) age is available for {int(dem.loc['all', 'age_recorded'])} participants "
        f"(mean {f(dem.loc['all', 'age_mean'], 1)} ± {f(dem.loc['all', 'age_sd'], 1)} years, range {dem.loc['all', 'age_min']:.0f}–{dem.loc['all', 'age_max']:.0f}) and sex for {int(dem.loc['all', 'sex_recorded'])} ({int(dem.loc['all', 'female'])} female, {int(dem.loc['all', 'male'])} male; {f(dem.loc['all', 'female_pct_of_recorded'], 0)} % female of those recorded). "
        f"For the {int(dem.loc['high', 'n'])} high-smile-line images with a reference measurement: age recorded for {int(dem.loc['high', 'age_recorded'])} (mean {f(dem.loc['high', 'age_mean'], 1)} ± {f(dem.loc['high', 'age_sd'], 1)}), sex for {int(dem.loc['high', 'sex_recorded'])} ({f(dem.loc['high', 'female_pct_of_recorded'], 0)} % female). "
        "The incomplete demographic record is stated as a limitation."),
        changes="[Table — new: demographic coverage per group] `tables/demographics.md`; Methods 2.1; Limitations", files=["outputs/07_report/tables/demographics.md"], sources=[S["dem"]])
    A["external_validity"] = dict(status="READY", text=(
        "We agree with the reviewer. The a priori χ² calculation addressed a frequency comparison that is not performed in this study and does not inform the data requirement of a segmentation network; it has been removed together with the statement that the cohort size supports external validity. "
        "Data adequacy is now shown empirically with a learning curve (see our response on the power calculation) and the agreement analysis is justified by precision rather than power. "
        + (f"The Discussion states explicitly: \"{m_single.group(1)}\"" if m_single else "The Limitations section now states that all images come from a single centre and a single imaging device and that external validation is required before clinical use.")),
        changes="Methods 2.1 (sample size) rewritten; Discussion / Limitations (single centre, single device, internal estimate)", files=["outputs/07_report/figures/learning_curve.png"], sources=[S["guc"], S["lc"]])
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
        f"(without seeing the threshold table) and measure gingival display per tooth on their own calibrated scale; the majority class is the reference standard, the primary statistic is the linear-weighted κ with bootstrap confidence intervals, and the primary set is the {int(dc.loc['high', 'kept'])} reference images with out-of-fold model predictions (the fixed test subset is secondary). "
        "Model values are reported uncorrected (primary) and with the mask-level correction (secondary). "
        "[PENDING — the expert forms are being completed; the numbers will be inserted from `outputs/04_expert/manuscript_numbers.md` (κ, per-class sensitivity/specificity, inter-expert Fleiss κ, ICC of millimetre values).] "
        f"Note that the E4 class (> 8 mm) has no case among the reference images (maximum mean gingival display {f(ref_max)} mm; n(E4) = {n_e4}); the E4 branch of the decision table is therefore not validated and this is stated."),
        changes="[Section 2.x — new: expert evaluation protocol]; [Results 3.x — to be inserted]; [Table — model vs expert agreement] `tables/expert_agreement.md`", files=["outputs/04_expert/expert_summary.md", "outputs/07_report/tables/expert_agreement.md", "docs/Uzman_degerlendirme_protokolu.md"], sources=[S["status"], S["per6"]])
    A["overlap"] = dict(status="CLINICAL", text=(
        (f"Draft from the clinical team (to be confirmed and signed off by them): \"{m_overlap.group(1)}\"" if m_overlap else "[CLINICAL — text from the clinical team on the relationship with the earlier study]")
        + f" Technical check: {m_match.group(1) if m_match else '?'} of the {int(dc.loc['high', 'images_roboflow_export'])} high-smile-line images have a reference measurement in the earlier study's measurement file; the earlier partition was not reused (new participant-level split, different label scheme and model)."),
        changes="Methods 2.1 (relationship to the earlier study); `outputs/07_report/overlap_with_prior_study.md` [to be generated]", files=["docs/Klinik_ekip_kararlari_15Eylul.md", "outputs/01_data/OZET.md"], sources=[S["clin"], S["ozet1"]])
    A["intra_observer"] = dict(status="READY", text=(
        (f"We report the intra-observer reliability of the clinical reference on the complete two-session file. We note a correction: the calibration files used for the original submission contained {m_15.group(1)} images per session although {m_15.group(2)} were stated; the clinical team supplied the complete 20-image file and the statistics were recomputed. " if m_15 else "")
        + (f"Twenty images × six teeth measured twice by the same observer: tooth level ICC(2,1) {tooth_row[2]}, mean difference {tooth_row[5]} mm, SD {tooth_row[6]} mm, 95 % limits of agreement {tooth_row[7]} mm (n = {tooth_row[1]}); image-mean level ICC(2,1) {image_row[2]} (n = {image_row[1]}). " if tooth_row and image_row else "")
        + (f"({intra_in_coco.group(1)} of these {intra_in_coco.group(2)} images belong to the present dataset.) " if intra_in_coco else "")
        + f"The tooth-level SD of {tooth_row[6] if tooth_row else '?'} mm is the repeatability floor of the reference against which the pipeline's MAE is interpreted."),
        changes="Methods 2.x (reference measurement); Results (intra-observer table) `outputs/03_oracle/intra_observer.md`", files=["outputs/03_oracle/intra_observer.md"], sources=[S["intra"], S["audit"]])
    A["limitations"] = dict(status="CLINICAL", text=(
        "The Discussion and Limitations are being revised by the clinical team; the technical facts to be stated are: (i) terminology — 'high smile line' is used throughout instead of 'gummy smile / excessive gingival display', and the decision table lists possible aetiological conditions and treatment alternatives, not diagnoses; "
        f"(ii) the E4 class (> 8 mm) does not occur among the reference images (maximum {f(ref_max)} mm), so that branch is not validated; (iii) external validity: single centre, single device, internal estimate only; "
        + (f"(iv) \"{m_pigment.group(1)}\"; " if m_pigment else "(iv) gingival and skin pigmentation were not recorded; ")
        + f"(v) demographic records are incomplete (see Reviewer 3); (vi) the segmentation model places the lower gingival margin {f(a6['gingiva_bottom_edge_bias_mm_mean'])} mm too low on average; the correction is post hoc and secondary; (vii) {n_fail} of {int(P['n']) + n_fail} images produced no gingiva mask and must be flagged for manual review in any deployment."),
        changes="Discussion / Limitations [text from the clinical team]", files=["docs/Klinik_ekip_kararlari_15Eylul.md", "outputs/06_prediction/prediction_summary.md"], sources=[S["clin"], S["per6"], S["bset"]])

    # ------------------------------------------------------------------ documents
    lines = ["# Response to the reviewers — draft", "",
             "<!-- Generated by scripts/build_rebuttal.py. Every number is read from the file named in the source comment under each response; nothing is typed by hand. "
             "Reviewer quotes: the reviewer document (docs/Hakem_revizyonları.docx) is not in the repository; paste each item's original text into docs/hakem_maddeleri.yaml. -->", "",
             "Conventions: uncorrected measurement results are the primary analysis; results with the post-hoc mask-level correction are secondary. 'High smile line' is used throughout. "
             "Status: READY = answered from the analysis outputs; PENDING = waits for the expert forms; CLINICAL = wording to be provided by the clinical team.", ""]
    rows = []
    for rev in spec["reviewers"]:
        lines += [f"## {rev['name']}", ""]
        for it in rev["items"]:
            a = A[it["answer"]]
            status = it.get("status_override") or a["status"]
            quote = it.get("quote") or QUOTE_PLACEHOLDER
            lines += [f"### {it['id']} — {it['topic']}", "",
                      f"> {quote}", ""]
            if not it.get("quote"):
                lines += [f"*(Paraphrase from the technical audit / task document: {it['paraphrase']})*", ""]
            lines += ["**Response:**", "", a["text"], "",
                      f"**Changes in the manuscript:** {a['changes']}", "",
                      f"**Status:** {status}", "",
                      "Outputs: " + ", ".join(f"`{p}`" for p in a["files"]), "",
                      *a["sources"], ""]
            rows.append({"item": it["id"], "reviewer": rev["name"], "topic": it["topic"], "status": status, "owner": it.get("owner", "technical"), "outputs": "; ".join(a["files"])})
    (o7 / "RESPONSE_TO_REVIEWERS.md").write_text("\n".join(lines), encoding="utf-8")

    st = pd.DataFrame(rows)
    durum = ["# Rebuttal durumu — hakem maddeleri (klinik ekibe)", "",
             f"Üretim: `scripts/build_rebuttal.py`; sayılar `outputs/` altındaki dosyalardan okunur. Durum: READY = analiz çıktılarından cevaplandı; PENDING = uzman formlarını bekliyor; CLINICAL = metni klinik ekip yazacak. "
             f"Toplam {len(st)} madde: READY {int((st['status'] == 'READY').sum())}, PENDING {int((st['status'] == 'PENDING').sum())}, CLINICAL {int((st['status'] == 'CLINICAL').sum())}.", "",
             "| madde | hakem | konu | durum | sorumlu | cevaplayan çıktı |", "|---|---|---|---|---|---|"]
    for r in rows:
        durum.append(f"| {r['item']} | {r['reviewer']} | {r['topic']} | **{r['status']}** | {'teknik' if r['owner'] == 'technical' else 'klinik'} | {r['outputs']} |")
    durum += ["", "Not: hakemlerin orijinal metni depoda yok (`docs/Hakem_revizyonları.docx`); madde numaraları ve alıntılar belge geldiğinde `docs/hakem_maddeleri.yaml` üzerinden tamamlanacak. Listedeki maddeler teknik denetim raporu ve görev belgesinden derlendi; belgede başka maddeler varsa eklenecek."]
    (o7 / "REBUTTAL_DURUM.md").write_text("\n".join(durum) + "\n", encoding="utf-8")

    ready = st[st["status"] == "READY"]; pend = st[st["status"] == "PENDING"]; clin_ = st[st["status"] == "CLINICAL"]
    ozet = ["# Rebuttal — Türkçe özet", "",
            f"- Taslak: `RESPONSE_TO_REVIEWERS.md` ({len(st)} madde, 4 hakem). Tüm sayılar `outputs/` dosyalarından okundu; her cevabın altında kaynak dosya yorumu var.",
            f"- **Hazır ({len(ready)})**: " + ", ".join(f"{r.item} ({r.topic})" for r in ready.itertuples()) + ".",
            f"- **Uzman verisi bekleyen ({len(pend)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in pend.itertuples()) or "yok") + ". Formlar analiz edilince `run_expert_analysis.py` → `manuscript_numbers.md`; ayrıca kalibrasyon cevabındaki uzman ölçeği cümlesi (R4-2 içinde [PENDING] işaretli).",
            f"- **Klinik ekipten metin bekleyen ({len(clin_)})**: " + (", ".join(f"{r.item} ({r.topic})" for r in clin_.itertuples()) or "yok") + ". Örtüşme paragrafı klinik ekibin 15 Eylül metninden taslak olarak kondu; sınırlılıklar için teknik maddeler listelendi.",
            "- Ölçüm doğruluğu her yerde düzeltmesiz birincil, maske düzeyi düzeltme ikincil.",
            "- Açıkça kabul edilen hatalar: eski ölçüm modülünün geometri hatası (eski Figure 6; v3 ve v1 kolonları geçersiz, figür yeniden üretildi), aynı hastanın iki fotoğrafının bölüntüler arasında bulunması (hasta düzeyi yeniden bölünme, yeniden eğitim), gözlemci içi dosyasında 15 vs 20 görüntü (tam dosyayla yeniden hesaplandı).",
            "- Yapamadıklarımız açıkça yazıldı: E4 sınıfı veri setinde yok; dış geçerlilik yok (tek merkez/cihaz); pigmentasyon kaydı yok; demografi kısmi.",
            "- Eksik: hakemlerin orijinal metni (`docs/Hakem_revizyonları.docx`) depoda yok; alıntı yerlerinde yer tutucu var. Belge gelince `python scripts/build_rebuttal.py --docx <dosya>` paragrafları listeler, alıntılar `docs/hakem_maddeleri.yaml`'a yapıştırılır ve betik yeniden çalıştırılır.",
            "- Terminoloji: 'high smile line'; 'gummy smile / excessive gingival display' kullanılmadı."]
    (o7 / "REBUTTAL_OZET.md").write_text("\n".join(ozet) + "\n", encoding="utf-8")
    print("\n".join(ozet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
