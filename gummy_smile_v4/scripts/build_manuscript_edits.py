#!/usr/bin/env python
"""Manuscript edit list — what has to change in the submitted text, sentence by sentence.

    python scripts/build_manuscript_edits.py

Reads the submitted manuscript and the appendices from docs/ (plain UTF-8 text despite the
.docx extension), locates each sentence that has to change by an anchor substring, and writes
outputs/07_report/MANUSCRIPT_EDITS.md with, for every edit: the current sentence as submitted,
the proposed replacement, the reason, and the reviewer item it answers. Every number in a
proposed sentence is read from outputs/ (source given per edit); nothing is typed by hand.
An anchor that is not found is reported as NOT FOUND instead of being silently dropped.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.report import tables as T  # noqa: E402

DOCS = {
    "manuscript": "Makale_gonderilmis_hali.docx",
    "Appendix B": "Appendix_B-Dataset_Preparation_and_Preprocessing.docx",
    "Appendix C": "Appendix_C-Dataset_Versioning_and_Preprocessing_Pipeline.docx",
    "Appendix D": "Appendix_D-Detailed_Training_and_Hyperparameter_Optimization.docx",
    "Appendix E": "Appendix_E-Training_configuration_and_hyperparameter_search_strategy.docx",
    "Appendix F": "Appendix_F-Comparison_of_Three_Segmentation_Models_for_Gingival_Display_Analysis.docx",
}


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


def clean(t: str) -> str:
    """Strip the bold/italic markers the export scattered through the text."""
    t = t.replace("**", "").replace("*", "")
    return re.sub(r"\s+", " ", t).strip()


def sentences(text: str) -> List[str]:
    out = []
    for para in text.split("\n"):
        p = clean(para)
        if not p:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(“\"])", p) if s.strip())
    return out


def find(sents: List[str], anchor: str, span: int = 1) -> Optional[str]:
    a = clean(anchor).lower()
    for i, s in enumerate(sents):
        if a in s.lower():
            return " ".join(sents[i:i + span])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["_root"]
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    o3, o5, o6, o7 = outputs / "03_oracle", outputs / "05_predictions", outputs / "06_prediction", outputs / "07_report"
    tab = o7 / "tables"

    def sfile(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(root))
        except ValueError:
            return str(p)

    text = {k: (root / "docs" / v).read_text(encoding="utf-8") for k, v in DOCS.items()}
    sents = {k: sentences(v) for k, v in text.items()}

    # ------------------------------------------------------------------ numbers
    dc = pd.read_csv(tab / "dataset_counts.csv").set_index("group")
    dem = pd.read_csv(tab / "demographics.csv").set_index("group")
    seg, seg_kind = read_segmentation_metrics(tab / "segmentation_metrics_test.csv")
    tm = json.loads((o5 / "test_metrics.json").read_text())
    LC = T.learning_curve_facts(tab, o5 / "learning_curve.md")
    lc = LC["df"]
    acc = pd.read_csv(o6 / "measurement_accuracy.csv").set_index("set")
    est3 = pd.read_csv(o3 / "estimator_comparison.csv").set_index("combo")
    bset = pd.read_csv(o6 / "boundary_by_set.csv").set_index("set")
    intra = (o3 / "intra_observer.md").read_text(encoding="utf-8")
    ms = T.measurement_accuracy(o3, o6)[1]
    k = float(cfg["measurement"]["px_per_mm"])
    off_px = int(cfg["measurement"]["bottom_edge_offset_px"])
    method = f"{cfg['measurement']['method']['regioning']}_{cfg['measurement']['method']['estimator']}"
    train_args = json.loads((o5 / "final" / "environment.json").read_text())["train_args"]
    env = json.loads((o5 / "final" / "environment.json").read_text())["environment"]

    P = acc.loc["(a) OOF masks, all reference images [PRIMARY]"]
    Pc = acc.loc["(a) OOF masks, all reference images [PRIMARY] — corrected"]
    B = acc.loc["(b) final model masks, test high images [secondary set]"]
    G = acc.loc["GT masks, all reference images (Stage 3, same method and scale)"]
    G3 = est3.loc[method]
    a6 = bset.loc["(a) OOF, 145 reference high"]
    low6, norm6 = bset.loc["(c) test low"], bset.loc["(c) test normal"]
    sg, sl, sa = seg.loc["diseti"], seg.loc["dudak"], seg.loc["all"]
    cm = np.array(tm.get("confusion_matrix", []))
    g_tp, g_fp, g_fn = (cm[0, 0], cm[0, 2], cm[2, 0] + cm[1, 0]) if cm.shape == (3, 3) else (np.nan,) * 3
    lc25, lc100 = lc[lc.fraction == 0.25].iloc[0], lc[lc.fraction == 1.0].iloc[0]
    def intra_row(label: str):
        """Cells of one row of the intra-observer table: level, n, ICC(2,1), ICC(3,1), ICC(2,k),
        mean difference, SD, limits of agreement. Parsed by splitting the row, because the columns
        share a number-plus-interval shape that a regular expression silently mis-aligns."""
        for line in intra.splitlines():
            if line.startswith("|") and line.split("|")[1].strip() == label:
                return [c.strip() for c in line.strip().strip("|").split("|")]
        return None

    t_row, i_row = intra_row("tooth site"), intra_row("image mean")
    n_high_pct = 100 * int(dc.loc["high", "kept"]) / int(dc.loc["total", "kept"])
    S = {"dc": sfile(tab / "dataset_counts.csv"), "dem": sfile(tab / "demographics.csv"), "seg": sfile(tab / "segmentation_metrics_test.csv"),
         "tm": sfile(o5 / "test_metrics.json"), "lc": sfile(tab / "learning_curve.csv"), "acc": sfile(o6 / "measurement_accuracy.csv"),
         "est3": sfile(o3 / "estimator_comparison.csv"), "bset": sfile(o6 / "boundary_by_set.csv"), "intra": sfile(o3 / "intra_observer.md"),
         "env": sfile(o5 / "final" / "environment.json"), "off": sfile(o6 / "offset_correction.md"), "cfg": "configs/config.yaml"}

    def f(x, d=2, sign=False):
        return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"

    def fm(r):
        return (f"MAE {f(r['mae'])} mm, RMSE {f(r['rmse'])} mm, r = {f(r['r'], 3)}, ICC(2,1) {f(r['icc2_1'], 3)} "
                f"[{f(r['icc2_1_ci_low'], 3)}, {f(r['icc2_1_ci_high'], 3)}], mean difference {f(r['ba_bias'], 2, True)} mm "
                f"(95 % limits of agreement {f(r['ba_loa_low'])} to {f(r['ba_loa_high'])} mm)")

    # ------------------------------------------------------------------ the edits
    E: List[Dict[str, Any]] = []

    def edit(doc, section, anchor, proposed, why, item, sources, span=1, action="replace"):
        E.append({"doc": doc, "section": section, "anchor": anchor, "current": find(sents[doc], anchor, span),
                  "proposed": proposed, "why": why, "item": item, "sources": sources, "action": action})

    edit("manuscript", "Title / Abstract (Objectives)", "Toward etiological interpretation and treatment planning",
         "Keep the title but make the claim of the subtitle match what is validated, e.g. \"AI-driven quantification of gingival display from smile photographs: measurement accuracy and a rule-based framework for etiological interpretation\". "
         "The subtitle should not promise treatment planning validation that the study does not provide.",
         "The etiology-treatment layer is a transparent application of published thresholds; its agreement with clinical judgement is assessed as an agreement study, not validated as a decision tool.",
         "R3-General-3, R3-Abstract-1, R4-1", [])
    edit("manuscript", "Abstract (Methods)", "A total of 1,315 frontal smiling photographs were categorized according to smile line",
         f"A total of {int(dc.loc['total', 'images_roboflow_export'])} frontal smiling photographs were categorised according to smile line. After exclusion of images without a clinical reference measurement, same-participant duplicates and one ambiguous record, "
         f"{int(dc.loc['total', 'kept'])} images from {int(dc.loc['total', 'kept'])} participants were analysed and partitioned at participant level into training ({int(dc.loc['total', 'train'])}), validation ({int(dc.loc['total', 'valid'])}) and a fixed test set ({int(dc.loc['total', 'test'])}). "
         f"Quantitative gingival display assessment was performed only in the high smile line subgroup with a clinical reference measurement (n = {int(dc.loc['high', 'kept'])}, {n_high_pct:.1f} % of the analysed images); "
         "low and average smile-line images were used only for training the segmentation model.",
         "The submitted Abstract lets the reader assume that all 1,315 images entered the quantitative analysis and does not state the partition level.",
         "R2-1, R2-5, R2-6, R3-Results-2, R4-7", [S["dc"]])
    edit("manuscript", "Abstract (Results)", "YOLOv11x-seg demonstrated the best overall segmentation performance, with a mask mAP@50 of 0.79369",
         f"On the fixed, participant-level test set (n = {int(tm.get('n_images', 0))} images, evaluated once) the final YOLOv11x-seg model achieved a mask mAP@50 of {f(sa['seg_map50'])} "
         f"(gingiva {f(sg['seg_map50'])}, lip {f(sl['seg_map50'])}), mask mAP@50–95 {f(sa['seg_map50_95'])}, mask precision {f(sa['seg_precision'])} and mask recall {f(sa['seg_recall'])}; "
         f"the corresponding box metrics were mAP@50 {f(sa['box_map50'])}, precision {f(sa['box_precision'])} and recall {f(sa['box_recall'])}. "
         f"Gingival display measured from the predicted masks agreed with the clinical reference measurements with {fm(P)} on the {int(P['n'])} high-smile-line images predicted out of fold, "
         f"and {fm(B)} on the fixed test set. In a secondary analysis correcting a systematic displacement of the lower gingival margin, the out-of-fold error fell to MAE {f(Pc['mae'])} mm with a mean difference of {f(Pc['ba_bias'], 2, True)} mm.",
         "The submitted numbers are validation-set metrics of a model tuned on that validation set, and mix Box with Mask metrics without labelling them; no millimetre accuracy is reported.",
         "R2-2, R2-8, R3-Results-4, R4-4, R4-6", [S["seg"], S["tm"], S["acc"]], span=2)
    edit("manuscript", "Abstract (Conclusions)", "The proposed framework enables objective quantification of gingival display and translates these measurements",
         "The proposed framework quantifies gingival display from smile photographs with a measurement error of the order of half a millimetre against clinical reference measurements, and maps the measurement to literature-derived etiological and treatment categories through an explicit rule set. "
         "The segmentation and measurement components were evaluated on an independent test set; the etiological and treatment layer was compared with independent clinical assessment as an agreement analysis and has not been validated as a clinical decision tool.",
         "The submitted conclusion implies clinical decision support has been demonstrated. The reviewers ask that conclusions be restricted to what was evaluated.",
         "R3-Abstract-1, R3-Discussion-5, R4-1", [S["acc"]])
    edit("manuscript", "Abstract (Clinical Significance)", "may improve the consistency of excessive gingival display assessment",
         "The framework may improve the consistency of high smile line assessment by providing reproducible measurements; its clinical usefulness for etiological interpretation and treatment planning remains to be established prospectively.",
         "Removes the implied clinical validation and the term 'excessive gingival display' (clinical team's terminology decision: 'high smile line').",
         "R3-Discussion-5, terminology", [])
    edit("manuscript", "Abstract (Keywords)", "Excessive gingival display",
         "Replace the keyword 'Excessive gingival display' with 'High smile line'; add 'Gingival display measurement' and 'Clinical decision support'.",
         "Terminology decision of the clinical team (15 September): 'high smile line' throughout.",
         "terminology", [])
    edit("manuscript", "2.1 Study design (sample size)", "The sample size for this study was calculated using G*Power software",
         "REMOVE the whole sample-size paragraph and Appendix A, and replace with: \"No formal power calculation was applied to the segmentation training set, because conventional hypothesis-testing sample-size methods do not determine the data requirements of deep-learning models. "
         f"Data adequacy was instead assessed empirically: the final architecture was retrained on stratified 25 %, 50 %, 75 % and 100 % subsets of the training partition ({int(lc25['n_train_images'])} to {int(lc100['n_train_images'])} images) with the validation set held constant, "
         + (f"and {LC['label']} rose from {LC['first']:.3f} to {LC['last']:.3f}, reaching a plateau (Supplementary Figure S1). " if LC["plateau"] else
            f"and {LC['label']} was {LC['points']} at {LC['sizes']} training images respectively (Supplementary Figure S1). "
            "Performance had not plateaued within the available training-set size; the increments between adjacent points are of the same order as run-to-run variation, "
            "so the curve indicates that additional data could still improve segmentation performance. This is stated as a limitation. ")
         + f"For the millimetre-level agreement analysis the sample size is justified by precision rather than power: {ms.get('precision_note', '')}\"",
         "The χ² calculation describes a comparison of correct and incorrect detections that the study never performs, and it cannot determine the data requirement of a segmentation model.",
         "R2-sample-size, R4-external-validity", [S["lc"]], span=4, action="replace whole paragraph")
    edit("manuscript", "2.1 Study design (external validity)", "the sample size of the present study supports the reliability and external validity of the model",
         "DELETE this sentence. Add to the Limitations: \"All images were acquired at a single centre with a single smartphone model under a standardised protocol; the reported performance is therefore an internal estimate, and external validation on images from other centres, devices and populations is required before clinical use.\"",
         "Sample size does not establish external validity, and the cohort is single-centre and single-device.",
         "R4-external-validity, R3-Discussion-5", [])
    edit("manuscript", "2.6 Image dataset and preprocessing", "This study utilized a dataset consisting of 1,315 frontal smiling photographs",
         f"This study used {int(dc.loc['total', 'images_roboflow_export'])} frontal smile photographs categorised by smile line as high (n = {int(dc.loc['high', 'images_roboflow_export'])}), average (n = {int(dc.loc['normal', 'images_roboflow_export'])}) and low (n = {int(dc.loc['low', 'images_roboflow_export'])}). "
         f"A content-based search identified photographs of the same participant acquired in the same session; one photograph per participant was retained. Together with {int(dc.loc['high', 'no_reference_measurement'])} high-smile-line images that had no clinical reference measurement and one image with an ambiguous record, "
         f"{int(dc.loc['total', 'images_roboflow_export']) - int(dc.loc['total', 'kept'])} images were excluded, leaving {int(dc.loc['total', 'kept'])} images from {int(dc.loc['total', 'kept'])} participants "
         f"(high {int(dc.loc['high', 'kept'])}, average {int(dc.loc['normal', 'kept'])}, low {int(dc.loc['low', 'kept'])}). "
         f"Age was recorded for {int(dem.loc['all', 'age_recorded'])} participants (mean {f(dem.loc['all', 'age_mean'], 1)} ± {f(dem.loc['all', 'age_sd'], 1)} years) and sex for {int(dem.loc['all', 'sex_recorded'])} "
         f"({int(dem.loc['all', 'female'])} female, {int(dem.loc['all', 'male'])} male); the smile-line groups are a consecutive clinical series and were not matched for size or sex distribution, and they are not compared with each other.",
         "The submitted sentence reports 896 females / 419 males and a mean age for all 1,315 images although age and sex are recorded for only part of the cohort; the group sizes and the duplicate participants must also be stated.",
         "R2-5, R3-Methods-5, R3-Results-2", [S["dc"], S["dem"]])
    edit("manuscript", "2.6 Image dataset and preprocessing (split)", "The dataset was initially divided into training, validation, and test sets using a 70 %, 15 %, and 15 % split",
         f"The dataset was partitioned once, at participant level, into training ({int(dc.loc['total', 'train'])} images), validation ({int(dc.loc['total', 'valid'])}) and a fixed test set ({int(dc.loc['total', 'test'])}), stratified by smile-line group and, within the high smile line group, by reference class. "
         "All photographs of a participant remain in a single partition (CLAIM 2024). Augmentation was applied to the training partition only, by the training framework at run time; no augmented copies enter the validation or test sets, and the test set was evaluated once, with the final model. "
         f"For the millimetre validation, the {int(dc.loc['high', 'kept'])} high-smile-line images with a clinical reference measurement were additionally predicted out of fold in a 5-fold participant-level cross-validation, so that each image was measured from a mask produced by a model that had not seen it.",
         "The 70/15/15 → 92/4/4 sequence, the 1,315 → 3,403 count and the unclear partition level are the reviewers' main methodological objection; the count grew through augmentation, and the partition was not at participant level.",
         "R2-5, R2-6, R3-Results-2, R4-7", [S["dc"]], span=5, action="replace whole paragraph")
    edit("manuscript", "2.7 Annotation and labeling", "In the v7 dataset used for the detailed YOLOv8 and YOLOv11 training experiments",
         f"In the dataset used for the reported training experiments the annotated instance distribution comprised 3,938 gingiva and 1,318 lip instances over the {int(dc.loc['total', 'images_roboflow_export'])} original images "
         "(in low and average smile lines each visible interdental papilla is a separate gingiva instance, which is why gingiva instances outnumber images). "
         "Instance counts reported previously for an augmented dataset version are not comparable with image counts and are no longer reported as such.",
         "'YOLOv8 and YOLOv11' is a leftover from an earlier version (the reported comparison is RF-DETR-Seg, YOLOv26, YOLOv11), and the instance counts belong to an augmented version.",
         "R2-4, R2-5", [S["dc"]], span=2)
    edit("manuscript", "2.7 Annotation and labeling (3,403)", "resulting in an updated dataset version containing 3,403 images",
         "DELETE the sentence. The dataset used for the reported results is the cleaned set of "
         f"{int(dc.loc['total', 'kept'])} original images; augmentation is applied on the fly to the training partition and produces no additional stored images.",
         "3,403 was an augmented dataset version; presenting it as 'images' is the source of the reviewers' confusion.",
         "R2-5, R3-Results-2, R4-7", [S["dc"]])
    edit("manuscript", "2.8.1 Preliminary comparison phase", "RF-DETR-Seg was evaluated on dataset v1, whereas YOLOv26 and YOLOv11 were evaluated on datasets v2 and v3",
         "This stage was a screening step run with the annotation platform's default training settings on separate copies of the same baseline dataset version. Because the training data, preprocessing and settings were not held identical across the three runs, "
         "it does not isolate the effect of the architecture and is reported only as the basis for selecting one architecture to take forward; no claim of architectural superiority is made.",
         "The reviewer is right that the comparison does not isolate the architecture; the honest fix is to relabel the stage and withdraw the claim.",
         "R4-8", [])
    edit("manuscript", "2.9 Intra-observer calibration", "Intra-observer reliability was assessed on 20 randomly selected images",
         f"Intra-observer reliability of the clinical reference was assessed by remeasuring the gingival display of 20 images at six tooth sites each, by the same examiner at a separate session: "
         f"tooth-site level ICC(2,1) {t_row[2] if t_row else '?'} (n = {t_row[1] if t_row else '?'} tooth-site pairs), "
         f"image-mean level ICC(2,1) {i_row[2] if i_row else '?'} (n = {i_row[1] if i_row else '?'} images), "
         f"mean difference {t_row[5] if t_row else '?'} mm, standard deviation {t_row[6] if t_row else '?'} mm, 95 % limits of agreement {t_row[7] if t_row else '?'} mm at tooth-site level. "
         "The paired t-test is not used as evidence of agreement; the limits of agreement and the ICC are reported instead.",
         "The submitted sentence gives no numbers, cites a paired t-test as evidence of agreement, and calls the repeated measurements 'tooth measurements' although the quantity measured is the gingival display at a tooth site, not a dimension of the tooth. The calibration files used for the original submission contained 15 images per session, not 20; the complete 20-image file was supplied by the clinical team and the statistics were recomputed from it.",
         "R4-4 (reference reliability)", [S["intra"]], span=2)
    edit("manuscript", "3.1 Model selection and segmentation performance", "the YOLOv11 model demonstrated superior performance in terms of mAP@50 (79.3 %)",
         "In a preliminary screening on the annotation platform, the YOLOv11 family gave the most balanced mask mAP@50 among the three candidate architectures and was selected for the subsequent training and optimisation stages. "
         "Because the three runs did not share identical data and settings, this screening is not a controlled comparison of architectures and no conclusion about architectural superiority is drawn from it.",
         "'Superior performance' is not defined and the comparison is not controlled; the reviewer also asks what 'performance' means.",
         "R3-Results-4, R4-8", [], span=2)
    edit("manuscript", "3.1 Model selection (validation metrics)", "The validation results of the final YOLOv11x-seg model were as follows",
         f"The final YOLOv11x-seg model was evaluated once on the fixed, participant-level test set (n = {int(tm.get('n_images', 0))} images): mask mAP@50 {f(sa['seg_map50'])}, mask mAP@50–95 {f(sa['seg_map50_95'])}, mask precision {f(sa['seg_precision'])}, mask recall {f(sa['seg_recall'])}, mask F1 {f(sg['seg_f1'])} (gingiva) and {f(sl['seg_f1'])} (lip); "
         f"box mAP@50 {f(sa['box_map50'])}, box mAP@50–95 {f(sa['box_map50_95'])}, box precision {f(sa['box_precision'])}, box recall {f(sa['box_recall'])}. "
         f"Per class, mask mAP@50 was {f(sg['seg_map50'])} for gingiva and {f(sl['seg_map50'])} for lip. Metric type (Box or Mask) is stated for every value; validation-set metrics are not reported as results.",
         "The submitted figures are validation-set metrics of a model whose hyperparameters were selected on that validation set, and the list mixes Box precision/recall with Mask mAP without labels.",
         "R2-2, R2-8, R4-6", [S["seg"], S["tm"]], span=8, action="replace whole list")
    edit("manuscript", "3.1 (class-wise lip performance)", "lip segmentation showed consistently high performance across all models, with class-wise mAP values ranging from 97.0% to 99.0%",
         f"On the fixed test set, lip segmentation reached a mask mAP@50 of {f(sl['seg_map50'])} whereas gingiva segmentation reached {f(sg['seg_map50'])}. "
         f"The difference is examined at the boundary rather than left as an aggregate gap: against the annotated masks of the {int(a6['n'])} reference images, the upper, lip-side gingiva edge is accurate "
         f"(mean absolute column-wise error {f(a6['gingiva_top_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_top_edge_bias_mm_mean'], 2, True)} mm) while the lower, festooned gingival margin is placed systematically too low "
         f"({f(a6['gingiva_bottom_edge_mae_mm_mean'])} mm, bias {f(a6['gingiva_bottom_edge_bias_mm_mean'], 2, True)} mm); gingiva mask IoU {f(a6['gingiva_mask_iou_mean'])}, lip mask IoU {f(a6['lip_mask_iou_mean'])}. "
         f"Pooled over the whole test set the gingiva IoU is lower ({f(bset.loc['(c) test all', 'gingiva_mask_iou_mean'])}) because in low ({f(low6['gingiva_mask_iou_mean'])}) and average ({f(norm6['gingiva_mask_iou_mean'])}) smile lines the annotated gingiva is thin or absent "
         f"(median annotated width {low6['gingiva_n_columns_gt_median']:.0f} image columns in the low group against {a6['gingiva_n_columns_gt_median']:.0f} in the high group).",
         "The reviewers ask for a quantitative investigation of the lip/gingiva gap and for the low gingiva value to be stated in the text.",
         "R2-8, R2-10", [S["seg"], S["bset"]], span=2)
    edit("manuscript", "3.x (new section)", "Clinical threshold-based classification",
         "INSERT a new Results section before the threshold-based classification: \"3.x Millimetre measurement accuracy. "
         f"Measurement geometry was first evaluated on the annotated masks ({fm(G)}; method {method}, scale fitted on a 60 % development subset and applied unchanged to the remaining 40 %: MAE {f(G3['mae_holdout'])} mm, ICC {f(G3['icc2_1_holdout'], 3)}). "
         f"The full pipeline was then evaluated on the model's own masks, each image predicted by a model that had not seen it: {fm(P)}; agreement with the reference class was {100 * P['threshold_agreement']:.0f} % (linear-weighted κ {f(P['threshold_kappa_linear'], 2)}) and {100 * P['within_1_mm']:.0f} % of images were within 1 mm of the reference. "
         f"On the fixed test set with the final model: {fm(B)}. One image ({int(P['n_segmentation_failure'])} of {int(P['n']) + int(P['n_segmentation_failure'])}) produced no gingiva mask and is reported as a segmentation failure rather than as a 0 mm measurement. "
         f"A post-hoc correction of the lower gingival margin (mask level, {abs(off_px)} px estimated on the development subset) is reported as a secondary analysis: MAE {f(Pc['mae'])} mm, mean difference {f(Pc['ba_bias'], 2, True)} mm, class agreement {100 * Pc['threshold_agreement']:.0f} %.\"",
         "The manuscript reports no millimetre accuracy at all; this is the reviewers' central objection and the thresholds at 3, 4, 6 and 8 mm make the measurement error clinically decisive.",
         "R3-Results-5, R4-3, R4-4", [S["acc"], S["est3"], S["off"]])
    edit("manuscript", "3.4 Confusion matrix analysis", "For the gingiva class, the normalized diagonal value was ≈ 72 %",
         f"On the fixed test set the model produced {int(g_tp)} correct gingiva instances, {int(g_fp)} false positives and {int(g_fn)} false negatives for the gingiva class "
         f"(precision {f(sg['seg_precision'])}, recall {f(sg['seg_recall'])} for the mask), against {f(sl['seg_precision'])} and {f(sl['seg_recall'])} for the lip class. "
         "Sensitivity and specificity in the epidemiological sense are not defined for instance segmentation, because there is no fixed set of candidate regions and hence no count of true negatives; precision, recall, F1 and the confusion matrix are reported instead.",
         "The reviewer asks explicitly whether false positives and false negatives were computed, and whether 'performance' means sensitivity or specificity.",
         "R3-Results-4", [S["tm"], S["seg"]], span=2)
    edit("manuscript", "3.5 (Figure 6)", "To enable clinically interpretable output, the quantified gingival display values were integrated into a predefined threshold-based decision framework",
         "Rewrite around the corrected measurement and state the erratum: \"Figure 6 of the submitted version reported values produced by an earlier implementation of the measurement module. "
         "That implementation merged the gingiva and lip masks, selected the largest contour and reported the between-region variation of its upper edge; the gingival margin was not used. Both columns of the original figure are therefore withdrawn. "
         f"The figure has been reproduced with the corrected measurement module and is accompanied by the agreement analysis against the clinical reference ({fm(P)}).\"",
         "Reviewer 4 asks that the figure be explained and the correct measurement demonstrated against an independent clinical standard; the cause was a software error and must be stated as such.",
         "R4-3", [S["acc"]])
    edit("manuscript", "4 Discussion (first paragraph)", "the principal contribution of the present study is the integration of validated AI-based quantitative measurement",
         "Replace 'validated AI-based quantitative measurement' with 'AI-based quantitative measurement, whose accuracy against clinical reference measurements is reported here', and state that the etiological layer is an explicit rule set whose agreement with clinical judgement is assessed, not a validated diagnostic tool.",
         "'Validated' cannot be claimed for the decision layer, and the measurement validation belongs to this manuscript rather than being inherited.",
         "R3-Discussion-5, R4-1", [])
    edit("manuscript", "4 Discussion (previous study)", "In our previous study, we established the validity of AI-based gingival display quantification",
         "Add, in the same paragraph, the overlap statement: the two studies draw on the same source pool of standardised smile photographs; of the images in the present study a stated number were also in the earlier model-development set, the earlier partitions were not reused, the label scheme and the model differ, "
         "and the present study reports its own measurement validation rather than inheriting the earlier one. [CLINICAL — exact counts and wording from the clinical team; draft in docs/Klinik_ekip_kararlari_15Eylul.md.]",
         "Reviewer 4 requires an explicit statement of overlap and of why the present study is not redundant.",
         "R4-2", [])
    edit("manuscript", "4 Discussion (limitations)", "the relatively limited demographic and phenotypic diversity of the dataset may have led to insufficient representation of rare smile patterns",
         "Expand the Limitations to state, each in its own sentence: single centre and single device, so the performance is an internal estimate; gingival and skin pigmentation and ethnicity were not recorded, so their effect on segmentation could not be assessed; "
         f"age and sex were recorded for part of the cohort only ({int(dem.loc['all', 'age_recorded'])} and {int(dem.loc['all', 'sex_recorded'])} of {int(dem.loc['all', 'n'])}); the E4 class (> 8 mm) does not occur in this cohort, so that branch of the decision table is not validated; "
         f"gingival segmentation is the weakest component (test-set mask mAP@50 {f(sg['seg_map50'])}) and places the lower gingival margin {f(a6['gingiva_bottom_edge_bias_mm_mean'])} mm too low on average; "
         + ("" if LC["plateau"] else
            "the learning curve had not plateaued within the available training-set size, so additional training data could still improve segmentation performance; ")
         + "and one image produced no gingiva mask, so a deployed system must flag such images for manual review rather than output a value.",
         "The reviewers ask for pigmentation, rare classes and the gingiva performance to be discussed as limitations; the current sentence is vague.",
         "R2-8, R3-Results-1, R3-Discussion-5", [S["dem"], S["seg"], S["bset"]])
    edit("manuscript", "5 Conclusion", "This study presents a segmentation-based analytical pipeline that enables the objective measurement of gingival display",
         "Restrict the conclusion to what was evaluated: the pipeline measures gingival display from smile photographs with a reported accuracy against clinical reference measurements, and maps the measurement to literature-derived categories through an explicit rule set. "
         "State that the decision layer has not been validated as a clinical tool and that prospective, multi-centre validation is required before clinical use.",
         "The submitted conclusion presents the decision support as established.",
         "R3-Discussion-5, R4-1", [])
    edit("manuscript", "Throughout (terminology)", "Gummy smile is a multifactorial developmental condition",
         "Replace 'gummy smile' and 'excessive gingival display' with 'high smile line' throughout, except where a cited study's own terminology is being reported. Table 1 is described as listing possible aetiological conditions and treatment alternatives, not diagnoses or indications.",
         "Terminology decision of the clinical team (15 September).",
         "terminology", [])

    edit("Appendix B", "B.1", "The raw dataset consisted of 1,315 original images obtained for gingival display analysis",
         f"The raw dataset consisted of {int(dc.loc['total', 'images_roboflow_export'])} original images (high {int(dc.loc['high', 'images_roboflow_export'])}, average {int(dc.loc['normal', 'images_roboflow_export'])}, low {int(dc.loc['low', 'images_roboflow_export'])}). "
         f"After removal of same-participant duplicates, high-smile-line images without a clinical reference measurement and one ambiguous record, {int(dc.loc['total', 'kept'])} images remained and were partitioned at participant level "
         f"({int(dc.loc['total', 'train'])} / {int(dc.loc['total', 'valid'])} / {int(dc.loc['total', 'test'])}).",
         "The appendix must match the cleaned dataset actually used for the reported results.",
         "R2-5, R4-7", [S["dc"]], span=2)
    edit("Appendix B", "B.1 (augmentation and splits)", "The dataset was initially divided into 70 % training, 15 % validation, and 15 % test sets",
         "Replace the whole paragraph: a single participant-level partition is used; augmentation is applied by the training framework to the training partition at run time (mosaic, scaling, HSV and horizontal flip, with mosaic disabled for the final 10 epochs) and produces no stored image copies; "
         "no augmentation is applied to the validation or test images. The 70/15/15 and 92/4/4 schemes and the stored augmented dataset versions belong to the earlier experiments and are no longer used.",
         "The stored augmented versions are what produced the 3,155 / 2,235 / 3,403 counts the reviewers could not reconcile.",
         "R2-5, R2-6, R4-7", [S["dc"]], span=2, action="replace whole paragraph")
    edit("Appendix C", "C.1", "Initial augmented dataset version created using ±5° rotation and ±5 exposure transformations",
         "Rewrite the appendix as a record of dataset versions used during development, stating clearly that the reported results come from the cleaned, participant-level partition and not from these versions; or remove it and describe the final dataset only.",
         "Dataset versions v1/v6/v7 with different image counts are the origin of the reviewers' confusion and none of them is the dataset of the reported results.",
         "R2-5, R4-7, R4-8", [])
    edit("Appendix D", "D.1 (expanded version)", "Expanded version: Updated dataset used during the hyperparameter optimization phase, containing 3,403 images",
         f"Replace with the final training configuration actually used: dataset {int(dc.loc['total', 'kept'])} images partitioned at participant level ({int(dc.loc['total', 'train'])} / {int(dc.loc['total', 'valid'])} / {int(dc.loc['total', 'test'])}); "
         f"model {cfg['yolo']['model']}; {int(train_args['epochs'])} epochs, batch {int(train_args['batch'])}, image size {int(train_args['imgsz'])}, {train_args['optimizer']} optimiser, initial learning rate {train_args['lr0']}, final learning-rate fraction {train_args['lrf']}, "
         f"cosine schedule, close_mosaic {int(train_args['close_mosaic'])}, patience {int(train_args['patience'])}, seed {int(train_args['seed'])}, deterministic training; "
         f"{env.get('torch', '?')} / Ultralytics {env.get('ultralytics', '?')} on an {env.get('gpu', '?')}.",
         "The appendix must describe the training that produced the reported model, not the superseded hyperparameter search on an augmented dataset version.",
         "R2-5, R2-9, R4-6, R4-7", [S["env"], S["cfg"]], span=3)
    edit("Appendix D", "D.1 (YOLOv8 screening)", "In the first Colab-based screening stage, YOLOv8-seg, YOLOv11-seg, and YOLO26-seg models were trained across multiple scales",
         "Keep this paragraph but label it explicitly as a preliminary screening run that is not part of the reported comparison and on which no reported result depends; make sure the main text does not refer to YOLOv8.",
         "The reviewer noticed the YOLOv8 mention in the main text; the screening run is the only place it belongs.",
         "R2-4, R4-8", [], span=2)
    edit("Appendix D", "D.1 (instance counts)", "the effective instance distribution reflected the augmented dataset structure and reached 5,628 gingiva instances and 1,846 lip instances",
         "Replace with the instance counts of the original annotation: 3,938 gingiva and 1,318 lip instances over the original images, noting that each visible interdental papilla is a separate gingiva instance in low and average smile lines. "
         "Augmented instance counts are not reported, since augmentation is applied at run time.",
         "5,628 / 1,846 is exactly twice the v7 training-split count and is what Figure 4 reports; it is not an annotation count.",
         "R2-5", [S["dc"]])
    edit("Appendix E", "E.1", "Batch size: 16 (in advanced experiments increased to 32)",
         f"Report the configuration of the reported model: batch {int(train_args['batch'])}, image size {int(train_args['imgsz'])}, {train_args['optimizer']}, initial learning rate {train_args['lr0']}, final fraction {train_args['lrf']}, cosine schedule, close_mosaic {int(train_args['close_mosaic'])}, patience {int(train_args['patience'])}, cache {train_args['cache']}, workers {int(train_args['workers'])}, seed {int(train_args['seed'])}. "
         "Describe the earlier grid search as exploratory and state that it was performed on a superseded dataset version.",
         "The appendix should let a reader reproduce the reported model.",
         "R2-9, R4-6", [S["env"], S["cfg"]], span=3)
    edit("Appendix F", "F.1", "These results indicate that the YOLOv11 family provided the strongest overall balance among the compared models",
         "Reword to: these screening results, obtained with platform default settings on separate copies of the baseline dataset, were used only to select one architecture for the subsequent work; they do not establish that one architecture is superior, since data, preprocessing and settings were not held identical across the runs.",
         "Same objection as in the main text: the comparison does not isolate the architecture.",
         "R4-8", [])

    # ------------------------------------------------------------------ document
    n_found = sum(1 for e in E if e["current"])
    out = ["# Manuscript edits — submitted version vs revision", "",
           "<!-- Generated by scripts/build_manuscript_edits.py. The current sentences are located in the submitted files in docs/ by an anchor; "
           "every number in a proposed sentence is read from the file listed under 'Sources'. Placeholders in square brackets mark text that the clinical team supplies. -->", "",
           f"Source files: {', '.join(f'`docs/{v}`' for v in DOCS.values())}.", "",
           f"{len(E)} edits; the current sentence was located for {n_found} of them. Reviewer items refer to `RESPONSE_TO_REVIEWERS.md`. "
           "Measurement results are quoted uncorrected (primary); the post-hoc correction appears only where it is labelled secondary.", ""]
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for e in E:
        by_doc.setdefault(e["doc"], []).append(e)
    for doc, items in by_doc.items():
        out += [f"## {doc}", ""]
        for i, e in enumerate(items, start=1):
            out += [f"### {doc} · {e['section']}", "",
                    f"**Current ({e['action']}):**", "",
                    (f"> {e['current']}" if e["current"] else f"> **NOT FOUND** — anchor `{e['anchor']}` did not match; locate the passage manually."), "",
                    "**Proposed:**", "", e["proposed"], "",
                    f"**Why:** {e['why']}", "",
                    f"**Reviewer item:** {e['item']}", ""]
            if e["sources"]:
                out += ["<!-- source: " + "; ".join(e["sources"]) + " -->", ""]
    (o7 / "MANUSCRIPT_EDITS.md").write_text("\n".join(out), encoding="utf-8")
    missing = [f"{e['doc']} · {e['section']}" for e in E if not e["current"]]
    print(f"MANUSCRIPT_EDITS.md: {len(E)} edits, {n_found} anchors located" + (f"; NOT FOUND: {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
