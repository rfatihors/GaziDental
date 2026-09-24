#!/usr/bin/env python
"""Rebuild docs/hakem_maddeleri.yaml from the FULL reviewer letter.

    python scripts/build_reviewer_items.py [--docx docs/Hakem_Yorumları.docx]

The quotes are not typed: each item names the lines of the extracted letter it consists of, and the
quote is sliced out of that text, so it is verbatim by construction and a changed source file is
caught by the anchor check rather than by proof-reading. ``docs/Hakem_revizyonları.docx`` is the
earlier, abridged summary (31 items against 59) and is **not** consulted: where the two disagree the
full letter wins.

Every item also carries a short anchor that must appear in its own quote, so a shifted line map
fails loudly instead of silently attaching the wrong text to an item.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# id, line range (1-based, inclusive), topic, answer key, owner, anchor that must be inside the quote
# owner: technical = answered from outputs/; clinical = wording from the clinical team.
R3 = [
    ("R3-General-1", (5, 5), "Rewrite the introduction with a specific aim and references for all statements", "intro_rewrite", "clinical", "rewrite the introduction"),
    ("R3-General-2", (6, 6), "Clarify the materials and methods", "methods_clarity", "clinical", "clarify the materials"),
    ("R3-General-3", (7, 7), "Narrow the manuscript to a validated segmentation/measurement study, or compare with clinical assessment", "narrow_or_compare", "technical", "narrowing the manuscript"),
    ("R3-General-4", (8, 8), "Rewrite the discussion on the study outcomes; draw relevant rather than general conclusions", "discussion_rewrite", "clinical", "rewrite the discussion"),
    ("R3-General-5", (9, 9), "Improve the language of the manuscript", "language", "clinical", "improve the language"),
    ("R3-Title-1", (12, 12), "The title promises more than was validated", "title_claim", "clinical", "The title includes more"),
    ("R3-Abstract-1", (14, 14), "Abstract reports no millimetric accuracy and no validation of the etiological/treatment categories", "abstract_claims", "technical", "do not report millimetric"),
    ("R3-Abstract-2", (15, 15), "Abstract must state the limitations: single centre, one examiner, no clinical assessment", "abstract_limitations", "technical", "Add the study limitations"),
    ("R3-Intro-1", (17, 17), "A reference is missing from the paragraph describing the state of the literature", "intro_rewrite", "clinical", "add a reference or references"),
    ("R3-Intro-2", (18, 18), "The stated scientific gap is general and does not match what the study evaluates", "intro_gap", "clinical", "which describes the scientific gap"),
    ("R3-Intro-3", (19, 19), "Several objectives are presented at once; identify one primary objective", "intro_objectives", "clinical", "one primary objective"),
    ("R3-Methods-1", (21, 21), "Study design, retrospective/prospective, recruitment process and dates", "study_design", "clinical", "cross-sectional study"),
    ("R3-Methods-2", (22, 22), "Inclusion criteria are not given, only exclusion criteria", "inclusion_criteria", "clinical", "inclusion criteria"),
    ("R3-Methods-3", (23, 23), "Sample size belongs in Methods; the numbers of patients and images in Results", "sample_size_placement", "technical", "should only be presented only in the Methods"),
    ("R3-Methods-4", (24, 24), "Why 1,315 patients and not another number", "why_1315", "technical", "why you decided to include 1315"),
    ("R3-Methods-5", (25, 25), "Unequal numbers of patients per smile-line group; equal gender distribution requested", "group_sizes", "technical", "low, medium, and high smile-line groups"),
    ("R3-Methods-6", (26, 26), "Was the periodontal probe in Figure 1 used for calibration? The method is not described", "probe_calibration", "technical", "periodontal probe"),
    ("R3-Methods-7", (27, 27), "Why low and average smile-line images were needed", "why_low_normal", "technical", "low and average smile lines were needed"),
    ("R3-Methods-8", (28, 28), "Cut-off and clinically acceptable error; 0, 1 and 3 mm would not be called a gummy smile", "clinical_cutoff", "technical", "clinically acceptable error"),
    ("R3-Methods-9", (29, 29), "Was a systematic search used to synthesise the evidence behind Table 1?", "table1_evidence", "clinical", "systematic search"),
    ("R3-Methods-10", (30, 33), "Table 1 has clinical deficiencies (a: is < 4 mm a problem; b: aetiology from millimetres alone; c: short upper lip needs a lip measurement)", "table1_logic", "clinical", "clinical deficiencies"),
    ("R3-Methods-11", (34, 34), "Abbreviations such as YOLO must be expanded at first mention", "abbreviations", "clinical", "Abbreviations such as YOLO"),
    ("R3-Results-1", (36, 36), "Demographics not reported; ethnicity and pigmentation of skin and gingiva may affect segmentation", "pigmentation", "clinical", "pigmentation of the skin and gingiva"),
    ("R3-Results-2", (37, 37), "Exact number of participants and images for training, validation and testing", "image_counts", "technical", "1315 and 3403"),
    ("R3-Results-3", (38, 38), "Section 3.5 belongs in Methods; only the validation results belong in Results", "section_35_to_methods", "technical", "Section 3.5"),
    ("R3-Results-4", (39, 40), "What 'performance' means; sensitivity/specificity; false positives and false negatives", "metric_meaning", "technical", "What is meant by (performance)"),
    ("R3-Results-5", (41, 41), "Distinguish segmentation performance from millimetre measurement ability", "separate_evaluations", "technical", "distinguish between evaluating the segmentation"),
    ("R3-Discussion-1", (43, 43), "The discussion repeats literature that belongs in the introduction", "discussion_rewrite", "clinical", "repeats findings from the literature"),
    ("R3-Discussion-2", (44, 44), "The discussion should focus on this study's results and discuss the model's validity", "discussion_rewrite", "clinical", "focus on the results of this study"),
    ("R3-Discussion-3", (45, 45), "Limitations must include selection bias, one examiner and demographic bias", "limitations", "technical", "potential selection bias"),
    ("R3-Discussion-4", (46, 46), "A conclusion appears twice: at the end of the discussion and as its own section", "duplicate_conclusion", "clinical", "separate Conclusion section"),
    ("R3-Discussion-5", (47, 47), "Conclusion must not state the model is clinically validated", "clinical_validation_claim", "clinical", "clinically validated"),
    ("R3-References-1", (49, 49), "Why reference 11 was used in the sample size calculation", "reference_11", "technical", "reference (11)"),
]

R1 = [
    ("R1-novelty", (58, 58), "Technical novelty is incremental and integrative rather than foundational", "novelty", "clinical", "incremental and integrative"),
    ("R1-1", (59, 59), "No system block diagram; how Fig. 6 treatment suggestions are reached", "diagram", "technical", "block diagram"),
    ("R1-2", (60, 60), "No segmentation result images", "segmentation_examples", "technical", "no segmentation result images"),
    ("R1-discussion-length", (61, 61), "The Discussion is too long and repetitive", "discussion_rewrite", "clinical", "too long"),
]

R4 = [
    ("R4-1", (70, 70), "The etiology-treatment module is not validated; no clinical gold standard, no agreement with clinicians", "expert_validation", "technical", "constructing a decision rule"),
    ("R4-1b", (77, 77), "No cohort in which clinicians independently establish the cause and the plan; no sensitivity, specificity or kappa", "expert_validation", "technical", "no clinical gold standard"),
    ("R4-2", (73, 75), "Overlap with the authors' J Dent 2026 study, same ethics number; originality and redundant publication", "overlap", "clinical", "same ethics approval number"),
    ("R4-3", (79, 79), "Figure 6 — v3_yolo vs v1_xgboost give radically different measurements", "figure6", "technical", "v1_xgboost"),
    ("R4-4", (80, 80), "Millimetre-level accuracy not demonstrated (MAE, RMSE, Bland-Altman, ICC)", "mm_accuracy", "technical", "Bland–Altman"),
    ("R4-5", (81, 81), "Pixel-to-millimetre conversion not described", "calibration", "technical", "pixel-to-millimeter conversion"),
    ("R4-6", (82, 82), "Validation-set metrics reported as final results instead of test-set metrics", "validation_vs_test", "technical", "fixed test set"),
    ("R4-7", (83, 83), "Unclear splitting scheme (70/15/15 then 92/4/4), 1,315 to 3,403, patient-level partitioning (CLAIM 2024)", "leakage", "technical", "CLAIM 2024"),
    ("R4-8", (84, 84), "Architecture comparison is not fair (different dataset versions)", "architecture_comparison", "technical", "not entirely fair"),
    ("R4-9", (78, 78), "The etiology of a gummy smile cannot be derived from millimetres alone; this contradicts the logic of Table 1", "table1_logic", "clinical", "cannot be derived solely"),
    ("R4-external-validity", (85, 85), "The G*Power calculation does not establish sufficiency or external validity", "external_validity", "technical", "external validity"),
]

R2 = [
    ("R2-1", (88, 88), "Abstract implies all 1,315 images were used for the quantitative analysis", "abstract_scope", "technical", "may imply that all 1,315"),
    ("R2-2", (89, 89), "Box metrics vs Mask metrics not identified", "metric_types", "technical", "Box metrics"),
    ("R2-3", (90, 90), "Whether a G*Power chi-square calculation is appropriate for a deep-learning segmentation model", "power", "technical", "G*Power, chi-square test"),
    ("R2-4", (91, 91), '"YOLOv8 and YOLOv11" inconsistent with the rest of the manuscript', "yolov8", "technical", "YOLOv8"),
    ("R2-5", (92, 92), "1,315 vs 3,403 images and the instance counts in Figure 4", "image_counts", "technical", "3,403 images"),
    ("R2-6", (93, 93), "Image- or patient-level split; data leakage; small test set", "leakage", "technical", "data leakage"),
    ("R2-7", (94, 94), "Single examiner; no inter-rater reliability, so a possible ground-truth bias", "inter_rater", "technical", "inter-rater reliability"),
    ("R2-8", (95, 95), "Gingiva mAP@50 of 0.587 not reported in the text; should be discussed as a limitation", "gingiva_map_low", "technical", "0.587"),
    ("R2-9", (96, 96), "No meaningful performance gain despite tuning, expansion and scaling", "no_improvement", "technical", "lack of meaningful performance gain"),
    ("R2-10", (97, 97), "Lip vs gingiva mAP gap should be investigated quantitatively", "boundary", "technical", "boundary IoU"),
    ("R2-11", (98, 98), "Difference and novelty against the authors' previous study (Ref. 30, J Dent 2026)", "overlap", "clinical", "Çankaya"),
    ("R2-12", (99, 99), "Intra-observer ICC value, 95 % confidence interval and ICC type not given", "intra_icc", "technical", "ICC type"),
]

REVIEWERS = [("Reviewer 1", R1), ("Reviewer 2", R2), ("Reviewer 3", R3), ("Reviewer 4", R4)]

HEADER = """# Reviewer items for the rebuttal (scripts/build_rebuttal.py).
#
# GENERATED by scripts/build_reviewer_items.py from docs/Hakem_Yorumları.docx, the FULL reviewer
# letter. Do not edit by hand: change the line map in that script and re-run it.
#
# `quote` is sliced out of the letter, so it is verbatim by construction. The earlier
# docs/Hakem_revizyonları.docx is an abridged summary (31 items against {n}) and is not consulted;
# where the two disagree the full letter wins.
#
# Numbering is the reviewers' own. Reviewer 2 numbers 1-12 continuously; Reviewer 3 restarts inside
# each section, so its ids carry the section name; Reviewers 1 and 4 wrote continuous prose, split
# into items here with the corresponding passage quoted. `owner`: technical (answered from outputs/)
# or clinical (wording from the clinical team). Several items share an `answer` where the reviewers
# raise the same point; the response document cross-references them.
reviewers:
"""


def letter_lines(docx: Path) -> list[str]:
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf8")
    x = re.sub(r"<w:br[^>]*/>", "\n", xml)
    x = re.sub(r"</w:p>|</w:tc>", "\n", x)
    x = re.sub(r"<w:tab[^>]*/>", " ", x)
    x = html.unescape(re.sub(r"<[^>]+>", "", x))
    return [l.rstrip() for l in x.split("\n") if l.strip()]


def wrap(text: str, indent: str = "          ", width: int = 96) -> str:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(indent + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", default="docs/Hakem_Yorumları.docx")
    ap.add_argument("--out", default="docs/hakem_maddeleri.yaml")
    args = ap.parse_args()
    lines = letter_lines(ROOT / args.docx)
    n_items = sum(len(items) for _, items in REVIEWERS)
    body = [HEADER.format(n=n_items)]
    used: set[int] = set()
    for name, items in REVIEWERS:
        body.append(f"  - name: {name}\n    items:")
        for iid, (a, b), topic, answer, owner, anchor in items:
            quote = " ".join(lines[a - 1:b])
            if anchor not in quote:
                raise SystemExit(f"{iid}: anchor {anchor!r} is not in lines {a}-{b} of the letter — the line map is stale:\n  {quote[:200]}")
            used.update(range(a, b + 1))
            t = topic.replace("'", "''")
            body.append(f"      - id: {iid}\n        topic: '{t}'\n        answer: {answer}\n        owner: {owner}\n"
                        f"        source_lines: {a}-{b}\n        quote: |\n{wrap(quote)}")
        body.append("")
    (ROOT / args.out).write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    print(f"{args.out}: {n_items} items from {len(lines)} lines of {args.docx}")
    for name, items in REVIEWERS:
        print(f"  {name}: {len(items)}")
    skipped = [f"{i}: {lines[i-1][:70]}" for i in range(1, len(lines) + 1) if i not in used]
    print(f"  letter lines not attached to an item: {len(skipped)} (headings and the summary prose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
