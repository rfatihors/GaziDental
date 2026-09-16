#!/usr/bin/env python
"""Stage 4 — expert-agreement analysis (analysis plan §2–3, protocol §5).

Real run:      python scripts/run_expert_analysis.py
               (forms in data/expert/Uzman_{1,2,3}_form.xlsx, model table from outputs/03_oracle
                or --model-table outputs/06_prediction/per_image_results.csv)
Synthetic run: python scripts/run_expert_analysis.py --synthetic
               (forms generated from the real key and the real reference values; same code path)

Outputs (outputs/04_expert/ or --out): form_qc.csv, reference_standard.csv, consensus_pending.csv,
class_agreement.csv, per_class.csv, strata.csv, intra_expert.csv, inter_expert.json,
mm_agreement.csv, tooth_level_long.csv, mixed_models.md, scale_agreement.md,
expert_summary.md, manuscript_numbers.md, OZET.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import input_path, load_config, resolve  # noqa: E402
from gsv4.eval.expert import (  # noqa: E402
    class_agreement, inter_expert, intra_expert, mm_agreement, model_table, reference_standard,
    strata_by_threshold, tooth_long, tooth_mixed_models,
)
from gsv4.io.forms import form_qc_summary, join_key, read_form, split_repeats  # noqa: E402
from gsv4.measure.calibration import offset_mm_at  # noqa: E402


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "(none)"
    cols = list(df.columns)
    int_cols = {c for c in cols if pd.api.types.is_integer_dtype(df[c])}
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in int_cols:
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else fmt.format(v))
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def ci(v, lo, hi, fmt="{:.3f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{fmt.format(v)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--forms-dir", default=None, help="directory with Uzman_{1,2,3}_form.xlsx (default paths.expert_dir)")
    ap.add_argument("--model-table", default=None, help="per_image_results.csv (default outputs/03_oracle)")
    ap.add_argument("--out", default="04_expert")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic forms (and a synthetic model table if none exists)")
    ap.add_argument("--synthetic-p-agree", type=float, default=0.8)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    n_boot = args.n_boot or int(cfg["expert"]["bootstrap_samples"])
    out_dir = resolve(cfg, Path(cfg["paths"]["outputs"]) / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(input_path(cfg, "anonymisation_key_csv"), encoding="utf-8-sig")
    model_path = Path(args.model_table) if args.model_table else resolve(cfg, Path(cfg["paths"]["outputs"]) / "03_oracle" / "per_image_results.csv")
    forms_dir = Path(args.forms_dir) if args.forms_dir else resolve(cfg, cfg["paths"]["expert_dir"])

    if args.synthetic:
        from tests.synth import synthetic_model_output, synthetic_truth, write_synthetic_forms

        forms_dir = out_dir / "synthetic_forms"
        if model_path.exists():
            per = pd.read_csv(model_path)
            truth = pd.DataFrame({"image": per["image"], "true_mean_mm": per["ref_mm"], "true_label": per["ref_label"],
                                  **{f"true_mm_{i}": per[f"ref_mm_{i}"] for i in range(1, 7)}})
            truth["true_class"] = truth["true_label"].map(lambda l: l.split("-")[0] if isinstance(l, str) and l.startswith("E") else "E1")
            truth["true_px_per_mm"] = float(cfg["measurement"]["px_per_mm"] or 16.84) * np.exp(np.random.default_rng(seed).normal(0, 0.05, len(truth)))
            model_source = f"real model table {model_path}"
        else:
            truth = synthetic_truth(key, seed=seed)
            per = synthetic_model_output(truth, seed=seed)
            model_path = out_dir / "synthetic_per_image_results.csv"
            per.to_csv(model_path, index=False)
            model_source = "synthetic model output (no per_image_results.csv found)"
        write_synthetic_forms(forms_dir, key, truth, seed=seed, p_agree=args.synthetic_p_agree, messy=True,
                              per_expert=[{"p_agree": args.synthetic_p_agree}, {"p_agree": args.synthetic_p_agree - 0.1, "class_bias": 1}, {"p_agree": args.synthetic_p_agree + 0.05}])
        data_note = f"SYNTHETIC forms (p_agree ≈ {args.synthetic_p_agree}, expert 2 biased upward, messy cells) written to {forms_dir}; {model_source}."
    else:
        data_note = f"real forms from {forms_dir}; model table {model_path}"
        if not model_path.exists():
            raise SystemExit(f"model table not found: {model_path}")
    per = pd.read_csv(model_path)
    per["image"] = per["image"].astype(str)

    # ---- forms
    forms, first, pairs, qc_rows = {}, {}, {}, []
    for e in (1, 2, 3):
        p = forms_dir / f"Uzman_{e}_form.xlsx"
        if not p.exists():
            raise SystemExit(f"form not found: {p}")
        f = join_key(read_form(p, cfg["expert"]), key)
        forms[e] = f
        first[e], pairs[e] = split_repeats(f)
        qc = form_qc_summary(f)
        qc_rows.append({"expert": e, **{k: v for k, v in qc.items() if k != "flag_counts"}, "flags": "; ".join(f"{k}: {v}" for k, v in qc["flag_counts"].items())})
    form_qc = pd.DataFrame(qc_rows)
    form_qc.to_csv(out_dir / "form_qc.csv", index=False)
    pd.concat([f.assign(expert=e) for e, f in forms.items()]).to_csv(out_dir / "forms_long.csv", index=False)

    # ---- subsets
    splits_path = resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "splits.json")
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {"by_split": {"test": []}}
    all_images = [i for i in key.loc[key["tekrar"] == 0, "orijinal"].astype(str) if i in set(per["image"])]
    test_images = [u.split("/", 1)[1] for u in splits["by_split"].get("test", []) if u.startswith("high/")]
    test_images = [i for i in test_images if i in all_images]
    # Pre-specified (12 Sep 2026, before real forms): primary = all measured images (Stage 6:
    # 5-fold out-of-fold predicted masks), secondary = the fixed test subset.
    subsets = {f"all {len(all_images)} images, OOF (primary)": all_images, "fixed test subset (secondary)": test_images}
    prespec_note = f"""## Pre-specification note
The protocol originally named the fixed test subset (n = {len(test_images)}) as the primary set for the class-agreement analysis and all {len(all_images)} images as secondary. This was reversed **before any real expert form was available**, on the basis of the synthetic dry run: with n = {len(test_images)} the bootstrap 95 % CI of the linear-weighted κ spanned roughly −0.15 to 0.84, i.e. the primary estimate would have been uninformative. The 5-fold out-of-fold predictions are equally unbiased (every image is predicted by a model that never saw it or its same-patient twin), so the primary set is now all {len(all_images)} reference images with OOF predicted masks; the fixed test subset (final model) is reported as secondary. The same rule applies in Stage 6. In this run the model table is: {'GT masks / synthetic dry run' if args.synthetic else str(model_path)}."""

    # ---- expert side
    consensus_path = forms_dir / "consensus.csv"
    consensus = pd.read_csv(consensus_path) if consensus_path.exists() else None
    reference = reference_standard(first, consensus)
    reference.to_csv(out_dir / "reference_standard.csv", index=False)
    pending = reference[reference["reference_kind"] == "consensus_pending"]
    pending.to_csv(out_dir / "consensus_pending.csv", index=False)
    clinical_ref = per.set_index("image")["ref_mm"] if "ref_mm" in per.columns else None
    inter = inter_expert(first, clinical_ref, all_images, n_boot, seed)
    expert_mean_mm, expert_scale = inter.pop("expert_mean_mm"), inter.pop("expert_scale_mean")
    (out_dir / "inter_expert.json").write_text(json.dumps({k: (v if not isinstance(v, dict) else v) for k, v in inter.items()}, indent=1, default=float), encoding="utf-8")
    intra = intra_expert(pairs, n_boot, seed)
    intra.to_csv(out_dir / "intra_expert.csv", index=False)

    # ---- model side: global scale uncorrected = PRIMARY; corrected (config offset_px, converted with
    # the scale in use) = secondary; expert per-image scale (uncorrected and corrected) = secondary.
    # the mask-level correction lives in the Stage-6 table (<method>_px_corrected columns); a Stage-3
    # table (ground-truth masks) has none, so corrected = uncorrected there and the note says so
    model = model_table(per, expert_scale)
    predicted_table = bool(model["model_has_correction"].iloc[0])
    offset_px = float(model["model_bottom_edge_offset_px"].iloc[0])
    scales = ["global", "global_corrected"] + (["expert", "expert_corrected"] if model["model_mm_expert"].notna().sum() >= 10 else [])
    scale_note = (("" if predicted_table else "**Model table comes from ground-truth masks (Stage 3 / synthetic dry run): the post-hoc mask-level correction exists only for predicted masks, so the corrected columns equal the uncorrected ones here.** ")
                  + f"Model scales: `global` = uncorrected, global {cfg['measurement']['px_per_mm']} px/mm (PRIMARY); `global_corrected` = the mask-level lower-edge correction of Stage 6 "
                  f"(config `bottom_edge_offset_px` = {offset_px:+.0f} px; re-measured pixel values `<method>_px_corrected`, {offset_mm_at(float(cfg['measurement']['px_per_mm']), offset_px):+.2f} mm at the global scale; secondary); "
                  f"`expert` / `expert_corrected` = the experts' mean per-image scale applied to the uncorrected / corrected pixel values (secondary). "
                  "Because the correction is in pixels at mask level, no mm constant is involved and the per-image scale applies unchanged.")
    cls, per_class = class_agreement(model, reference, subsets, n_boot, seed, scales=scales)
    cls.to_csv(out_dir / "class_agreement.csv", index=False)
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    strata = strata_by_threshold(model, reference, all_images)
    strata.to_csv(out_dir / "strata.csv", index=False)
    mm_rows = pd.concat([mm_agreement(model, first, expert_mean_mm, clinical_ref, all_images, s) for s in scales])
    mm_rows.to_csv(out_dir / "mm_agreement.csv", index=False)
    long = tooth_long(model, first, all_images, "global")
    long.to_csv(out_dir / "tooth_level_long.csv", index=False)
    mixed = tooth_mixed_models(long)
    long_c = tooth_long(model, first, all_images, "global_corrected")
    long_c.to_csv(out_dir / "tooth_level_long_corrected.csv", index=False)
    mixed_c = tooth_mixed_models(long_c)

    # frame-outside subset: does the expert scale close the gap to the clinical reference?
    frame_txt = ""
    if "frame_ok" in per.columns and "expert" in scales:
        m2 = model.set_index("image")
        rows = []
        for name, mask in (("inside 2698x1799", m2["frame_ok"].astype(bool)), ("outside 2698x1799", ~m2["frame_ok"].astype(bool))):
            part = m2[mask].dropna(subset=["model_mm_global", "model_mm_expert", "ref_mm"])
            rows.append({"frame": name, "n": len(part), "mae_global_scale": float(np.abs(part["model_mm_global"] - part["ref_mm"]).mean()),
                         "mae_expert_scale": float(np.abs(part["model_mm_expert"] - part["ref_mm"]).mean()),
                         "mean_expert_px_per_mm": float(part["expert_px_per_mm"].mean())})
        frame_df = pd.DataFrame(rows)
        frame_df.to_csv(out_dir / "frame_scale_comparison.csv", index=False)
        frame_txt = md_table(frame_df)

    # ---- reports
    mixed_md = "# Tooth-level mixed models (model − expert mean, per tooth; random intercept per patient)\n\nUncorrected model values (PRIMARY) first, then the corrected values (secondary, `global_corrected`).\n\n"
    for r in [dict(m, formula=m["formula"] + "  [uncorrected, PRIMARY]") for m in mixed] + [dict(m, formula=m["formula"] + "  [corrected, secondary]") for m in mixed_c]:
        if "error" in r:
            mixed_md += f"`{r['formula']}`: failed — {r['error']}\n\n"
            continue
        fe = r["fixed_effects"].reset_index().rename(columns={"index": "term"})
        mixed_md += (f"## `{r['formula']}`  (n_obs = {r['n_obs']}, patients = {r['n_groups']}, estimator = {r['estimator']}, converged = {r['converged']})\n\n"
                     + (f"Note: {r['note']}\n\n" if r.get('note') else "") + f"{md_table(fe)}\n\n"
                     f"variance: between-patient {r['var_patient']:.4f}, residual {r['var_resid']:.4f}; derived tooth-level ICC(patient) = {r['icc_patient']:.3f}; AIC {r['aic']:.1f}\n\n")
    mixed_md += ("`alignment_uncertain` = images where method C could not place the six zeniths and equal-split regions were used; "
                 "its fixed effect tests whether tooth-level disagreement is larger on those images.\n")
    (out_dir / "mixed_models.md").write_text(mixed_md, encoding="utf-8")

    scale_md = f"""# Expert per-image scale (Ölçek, pixel/mm) — agreement and use

Experts' mean per-image scale: n = {inter.get('scale_n', 0)}, mean {inter.get('scale_mean', float('nan')):.2f} px/mm (SD between images {inter.get('scale_sd_between_images', float('nan')):.2f}); global oracle scale {cfg['measurement']['px_per_mm']} px/mm.
Agreement among the three experts on the scale: ICC(2,1) {ci(inter.get('scale_icc2_1'), inter.get('scale_icc2_1_ci_low'), inter.get('scale_icc2_1_ci_high'))}; within-image CV median {inter.get('scale_cv_within_image_median', float('nan')):.3f} (mean {inter.get('scale_cv_within_image_mean', float('nan')):.3f}). This directly measures the precision of probe-based calibration in these photographs.
Intra-expert scale repeatability (20 repeats): {', '.join(f"expert {int(r['expert'])}: ICC {r.get('scale_icc2_1', float('nan')):.3f}, CV {r.get('scale_cv_repeat', float('nan')):.3f}" for _, r in intra.iterrows())}.

{scale_note} All four appear in class_agreement.csv and mm_agreement.csv.

## Frame-outside images: does the per-image expert scale close the gap to the clinical reference?
{frame_txt or '(expert scale not available)'}
"""
    (out_dir / "scale_agreement.md").write_text(scale_md, encoding="utf-8")

    prim = cls[(cls["scale"] == "global") & (cls["subset"].str.contains("primary")) & (cls["scoring"] == "strict")]
    prim = prim.iloc[0] if len(prim) else None
    sec = cls[(cls["scale"] == "global") & (cls["subset"].str.contains("secondary")) & (cls["scoring"] == "strict")]
    sec = sec.iloc[0] if len(sec) else None
    mm_glob = mm_rows[(mm_rows["scale"] == "global")].set_index("comparator")
    mm_glob_c = mm_rows[(mm_rows["scale"] == "global_corrected")].set_index("comparator")

    def num(label, v, lo, hi, n, fmt="{:.3f}"):
        if v is None or pd.isna(v):
            return f"| {label} | n/a | n/a | {n} |"
        return f"| {label} | {fmt.format(v)} | [{fmt.format(lo)}, {fmt.format(hi)}] | {n} |"

    numbers = ["# Numbers for the manuscript (value, 95 % CI, n)", "", "| metric | value | 95 % CI | n |", "|---|---|---|---|"]
    if prim is not None:
        numbers += [num("Model vs expert majority, linear-weighted κ (primary: all images, OOF, strict)", prim["kappa_linear"], prim["kappa_linear_ci_low"], prim["kappa_linear_ci_high"], int(prim["n"])),
                    num("… unweighted κ", prim["kappa_unweighted"], prim["kappa_unweighted_ci_low"], prim["kappa_unweighted_ci_high"], int(prim["n"])),
                    num("… observed agreement", prim["observed_agreement"], prim["observed_agreement_ci_low"], prim["observed_agreement_ci_high"], int(prim["n"])),
                    num("… PABAK", prim["pabak"], prim["pabak_ci_low"], prim["pabak_ci_high"], int(prim["n"]))]
    if sec is not None:
        numbers += [num("Model vs expert majority, linear-weighted κ (secondary: fixed test subset, strict)", sec["kappa_linear"], sec["kappa_linear_ci_low"], sec["kappa_linear_ci_high"], int(sec["n"]))]
    prim_c = cls[(cls["scale"] == "global_corrected") & (cls["subset"].str.contains("primary")) & (cls["scoring"] == "strict")]
    if len(prim_c):
        r = prim_c.iloc[0]
        numbers += [num(f"… corrected model values (offset_px {offset_px:+.0f}, secondary), linear-weighted κ (primary set, strict)", r["kappa_linear"], r["kappa_linear_ci_low"], r["kappa_linear_ci_high"], int(r["n"]))]
    len_row = cls[(cls["scale"] == "global") & (cls["subset"].str.contains("primary")) & (cls["scoring"] == "lenient")]
    if len(len_row):
        r = len_row.iloc[0]
        numbers += [num("… lenient scoring, linear-weighted κ (primary set)", r["kappa_linear"], r["kappa_linear_ci_low"], r["kappa_linear_ci_high"], int(r["n"]))]
    numbers += [num("Inter-expert Fleiss κ (class)", inter.get("fleiss_kappa"), inter.get("fleiss_kappa_ci_low"), inter.get("fleiss_kappa_ci_high"), inter.get("n_class", 0))]
    if "mm_icc2_1" in inter:
        numbers += [num("Inter-expert ICC(2,1), image-mean mm (3 experts)", inter["mm_icc2_1"], inter["mm_icc2_1_ci_low"], inter["mm_icc2_1_ci_high"], inter["mm_n"]),
                    num("Inter-expert ICC(2,k), image-mean mm (3 experts)", inter["mm_icc2_k"], inter["mm_icc2_k_ci_low"], inter["mm_icc2_k_ci_high"], inter["mm_n"])]
    if "mm4_icc2_1" in inter:
        numbers += [num("ICC(2,1), 3 experts + clinical reference", inter["mm4_icc2_1"], inter["mm4_icc2_1_ci_low"], inter["mm4_icc2_1_ci_high"], inter["mm4_n"])]
    for comp, label in (("expert_mean", "Model vs expert mean, ICC(2,1) mm"), ("clinical_reference", "Model vs clinical reference, ICC(2,1) mm")):
        for tbl, tag in ((mm_glob, "uncorrected, PRIMARY"), (mm_glob_c, "corrected, secondary")):
            if comp in tbl.index and pd.notna(tbl.at[comp, "icc2_1"]):
                r = tbl.loc[comp]
                numbers += [num(f"{label} — {tag}", r["icc2_1"], r["icc2_1_ci_low"], r["icc2_1_ci_high"], int(r["n"])),
                            num(f"… bias (model − {comp.replace('_', ' ')}), mm — {tag}", r["bias"], r["bias_ci_low"], r["bias_ci_high"], int(r["n"]), "{:+.2f}")]
    for _, r in intra.iterrows():
        numbers += [num(f"Intra-expert κ (linear), expert {int(r['expert'])}", r.get("kappa_linear"), r.get("kappa_linear_ci_low"), r.get("kappa_linear_ci_high"), int(r.get("n_class_pairs", 0))),
                    num(f"Intra-expert ICC(2,1) image mm, expert {int(r['expert'])}", r.get("icc2_1_image"), r.get("icc2_1_image_ci_low"), r.get("icc2_1_image_ci_high"), int(r.get("n_mm_pairs", 0)))]
    if "scale_icc2_1" in inter:
        numbers += [num("Expert scale agreement ICC(2,1), px/mm", inter["scale_icc2_1"], inter["scale_icc2_1_ci_low"], inter["scale_icc2_1_ci_high"], inter["scale_n"])]
    for r in mixed:
        if "error" not in r and r["formula"].startswith("diff ~ 1 "):
            fe = r["fixed_effects"].loc["Intercept"]
            numbers += [num("Tooth-level bias (mixed model intercept), mm — uncorrected (PRIMARY)", fe["estimate"], fe["ci_low"], fe["ci_high"], r["n_obs"], "{:+.3f}")]
    for r in mixed_c:
        if "error" not in r and r["formula"].startswith("diff ~ 1 "):
            fe = r["fixed_effects"].loc["Intercept"]
            numbers += [num("Tooth-level bias (mixed model intercept), mm — corrected (secondary)", fe["estimate"], fe["ci_low"], fe["ci_high"], r["n_obs"], "{:+.3f}")]
    numbers_md = "\n".join(numbers) + "\n\nE4 has no reference case in this dataset (per_class.csv shows n = 0); the E4/T4 branch is not validated.\n\n" + prespec_note + "\n"
    (out_dir / "manuscript_numbers.md").write_text(numbers_md, encoding="utf-8")

    pc_prim = per_class[(per_class["scale"] == "global") & (per_class["subset"].str.contains("primary"))][["class", "n_reference", "n_predicted", "sensitivity", "sensitivity_ci_low", "sensitivity_ci_high", "specificity", "specificity_ci_low", "specificity_ci_high"]]
    summary = f"""# Expert-agreement analysis — summary

Data: {data_note}

{prespec_note}

## Forms and quality control
{md_table(form_qc)}
Rows are never dropped: empty rows, blank teeth, zero values, missing confidence and unparsable cells are flagged in `forms_long.csv` (`form_flags`) and excluded only from the statistic that needs the missing field.

## Reference standard
Majority of three primary classes; unanimous {int((reference['reference_kind'] == 'unanimous').sum())}, majority {int((reference['reference_kind'] == 'majority').sum())}, consensus (blinded) {int((reference['reference_kind'] == 'consensus').sum())}, **consensus pending {len(pending)}** (excluded from the primary analysis; `consensus_pending.csv`; drop a `consensus.csv` with columns image,class into the forms directory to resolve), insufficient votes {int((reference['reference_kind'] == 'insufficient_votes').sum())}.

## Model vs expert reference — class (model = selected method mm + rule engine)
{scale_note}
Scoring: strict = model's first candidate; lenient = agreement if the expert class is among the model's candidates; lenient2 = also the expert's second candidate (reported only). Bootstrap CIs: {n_boot} resamples, seed {seed}.

{md_table(cls[["scale", "subset", "scoring", "n", "n_consensus_pending_excluded", "n_model_unclassified", "kappa_linear", "kappa_linear_ci_low", "kappa_linear_ci_high", "kappa_unweighted", "observed_agreement", "pabak"]])}

### Per class (primary set, strict, global scale; counts and Wilson 95 % CIs)
{md_table(pc_prim)}

### Disagreements by distance of the model value to the nearest clinical threshold (3, 4, 6, 8 mm)
{md_table(strata)}

## Inter-expert agreement
Fleiss κ (class, n = {inter.get('n_class')}): {ci(inter.get('fleiss_kappa'), inter.get('fleiss_kappa_ci_low'), inter.get('fleiss_kappa_ci_high'))}; pairwise linear κ: {inter.get('pairwise_kappa_linear')}.
Image-mean mm, 3 experts: ICC(2,1) {ci(inter.get('mm_icc2_1'), inter.get('mm_icc2_1_ci_low'), inter.get('mm_icc2_1_ci_high'))}, ICC(2,k) {ci(inter.get('mm_icc2_k'), inter.get('mm_icc2_k_ci_low'), inter.get('mm_icc2_k_ci_high'))} (n = {inter.get('mm_n')}).
With the clinical reference observer as 4th rater: ICC(2,1) {ci(inter.get('mm4_icc2_1'), inter.get('mm4_icc2_1_ci_low'), inter.get('mm4_icc2_1_ci_high'))}, ICC(2,k) {ci(inter.get('mm4_icc2_k'), inter.get('mm4_icc2_k_ci_low'), inter.get('mm4_icc2_k_ci_high'))}.

## Intra-expert agreement (20 repeated images)
{md_table(intra)}

## Model vs experts — millimetres
{md_table(mm_rows)}

Tooth-level analysis: `mixed_models.md` (random intercept per patient; tooth position and `alignment_uncertain` as fixed effects). Scale comparison: `scale_agreement.md`. Numbers for the manuscript: `manuscript_numbers.md`.
"""
    (out_dir / "expert_summary.md").write_text(summary, encoding="utf-8")
    ozet = f"""# Aşama 4 — Türkçe özet

- Veri: {data_note}
- Formlar: satır düşürülmedi; uzman başına boş satır {form_qc['n_rows_empty'].tolist()}, eksik sınıf {form_qc['n_class_missing'].tolist()}, eksik ölçek {form_qc['n_scale_missing'].tolist()}.
- Referans: çoğunluk; konsensüs bekleyen {len(pending)}.
- Birincil (145 görüntü, OOF, katı, global ölçek): doğrusal ağırlıklı κ {ci(prim['kappa_linear'], prim['kappa_linear_ci_low'], prim['kappa_linear_ci_high']) if prim is not None else 'n/a'}, n = {int(prim['n']) if prim is not None else 0}. İkincil (sabit test): κ {ci(sec['kappa_linear'], sec['kappa_linear_ci_low'], sec['kappa_linear_ci_high']) if sec is not None else 'n/a'}.
- Uzmanlar arası Fleiss κ {ci(inter.get('fleiss_kappa'), inter.get('fleiss_kappa_ci_low'), inter.get('fleiss_kappa_ci_high'))}; mm ICC(2,1) {inter.get('mm_icc2_1', float('nan')):.3f}; ölçek ICC(2,1) {inter.get('scale_icc2_1', float('nan')):.3f}, görüntü içi CV medyan {inter.get('scale_cv_within_image_median', float('nan')):.3f}.
- Model–uzman ortalaması mm (düzeltmesiz, birincil): ICC(2,1) {mm_glob.at['expert_mean', 'icc2_1'] if 'expert_mean' in mm_glob.index else float('nan'):.3f}, sapma {mm_glob.at['expert_mean', 'bias'] if 'expert_mean' in mm_glob.index else float('nan'):+.2f} mm; düzeltilmiş (ikincil, offset_px {offset_px:+.0f}): ICC {mm_glob_c.at['expert_mean', 'icc2_1'] if 'expert_mean' in mm_glob_c.index else float('nan'):.3f}, sapma {mm_glob_c.at['expert_mean', 'bias'] if 'expert_mean' in mm_glob_c.index else float('nan'):+.2f} mm.
- Karma modeller: {len([r for r in mixed if 'error' not in r])}/3 kuruldu; {len([r for r in mixed if r.get('estimator', '').startswith('MixedLM')])} tanesi MixedLM, gerisi sınır durumu (hasta varyansı ≈ 0) → küme-dayanıklı OLS.
- Üretilen dosyalar: form_qc.csv, forms_long.csv, reference_standard.csv, consensus_pending.csv, class_agreement.csv, per_class.csv, strata.csv, intra_expert.csv, inter_expert.json, mm_agreement.csv, tooth_level_long.csv, mixed_models.md, scale_agreement.md, frame_scale_comparison.csv, expert_summary.md, manuscript_numbers.md.
"""
    (out_dir / "OZET.md").write_text(ozet, encoding="utf-8")
    print(ozet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
