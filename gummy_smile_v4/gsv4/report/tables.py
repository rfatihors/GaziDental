"""Manuscript tables (Stage 7) as CSV + Markdown. Each builder returns
``(DataFrame or None, status dict)``; missing inputs give a pending status."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
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


def dataset_counts(manifest: pd.DataFrame, splits: Dict[str, Any], pairs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for g, part in manifest.groupby("group"):
        kept = part[part["keep"]]
        drop = part[~part["keep"]]["drop_reason"].str.split(":").str[0].value_counts()
        rec = {"group": g, "images_roboflow_export": len(part), "no_reference_measurement": int(drop.get("no_reference_measurement", 0)),
               "same_patient_duplicate": int(drop.get("duplicate_of", 0)), "ambiguous_name": int(drop.get("row_ambiguous", 0)),
               "kept": len(kept), "with_clinical_reference": int(kept["has_reference_measurement"].sum())}
        for s in ("train", "valid", "test"):
            rec[s] = int((kept["split"] == s).sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    tot = {c: int(df[c].sum()) for c in df.columns if c != "group"}
    df = pd.concat([df, pd.DataFrame([{"group": "total", **tot}])], ignore_index=True)
    meta = {"same_patient_pairs_detected": int(len(pairs)), "pairs_cross_split_in_original_export": int(pairs["cross_split"].astype(str).str.lower().eq("true").sum()) if "cross_split" in pairs.columns else None,
            "pairs_involving_original_test": int(pairs["involves_test"].astype(str).str.lower().eq("true").sum()) if "involves_test" in pairs.columns else None,
            "seed": splits.get("seed"), "cv_folds": splits.get("cv_folds", {}).get("n_folds")}
    return df, {"status": "done", "source": "data/manifest", **meta}


def demographics(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    k = manifest[manifest["keep"]]
    rows = []
    for g, part in list(k.groupby("group")) + [("all", k)] + [("reference set (145)", k[k["has_reference_measurement"]])]:
        rows.append({"group": g, "n": len(part), "age_recorded": int(part["age"].notna().sum()), "age_mean": part["age"].mean(), "age_sd": part["age"].std(),
                     "age_min": part["age"].min(), "age_max": part["age"].max(), "sex_recorded": int(part["sex"].notna().sum()),
                     "female": int((part["sex"] == "F").sum()), "male": int((part["sex"] == "M").sum()),
                     "female_pct_of_recorded": 100 * (part["sex"] == "F").sum() / max(1, part["sex"].notna().sum())})
    return pd.DataFrame(rows), {"status": "done", "source": "data/manifest/dataset_manifest.csv", "note": "age/sex recorded for part of the cohort only; reported on the n with a record"}


def measurement_accuracy(oracle_dir: Path, prediction_dir: Optional[Path]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    est = oracle_dir / "estimator_comparison.csv"
    if not est.exists():
        return None, {"status": "pending", "needs": str(est)}
    e = pd.read_csv(est).set_index("combo")
    sens = pd.read_csv(oracle_dir / "sensitivity.csv") if (oracle_dir / "sensitivity.csv").exists() else pd.DataFrame()
    summary = (oracle_dir / "oracle_summary.md").read_text() if (oracle_dir / "oracle_summary.md").exists() else ""
    sel = summary.split("## Selected method: **")[1].split("**")[0] if "## Selected method: **" in summary else e["mae_dev"].idxmin()
    rows = []

    def row(label, r, n, extra=None):
        rec = {"analysis": label, "n": int(n), "px_per_mm": r.get("px_per_mm_dev", np.nan), "mae_mm": r["mae_holdout"] if "mae_holdout" in r else r["mae"],
               "rmse_mm": r["rmse_holdout"] if "rmse_holdout" in r else r["rmse"], "r": r["r_holdout"] if "r_holdout" in r else r["r"],
               "icc2_1": r["icc2_1_holdout"] if "icc2_1_holdout" in r else r["icc2_1"],
               "icc2_1_ci_low": r["icc2_1_ci_low_holdout"] if "icc2_1_ci_low_holdout" in r else r["icc2_1_ci_low"],
               "icc2_1_ci_high": r["icc2_1_ci_high_holdout"] if "icc2_1_ci_high_holdout" in r else r["icc2_1_ci_high"],
               "bias_mm": r["ba_bias_holdout"] if "ba_bias_holdout" in r else r["ba_bias"],
               "loa_low_mm": r["ba_loa_low_holdout"] if "ba_loa_low_holdout" in r else r["ba_loa_low"],
               "loa_high_mm": r["ba_loa_high_holdout"] if "ba_loa_high_holdout" in r else r["ba_loa_high"]}
        rec.update(extra or {})
        return rec

    rows.append(row(f"GT masks, holdout, {sel} (selected)", e.loc[sel], e.loc[sel, "n_holdout"]))
    if "A_p25" in e.index and sel != "A_p25":
        rows.append(row("GT masks, holdout, A_p25 (sensitivity: no fallback)", e.loc["A_p25"], e.loc["A_p25", "n_holdout"]))
    for _, s in sens.iterrows():
        if s["subset"].startswith("dev + holdout") or "only 2698" in s["subset"] or "without dash" in s["subset"]:
            rows.append(row(f"GT masks, {s['subset']}", s, s["n"], {"px_per_mm": e.loc[sel, "px_per_mm_dev"]}))
    status = {"status": "done", "source": str(oracle_dir), "selected_method": sel}
    acc6 = prediction_dir / "measurement_accuracy.csv" if prediction_dir is not None else None
    if acc6 is not None and acc6.exists():
        # Stage 6 (scripts/run_prediction_eval.py): method and scale fixed from Stage 3, nothing re-fitted
        pa = pd.read_csv(acc6)
        wanted = [("(a) OOF masks, all reference images", f"Predicted masks (OOF, fold models), all reference images, {sel} (PRIMARY)"),
                  ("(a) OOF, Stage-3 holdout", f"Predicted masks (OOF), Stage-3 holdout images, {sel}"),
                  ("(b) final model masks", f"Predicted masks (final model), test-set high images, {sel} (secondary)")]
        for prefix, label in wanted:
            hit = pa[pa["set"].str.startswith(prefix)]
            if len(hit) and pd.notna(hit.iloc[0].get("mae", np.nan)):
                r6 = hit.iloc[0]
                rows.append(row(label, r6, r6["n"], {"px_per_mm": e.loc[sel, "px_per_mm_dev"], "kappa_linear": r6.get("threshold_kappa_linear", np.nan)}))
        status["prediction_rows"] = "done"
        status["prediction_source"] = str(acc6)
    else:
        rows.append({"analysis": "Predicted masks (OOF, 145 images) — PENDING: Stage 6 (scripts/run_prediction_eval.py)", "n": 0})
        status["prediction_rows"] = "pending"
    df = pd.DataFrame(rows)
    # precision-based sample-size justification with the observed numbers (Bonett 2002; Bland & Altman 1999)
    r0 = rows[0]
    n, rho, sd = r0["n"], r0["icc2_1"], (r0["loa_high_mm"] - r0["loa_low_mm"]) / (2 * 1.96)
    n145 = 145
    se_icc = np.sqrt(2 * (1 - rho) ** 2 * (1 + rho) ** 2 / (2 * (n145 - 1)))
    status["precision_note"] = (f"With n = {n145} reference images, an ICC of {rho:.2f} has a 95 % CI half-width of ≈ {1.96 * se_icc:.3f} (Bonett 2002, k = 2); "
                                f"each Bland–Altman limit of agreement has a half-width of ≈ {1.96 * sd * np.sqrt(3 / n145):.2f} mm for the observed between-method SD of {sd:.2f} mm (Bland & Altman 1999).")
    return df, status


def segmentation_metrics(pred_dir: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    p = pred_dir / "test_metrics.json"
    if not p.exists():
        return None, {"status": "pending", "needs": f"{p} (evaluate_test on the workstation)"}
    m = json.loads(p.read_text())
    if "per_class" not in m:
        # the workstation ran the boundary stage only (--metrics-only); validation metrics are still missing
        return None, {"status": "pending", "needs": f"{p} without a per_class block — re-run gsv4.train.evaluate_test on the workstation (validation pass)",
                      "boundary_by_group": m.get("boundary_by_group")}
    rows = []
    for cname, vals in m["per_class"].items():
        rows.append({"class": cname, **vals})
    df = pd.DataFrame(rows)
    st = {"status": "done", "source": str(p), "n_test_images": m.get("n_images"), "weights": m.get("weights")}
    if "boundary_by_group" in m:
        st["boundary_by_group"] = m["boundary_by_group"]
    return df, st


def learning_curve(pred_dir: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    p = pred_dir / "learning_curve.csv"
    if not p.exists() or "done" not in pd.read_csv(p).columns or not pd.read_csv(p)["done"].all():
        return None, {"status": "pending", "needs": f"{p} with all four points trained (learning_curve --collect on the workstation)"}
    df = pd.read_csv(p)
    verdict = ""
    md = pred_dir / "learning_curve.md"
    if md.exists():
        verdict = md.read_text().split("→ **")[1].split("**")[0] if "→ **" in md.read_text() else ""
    return df, {"status": "done", "source": str(p), "verdict": verdict}


def expert_agreement(expert_dir: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    summ = expert_dir / "expert_summary.md"
    if not summ.exists():
        return None, {"status": "pending", "needs": "expert forms in data/expert + run_expert_analysis.py"}
    text = summ.read_text()
    if "SYNTHETIC" in text:
        return None, {"status": "pending", "needs": "real expert forms (current outputs are a synthetic dry run)", "dry_run": str(summ)}
    nums = expert_dir / "manuscript_numbers.md"
    rows = []
    for line in nums.read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| metric") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 4:
                rows.append({"metric": cells[0], "value": cells[1], "ci95": cells[2], "n": cells[3]})
    return pd.DataFrame(rows), {"status": "done", "source": str(nums)}
