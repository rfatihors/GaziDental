"""Manuscript tables (Stage 7) as CSV + Markdown. Each builder returns
``(DataFrame or None, status dict)``; missing inputs give a pending status."""
from __future__ import annotations

import json
import re
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
        has_corr = bool(pa["set"].astype(str).str.contains("— corrected").any())
        prim_note = "uncorrected (PRIMARY)" if has_corr else "PRIMARY (no post-hoc correction)"
        wanted = [("(a) OOF masks, all reference images [PRIMARY]", f"Predicted masks (OOF, fold models), all reference images, {sel} — {prim_note}"),
                  ("(a) OOF masks, all reference images [PRIMARY] — corrected", f"Predicted masks (OOF, fold models), all reference images, {sel} — corrected, mask-level lower edge (secondary)"),
                  ("(a) OOF, Stage-3 holdout images only (scale never fitted on these)", f"Predicted masks (OOF), Stage-3 holdout images, {sel}"),
                  ("(a) OOF, Stage-3 holdout images only (scale never fitted on these) — corrected", f"Predicted masks (OOF), Stage-3 holdout images, {sel} — corrected"),
                  ("(b) final model masks, test high images [secondary set]", f"Predicted masks (final model), test-set high images, {sel}"),
                  ("(b) final model masks, test high images [secondary set] — corrected", f"Predicted masks (final model), test-set high images, {sel} — corrected")]
        for name, label in wanted:
            hit = pa[pa["set"] == name]
            if len(hit) and pd.notna(hit.iloc[0].get("mae", np.nan)):
                r6 = hit.iloc[0]
                rows.append(row(label, r6, r6["n"], {"px_per_mm": e.loc[sel, "px_per_mm_dev"], "correction": r6.get("correction", ""),
                                                     "model": r6.get("model", ""), "masks_dir": r6.get("masks_dir", ""),
                                                     "kappa_linear": r6.get("threshold_kappa_linear", np.nan), "images_with_zeroed_columns": r6.get("images_with_zeroed_columns", np.nan)}))
        status["prediction_rows"] = "done"
        status["prediction_source"] = str(acc6)
        models = sorted({m for m in pa.get("model", pd.Series(dtype=str)).dropna().unique() if "COCO annotations" not in str(m)})
        status["prediction_model"] = ", ".join(models) if models else "not recorded"
        status["post_hoc_correction"] = "yes (secondary rows)" if has_corr else "none — single result set"
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


def segmentation_metrics(pred_dir: Path, stage6_dir: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Per-class detection/segmentation metrics of the final model, from ITS OWN evaluator.

    ``stage6_dir/segmentation_metrics.json`` is written by scripts/run_prediction_eval.py and carries
    the model, the mask directory, the evaluator and the source file; it is preferred over the
    convention-based path so that another model's numbers can never be picked up by accident.
    """
    prov = stage6_dir / "segmentation_metrics.json" if stage6_dir is not None else None
    if prov is not None and prov.exists():
        block = json.loads(prov.read_text())
        if not block.get("available"):
            return None, {"status": "pending", "model": block.get("model"), "evaluator": block.get("evaluator"),
                          "needs": f"{block.get('source')} — the final model's own evaluation has not run "
                                   "(rfdetr_train_predict.py --predict-only --evaluate for RF-DETR, "
                                   "gsv4.train.evaluate_test for YOLO); no other model's metrics are substituted"}
        m = block["metrics"]
        if "per_class" not in m:                      # a flat COCO block: one row per metric
            flat = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            df = pd.DataFrame({"metric": list(flat), "value": list(flat.values())})
            return df, {"status": "done", "source": block.get("source"), "model": block.get("model"),
                        "evaluator": block.get("evaluator"), "masks": block.get("masks"),
                        "note": "this framework's own COCO evaluation; not row-comparable with another framework's mAP"}
        p, extra = Path(str(block.get("source"))), {"model": block.get("model"), "evaluator": block.get("evaluator"), "masks": block.get("masks")}
    else:
        p, extra = pred_dir / "test_metrics.json", {"evaluator": "Ultralytics val()"}
        if not p.exists():
            return None, {"status": "pending", "needs": f"{p} (evaluate_test on the workstation)", **extra}
        m = json.loads(p.read_text())
    if "per_class" not in m:
        # the workstation ran the boundary stage only (--metrics-only); validation metrics are still missing
        return None, {"status": "pending", "needs": f"{p} without a per_class block — re-run gsv4.train.evaluate_test on the workstation (validation pass)",
                      "boundary_by_group": m.get("boundary_by_group")}
    from gsv4.train.evaluate_test import reported_metrics

    reported, settings = reported_metrics(m)
    label = {"standard": "standard (reported mAP)", "operating_point_legacy": "operating point (legacy file, see note)"}.get(
        settings.get("kind", ""), str(settings.get("kind", "unknown")))
    blocks = [(label, reported, settings)]
    if "per_class_operating_point" in m:
        blocks.append(("operating point (pipeline)", m["per_class_operating_point"], m.get("eval_settings_operating_point", {})))
    rows = []
    for label, block, st_ in blocks:
        for cname, vals in block.items():
            rows.append({"settings": label, "conf": st_.get("conf"), "max_det": st_.get("max_det"), "class": cname, **vals})
    df = pd.DataFrame(rows)
    st = {"status": "done", "source": str(p), **extra, "n_test_images": m.get("n_images"), "weights": m.get("weights"),
          "settings_of_the_reported_map": settings.get("kind"),
          "note": ("mAP is reported at the standard evaluation settings (conf 0.001, NMS IoU 0.7, max_det 300); "
                   "the operating-point rows are the configuration the measurement pipeline runs at"
                   if settings.get("kind") == "standard" else
                   "WARNING: this file predates the two-settings evaluation, so the numbers below were computed at the "
                   "pipeline operating point (conf 0.25, max_det 20), which truncates the precision-recall curve and "
                   "understates mAP — re-run gsv4.train.evaluate_test to get the standard-settings figures")}
    if "boundary_by_group" in m:
        st["boundary_by_group"] = m["boundary_by_group"]
    return df, st


def segmentation_metrics_facts(tab: Path) -> Dict[str, Any]:
    """The test-set segmentation table in whichever shape Stage 7 wrote it, plus its provenance.

    Two evaluators write this table and they do not produce the same quantities:

    * **per_class** — Ultralytics ``val()`` (YOLO): one row per class, box and mask precision, recall
      and mAP, at the evaluation settings named in the `settings` column;
    * **coco_pooled** — RF-DETR's own COCO evaluation (pycocotools): ``metric,value`` rows, pooled
      over the two classes. pycocotools computes a per-category AP internally but ``evaluate()``
      returns only the pooled summary, so there is **no** per-class mAP in this file.

    The caller must not paper over the difference: ``per_class`` says whether class-level mAP exists,
    ``evaluator`` and ``format`` say where the numbers come from, and every answer that quotes them is
    expected to name both. Per-class evidence for a pooled table comes from the framework-independent
    boundary/IoU table instead (`boundary_by_set.csv`), which is per class by construction.
    """
    csv, meta = tab / "segmentation_metrics_test.csv", tab / "segmentation_metrics_test.md"
    if meta.exists() and "PENDING — needs:" in meta.read_text(encoding="utf-8"):
        raise SystemExit(f"{meta} is pending: {meta.read_text(encoding='utf-8').split('needs:')[-1].strip()}\n"
                         "Run Stage 6 for this model and re-run scripts/build_report.py first.")
    F: Dict[str, Any] = {k: _md_meta(meta, k) for k in ("source", "model", "evaluator", "masks", "note", "settings")}
    df = pd.read_csv(csv)
    if "class" in df.columns:
        if "settings" in df.columns:
            std = df[df["settings"].astype(str).str.startswith("standard")]
            block, kind = (std, "standard") if len(std) else (df, str(df["settings"].iloc[0]))
        else:
            block, kind = df, "operating_point_legacy"
        t = block.set_index("class")
        F.update(format="per_class", per_class=True, settings_kind=kind, table=t,
                 classes={"gingiva": t.loc["diseti"].to_dict(), "lip": t.loc["dudak"].to_dict()},
                 pooled=t.loc["all"].to_dict())
        F["evaluator"] = F["evaluator"] or "Ultralytics val() (box and mask precision, recall, F1, mAP@50, mAP@50-95)"
    else:
        v = {str(m): float(x) for m, x in zip(df["metric"], df["value"])}

        def g(name: str) -> float:
            return next((x for k, x in v.items() if k.rsplit("/", 1)[-1] == name), float("nan"))

        F.update(format="coco_pooled", per_class=False, settings_kind="coco_standard", table=df, classes={},
                 pooled={"seg_map50": g("segm_mAP_50"), "seg_map50_95": g("segm_mAP_50_95"),
                         "box_map50": g("mAP_50"), "box_map50_95": g("mAP_50_95"), "box_map75": g("mAP_75"),
                         "mar": g("mAR"), "precision": g("precision"), "recall": g("recall"), "f1": g("F1")})
        F["evaluator"] = F["evaluator"] or "the model's own COCO evaluation (pycocotools, iouType='segm')"
    F["source"] = F["source"] or str(csv)
    F["provenance"] = f"{F['evaluator']}" + ("" if F["per_class"] else ", pooled over the two classes")
    return F


def learning_curve(pred_dir: Path, producer: str = "gsv4.train.learning_curve --collect on the workstation") -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """The learning-curve points of whichever model this report is built for.

    Two producers write ``learning_curve.csv`` and the table takes either: the YOLO curve
    (`gsv4/train/learning_curve.py`), whose rows appear one training run at a time and carry a
    ``done`` flag, and the RF-DETR curve (`scripts/build_rfdetr_learning_curve.py`), which is written
    only once all four points exist and instead names its metric, split and evaluator. ``producer``
    is the command the caller expects the file from, and it is what the pending row tells the reader
    to run.
    """
    p = pred_dir / "learning_curve.csv"
    if not p.exists():
        return None, {"status": "pending", "needs": f"{p} — write it with {producer}"}
    df = pd.read_csv(p)
    if "done" in df.columns and not df["done"].all():
        return None, {"status": "pending", "needs": f"{p} with all four points trained ({producer})"}
    st: Dict[str, Any] = {"status": "done", "source": str(p)}
    for key in ("metric", "split", "evaluator"):     # the RF-DETR curve names these in the file itself
        if key in df.columns and df[key].notna().any():
            st[key] = str(df[key].dropna().iloc[0])
    md = pred_dir / "learning_curve.md"
    if md.exists():
        text = md.read_text()
        st["verdict"] = text.split("→ **")[1].split("**")[0] if "→ **" in text else ""
        st["reading"] = str(md)
    return df, st


def _and_list(items: list) -> str:
    return " and ".join([", ".join(items[:-1]), items[-1]]) if len(items) > 1 else "".join(items)


def _md_meta(md: Path, key: str) -> Optional[str]:
    if not md.exists():
        return None
    m = re.search(rf"^- {re.escape(key)}: (.+)$", md.read_text(encoding="utf-8"), flags=re.M)
    return m.group(1).strip() if m else None


def learning_curve_facts(tab: Path, fallback_md: Path) -> Dict[str, Any]:
    """The curve the report table carries: its metric, its points and its verdict, whichever produced it.

    `tables/learning_curve.md` names the metric and the verdict of the curve that Stage 7 copied in —
    the YOLO curve (gingiva mask mAP@50) or the RF-DETR one (`val/segm_mAP_50`, both classes). Nothing
    here assumes which; the plateau wording downstream follows the verdict in the file, never the other
    way round.
    """
    meta = tab / "learning_curve.md"
    if meta.exists() and "PENDING — needs:" in meta.read_text(encoding="utf-8"):
        # the CSV beside it is then the previous model's curve, left behind by an earlier build
        raise SystemExit(f"{meta} is pending: {meta.read_text(encoding='utf-8').split('needs:')[-1].strip()}\n"
                         "The rebuttal is not written from another model's learning curve; produce that file and "
                         "re-run scripts/build_report.py first.")
    df = pd.read_csv(tab / "learning_curve.csv")
    metric = _md_meta(meta, "metric") or ("diseti_seg_map50" if "diseti_seg_map50" in df.columns else "")
    if metric not in df.columns:
        raise SystemExit(f"{tab / 'learning_curve.csv'} carries no readable curve metric ({metric!r}); "
                         "rebuild the report (scripts/build_report.py) so the table names its metric.")
    short = metric.rsplit("/", 1)[-1]
    label = ("gingiva mask mAP@50" if metric == "diseti_seg_map50" else
             f"mask mAP@50 over both classes (COCO, `{metric}`)" if short.startswith("segm_mAP_50") and not short.endswith("50_95") else
             f"`{metric}`")
    verdict = (_md_meta(meta, "verdict") or "").strip()
    reading = Path(_md_meta(meta, "reading") or fallback_md)
    rule = ""
    if reading.exists():
        line = next((ln for ln in reading.read_text(encoding="utf-8").splitlines() if "\u2192 **" in ln), "")
        rule = line.split("\u2192 **")[0].strip()          # the gains, not just the text before the first arrow
    df = df.sort_values("fraction")
    return {"df": df, "metric": metric, "label": label, "verdict": verdict,
            "plateau": verdict.startswith("plateau"), "rule": rule,
            "points": _and_list([f"{v:.3f}" for v in df[metric]]),
            "sizes": _and_list(["?" if pd.isna(n) else str(int(n)) for n in df["n_train_images"]]),
            "first": float(df[metric].iloc[0]), "last": float(df[metric].iloc[-1])}


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


def estimator_sensitivity(oracle_dir: Path, selected_combo: str, px_per_mm: float,
                          predicted_oracle_dir: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Every measurement-method × estimator combination of Stage 3, for the appendix.

    This is a *sensitivity* table, not a selection: the method and the scale were chosen once, on
    ground-truth masks, on the development subset, by the rule recorded in
    ``outputs/03_oracle/oracle_summary.md``; they were never re-selected or re-fitted on predicted
    masks. The holdout columns are reported so that a reader can see how little the choice depends
    on the combination, and they took no part in making it.
    """
    p = oracle_dir / "estimator_comparison.csv"
    if not p.exists():
        return None, {"status": "pending", "needs": f"{p} — run scripts/run_oracle.py --write-config (Stage 3)"}
    raw = pd.read_csv(p)
    cols = {"combo": "combo", "regioning": "regioning", "estimator": "estimator", "anchored": "lip_anchored",
            "px_per_mm_dev": "px_per_mm (dev fit)", "mae_dev": "MAE dev, mm", "mae_holdout": "MAE holdout, mm",
            "icc2_1_dev": "ICC(2,1) dev", "icc2_1_holdout": "ICC(2,1) holdout",
            "ba_bias_dev": "BA bias dev, mm", "ba_bias_holdout": "BA bias holdout, mm"}
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        return None, {"status": "pending", "needs": f"{p} lacks {missing} — re-run Stage 3"}
    df = raw[list(cols)].rename(columns=cols).sort_values("MAE dev, mm").reset_index(drop=True)
    df.insert(0, "selected", ["**yes**" if c == selected_combo else "" for c in df["combo"]])
    st: Dict[str, Any] = {"status": "done", "source": str(p), "n_combinations": int(len(df)),
                          "selected": selected_combo, "selected_px_per_mm": f"{px_per_mm:.2f}",
                          "sorted_by": "dev MAE (the quantity the selection used)"}
    summary = oracle_dir / "oracle_summary.md"
    if summary.exists():
        txt = summary.read_text(encoding="utf-8")
        m = re.search(r"^selection on dev MAE .*$", txt, flags=re.M)
        if m:
            st["selection_rule"] = m.group(0).strip().rstrip(".")
        m = re.search(r"could not be established on (\d+) of (\d+) images \((\d+) %", txt)
        if m:
            st["regioning_fallback"] = f"{m.group(3)} % ({m.group(1)} of {m.group(2)} images)"
        m = re.search(r"dev images where C succeeded \(n = (\d+)\), dev MAE is ([\d.]+) mm for `([A-Z]_\w+)` vs ([\d.]+) mm for `([A-Z]_\w+)`", txt)
        if m:
            st["fallback_check"] = (f"on the {m.group(1)} dev images where the regioning succeeded, dev MAE "
                                    f"{m.group(2)} mm for {m.group(3)} against {m.group(4)} mm for {m.group(5)}")
    # PLAN.md 7: the fallback rate is re-measured on the predicted masks; above 30 % the choice would
    # have been re-evaluated against A_p25. Reported here so the appendix carries both rates.
    if predicted_oracle_dir is not None and (predicted_oracle_dir / "oracle_summary.md").exists():
        m = re.search(r"could not be established on (\d+) of (\d+) images \((\d+) %",
                      (predicted_oracle_dir / "oracle_summary.md").read_text(encoding="utf-8"))
        if m:
            st["regioning_fallback_predicted_masks"] = f"{m.group(3)} % ({m.group(1)} of {m.group(2)} images)"
            st["fallback_reeval_threshold"] = "30 % (PLAN.md 7), not reached"
    st["note"] = ("the method and the scale were selected here, once, on GROUND-TRUTH masks on the dev subset; "
                  "they were never re-selected or re-fitted on predicted masks (outputs/09_final_rfdetr/PLAN.md 5 and Amendment 1)")
    return df, st
