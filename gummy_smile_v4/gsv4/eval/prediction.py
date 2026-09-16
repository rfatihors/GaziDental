"""Stage 6 — accuracy of the full pipeline on *predicted* masks (spec §Stage 6).

Everything here is deterministic table arithmetic on per-image results; the measurement
itself is ``gsv4.eval.oracle.measure_images`` with the method and global scale **fixed
in config.yaml** (selected in Stage 3 on ground-truth masks). Nothing is re-selected or
re-fitted on predicted masks — that would be a second look at the reference.

Sets (pre-registered, 12 Sep 2026):
* (a) all reference images with 5-fold out-of-fold masks — **primary**;
* (b) the fixed test subset of (a) with final-model masks — secondary;
* (c) the whole fixed test set (segmentation quality only; no reference measurement for
  low / normal smile lines, and a thin or absent gingiva there makes IoU uninformative).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from gsv4.eval.agreement import bland_altman, icc_two_raters, pearson
from gsv4.eval.kappa import bootstrap_ci
from gsv4.eval.oracle import LABEL_ORDER
from gsv4.measure.qc import QCFlag
from gsv4.rules.thresholds import label_for_mm

EXPECTED_MASK_SOURCE = "yolo:masks.data"
FALLBACK_REEVALUATE_FRAC = 0.30   # pre-registered: above this, C_p25 is re-evaluated against A_p25

EDGE_COLS = ("gingiva_top_edge_mae_px", "gingiva_top_edge_median_px", "gingiva_top_edge_bias_px",
             "gingiva_bottom_edge_mae_px", "gingiva_bottom_edge_median_px", "gingiva_bottom_edge_bias_px", "gingiva_thickness_mae_px")


# ----------------------------------------------------------------------------- OOF integrity
def check_oof(oof: pd.DataFrame, reference: pd.DataFrame, mask_dir: Path) -> Dict[str, Any]:
    """Integrity of the out-of-fold prediction table against the reference image set.

    ``reference`` needs ``image``, ``uid``, ``cv_fold``. Every problem is a sentence in
    ``problems``; ``ok`` is True only when the list is empty.
    """
    problems: List[str] = []
    n = len(oof)
    n_ref = len(reference)
    if n != n_ref:
        problems.append(f"{n} OOF rows but {n_ref} reference images")
    dup = oof["image"][oof["image"].duplicated()].tolist()
    if dup:
        problems.append(f"duplicated OOF images: {dup}")
    missing = sorted(set(reference["image"]) - set(oof["image"]))
    extra = sorted(set(oof["image"]) - set(reference["image"]))
    if missing:
        problems.append(f"reference images without OOF prediction: {missing}")
    if extra:
        problems.append(f"OOF rows that are not reference images: {extra}")
    src = oof["mask_source"].value_counts().to_dict()
    bad_src = {k: v for k, v in src.items() if k != EXPECTED_MASK_SOURCE}
    if bad_src:
        problems.append(f"mask_source other than {EXPECTED_MASK_SOURCE}: {bad_src}")
    no_png = [i for i in oof["image"] if not (Path(mask_dir) / f"{i}_gingiva.png").exists()]
    if no_png:
        problems.append(f"gingiva PNG missing for {len(no_png)} images: {no_png[:5]}")
    fold_mismatch: List[str] = []
    if "fold" in oof.columns and "cv_fold" in reference.columns:
        exp = reference.set_index("image")["cv_fold"]
        for img, f in zip(oof["image"], oof["fold"]):
            if img in exp.index and int(exp[img]) != int(f):
                fold_mismatch.append(img)
        if fold_mismatch:
            problems.append(f"fold of {len(fold_mismatch)} images differs from splits.json: {fold_mismatch[:5]}")
    empty = oof.loc[oof["n_gingiva"] == 0, "image"].tolist() if "n_gingiva" in oof.columns else []
    return {"ok": not problems, "problems": problems, "n_rows": n, "n_reference": n_ref, "mask_source": src,
            "folds": oof["fold"].value_counts().sort_index().to_dict() if "fold" in oof.columns else {},
            "empty_gingiva_prediction": empty, "n_missing_png": len(no_png)}


# ----------------------------------------------------------------------------- agreement vs reference
def agreement_metrics(pred_mm: Sequence[float], ref_mm: Sequence[float], n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """MAE, RMSE, r, ICC(2,1), Bland–Altman, threshold-label agreement and linear-weighted
    kappa (bootstrap CI) of a pipeline measurement against the clinical reference."""
    p = np.asarray(pred_mm, dtype=float)
    r = np.asarray(ref_mm, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    out: Dict[str, float] = {"n": int(ok.sum()), "n_segmentation_failure": int((~ok).sum())}  # no gingiva instance -> no mm value
    if len(p) < 3:
        return out
    d = p - r
    out.update({"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt(np.mean(d**2))), "median_abs_err": float(np.median(np.abs(d)))})
    out.update(pearson(p, r))
    icc = icc_two_raters(p, r)
    out.update({k: icc[k] for k in ("icc2_1", "icc2_1_ci_low", "icc2_1_ci_high", "icc3_1")})
    ba = bland_altman(p, r)
    out.update({f"ba_{k}": ba[k] for k in ("bias", "bias_ci_low", "bias_ci_high", "sd", "loa_low", "loa_high", "prop_slope", "prop_p")})
    pl = np.asarray([label_for_mm(v) for v in p])
    rl = np.asarray([label_for_mm(v) for v in r])
    out["threshold_agreement"] = float(np.mean(pl == rl))
    out["threshold_kappa_linear"] = weighted_kappa(rl, pl)
    ci = bootstrap_ci(lambda idx: weighted_kappa(rl[idx], pl[idx]), len(p), n_boot, seed)
    out["threshold_kappa_linear_ci_low"], out["threshold_kappa_linear_ci_high"] = ci["ci_low"], ci["ci_high"]
    out["within_0_5_mm"] = float(np.mean(np.abs(d) <= 0.5))
    out["within_1_mm"] = float(np.mean(np.abs(d) <= 1.0))
    return out


def weighted_kappa(ref_labels: Sequence[str], pred_labels: Sequence[str], order: Sequence[str] = LABEL_ORDER) -> float:
    """Linear-weighted Cohen kappa on the ordered combined labels (E1 < E1-E2 < … < E4)."""
    from sklearn.metrics import cohen_kappa_score

    lut = {c: i for i, c in enumerate(order)}
    a = [lut[x] for x in ref_labels]
    b = [lut[x] for x in pred_labels]
    if not a or len(set(a) | set(b)) < 2:
        return math.nan
    try:
        return float(cohen_kappa_score(a, b, weights="linear"))
    except ValueError:
        return math.nan


def label_confusion(ref_labels: Sequence[str], pred_labels: Sequence[str], order: Sequence[str] = LABEL_ORDER) -> pd.DataFrame:
    """Reference (rows) × pipeline (columns) label counts, only labels that occur."""
    ct = pd.crosstab(pd.Series(list(ref_labels), name="reference"), pd.Series(list(pred_labels), name="pipeline"))
    rows = [c for c in order if c in ct.index]
    cols = [c for c in order if c in ct.columns]
    return ct.loc[rows, cols]


def measurement_table(per: pd.DataFrame, sets: Dict[str, pd.Series], value_col: str = "selected_mm", ref_col: str = "ref_mm",
                      min_n: int = 5, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """One metrics row per named boolean subset of ``per`` (subsets smaller than ``min_n`` give n only)."""
    rows = []
    for name, mask in sets.items():
        part = per[mask.reindex(per.index).fillna(False).astype(bool)]
        rec: Dict[str, Any] = {"set": name}
        if len(part) < min_n:
            rec["n"] = int(len(part))
        else:
            rec.update(agreement_metrics(part[value_col], part[ref_col], n_boot=n_boot, seed=seed))
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- error decomposition
def error_decomposition(per: pd.DataFrame, pred_col: str = "selected_mm", gt_col: str = "gt_mm", ref_col: str = "ref_mm") -> Dict[str, Any]:
    """total = pipeline − reference; split into the segmentation part (pipeline − GT-mask
    measurement) and the geometry part (GT-mask measurement − reference), same method and
    scale throughout. Returns per-image columns and a summary with a variance split."""
    d = per[[pred_col, gt_col, ref_col]].astype(float).copy()
    d["e_total"] = d[pred_col] - d[ref_col]
    d["e_seg"] = d[pred_col] - d[gt_col]
    d["e_meas"] = d[gt_col] - d[ref_col]
    ok = d[["e_total", "e_seg", "e_meas"]].notna().all(axis=1)
    x = d[ok]
    summary: Dict[str, Any] = {"n": int(ok.sum())}
    for c in ("e_total", "e_seg", "e_meas"):
        summary[f"{c}_mae"] = float(x[c].abs().mean())
        summary[f"{c}_bias"] = float(x[c].mean())
        summary[f"{c}_sd"] = float(x[c].std(ddof=1)) if len(x) > 1 else math.nan
        summary[f"{c}_rmse"] = float(np.sqrt(np.mean(x[c] ** 2)))
    var_t = float(x["e_total"].var(ddof=1)) if len(x) > 1 else math.nan
    var_s = float(x["e_seg"].var(ddof=1)) if len(x) > 1 else math.nan
    var_m = float(x["e_meas"].var(ddof=1)) if len(x) > 1 else math.nan
    cov = float(np.cov(x["e_seg"], x["e_meas"])[0, 1]) if len(x) > 1 else math.nan
    summary.update({"var_total": var_t, "var_seg": var_s, "var_meas": var_m, "cov_seg_meas": cov,
                    "share_seg": var_s / var_t if var_t else math.nan, "share_meas": var_m / var_t if var_t else math.nan,
                    "share_cov": 2 * cov / var_t if var_t else math.nan,
                    "r_seg_meas": pearson(x["e_seg"], x["e_meas"])["r"] if len(x) > 2 else math.nan})
    return {"per_image": d[["e_total", "e_seg", "e_meas"]], "summary": summary}


def seg_error_vs_boundary(dec: pd.DataFrame, boundary: pd.DataFrame, px_per_mm: float,
                          cols: Sequence[str] = ("gingiva_mask_iou", "gingiva_top_edge_bias_px", "gingiva_bottom_edge_bias_px",
                                                 "gingiva_thickness_mae_px", "gingiva_columns_missed_frac", "gingiva_columns_spurious_frac")) -> pd.DataFrame:
    """Correlation (and slope in mm per mm for pixel columns) of the segmentation-induced
    measurement error with the boundary metrics of the same image (index = uid)."""
    j = dec[["e_seg"]].join(boundary[list(cols)], how="inner")
    rows = []
    for c in cols:
        ok = j[["e_seg", c]].notna().all(axis=1)
        if ok.sum() < 3:
            rows.append({"metric": c, "n": int(ok.sum())})
            continue
        xv = j.loc[ok, c].to_numpy(float)
        if c.endswith("_px"):
            xv = xv / px_per_mm
        yv = j.loc[ok, "e_seg"].to_numpy(float)
        pr = pearson(xv, yv)
        slope, intercept = np.polyfit(xv, yv, 1)
        rows.append({"metric": c.replace("_px", "_mm") if c.endswith("_px") else c, "n": int(ok.sum()), "r": pr["r"], "r_p": pr["r_p"],
                     "slope": float(slope), "intercept": float(intercept)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tooth level
def tooth_long(per: pd.DataFrame, teeth: Sequence[int], other_prefix: str = "ref_mm_", diff_name: str = "diff", group_col: str = "uid") -> pd.DataFrame:
    """Long table for the mixed model: one row per image × tooth with
    ``selected_region_i_mm − <other_prefix>i`` (reference tooth i, or the GT-mask region i)."""
    rows = []
    for _, r in per.iterrows():
        for i, t in enumerate(teeth, start=1):
            mc, oc = f"selected_region_{i}_mm", f"{other_prefix}{i}"
            if mc not in per.columns or oc not in per.columns or pd.isna(r[mc]) or pd.isna(r[oc]):
                continue
            rows.append({"patient": r[group_col], "tooth": int(t), "tooth_index": i, "model_mm": float(r[mc]), "other_mm": float(r[oc]),
                         "alignment_uncertain": bool(r.get("alignment_uncertain", False))})
    d = pd.DataFrame(rows)
    if len(d):
        d[diff_name] = d["model_mm"] - d["other_mm"]
    return d


def tooth_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (i, t), part in long.groupby(["tooth_index", "tooth"]):
        d = part["diff"].to_numpy(float)
        rows.append({"tooth_index": int(i), "tooth_fdi": int(t), "n": int(len(d)), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                     "sd": float(d.std(ddof=1)) if len(d) > 1 else math.nan, "r": pearson(part["model_mm"], part["other_mm"])["r"] if len(d) > 2 else math.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- fallback rule
def fallback_rate(per: pd.DataFrame, flag: str = QCFlag.ZENITH_DETECTION_FAILED.value) -> Dict[str, Any]:
    flags = per["qc_flags"].fillna("").astype(str)
    fb = flags.str.contains(flag)
    return {"n": int(len(per)), "n_fallback": int(fb.sum()), "frac": float(fb.mean()) if len(per) else math.nan,
            "reevaluate": bool(len(per) and fb.mean() > FALLBACK_REEVALUATE_FRAC), "threshold": FALLBACK_REEVALUATE_FRAC}


# ----------------------------------------------------------------------------- boundary / segmentation quality
def presence_categories(b: pd.DataFrame) -> pd.Series:
    """Per image: both / gt_only (missed) / pred_only (spurious) / neither, from the column counts."""
    g = b["gingiva_n_columns_gt"].fillna(0) > 0
    p = b["gingiva_n_columns_pred"].fillna(0) > 0
    return pd.Series(np.select([g & p, g & ~p, ~g & p], ["both", "gt_only", "pred_only"], default="neither"), index=b.index)


def boundary_set_summary(b: pd.DataFrame, name: str, px_per_mm: float) -> Dict[str, Any]:
    """Segmentation-quality summary of one image set from a boundary_error-style table.

    IoU and boundary IoU are NaN when both masks are empty (union 0) and the edge errors
    are NaN when no column has gingiva in both masks; ``n_*`` say how many images each
    statistic rests on. Pixel edge metrics are also given in mm at the global scale.
    """
    cat = presence_categories(b)
    out: Dict[str, Any] = {"set": name, "n": int(len(b)), "n_gt_gingiva": int((cat.isin(["both", "gt_only"])).sum()),
                           "n_both": int((cat == "both").sum()), "n_missed": int((cat == "gt_only").sum()),
                           "n_spurious": int((cat == "pred_only").sum()), "n_neither": int((cat == "neither").sum())}
    for c in ("gingiva_mask_iou", "gingiva_boundary_iou", "lip_mask_iou", "lip_boundary_iou"):
        v = b[c].dropna() if c in b.columns else pd.Series(dtype=float)
        out[f"{c}_n"] = int(len(v))
        out[f"{c}_mean"] = float(v.mean()) if len(v) else math.nan
        out[f"{c}_median"] = float(v.median()) if len(v) else math.nan
    for c in ("gingiva_top_edge_mae_px", "gingiva_top_edge_bias_px", "gingiva_bottom_edge_mae_px", "gingiva_bottom_edge_bias_px", "gingiva_thickness_mae_px"):
        v = b[c].dropna() if c in b.columns else pd.Series(dtype=float)
        out[f"{c}_n"] = int(len(v))
        out[f"{c}_mean"] = float(v.mean()) if len(v) else math.nan
        out[f"{c}_median"] = float(v.median()) if len(v) else math.nan
        out[c.replace("_px", "_mm_mean")] = float(v.mean() / px_per_mm) if len(v) else math.nan
        out[c.replace("_px", "_mm_median")] = float(v.median() / px_per_mm) if len(v) else math.nan
    for c in ("gingiva_columns_missed_frac", "gingiva_columns_spurious_frac"):
        v = b[c].dropna() if c in b.columns else pd.Series(dtype=float)
        out[f"{c}_mean"] = float(v.mean()) if len(v) else math.nan
    v = b["gingiva_n_columns_gt"].dropna() if "gingiva_n_columns_gt" in b.columns else pd.Series(dtype=float)
    out["gingiva_n_columns_gt_median"] = float(v.median()) if len(v) else math.nan
    return out


def boundary_sets_table(sets: Dict[str, pd.DataFrame], px_per_mm: float) -> pd.DataFrame:
    return pd.DataFrame([boundary_set_summary(b, name, px_per_mm) for name, b in sets.items()])


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else (str(int(v)) if float(v).is_integer() and abs(v) >= 1e3 else fmt.format(v)))
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
