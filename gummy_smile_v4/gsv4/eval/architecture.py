"""Statistics for the architecture comparison (outputs/08_architecture/PROTOCOL.md).

Every architecture predicts the same images, so the comparison is paired and the relevant
variance is that of the per-image difference between two architectures, not that of the
difference between a model and the clinical reference. All functions here work on a long
table with one row per (model, seed, image).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DECISION_THRESHOLD_MM = 0.15   # pre-registered in PROTOCOL.md §8, fixed before any run
N_BOOT = 2000


def per_image_errors(df: pd.DataFrame, value_col: str = "selected_mm", ref_col: str = "ref_mm") -> pd.DataFrame:
    """Add the signed and absolute millimetre error of every row."""
    out = df.copy()
    out["error_mm"] = out[value_col].astype(float) - out[ref_col].astype(float)
    out["abs_error_mm"] = out["error_mm"].abs()
    return out


def seed_table(df: pd.DataFrame, by: Sequence[str] = ("model", "seed")) -> pd.DataFrame:
    """One row per model and seed: n, MAE, bias, RMSE."""
    rows = []
    for key, part in df.groupby(list(by), sort=True):
        e = part["error_mm"].dropna().to_numpy(float)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update({"n": int(len(e)), "mae_mm": float(np.abs(e).mean()) if len(e) else math.nan,
                    "bias_mm": float(e.mean()) if len(e) else math.nan,
                    "rmse_mm": float(np.sqrt(np.mean(e ** 2))) if len(e) else math.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def model_table(seeds: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Mean and standard deviation over seeds, per model (PROTOCOL.md §5)."""
    rows = []
    for m, part in seeds.groupby(by, sort=True):
        rec: Dict[str, Any] = {by: m, "n_seeds": int(len(part))}
        for c in ("mae_mm", "bias_mm", "rmse_mm"):
            rec[f"{c}_mean"] = float(part[c].mean())
            rec[f"{c}_sd"] = float(part[c].std(ddof=1)) if len(part) > 1 else math.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def seed_averaged(df: pd.DataFrame, value_col: str = "selected_mm", key: str = "uid") -> pd.DataFrame:
    """One value per model and image, averaged over seeds: the architecture's expected output."""
    g = df.groupby(["model", key], sort=True)
    out = g[[value_col, "ref_mm"]].mean().reset_index()
    out["n_seeds"] = g.size().to_numpy()
    return out


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, value_col: str = "selected_mm", key: str = "uid",
                      n_boot: int = N_BOOT, seed: int = 42) -> Dict[str, Any]:
    """Difference in mean absolute error between two models on the images they share.

    Positive means ``a`` has the larger error, i.e. ``b`` is better. The confidence interval
    comes from a bootstrap over the shared images (paired: an image is resampled for both
    models at once), which is the interval the decision rule of PROTOCOL.md §8 refers to.
    """
    left = a.set_index(key)
    right = b.set_index(key)
    shared = [i for i in left.index if i in right.index]
    if not shared:
        return {"n": 0}
    ea = (left.loc[shared, value_col].astype(float) - left.loc[shared, "ref_mm"].astype(float)).abs().to_numpy()
    eb = (right.loc[shared, value_col].astype(float) - right.loc[shared, "ref_mm"].astype(float)).abs().to_numpy()
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb = ea[ok], eb[ok]
    d = ea - eb
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]) if len(d) else np.array([])
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) else (math.nan, math.nan)
    return {"n": int(len(d)), "mae_a": float(ea.mean()), "mae_b": float(eb.mean()), "diff_mae_mm": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "sd_paired_mm": float(d.std(ddof=1)) if len(d) > 1 else math.nan,
            "excludes_zero": bool(len(boot) and (lo > 0 or hi < 0))}


def decision(comparisons: Sequence[Dict[str, Any]], baseline: str, threshold_mm: float = DECISION_THRESHOLD_MM) -> Dict[str, Any]:
    """Apply the pre-registered rule of PROTOCOL.md §8 to the comparisons against the baseline.

    ``comparisons`` are dicts from :func:`paired_difference` with ``model_a`` = the baseline and
    ``model_b`` = the challenger, so ``diff_mae_mm`` > 0 means the challenger is better by that
    many millimetres. The final model changes only when a challenger leads by more than
    ``threshold_mm`` *and* the paired interval excludes zero. Both conditions must hold.
    """
    winners = [c for c in comparisons
               if c.get("n") and c.get("diff_mae_mm", 0) > threshold_mm and c.get("excludes_zero")]
    best = max(winners, key=lambda c: c["diff_mae_mm"]) if winners else None
    return {"baseline": baseline, "threshold_mm": threshold_mm, "change_final_model": bool(best),
            "winner": best.get("model_b") if best else None,
            "lead_mm": best.get("diff_mae_mm") if best else None,
            "rule": (f"a challenger replaces {baseline} only when it leads by more than {threshold_mm} mm in mean absolute "
                     "error and the paired 95 % bootstrap interval of that lead excludes zero (both conditions)"),
            "outcome": (f"{best['model_b']} leads {baseline} by {best['diff_mae_mm']:.3f} mm "
                        f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}]: Stage 6 is repeated with it (PROTOCOL.md §8)"
                        if best else f"no challenger meets both conditions: the final model stays {baseline}")}


def edge_table(df: pd.DataFrame, px_per_mm: float,
               cols: Sequence[str] = ("gingiva_top_edge_mae_px", "gingiva_top_edge_bias_px",
                                      "gingiva_bottom_edge_mae_px", "gingiva_bottom_edge_bias_px")) -> pd.DataFrame:
    """Gingival edge errors per model and seed, in millimetres at the global scale."""
    rows = []
    for key, part in df.groupby(["model", "seed"], sort=True):
        rec = {"model": key[0], "seed": key[1], "n": int(len(part))}
        for c in cols:
            v = part[c].dropna().astype(float) / float(px_per_mm) if c in part.columns else pd.Series(dtype=float)
            rec[c.replace("_px", "_mm")] = float(v.mean()) if len(v) else math.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def integrity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: signs that the predictor's classes were read wrongly.

    A run that produced no lip mask on any image, or that dropped instances for having no role, or
    whose gingival edge error is entirely systematic (mean absolute error equal to the absolute
    bias, i.e. every image shifted the same way), is not measuring what it claims to. The first
    architecture-comparison run showed all three at once, because RF-DETR's renumbered label space
    was read through the dataset's category table and the lip band was stored as the gingiva mask.
    """
    rows = []
    for model, part in df.groupby("model", sort=True):
        n = len(part)
        lip_col = next((c for c in ("n_lip_instances", "n_lip") if c in part.columns), None)
        no_lip = int((part[lip_col] == 0).sum()) if lip_col else 0
        ignored = int(part["n_ignored"].fillna(0).sum()) if "n_ignored" in part.columns else 0
        top = part["gingiva_top_edge_mae_px"].mean() if "gingiva_top_edge_mae_px" in part.columns else math.nan
        top_bias = part["gingiva_top_edge_bias_px"].mean() if "gingiva_top_edge_bias_px" in part.columns else math.nan
        fully_systematic = bool(np.isfinite(top) and np.isfinite(top_bias) and top > 0 and abs(abs(top_bias) - top) < 1e-6)
        problems = []
        if n and no_lip == n:
            problems.append("no lip mask on any image")
        if ignored:
            problems.append(f"{ignored} instance(s) dropped for having no role")
        if fully_systematic:
            problems.append("the gingival edge error is entirely systematic (MAE equals |bias|)")
        rows.append({"model": model, "n_rows": n, "images_without_lip": no_lip, "instances_ignored": ignored,
                     "edge_error_fully_systematic": fully_systematic, "suspect": bool(problems),
                     "problems": "; ".join(problems)})
    return pd.DataFrame(rows)


def bias_scatter_table(df: pd.DataFrame, value_col: str = "selected_mm", key: str = "uid") -> pd.DataFrame:
    """How much of each model's error is a systematic shift and how much is scatter.

    A lead in mean absolute error can come from either, and the two have very different
    consequences: a constant shift is removable by calibration (the pipeline already applies one at
    mask level), scatter is not. The seed-averaged per-image error is split into its mean (bias) and
    what is left once that mean is removed.
    """
    rows = []
    for model, part in df.groupby("model", sort=True):
        g = part.groupby(key).agg(pred=(value_col, "mean"), ref=("ref_mm", "mean"))
        e = (g["pred"] - g["ref"]).to_numpy(float)
        e = e[np.isfinite(e)]
        if not len(e):
            continue
        rows.append({"model": model, "n": int(len(e)), "mae_mm": float(np.abs(e).mean()), "bias_mm": float(e.mean()),
                     "sd_of_error_mm": float(e.std(ddof=1)) if len(e) > 1 else math.nan,
                     "mae_without_own_bias_mm": float(np.abs(e - e.mean()).mean()),
                     "removable_by_calibration_mm": float(np.abs(e).mean() - np.abs(e - e.mean()).mean()),
                     "within_0_5_mm": float((np.abs(e) <= 0.5).mean()), "within_1_mm": float((np.abs(e) <= 1.0).mean())})
    return pd.DataFrame(rows)


def error_correlation(df: pd.DataFrame, value_col: str = "selected_mm", key: str = "uid") -> pd.DataFrame:
    """Correlation of the per-image error between architectures (seed-averaged).

    A high correlation means the architectures fail on the same images and differ by little more
    than a shift; a low one means they fail differently and the comparison is about more than a
    constant.
    """
    g = df.groupby(["model", key]).agg(pred=(value_col, "mean"), ref=("ref_mm", "mean")).reset_index()
    g["err"] = g["pred"] - g["ref"]
    return g.pivot(index=key, columns="model", values="err").corr()


# Mask-pixel geometry of one configuration, in millimetres on the original image. Ultralytics
# letterboxes, so both axes share a scale; RF-DETR resizes to a square, so a 3:2 photograph gets a
# finer vertical grid than horizontal. The measured quantity is a vertical thickness, so the
# vertical figure is the one that matters (PROTOCOL_ADDENDUM_resolution.md §1).
def mask_pixel_mm(input_size: int, px_per_mm: float, image_wh: Sequence[int] = (2698, 1799),
                  downsample: int = 4, letterbox: bool = True) -> Dict[str, float]:
    w, h = float(image_wh[0]), float(image_wh[1])
    if letterbox:
        scale = input_size / max(w, h)
        sx = sy = scale
    else:
        sx, sy = input_size / w, input_size / h
    return {"grid": input_size // downsample,
            "horizontal_mm": (downsample / sx) / px_per_mm,
            "vertical_mm": (downsample / sy) / px_per_mm}


def resolution_verdict(control_b: Dict[str, Any], control_a: Dict[str, Any], mae_by_config: Dict[str, float],
                       threshold_mm: float = DECISION_THRESHOLD_MM) -> Dict[str, Any]:
    """Apply the pre-registered reading of PROTOCOL_ADDENDUM_resolution.md §4.

    ``control_b`` compares YOLOv11x at 640 (a) with RF-DETR at 432 (b), ``control_a`` compares
    YOLOv11x at 1024 (a) with RF-DETR at 624 (b); both come from :func:`paired_difference`, whose
    ``diff_mae_mm`` is positive when ``a`` has the larger error. ``mae_by_config`` is the uncorrected
    mean absolute error of every configuration, used only by rule 5.
    """
    def ok(c):
        return bool(c) and c.get("n")

    b_says = ok(control_b) and (control_b["diff_mae_mm"] < threshold_mm or not control_b["excludes_zero"])
    a_says = ok(control_a) and (not control_a["excludes_zero"] or control_a["diff_mae_mm"] < 0)
    if not (ok(control_a) and ok(control_b)):
        return {"rule": "incomplete", "resolution_explains_b": b_says, "resolution_explains_a": a_says,
                "outcome": "a control is missing; no conclusion may be drawn yet", "final_model": None}
    if b_says and a_says:
        return {"rule": 3, "resolution_explains_b": True, "resolution_explains_a": True, "final_model": "yolo11x-seg@1024",
                "outcome": ("both controls point to resolution: the lead is explained by the finer vertical mask grid, "
                            "the final model becomes YOLOv11x-seg at imgsz 1024 and Stage 6 is repeated with it")}
    if not b_says and not a_says:
        return {"rule": 4, "resolution_explains_b": False, "resolution_explains_a": False, "final_model": "rfdetr-seg-large",
                "outcome": ("the lead survives at matched resolution: PROTOCOL.md §8 applies as written, the final model "
                            "becomes RF-DETR-Seg Large and Stage 6 is repeated with it")}
    best = min(mae_by_config, key=mae_by_config.get) if mae_by_config else None
    return {"rule": 5, "resolution_explains_b": b_says, "resolution_explains_a": a_says, "final_model": best,
            "outcome": (f"the controls disagree (B says {'resolution' if b_says else 'architecture'}, A says "
                        f"{'resolution' if a_says else 'architecture'}); both are reported, the configuration with the "
                        f"lowest uncorrected mean absolute error is chosen ({best}), and the disagreement is stated in "
                        "the manuscript as an unresolved uncertainty")}
