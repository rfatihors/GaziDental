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
