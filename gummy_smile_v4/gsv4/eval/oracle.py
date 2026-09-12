"""Oracle validation harness (spec §4): measurement geometry vs clinical ImageJ reference
on ground-truth masks — or on any directory of predicted class masks (Stage 6).

Protocol (fixed before looking at the data):
* dev / holdout = 60 / 40 % of the reference images, seed from config, lists on disk;
* the global ``px_per_mm`` of every combination is fitted on **dev only** (regression
  through the origin) and applied unchanged to holdout;
* the method is selected on **dev MAE**; combinations within ``tolerance_mm`` of the
  best are tied and the simplest wins (regioning A > B > C, gingiva thickness >
  lip-anchored, estimator p25 > median > p10 > min > max);
* holdout numbers are reported once, for all combinations, never used for selection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from gsv4.eval.agreement import bland_altman, icc_two_raters, loo_scale, pearson, scale_through_origin
from gsv4.io.coco import load_annotations
from gsv4.masks.extract import ClassMasks, from_coco, from_png
from gsv4.measure.gingival_display import ESTIMATORS, REGIONINGS, measure_gingival_display
from gsv4.measure.qc import QCFlag, flags_to_str
from gsv4.rules.thresholds import label_for_mm

COMBOS: List[Tuple[str, str, bool]] = [(r, e, a) for r in REGIONINGS for e in ESTIMATORS for a in (False, True)]
SIMPLICITY = {"regioning": {"A": 0, "B": 1, "C": 2}, "estimator": {"p25": 0, "median": 1, "p10": 2, "min": 3, "max": 4}}
LABEL_ORDER = ["NO_VISIBLE_GINGIVA", "E1", "E1-E2", "E2-E3", "E3", "E4"]


def combo_name(regioning: str, estimator: str, anchored: bool) -> str:
    return f"{regioning}_{estimator}" + ("_lipanchored" if anchored else "")


def dev_holdout_split(uids: Sequence[str], seed: int, dev_frac: float = 0.6) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    order = sorted(uids)
    rng.shuffle(order)
    n_dev = int(round(dev_frac * len(order)))
    return {"dev": sorted(order[:n_dev]), "holdout": sorted(order[n_dev:])}


# ----------------------------------------------------------------------------- measurement
def _gt_masks(row: pd.Series, coco_root: Path, coco_cfg: Dict[str, Any], cache: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[ClassMasks, Tuple[int, int]]:
    key = (row["group"], row["orig_split"])
    if key not in cache:
        cache[key] = load_annotations(coco_root, row["group"], row["orig_split"], coco_cfg)
    ann = cache[key]
    im = next(i for i in ann["images"] if i["file_name"] == row["file_name"])
    shape = (int(im["height"]), int(im["width"]))
    return from_coco(ann["annotations"], int(im["id"]), shape, coco_cfg["category_ids"]), shape


def _png_masks(row: pd.Series, mask_dir: Path) -> Tuple[Optional[ClassMasks], Tuple[int, int]]:
    shape = (int(row["height"]), int(row["width"]))
    g = mask_dir / f"{row['image']}_gingiva.png"
    if not g.exists():
        return None, shape
    lip = mask_dir / f"{row['image']}_lip.png"
    return from_png(g, lip if lip.exists() else None, expected_shape=shape), shape


def measure_images(rows: pd.DataFrame, cfg: Dict[str, Any], coco_root: Path, mask_source: str = "gt") -> pd.DataFrame:
    """Run the measurement on every row (GT COCO masks or ``<mask_source>/<image>_gingiva.png``)."""
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    out = []
    for _, r in rows.iterrows():
        if mask_source == "gt":
            masks, shape = _gt_masks(r, coco_root, cfg["coco"], cache)
        else:
            masks, shape = _png_masks(r, Path(mask_source))
        rec: Dict[str, Any] = {"uid": r["uid"], "image": r["image"], "group": r["group"], "width": shape[1], "height": shape[0]}
        if masks is None:
            rec.update({"qc_flags": QCFlag.NO_GINGIVA_MASK.value, "mask_missing": True})
            out.append(rec)
            continue
        masks.check_shape(shape)
        res = measure_gingival_display(masks.gingiva, masks.lip, px_per_mm=None, cfg=cfg["measurement"])
        rec.update(res.to_row())
        rec["mask_missing"] = False
        rec["n_gingiva_instances"] = masks.n_gingiva_instances
        rec["n_lip_instances"] = masks.n_lip_instances
        if not bool(r.get("frame_ok", True)):
            rec["qc_flags"] = flags_to_str(list(res.flags) + [QCFlag.FRAME_UNCERTAIN])
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- evaluation
def _metrics(pred_mm: np.ndarray, ref_mm: np.ndarray, ref_label: Sequence[str]) -> Dict[str, float]:
    ok = np.isfinite(pred_mm) & np.isfinite(ref_mm)
    p, r = pred_mm[ok], ref_mm[ok]
    d = p - r
    m: Dict[str, float] = {"n": int(ok.sum()), "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt(np.mean(d**2)))}
    m.update({f"{k}": v for k, v in pearson(p, r).items()})
    icc = icc_two_raters(p, r)
    m.update({k: icc[k] for k in ("icc2_1", "icc2_1_ci_low", "icc2_1_ci_high", "icc3_1")})
    ba = bland_altman(p, r)
    m.update({f"ba_{k}": ba[k] for k in ("bias", "bias_ci_low", "bias_ci_high", "sd", "loa_low", "loa_high", "prop_slope", "prop_p")})
    pl = [label_for_mm(v) for v in p]
    rl = [x for x, o in zip(ref_label, ok) if o]
    m["threshold_agreement"] = float(np.mean([a == b for a, b in zip(pl, rl)]))
    try:
        m["threshold_kappa_linear"] = float(cohen_kappa_score([LABEL_ORDER.index(x) for x in rl], [LABEL_ORDER.index(x) for x in pl], weights="linear"))
    except (ValueError, IndexError):
        m["threshold_kappa_linear"] = float("nan")
    return m


def evaluate_combos(df: pd.DataFrame, split: Dict[str, List[str]], combos: Iterable[Tuple[str, str, bool]] = COMBOS) -> pd.DataFrame:
    """Fit each combination's scale on dev, report dev and holdout metrics."""
    df = df.set_index("uid")
    if "ref_label" not in df.columns:
        df["ref_label"] = df["ref_mm"].map(label_for_mm)
    dev = df.loc[[u for u in split["dev"] if u in df.index]]
    hold = df.loc[[u for u in split["holdout"] if u in df.index]]
    rows = []
    for reg, est, anch in combos:
        col = f"{combo_name(reg, est, anch)}_px"
        if col not in df.columns:
            continue
        ok_dev = dev[col].notna() & dev["ref_mm"].notna()
        scale = scale_through_origin(dev.loc[ok_dev, col], dev.loc[ok_dev, "ref_mm"])
        k = scale["px_per_mm"]
        rec: Dict[str, Any] = {"regioning": reg, "estimator": est, "anchored": anch, "combo": combo_name(reg, est, anch),
                               "px_per_mm_dev": k, "scale_r2_dev": scale["r2"], "scale_resid_sd_mm": scale["resid_sd_mm"]}
        loo = loo_scale(dev.loc[ok_dev, col], dev.loc[ok_dev, "ref_mm"])
        rec.update({"loo_px_per_mm_sd": loo["px_per_mm_sd"], "loo_mae_dev": loo["loo_mae_mm"]})
        for name, part in (("dev", dev), ("holdout", hold)):
            m = _metrics((part[col] / k).to_numpy(dtype=float), part["ref_mm"].to_numpy(dtype=float), list(part["ref_label"]))
            rec.update({f"{kk}_{name}": vv for kk, vv in m.items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def select_method(results: pd.DataFrame, tolerance_mm: float = 0.02) -> Tuple[Dict[str, Any], str]:
    """Best dev MAE; ties within ``tolerance_mm`` resolved by simplicity."""
    r = results.copy()
    r["simplicity"] = (
        r["regioning"].map(SIMPLICITY["regioning"]) * 100
        + r["anchored"].astype(int) * 10
        + r["estimator"].map(SIMPLICITY["estimator"])
    )
    best = r["mae_dev"].min()
    tied = r[r["mae_dev"] <= best + tolerance_mm].sort_values(["simplicity", "mae_dev"])
    chosen = tied.iloc[0]
    why = (f"selection on dev MAE (n = {int(chosen['n_dev'])}); best dev MAE = {best:.3f} mm; "
           f"{len(tied)} combination(s) within {tolerance_mm:.2f} mm of it ({', '.join(tied['combo'])}); "
           f"the simplest of those is chosen (A > B > C, gingiva thickness > lip-anchored, p25 > median > p10 > min > max).")
    return {"regioning": chosen["regioning"], "estimator": chosen["estimator"], "anchored": bool(chosen["anchored"]),
            "combo": chosen["combo"], "px_per_mm": float(chosen["px_per_mm_dev"]), "mae_dev": float(chosen["mae_dev"])}, why


def sensitivity(df: pd.DataFrame, split: Dict[str, List[str]], combo: str, k: float) -> pd.DataFrame:
    """Holdout metrics of the chosen combination on the pre-defined subsets (scale fixed)."""
    col = f"{combo}_px"
    d = df.set_index("uid")
    hold = d.loc[[u for u in split["holdout"] if u in d.index]]
    both = d.loc[[u for u in split["dev"] + split["holdout"] if u in d.index]]
    subsets = {
        "holdout: all": hold,
        "holdout: without dash-zero rows": hold[~hold["has_dash_zero"].astype(bool)],
        "holdout: only 2698x1799 (+/-2 px) frames": hold[hold["frame_ok"].astype(bool)],
        "holdout: frames outside 2698x1799": hold[~hold["frame_ok"].astype(bool)],
        "holdout: without ambiguous 100-999 cells": hold[~hold["has_ambiguous_100_999"].astype(bool)],
        "dev + holdout (scale still from dev)": both,
    }
    rows = []
    for name, part in subsets.items():
        if len(part) < 5:
            rows.append({"subset": name, "n": len(part)})
            continue
        m = _metrics((part[col] / k).to_numpy(dtype=float), part["ref_mm"].to_numpy(dtype=float), list(part["ref_label"]))
        rows.append({"subset": name, **m})
    return pd.DataFrame(rows)


def per_image_scale(df: pd.DataFrame, combo: str) -> pd.DataFrame:
    """Per-image px/mm ratio for one combination, with the frame group."""
    d = df[["uid", "frame_ok", "width", "height", "ref_mm", f"{combo}_px"]].copy()
    d["px_per_mm_image"] = d[f"{combo}_px"] / d["ref_mm"]
    d.loc[~np.isfinite(d["px_per_mm_image"]) | (d["ref_mm"] <= 0), "px_per_mm_image"] = np.nan
    return d


def tooth_level(df: pd.DataFrame, combo: str, k: float) -> pd.DataFrame:
    """Secondary: region i (left-to-right) vs reference tooth i, MAE/bias per tooth."""
    rows = []
    for i in range(1, 7):
        pc, rc = f"{combo}_region_{i}_px", f"ref_mm_{i}"
        if pc not in df.columns or rc not in df.columns:
            continue
        p = df[pc].to_numpy(dtype=float) / k
        r = df[rc].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(r)
        rows.append({"tooth_index": i, "n": int(ok.sum()), "mae": float(np.abs(p[ok] - r[ok]).mean()),
                     "bias": float((p[ok] - r[ok]).mean()), "r": pearson(p[ok], r[ok])["r"]})
    return pd.DataFrame(rows)
