"""Boundary quality of a predicted mask against ground truth (Reviewer 2 #10):
boundary IoU and per-column distance error of the gingiva's upper and lower edges."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from gsv4.measure.profile import column_profile


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    return m & ~binary_erosion(m, border_value=0)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_px: int = 5) -> float:
    """IoU of the dilated boundaries (Cheng et al. 2021 style, fixed pixel dilation)."""
    bp = binary_dilation(mask_boundary(pred), iterations=dilation_px)
    bg = binary_dilation(mask_boundary(gt), iterations=dilation_px)
    union = (bp | bg).sum()
    return float((bp & bg).sum() / union) if union else float("nan")


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else float("nan")


def edge_distance_errors(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Column-wise |top_pred - top_gt| and |bottom_pred - bottom_gt| (px) over columns where
    both masks have gingiva; plus how many GT columns the prediction misses and vice versa."""
    tp, top_p, bot_p = column_profile(pred)
    tg, top_g, bot_g = column_profile(gt)
    both = (tp > 0) & (tg > 0)
    out: Dict[str, float] = {
        "n_columns_gt": int((tg > 0).sum()), "n_columns_pred": int((tp > 0).sum()), "n_columns_both": int(both.sum()),
        "columns_missed_frac": float(((tg > 0) & (tp == 0)).sum() / max(1, (tg > 0).sum())),
        "columns_spurious_frac": float(((tp > 0) & (tg == 0)).sum() / max(1, (tp > 0).sum())),
    }
    if both.any():
        d_top = np.abs(top_p[both] - top_g[both]).astype(float)
        d_bot = np.abs(bot_p[both] - bot_g[both]).astype(float)
        s_top = (top_p[both] - top_g[both]).astype(float)
        s_bot = (bot_p[both] - bot_g[both]).astype(float)
        out.update({
            "top_edge_mae_px": float(d_top.mean()), "top_edge_median_px": float(np.median(d_top)), "top_edge_bias_px": float(s_top.mean()),
            "bottom_edge_mae_px": float(d_bot.mean()), "bottom_edge_median_px": float(np.median(d_bot)), "bottom_edge_bias_px": float(s_bot.mean()),
            "thickness_mae_px": float(np.abs(tp[both] - tg[both]).mean()),
        })
    else:
        out.update({k: float("nan") for k in ("top_edge_mae_px", "top_edge_median_px", "top_edge_bias_px", "bottom_edge_mae_px", "bottom_edge_median_px", "bottom_edge_bias_px", "thickness_mae_px")})
    return out


def boundary_report(pred: np.ndarray, gt: np.ndarray, dilation_px: int = 5) -> Dict[str, float]:
    return {"mask_iou": mask_iou(pred, gt), "boundary_iou": boundary_iou(pred, gt, dilation_px), **edge_distance_errors(pred, gt)}
