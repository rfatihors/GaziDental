from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class MetricBundle:
    mae: float
    rmse: float
    icc: float
    bland_altman_mean: float
    bland_altman_lower: float
    bland_altman_upper: float


def _icc_two_way_random(data: np.ndarray) -> float:
    if data.shape[1] != 2:
        raise ValueError("ICC calculation expects exactly two raters (manual vs predicted).")
    n, k = data.shape
    mean_per_target = np.mean(data, axis=1, keepdims=True)
    mean_per_rater = np.mean(data, axis=0, keepdims=True)
    grand_mean = np.mean(data)
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_between_targets = k * ((mean_per_target - grand_mean) ** 2).sum()
    ss_between_raters = n * ((mean_per_rater - grand_mean) ** 2).sum()
    ss_residual = ss_total - ss_between_targets - ss_between_raters
    df_between_targets = n - 1
    df_between_raters = k - 1
    df_residual = (n - 1) * (k - 1)
    ms_between_targets = ss_between_targets / df_between_targets if df_between_targets else 0.0
    ms_between_raters = ss_between_raters / df_between_raters if df_between_raters else 0.0
    ms_residual = ss_residual / df_residual if df_residual else 0.0
    numerator = ms_between_targets - ms_residual
    denominator = ms_between_targets + (k - 1) * ms_residual + (k * (ms_between_raters - ms_residual) / n)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _bland_altman_limits(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - truth
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    return {
        "mean": mean_diff,
        "lower": float(loa_lower),
        "upper": float(loa_upper),
    }


def evaluate_measurements(truth: np.ndarray, pred: np.ndarray) -> MetricBundle:
    mae = float(mean_absolute_error(truth, pred))
    rmse = float(mean_squared_error(truth, pred, squared=False))
    icc = _icc_two_way_random(np.column_stack([truth, pred]))
    bland = _bland_altman_limits(truth, pred)
    return MetricBundle(
        mae=mae,
        rmse=rmse,
        icc=icc,
        bland_altman_mean=bland["mean"],
        bland_altman_lower=bland["lower"],
        bland_altman_upper=bland["upper"],
    )


def bundle_to_dict(bundle: MetricBundle) -> Dict[str, float]:
    return {
        "mae": bundle.mae,
        "rmse": bundle.rmse,
        "icc": bundle.icc,
        "bland_altman_mean": bundle.bland_altman_mean,
        "bland_altman_lower": bundle.bland_altman_lower,
        "bland_altman_upper": bundle.bland_altman_upper,
    }
