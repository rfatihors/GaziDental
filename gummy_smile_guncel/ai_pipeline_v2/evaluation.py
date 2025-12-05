"""
Evaluation module for comparing manual and automatic gingival measurements (v2).

This script loads merged measurements, aligns manual and automatic metrics,
computes regression-style errors (MAE, RMSE, R²) alongside ICC, and saves the
results inside the ai_pipeline_v2 workspace.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _load_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Merged measurement file not found: {path}")
    return pd.read_csv(path)


def _icc_two_way_random(data: np.ndarray) -> float:
    """Compute ICC(2,1) for two raters given an (n_subjects x 2) array."""
    if data.shape[1] != 2:
        raise ValueError("ICC calculation expects exactly two raters (manual vs auto).")

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


def _align_manual_auto(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    auto_cols = [col for col in df.columns if col.startswith("mm_model_")]
    if not auto_cols:
        raise ValueError("Automatic measurement columns (mm_model_*) are missing.")

    auto_mean = df[auto_cols].mean(axis=1)

    manual_candidates: Iterable[str] = [
        col
        for col in df.columns
        if col.startswith("manual_mm_") or col in {"mean_mm", "manual_mean_mm"}
    ]
    if manual_candidates:
        manual_series = df[manual_candidates[0]]
    elif "mean_mm" in df.columns:
        manual_series = df["mean_mm"]
    else:
        raise ValueError("Manual measurement column (mean_mm or manual_mm_*) is missing.")

    paired = pd.concat([manual_series, auto_mean], axis=1, keys=["manual", "auto"])
    paired = paired.dropna()
    if paired.empty:
        raise ValueError("No overlapping manual/auto measurements after dropping NaNs.")
    return paired["manual"], paired["auto"]


def evaluate_measurements(measurement_path: Optional[Path] = None) -> Dict[str, float]:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)
    measurement_path = measurement_path or base_dir / "data" / "auto_measurements.csv"

    df = _load_measurements(measurement_path)
    manual, auto = _align_manual_auto(df)

    mae = mean_absolute_error(manual, auto)
    rmse = mean_squared_error(manual, auto, squared=False)
    r2 = r2_score(manual, auto)
    icc = _icc_two_way_random(np.column_stack([manual.values, auto.values]))

    results = {"mae": mae, "rmse": rmse, "r2": r2, "icc": icc}
    output_path = base_dir / "results" / "measurement_eval.json"
    pd.Series(results).to_json(output_path, indent=2)

    print("Measurement evaluation metrics:")
    for key, value in results.items():
        print(f"- {key}: {value:.4f}")
    print(f"Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    evaluate_measurements()
    print("Sample usage: python evaluation.py")
