"""
Evaluation module for comparing manual and automatic gingival measurements (v2).

This script loads merged measurements, aligns manual and automatic metrics,
computes regression-style errors (MAE, RMSE, R²) alongside ICC, and optionally
benchmarks new severity predictions against legacy XGBoost outputs. All results
are saved inside the ai_pipeline_v2 workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    HAS_STACK = True
except ImportError:  # pragma: no cover - offline fallback
    HAS_STACK = False


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _load_measurements(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Merged measurement file not found: {path}")
    if HAS_STACK:
        return pd.read_csv(path)
    rows: List[Dict[str, str]] = []
    with path.open() as csvfile:
        headers = csvfile.readline().strip().split(",")
        for line in csvfile:
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    return rows


def _icc_two_way_random_array(data: List[List[float]]) -> float:
    if len(data) == 0 or len(data[0]) != 2:
        return float("nan")
    n = len(data)
    values = data
    mean_per_target = [sum(pair) / len(pair) for pair in values]
    grand_mean = sum([val for pair in values for val in pair]) / (n * 2)
    ss_between_targets = 2 * sum((m - grand_mean) ** 2 for m in mean_per_target)
    ss_total = sum((val - grand_mean) ** 2 for pair in values for val in pair)
    ss_between_raters = n * sum(
        (
            (sum(pair[j] for pair in values) / n - grand_mean)
            ** 2
        )
        for j in range(2)
    )
    ss_residual = ss_total - ss_between_targets - ss_between_raters
    df_between_targets = n - 1
    df_between_raters = 1
    df_residual = (n - 1) * (2 - 1)
    ms_between_targets = ss_between_targets / df_between_targets if df_between_targets else 0.0
    ms_residual = ss_residual / df_residual if df_residual else 0.0
    numerator = ms_between_targets - ms_residual
    denominator = ms_between_targets + (2 - 1) * ms_residual
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _icc_two_way_random(data):
    if HAS_STACK:
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
    return _icc_two_way_random_array(data)


def _align_manual_auto(df) -> Tuple[List[float], List[float]]:
    if HAS_STACK:
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
        return paired["manual"].tolist(), paired["auto"].tolist()

    # fallback list-based alignment
    auto_cols = [col for col in df[0].keys() if col.startswith("mm_model_")]
    manual_candidates = [col for col in df[0].keys() if col.startswith("manual_mm_") or col in {"mean_mm", "manual_mean_mm"}]
    manual_col = manual_candidates[0] if manual_candidates else "mean_mm"
    manual_values: List[float] = []
    auto_values: List[float] = []
    for row in df:
        if manual_col not in row or any(row.get(col) is None for col in auto_cols):
            continue
        try:
            manual_val = float(row.get(manual_col, "nan"))
            auto_val = sum(float(row.get(col, "nan")) for col in auto_cols) / len(auto_cols)
        except ValueError:
            continue
        if math.isnan(manual_val) or math.isnan(auto_val):
            continue
        manual_values.append(manual_val)
        auto_values.append(auto_val)
    return manual_values, auto_values


def _safe_accuracy(preds: List[float], truth: List[float]) -> float:
    paired = [(p, t) for p, t in zip(preds, truth) if not (math.isnan(p) or math.isnan(t))]
    if not paired:
        return float("nan")
    correct = sum(1 for p, t in paired if p == t)
    return correct / len(paired)


def _load_legacy_predictions(path: Path):
    if HAS_STACK:
        df = pd.read_csv(path)
        case_col = None
        for candidate in ["case_id", "patient_id", "image_id", "filename"]:
            if candidate in df.columns:
                case_col = candidate
                break
        pred_col = None
        for candidate in ["predicted_severity", "severity_pred", "prediction", "severity"]:
            if candidate in df.columns:
                pred_col = candidate
                break
        if pred_col is None:
            raise KeyError("Legacy file does not contain a severity prediction column.")
        predictions = df[pred_col].astype(float)
        if case_col:
            predictions.index = df[case_col].astype(str)
        truth = df["severity"].astype(float) if "severity" in df.columns else None
        return predictions, truth
    rows = []
    with path.open() as csvfile:
        headers = csvfile.readline().strip().split(",")
        for line in csvfile:
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    predictions = {row.get("case_id", str(idx)): float(row.get("predicted_severity", row.get("prediction", 0))) for idx, row in enumerate(rows)}
    truth = {row.get("case_id", str(idx)): float(row.get("severity", "nan")) for idx, row in enumerate(rows)} if any("severity" in r for r in rows) else None
    return predictions, truth


def _compute_basic_metrics(manual: List[float], auto: List[float]) -> Dict[str, float]:
    paired = list(zip(manual, auto))
    if not paired:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "icc": float("nan")}
    mae = sum(abs(m - a) for m, a in paired) / len(paired)
    rmse = math.sqrt(sum((m - a) ** 2 for m, a in paired) / len(paired))
    mean_manual = sum(manual) / len(manual)
    ss_tot = sum((m - mean_manual) ** 2 for m in manual)
    ss_res = sum((m - a) ** 2 for m, a in paired)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    icc = _icc_two_way_random_array([[m, a] for m, a in paired])
    return {"mae": mae, "rmse": rmse, "r2": r2, "icc": icc}


def evaluate_measurements(
    measurement_path: Optional[Path] = None,
    legacy_xgboost_csv: Optional[Path] = None,
    legacy_manual_csv: Optional[Path] = None,
) -> Dict[str, float]:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)
    measurement_path = measurement_path or base_dir / "data" / "auto_measurements.csv"

    df = _load_measurements(measurement_path)
    manual, auto = _align_manual_auto(df)

    if HAS_STACK:
        mae = mean_absolute_error(manual, auto)
        rmse = mean_squared_error(manual, auto, squared=False)
        r2 = r2_score(manual, auto)
        icc = _icc_two_way_random(np.column_stack([manual, auto]))
        results = {"mae": mae, "rmse": rmse, "r2": r2, "icc": icc}
    else:
        results = _compute_basic_metrics(manual, auto)

    output_path = base_dir / "results" / "measurement_eval.json"
    Path(output_path).write_text(json.dumps(results, indent=2))

    benchmark: Dict[str, float] = {}

    if legacy_xgboost_csv:
        legacy_preds, legacy_truth = _load_legacy_predictions(legacy_xgboost_csv)
        new_pred_path = base_dir / "data" / "severity_predictions.csv"
        if new_pred_path.exists():
            new_rows = _load_measurements(new_pred_path)
            if HAS_STACK:
                new_series = new_rows.set_index("case_id")["predicted_severity"].astype(float)
                truth_series = new_rows.set_index("case_id")["severity"].astype(float) if "severity" in new_rows.columns else None
            else:
                new_series = {row.get("case_id", str(idx)): float(row.get("predicted_severity", 0)) for idx, row in enumerate(new_rows)}
                truth_series = {row.get("case_id", str(idx)): float(row.get("severity", "nan")) for idx, row in enumerate(new_rows)} if any("severity" in r for r in new_rows) else None
            if HAS_STACK:
                common = new_series.index.intersection(legacy_preds.index)
                if len(common) > 0:
                    benchmark["prediction_agreement"] = float((new_series.loc[common] == legacy_preds.loc[common]).mean())
                if truth_series is not None:
                    new_acc = _safe_accuracy(new_series.tolist(), truth_series.tolist())
                    legacy_acc = _safe_accuracy(legacy_preds.tolist(), truth_series.reindex(legacy_preds.index).tolist())
                    benchmark["new_accuracy"] = new_acc
                    benchmark["legacy_accuracy"] = legacy_acc
                    benchmark["delta_accuracy"] = new_acc - legacy_acc if not math.isnan(new_acc) and not math.isnan(legacy_acc) else float("nan")
            else:
                common_keys = set(new_series.keys()) & set(legacy_preds.keys())
                if common_keys:
                    agreement = sum(1 for k in common_keys if new_series[k] == legacy_preds[k]) / len(common_keys)
                    benchmark["prediction_agreement"] = agreement
                if truth_series:
                    truth_list = [truth_series.get(k, math.nan) for k in new_series.keys()]
                    new_acc = _safe_accuracy(list(new_series.values()), truth_list)
                    legacy_acc = _safe_accuracy(list(legacy_preds.values()), [truth_series.get(k, math.nan) for k in legacy_preds.keys()])
                    benchmark["new_accuracy"] = new_acc
                    benchmark["legacy_accuracy"] = legacy_acc
                    benchmark["delta_accuracy"] = new_acc - legacy_acc if not math.isnan(new_acc) and not math.isnan(legacy_acc) else float("nan")

    if benchmark:
        benchmark_path = base_dir / "results" / "benchmark_comparison.json"
        benchmark_payload = {"measurement": results, "benchmark": benchmark}
        benchmark_path.write_text(json.dumps(benchmark_payload, indent=2))
        print(f"Benchmark results saved to: {benchmark_path}")

    print("Measurement evaluation metrics:")
    for key, value in results.items():
        try:
            print(f"- {key}: {float(value):.4f}")
        except (TypeError, ValueError):
            print(f"- {key}: {value}")
    print(f"Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate automatic measurements and optional legacy benchmarks.")
    parser.add_argument("--legacy_xgboost_csv", type=Path, default=None, help="Optional legacy XGBoost severity predictions CSV")
    parser.add_argument("--legacy_manual_csv", type=Path, default=None, help="Optional legacy manual measurement CSV")
    args = parser.parse_args()

    evaluate_measurements(
        legacy_xgboost_csv=args.legacy_xgboost_csv,
        legacy_manual_csv=args.legacy_manual_csv,
    )
    print("Sample usage: python evaluation.py --legacy_xgboost_csv ../old/results.csv")
