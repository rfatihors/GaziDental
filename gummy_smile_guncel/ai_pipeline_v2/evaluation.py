"""Evaluation utilities for ai_pipeline_v2.

Compares automatic measurements, clinician-cleaned measurements, and XGBoost
predictions. Outputs regression metrics (MAE, RMSE, ICC) alongside classification
metrics for etiology/treatment codes and saves all artefacts inside the
ai_pipeline_v2 workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error

from .rule_based_cds import assign_clinical_codes


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _load_dataframe(path: Path, required_cols: List[str], default_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if default_id not in df.columns:
        df.insert(0, default_id, [f"{default_id}_{idx}" for idx in range(len(df))])
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing} in {path}")
    return df


def _measurement_metrics(truth: pd.Series, pred: pd.Series) -> Dict[str, float]:
    mae = mean_absolute_error(truth, pred)
    rmse = mean_squared_error(truth, pred, squared=False)
    icc = _icc_two_way_random(np.column_stack([truth, pred]))
    return {"mae": float(mae), "rmse": float(rmse), "icc": float(icc)}


def _classification_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, object]:
    labels = sorted(list(set(true_labels) | set(pred_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    accuracy = (np.array(true_labels) == np.array(pred_labels)).mean() if true_labels else float("nan")
    confusion = {
        "labels": labels,
        "matrix": cm.tolist(),
    }
    return {"accuracy": float(accuracy), "confusion_matrix": confusion}


def _merge_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual_path = DATA_DIR / "manual_cleaned.csv"
    auto_path = DATA_DIR / "auto_measurements.csv"
    xgb_path = DATA_DIR / "xgboost_predictions.csv"

    manual_df = _load_dataframe(manual_path, ["mean_mm"], "patient_id")
    auto_df = _load_dataframe(auto_path, [], "patient_id")
    xgb_df = _load_dataframe(xgb_path, ["predicted_mean_mm"], "patient_id")

    mm_cols = [col for col in auto_df.columns if col.startswith("mm_model_")]
    if "mean_mm" not in auto_df.columns:
        if not mm_cols:
            raise KeyError("Auto measurements require mean_mm or mm_model_* columns.")
        auto_df["mean_mm"] = auto_df[mm_cols].mean(axis=1)

    return manual_df, auto_df, xgb_df


def evaluate() -> None:
    _ensure_dirs()
    manual_df, auto_df, xgb_df = _merge_sources()

    merged_auto = manual_df.merge(auto_df[["patient_id", "mean_mm"]], on="patient_id", how="inner", suffixes=("_manual", "_auto"))
    merged_xgb = manual_df.merge(xgb_df[["patient_id", "predicted_mean_mm"]], on="patient_id", how="inner")

    if merged_auto.empty or merged_xgb.empty:
        raise ValueError("Insufficient overlapping patient_ids for evaluation.")

    auto_metrics = _measurement_metrics(merged_auto["mean_mm_manual"], merged_auto["mean_mm_auto"])
    xgb_metrics = _measurement_metrics(merged_xgb["mean_mm"], merged_xgb["predicted_mean_mm"])

    etiology_true = merged_xgb["etiology_code"] if "etiology_code" in merged_xgb.columns else merged_auto.get("etiology_code")
    treatment_true = merged_xgb["treatment_code"] if "treatment_code" in merged_xgb.columns else merged_auto.get("treatment_code")

    if etiology_true is None or treatment_true is None:
        etiology_true = merged_auto.get("etiology_code") or manual_df.get("etiology_code")
        treatment_true = merged_auto.get("treatment_code") or manual_df.get("treatment_code")

    auto_codes = merged_auto["mean_mm_auto"].apply(assign_clinical_codes)
    auto_etiology_pred = [code[0] for code in auto_codes]
    auto_treatment_pred = [code[1] for code in auto_codes]

    xgb_codes = merged_xgb["predicted_mean_mm"].apply(assign_clinical_codes)
    xgb_etiology_pred = [code[0] for code in xgb_codes]
    xgb_treatment_pred = [code[1] for code in xgb_codes]

    etiology_true_list = etiology_true.tolist() if isinstance(etiology_true, pd.Series) else list(etiology_true)
    treatment_true_list = treatment_true.tolist() if isinstance(treatment_true, pd.Series) else list(treatment_true)

    etiology_auto_metrics = _classification_metrics(etiology_true_list, auto_etiology_pred)
    etiology_xgb_metrics = _classification_metrics(etiology_true_list, xgb_etiology_pred)
    treatment_auto_metrics = _classification_metrics(treatment_true_list, auto_treatment_pred)
    treatment_xgb_metrics = _classification_metrics(treatment_true_list, xgb_treatment_pred)

    benchmark_payload = {
        "auto_vs_manual": auto_metrics,
        "xgboost_vs_manual": xgb_metrics,
        "delta_accuracy": float(etiology_xgb_metrics["accuracy"] - etiology_auto_metrics["accuracy"]),
    }
    etiology_treatment_payload = {
        "etiology": {
            "auto": etiology_auto_metrics,
            "xgboost": etiology_xgb_metrics,
        },
        "treatment": {
            "auto": treatment_auto_metrics,
            "xgboost": treatment_xgb_metrics,
        },
    }

    benchmark_path = RESULTS_DIR / "benchmark_comparison.json"
    etiology_path = RESULTS_DIR / "etiology_treatment_metrics.json"

    benchmark_path.write_text(json.dumps(benchmark_payload, indent=2))
    etiology_path.write_text(json.dumps(etiology_treatment_payload, indent=2))

    print(f"Benchmark metrics saved to: {benchmark_path}")
    print(f"Etiology/treatment metrics saved to: {etiology_path}")


if __name__ == "__main__":
    evaluate()
