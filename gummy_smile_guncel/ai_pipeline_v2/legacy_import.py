"""
Legacy data importer for gummy smile pipeline v2.

This utility standardizes legacy XGBoost severity predictions and manual
measurement spreadsheets into the ai_pipeline_v2 workspace without touching the
old pipeline files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - fallback when pandas missing
    HAS_PANDAS = False


def _ensure_dirs(base_dir: Path) -> None:
    for directory in [base_dir / "data", base_dir / "results", base_dir / "models"]:
        directory.mkdir(parents=True, exist_ok=True)


def _detect_case_column(columns: List[str]) -> Optional[str]:
    for candidate in ["case_id", "patient_id", "image_id", "filename", "id"]:
        if candidate in columns:
            return candidate
    return None


def _standardize_predictions(df) -> "pd.DataFrame":
    case_col = _detect_case_column(df.columns)
    if case_col is None:
        df = df.copy()
        df["case_id"] = [f"case_{idx}" for idx in range(len(df))]
        case_col = "case_id"
    pred_col = None
    for candidate in [
        "predicted_severity",
        "legacy_predicted_severity",
        "prediction",
        "severity_pred",
        "severity_prediction",
    ]:
        if candidate in df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        raise KeyError("Legacy prediction file does not contain a severity prediction column.")

    severity_col = None
    for candidate in ["severity", "label", "truth", "target"]:
        if candidate in df.columns:
            severity_col = candidate
            break

    standardized = pd.DataFrame()
    standardized["case_id"] = df[case_col].astype(str)
    standardized["legacy_predicted_severity"] = pd.to_numeric(df[pred_col], errors="coerce")
    if severity_col:
        standardized["severity"] = pd.to_numeric(df[severity_col], errors="coerce")
    return standardized


def _standardize_manual_measurements(df) -> "pd.DataFrame":
    case_col = _detect_case_column(df.columns)
    if case_col is None:
        df = df.copy()
        df["case_id"] = [f"case_{idx}" for idx in range(len(df))]
        case_col = "case_id"

    measurement_cols: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if any(lower.startswith(prefix) for prefix in ["manual_mm_", "manual_mm", "mm", "manual_gingival_"]):
            if any(char.isdigit() for char in lower):
                measurement_cols.append(col)
    measurement_cols = measurement_cols[:6]
    if not measurement_cols:
        raise KeyError("Manual measurement file does not contain mm columns.")

    standardized = pd.DataFrame()
    standardized["case_id"] = df[case_col].astype(str)
    renamed_cols: Dict[str, str] = {}
    for idx, col in enumerate(measurement_cols, start=1):
        renamed_cols[col] = f"manual_mm_{idx}"
    numeric = df[measurement_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.rename(columns=renamed_cols)
    standardized = pd.concat([standardized, numeric], axis=1)
    manual_cols = list(numeric.columns)
    standardized["manual_mean_mm"] = numeric.mean(axis=1)
    standardized["manual_max_mm"] = numeric.max(axis=1)
    return standardized


def _load_csv(path: Path):
    if not HAS_PANDAS:
        import csv

        rows: List[Dict[str, str]] = []
        with path.open() as csvfile:
            headers = csvfile.readline().strip().split(",")
            for line in csvfile:
                values = line.strip().split(",")
                rows.append(dict(zip(headers, values)))
        return rows

    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _merge_predictions(base_dir: Path, legacy_df) -> Path:
    target_path = base_dir / "data" / "severity_predictions.csv"
    if HAS_PANDAS:
        legacy_std = _standardize_predictions(legacy_df)
        if target_path.exists():
            existing = pd.read_csv(target_path)
            case_col = _detect_case_column(existing.columns) or "case_id"
            if case_col != "case_id" and case_col in existing.columns:
                existing = existing.rename(columns={case_col: "case_id"})
            merged = existing.merge(legacy_std, on="case_id", how="outer", suffixes=("", "_legacy"))
        else:
            merged = legacy_std
        merged.to_csv(target_path, index=False)
    else:  # pragma: no cover - minimal fallback
        import csv

        rows = legacy_df if isinstance(legacy_df, list) else []
        headers = list(rows[0].keys()) if rows else ["case_id", "legacy_predicted_severity"]
        with target_path.open("w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    return target_path


def _save_manual_measurements(base_dir: Path, manual_df) -> Path:
    target_path = base_dir / "data" / "manual_measurements.csv"
    if HAS_PANDAS:
        standardized = _standardize_manual_measurements(manual_df)
        standardized.to_csv(target_path, index=False)
    else:  # pragma: no cover - minimal fallback
        import csv

        with target_path.open("w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(manual_df[0].keys()))
            writer.writeheader()
            writer.writerows(manual_df)
    return target_path


def import_legacy(legacy_xgboost: Optional[Path] = None, legacy_manual: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    base_dir = Path(__file__).resolve().parent
    _ensure_dirs(base_dir)

    pred_path: Optional[Path] = None
    manual_path: Optional[Path] = None

    if legacy_xgboost:
        legacy_data = _load_csv(legacy_xgboost)
        if HAS_PANDAS:
            pred_path = _merge_predictions(base_dir, legacy_data)
        else:
            pred_path = _merge_predictions(base_dir, legacy_data)
        print(f"Legacy predictions saved to: {pred_path}")

    if legacy_manual:
        manual_data = _load_csv(legacy_manual)
        if HAS_PANDAS:
            manual_path = _save_manual_measurements(base_dir, manual_data)
        else:
            manual_path = _save_manual_measurements(base_dir, manual_data)
        print(f"Manual measurements saved to: {manual_path}")

    return pred_path, manual_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize legacy outputs into ai_pipeline_v2 workspace.")
    parser.add_argument("--legacy-xgboost", type=Path, default=None, help="Path to legacy XGBoost severity predictions CSV")
    parser.add_argument("--legacy-manual", type=Path, default=None, help="Path to manual measurement CSV/Excel")
    args = parser.parse_args()

    import_legacy(args.legacy_xgboost, args.legacy_manual)
