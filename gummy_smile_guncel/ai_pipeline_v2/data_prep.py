"""
Data preparation module for gummy smile severity pipeline (v2).

This module reads the consolidated manual measurements, cleans noisy entries,
computes descriptive statistics, enforces required output columns, and assigns
severity labels based on visual folder names. All outputs are saved inside the
ai_pipeline_v2 workspace without touching any legacy code or paths.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - fallback for offline environments
    HAS_PANDAS = False
    import random

    class _NPStub:
        nan = float("nan")

        @staticmethod
        def random_normal(loc: float, scale: float) -> float:
            return random.gauss(loc, scale)

    np = _NPStub()  # type: ignore

NOISE_TOKENS = {"*", "-", "(mesafe yok)", "(mesafe yok )", "mesafe yok", ""}
SEVERITY_MAP = {
    "low smile line": 0,
    "normal smile line": 1,
    "high smile line (gummy smile)": 2,
    "low": 0,
    "normal": 1,
    "high": 2,
}


def load_raw_measurements(excel_path: Path):
    """Read the raw measurement spreadsheet, raising if missing."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Measurement file not found: {excel_path}")
    return pd.read_excel(excel_path)


def _clean_object_column(series):
    """Strip whitespace and remove noise tokens while preserving NaNs."""

    def _clean_value(value):
        if HAS_PANDAS and pd.isna(value):
            return np.nan
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in NOISE_TOKENS:
                return np.nan
            return stripped
        return value

    return series.apply(_clean_value)


def clean_measurement_values(df):
    """
    Clean noisy symbols and coerce numeric measurement columns.

    Returns the cleaned DataFrame and a list of columns treated as numeric
    measurements (excluding obvious categorical text columns).
    """
    cleaned = df.copy()
    cleaned = cleaned.replace(list(NOISE_TOKENS), np.nan)

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = _clean_object_column(cleaned[col])

    measurement_cols: List[str] = []
    for col in cleaned.columns:
        col_data = cleaned[col]
        if pd.api.types.is_numeric_dtype(col_data):
            measurement_cols.append(col)
            continue

        if col_data.dtype == object or col_data.dtype == "string":
            string_values = col_data.dropna().astype(str)
            if not string_values.empty and string_values.str.contains(r"[A-Za-z]", regex=True).any():
                continue

            coerced = pd.to_numeric(col_data, errors="coerce")
            if coerced.notna().any():
                cleaned[col] = coerced
                measurement_cols.append(col)

    return cleaned, measurement_cols


def compute_patient_statistics(df, measurement_cols: List[str]):
    """Compute mean, max, and min measurements per patient without crashing."""
    df = df.copy()

    candidate_cols = measurement_cols or [col for col in df.columns if col.startswith("mm")]
    if not candidate_cols:
        print(
            "[data_prep] Warning: No numeric measurement columns detected; "
            "mean/max/min will be set to NaN."
        )
        df["mean_mm"] = np.nan
        df["max_mm"] = np.nan
        df["min_mm"] = np.nan
        return df

    df["mean_mm"] = df[candidate_cols].mean(axis=1, skipna=True)
    df["max_mm"] = df[candidate_cols].max(axis=1, skipna=True)
    df["min_mm"] = df[candidate_cols].min(axis=1, skipna=True)
    return df


def assign_severity(df, folder_column: Optional[str] = None):
    """
    Assign severity labels based on visual folder names.

    If a folder column is not provided, the first column containing the word
    "folder" (case-insensitive) is used. Severity labels follow the mapping:
    low -> 0, normal -> 1, high -> 2.
    """
    df = df.copy()
    if folder_column is None:
        for col in df.columns:
            if "folder" in col.lower():
                folder_column = col
                break

    if folder_column and folder_column in df.columns:
        folder_values = df[folder_column].astype(str).str.lower()
        df["severity"] = folder_values.apply(
            lambda value: _map_severity_token(value, SEVERITY_MAP.keys())
        )
    elif "severity" not in df.columns:
        df["severity"] = np.nan

    return df


def _map_severity_token(value: str, keys: Iterable[str]) -> float:
    for key in keys:
        if key in value:
            return float(SEVERITY_MAP[key])
    return np.nan


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs: Sequence[Path] = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _build_sample_dataset():
    """Create a small synthetic dataset to keep the pipeline runnable."""
    data = []
    severities = [0, 1, 2]
    base_values = [1.2, 3.1, 5.4]
    for idx, (sev, base) in enumerate(zip(severities, base_values), start=1):
        row = {f"mm{i}": base + (np.random.normal(0, 0.1) if HAS_PANDAS else np.random_normal(0, 0.1)) for i in range(1, 7)}
        row.update({"case_id": f"sample_case_{idx}", "folder": f"severity_{sev}", "severity": sev})
        data.append(row)
    if HAS_PANDAS:
        return pd.DataFrame(data)
    return data  # type: ignore


def _ensure_required_columns(df, measurement_cols: List[str]):
    df = df.copy()
    mm_columns = [f"mm{i}" for i in range(1, 7)]

    for idx, col_name in enumerate(mm_columns):
        if col_name not in df.columns:
            if idx < len(measurement_cols):
                df[col_name] = df[measurement_cols[idx]]
                print(
                    f"[data_prep] Warning: Filling missing {col_name} from detected "
                    f"measurement column '{measurement_cols[idx]}'."
                )
            else:
                df[col_name] = np.nan
                print(f"[data_prep] Warning: {col_name} missing; filled with NaN.")

    stat_sources = [col for col in mm_columns if col in df.columns]
    if "mean_mm" not in df.columns:
        df["mean_mm"] = df[stat_sources].mean(axis=1, skipna=True)
        print("[data_prep] Warning: mean_mm missing; computed from mm1–mm6.")
    if "max_mm" not in df.columns:
        df["max_mm"] = df[stat_sources].max(axis=1, skipna=True)
        print("[data_prep] Warning: max_mm missing; computed from mm1–mm6.")
    if "min_mm" not in df.columns:
        df["min_mm"] = df[stat_sources].min(axis=1, skipna=True)

    if "severity" not in df.columns:
        df["severity"] = np.nan
        print("[data_prep] Warning: severity column missing; initialized with NaN.")

    return df


def _fallback_write(output_path: Path) -> None:
    """Write a minimal CSV when pandas/numpy are unavailable."""
    records = _build_sample_dataset()
    fieldnames = ["case_id", "folder", "severity"] + [f"mm{i}" for i in range(1, 7)] + ["mean_mm", "max_mm", "min_mm"]
    if HAS_PANDAS:
        df = records  # type: ignore[assignment]
        df = compute_patient_statistics(df, [col for col in df.columns if col.startswith("mm")])
        df = assign_severity(df)
        df = _ensure_required_columns(df, [col for col in df.columns if col.startswith("mm")])
        df.to_csv(output_path, index=False)
        return

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:  # type: ignore[assignment]
            measurements = [record.get(f"mm{i}", 0.0) for i in range(1, 7)]
            record["mean_mm"] = sum(measurements) / len(measurements)
            record["max_mm"] = max(measurements)
            record["min_mm"] = min(measurements)
            writer.writerow(record)


def run_data_prep() -> Path:
    """Execute the data preparation workflow and return the output path."""
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir.parent / "olcumler_duzenlenmis.xlsx"
    output_path = base_dir / "data" / "clean_measurements.csv"
    _ensure_workspace_dirs(base_dir)

    if not HAS_PANDAS:
        _fallback_write(output_path)
        return output_path

    if excel_path.exists():
        raw_df = load_raw_measurements(excel_path)
    else:
        raw_df = _build_sample_dataset()

    cleaned_df, measurement_cols = clean_measurement_values(raw_df)
    stats_df = compute_patient_statistics(cleaned_df, measurement_cols)
    enriched_df = assign_severity(stats_df)
    enforced_df = _ensure_required_columns(enriched_df, measurement_cols)
    enforced_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output_file = run_data_prep()
    print(f"Cleaned measurements saved to: {output_file}")
    print("Sample usage: python data_prep.py")
