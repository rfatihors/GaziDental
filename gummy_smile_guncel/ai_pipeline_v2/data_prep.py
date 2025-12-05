"""
Data preparation module for gummy smile severity pipeline (v2).

This module reads the consolidated manual measurements, cleans noisy entries,
computes descriptive statistics, and assigns severity labels based on visual
folder names. The cleaned dataset is saved inside the ai_pipeline_v2 workspace
without touching any legacy code or paths.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NOISE_TOKENS = {"*", "-", "(mesafe yok)", "(mesafe yok )", "mesafe yok", ""}
SEVERITY_MAP = {
    "low smile line": 0,
    "normal smile line": 1,
    "high smile line (gummy smile)": 2,
    "low": 0,
    "normal": 1,
    "high": 2,
}


def load_raw_measurements(excel_path: Path) -> pd.DataFrame:
    """Read the raw measurement spreadsheet."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Measurement file not found: {excel_path}")
    return pd.read_excel(excel_path)


def clean_measurement_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean noisy symbols and coerce numeric measurement columns.

    Returns the cleaned DataFrame and a list of columns treated as numeric
    measurements (excluding obvious categorical text columns).
    """
    cleaned = df.copy()
    cleaned = cleaned.replace(list(NOISE_TOKENS), np.nan)

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(list(NOISE_TOKENS), np.nan)

    measurement_cols: List[str] = []
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            measurement_cols.append(col)
            continue

        if cleaned[col].dtype == object:
            if cleaned[col].astype(str).str.contains(r"[A-Za-z]", na=False).any():
                continue

            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if coerced.notna().any():
                cleaned[col] = coerced
                measurement_cols.append(col)

    return cleaned, measurement_cols


def compute_patient_statistics(df: pd.DataFrame, measurement_cols: List[str]) -> pd.DataFrame:
    """Compute mean, max, and min measurements per patient."""
    if not measurement_cols:
        raise ValueError("No numeric measurement columns found for aggregation.")

    df = df.copy()
    df["mean_mm"] = df[measurement_cols].mean(axis=1)
    df["max_mm"] = df[measurement_cols].max(axis=1)
    df["min_mm"] = df[measurement_cols].min(axis=1)
    return df


def assign_severity(df: pd.DataFrame, folder_column: Optional[str] = None) -> pd.DataFrame:
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
    else:
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


def run_data_prep() -> Path:
    """Execute the data preparation workflow and return the output path."""
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir.parent / "olcumler_duzenlenmis.xlsx"
    output_path = base_dir / "data" / "clean_measurements.csv"
    _ensure_workspace_dirs(base_dir)

    raw_df = load_raw_measurements(excel_path)
    cleaned_df, measurement_cols = clean_measurement_values(raw_df)
    stats_df = compute_patient_statistics(cleaned_df, measurement_cols)
    enriched_df = assign_severity(stats_df)
    enriched_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output_file = run_data_prep()
    print(f"Cleaned measurements saved to: {output_file}")
    print("Sample usage: python data_prep.py")
