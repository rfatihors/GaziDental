"""Manual measurement standardization for the AI pipeline v2.

This module ingests clinician-provided measurements and etiology/treatment
inputs, harmonises column names, computes summary statistics, and enforces the
official E1–T1 clinical coding standard. The cleaned output is written to
``ai_pipeline_v2/data/manual_cleaned.csv`` for downstream training and
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .rule_based_cds import assign_clinical_codes


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise measurement column names to mm1…mm6 and compute mean_mm."""

    df = df.copy()
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"patient_{idx}" for idx in range(len(df))])

    column_map: Dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if lower.startswith("mm") and any(char.isdigit() for char in lower):
            digits: List[str] = [char for char in lower if char.isdigit()]
            if not digits:
                continue
            idx = int("".join(digits))
            if 1 <= idx <= 6:
                column_map[col] = f"mm{idx}"
        if lower.startswith("manual_mm_"):
            suffix = lower.replace("manual_mm_", "")
            if suffix.isdigit():
                idx = int(suffix)
                if 1 <= idx <= 6:
                    column_map[col] = f"mm{idx}"

    df = df.rename(columns=column_map)
    measurement_cols = [col for col in [f"mm{i}" for i in range(1, 7)] if col in df.columns]
    df[measurement_cols] = df[measurement_cols].apply(pd.to_numeric, errors="coerce")
    df["mean_mm"] = df[measurement_cols].mean(axis=1)
    return df[["patient_id", *measurement_cols, "mean_mm"]]


def _load_etiology_inputs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["patient_id", "etiology", "previous_treatment", "etiology_code", "treatment_code"])
    df = pd.read_excel(path)
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"patient_{idx}" for idx in range(len(df))])
    df = df.rename(columns={
        "previous": "previous_treatment",
        "previous treatment": "previous_treatment",
        "treatment": "previous_treatment",
    })
    for col in ["etiology", "previous_treatment", "etiology_code", "treatment_code"]:
        if col not in df.columns:
            df[col] = None
    return df[["patient_id", "etiology", "previous_treatment", "etiology_code", "treatment_code"]]


def _apply_official_codes(df: pd.DataFrame) -> pd.DataFrame:
    provided_etiology = df.get("etiology_code")
    provided_treatment = df.get("treatment_code")

    codes = df["mean_mm"].apply(assign_clinical_codes)
    df["etiology_code"] = [
        (str(val).upper() if isinstance(val, str) else None) or code[0]
        for val, code in zip(provided_etiology if provided_etiology is not None else [None] * len(df), codes)
    ]
    df["treatment_code"] = [
        (str(val).upper() if isinstance(val, str) else None) or code[1]
        for val, code in zip(provided_treatment if provided_treatment is not None else [None] * len(df), codes)
    ]
    return df


def run_manual_standardisation() -> Path:
    _ensure_dirs()

    manual_path = DATA_DIR / "manual_measurements.csv"
    etiology_path = DATA_DIR / "etiology_treatment_inputs.xlsx"

    manual_df = pd.read_csv(manual_path)
    measurements = _standardize_measurement_columns(manual_df)

    etiology_inputs = _load_etiology_inputs(etiology_path)
    merged = measurements.merge(etiology_inputs, on="patient_id", how="left")

    merged = _apply_official_codes(merged)
    merged["etiology"] = merged["etiology"].fillna("unknown").astype(str)
    merged["previous_treatment"] = merged["previous_treatment"].fillna("none").astype(str)

    output_path = DATA_DIR / "manual_cleaned.csv"
    merged.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output = run_manual_standardisation()
    print(f"Manual measurements cleaned and saved to: {output}")
