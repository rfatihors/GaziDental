"""Rule-based clinical decision support (pipeline v2).

This module reads automatic measurements and supplemental clinician inputs to
assign etiology/treatment codes according to the official E1–T1 standard. The
resulting decisions are stored in ``ai_pipeline_v2/data/clinical_decisions.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def assign_clinical_codes(mean_mm: float) -> Tuple[str, str]:
    """Apply the official E1–T1 clinical standard."""

    if mean_mm > 8:
        return "E4", "T4"
    if mean_mm < 4:
        return "E1", "T1"
    if mean_mm <= 6:
        return "E2", "T2"
    if mean_mm <= 8:
        return "E3", "T3"
    return "E1", "T1"


def _load_auto_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Auto measurements not found at {path}. Run auto_measurement.py first.")
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"patient_{idx}" for idx in range(len(df))])
    mm_cols = [col for col in df.columns if col.startswith("mm_model_")]
    if "mean_mm" not in df.columns:
        if not mm_cols:
            raise KeyError("mean_mm column is missing and mm_model_* columns are unavailable to compute it.")
        df["mean_mm"] = df[mm_cols].mean(axis=1)
    return df[["patient_id", "mean_mm"]]


def _load_etiology_inputs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["patient_id", "etiology", "previous_treatment"])
    df = pd.read_excel(path)
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"patient_{idx}" for idx in range(len(df))])
    df = df.rename(columns={
        "previous": "previous_treatment",
        "previous treatment": "previous_treatment",
        "treatment": "previous_treatment",
    })
    for col in ["etiology", "previous_treatment"]:
        if col not in df.columns:
            df[col] = None
    return df[["patient_id", "etiology", "previous_treatment"]]


def run_rule_based_decisions() -> Path:
    _ensure_dirs()

    auto_path = DATA_DIR / "auto_measurements.csv"
    clinician_input_path = DATA_DIR / "etiology_treatment_inputs.xlsx"
    output_path = DATA_DIR / "clinical_decisions.csv"

    auto_df = _load_auto_measurements(auto_path)
    clinician_df = _load_etiology_inputs(clinician_input_path)
    merged = auto_df.merge(clinician_df, on="patient_id", how="left")

    merged[["etiology_code", "treatment_code"]] = merged["mean_mm"].apply(assign_clinical_codes).tolist()
    merged["proposed_treatment"] = merged["treatment_code"]

    merged.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = run_rule_based_decisions()
    print(f"Clinical decisions saved to: {path}")
