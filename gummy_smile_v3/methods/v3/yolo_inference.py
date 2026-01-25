from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_v3_measurements(measurements_path: Path) -> pd.DataFrame:
    df = pd.read_csv(measurements_path)
    if "case_id" in df.columns and "patient_id" not in df.columns:
        df = df.rename(columns={"case_id": "patient_id"})
    if "mean_mm" not in df.columns:
        raise KeyError("mean_mm column is required in v3 measurements")
    return df
