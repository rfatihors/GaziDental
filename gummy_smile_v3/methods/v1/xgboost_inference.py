from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_v1_predictions(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    if "case_id" in df.columns and "patient_id" not in df.columns:
        df = df.rename(columns={"case_id": "patient_id"})
    if "predicted_mean_mm" not in df.columns:
        raise KeyError("predicted_mean_mm column is required in v1 predictions")
    return df
