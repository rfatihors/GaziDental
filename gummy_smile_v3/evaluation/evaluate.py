from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from gummy_smile_v3.measurement.measurement_metrics import bundle_to_dict, evaluate_measurements


def _load_manual(manual_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manual_path)
    if "case_id" in df.columns and "patient_id" not in df.columns:
        df = df.rename(columns={"case_id": "patient_id"})
    if "mean_mm" not in df.columns:
        raise KeyError("manual_measurements.csv must contain mean_mm column.")
    return df[["patient_id", "mean_mm"]]


def _extract_value(df: pd.DataFrame, patient_id: str, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    match = df.loc[df["patient_id"] == patient_id, column]
    if match.empty:
        return None
    value = match.iloc[0]
    return float(value) if pd.notna(value) else None


def evaluate_if_available(
    manual_path: Path,
    v1_df: pd.DataFrame,
    v3_df: pd.DataFrame,
    patient_id: str,
    output_path: Path,
) -> Dict[str, object]:
    if not manual_path.exists():
        return {"status": "SKIP", "reason": "manual_measurements.csv not found."}

    manual_df = _load_manual(manual_path)
    manual_value = _extract_value(manual_df, patient_id, "mean_mm")
    if manual_value is None:
        return {"status": "SKIP", "reason": "No manual measurement for this patient_id."}

    v1_value = _extract_value(v1_df, patient_id, "gum_visibility_mm")
    v3_value = _extract_value(v3_df, patient_id, "gum_visibility_mm")

    results: Dict[str, object] = {"status": "OK", "manual_mean_mm": manual_value}

    if v1_value is not None:
        metrics = evaluate_measurements(np.array([manual_value]), np.array([v1_value]))
        results["v1_vs_manual"] = bundle_to_dict(metrics)
    if v3_value is not None:
        metrics = evaluate_measurements(np.array([manual_value]), np.array([v3_value]))
        results["v3_vs_manual"] = bundle_to_dict(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
