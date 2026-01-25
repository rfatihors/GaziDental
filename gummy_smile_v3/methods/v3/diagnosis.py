from __future__ import annotations

"""Etiology/treatment recommendations based on mean gingival display."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class DiagnosisResult:
    etiology_code: str
    etiology: List[str]
    treatment_code: str
    treatment: List[str]


DIAGNOSIS_RULES = [
    {
        "label": "E1",
        "min_mm": None,
        "max_mm": 4.0,
        "etiology": [
            "Gecikmiş pasif erüpsiyon",
            "Gingival hipertrofi",
            "Kalın gingival biyotip",
        ],
        "treatment": [
            "Gingivektomi",
            "Gingivoplasti",
        ],
        "treatment_code": "T1",
    },
    {
        "label": "E2",
        "min_mm": 4.0,
        "max_mm": 6.0,
        "etiology": [
            "Hiperaktif (Hipermobil) üst dudak",
            "Kısa üst dudak (<20 mm)",
        ],
        "treatment": [
            "Lip repositioning",
            "Botox",
        ],
        "treatment_code": "T2",
    },
    {
        "label": "E3",
        "min_mm": 6.0,
        "max_mm": 8.0,
        "etiology": [
            "Dentoalveolar ekstrüzyon",
            "Derin kapanış",
        ],
        "treatment": [
            "Ortodontik tedavi",
            "Osteotomi",
        ],
        "treatment_code": "T3",
    },
    {
        "label": "E4",
        "min_mm": 8.0,
        "max_mm": None,
        "etiology": [
            "Artmış vertikal maxiller yükseklik",
        ],
        "treatment": [
            "LeFort I osteotomi",
        ],
        "treatment_code": "T4",
    },
]


def _match_rule(mean_mm: float) -> Optional[DiagnosisResult]:
    for rule in DIAGNOSIS_RULES:
        min_mm = rule["min_mm"]
        max_mm = rule["max_mm"]
        lower_ok = True if min_mm is None else mean_mm >= min_mm
        upper_ok = True if max_mm is None else mean_mm < max_mm
        if lower_ok and upper_ok:
            return DiagnosisResult(
                etiology_code=rule["label"],
                etiology=rule["etiology"],
                treatment_code=rule["treatment_code"],
                treatment=rule["treatment"],
            )
    return None


def generate_diagnosis(measurements_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(measurements_path)
    if "mean_mm" not in df.columns:
        raise KeyError("mean_mm column is required in measurement output")

    rows = []
    for _, row in df.iterrows():
        mean_mm = float(row["mean_mm"]) if pd.notna(row["mean_mm"]) else float("nan")
        if np.isnan(mean_mm):
            diagnosis = None
        else:
            diagnosis = _match_rule(mean_mm)

        rows.append(
            {
                "patient_id": row.get("patient_id"),
                "mean_mm": mean_mm,
                "etiology_code": diagnosis.etiology_code if diagnosis else None,
                "etiology": "; ".join(diagnosis.etiology) if diagnosis else None,
                "treatment_code": diagnosis.treatment_code if diagnosis else None,
                "treatment": "; ".join(diagnosis.treatment) if diagnosis else None,
            }
        )

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_path
