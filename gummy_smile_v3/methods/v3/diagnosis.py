from __future__ import annotations

"""Etiology/treatment recommendations based on mean gingival display."""

from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Iterable, List, Optional

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


def _severity_from_mean(mean_mm: float) -> Optional[str]:
    if np.isnan(mean_mm):
        return None
    if mean_mm < 2.0:
        return "low"
    if mean_mm < 4.0:
        return "normal"
    return "high"


def _severity_from_coco(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        severity = path.parent.parent.name
        if severity not in {"low", "normal", "high"}:
            continue
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        for image in data.get("images", []):
            filename = image.get("file_name")
            if not filename:
                continue
            rows.append(
                {
                    "patient_id": Path(filename).stem,
                    "severity_ground_truth": severity,
                }
            )
    return pd.DataFrame(rows)


def _severity_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "case_id" in df.columns and "patient_id" not in df.columns:
        df = df.rename(columns={"case_id": "patient_id"})
    severity_column = None
    for candidate in ("severity", "severity_label", "label"):
        if candidate in df.columns:
            severity_column = candidate
            break
    if severity_column is None:
        raise KeyError("Severity labels CSV must include severity, severity_label, or label column.")
    return df[["patient_id", severity_column]].rename(columns={severity_column: "severity_ground_truth"})


def _load_severity_metadata(
    coco_glob: Optional[str] = None,
    labels_csv: Optional[Path] = None,
) -> pd.DataFrame:
    frames = []
    if coco_glob:
        paths = [Path(path) for path in glob.glob(coco_glob, recursive=True)]
        frames.append(_severity_from_coco(paths))
    if labels_csv is not None:
        frames.append(_severity_from_csv(labels_csv))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.dropna(subset=["patient_id"])
    grouped = (
        combined.groupby("patient_id")["severity_ground_truth"]
        .apply(lambda values: values.dropna().unique().tolist())
        .reset_index()
    )
    grouped["severity_ground_truth"] = grouped["severity_ground_truth"].apply(
        lambda labels: labels[0] if len(labels) == 1 else None
    )
    return grouped


def generate_diagnosis(
    measurements_path: Path,
    output_path: Path,
    coco_glob: Optional[str] = None,
    labels_csv: Optional[Path] = None,
) -> Path:
    df = pd.read_csv(measurements_path)
    if "mean_mm" not in df.columns:
        raise KeyError("mean_mm column is required in measurement output")
    severity_df = _load_severity_metadata(coco_glob=coco_glob, labels_csv=labels_csv)
    if not severity_df.empty:
        df = df.merge(severity_df, on="patient_id", how="left")

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
                "severity_ground_truth": row.get("severity_ground_truth"),
                "severity_predicted": _severity_from_mean(mean_mm),
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
