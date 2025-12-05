"""
Rule-based clinical decision support for gummy smile management (v2).

This module applies simple thresholds on mean gingival display measurements to
assign treatment categories. All inputs and outputs are confined to the
ai_pipeline_v2 workspace. Falls back to CSV-only logic when pandas is not
available.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - offline fallback
    HAS_PANDAS = False

TREATMENT_RULES = [
    ("conservative", lambda mean_mm: mean_mm < 2),
    ("botox", lambda mean_mm: 2 <= mean_mm <= 4),
    ("surgery / combined", lambda mean_mm: mean_mm > 4),
]


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs: List[Path] = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _load_measurements(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Auto measurements not found at {path}. Run auto_measurement.py first."
        )
    if HAS_PANDAS:
        return pd.read_csv(path)
    rows = []
    with path.open() as csvfile:
        headers = csvfile.readline().strip().split(",")
        for line in csvfile:
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    return rows


def _ensure_mean_mm(df):
    if HAS_PANDAS:
        if "mean_mm" in df.columns:
            return df["mean_mm"]
        mm_cols = [col for col in df.columns if col.startswith("mm_model_")]
        if not mm_cols:
            raise KeyError(
                "mean_mm column is missing and no mm_model_* columns are available to compute it."
            )
        return df[mm_cols].mean(axis=1)
    mm_cols = [col for col in df[0].keys() if col.startswith("mm_model_")]
    if not mm_cols:
        return [float("nan") for _ in df]
    mean_values = []
    for row in df:
        vals = [float(row.get(col, 0)) for col in mm_cols]
        mean_values.append(sum(vals) / len(vals))
    return mean_values


def _apply_rules(mean_mm_series):
    categories = []
    for value in mean_mm_series:
        try:
            value = float(value)
        except (TypeError, ValueError):
            categories.append(None)
            continue
        category = None
        for label, rule_fn in TREATMENT_RULES:
            if value == value and rule_fn(value):  # check for NaN
                category = label
                break
        categories.append(category)
    if HAS_PANDAS:
        return pd.Series(categories, index=mean_mm_series.index, name="treatment_category")
    return categories


def run_rule_based_decisions() -> Path:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    input_path = base_dir / "data" / "auto_measurements.csv"
    output_path = base_dir / "data" / "clinical_decisions.csv"

    measurements = _load_measurements(input_path)
    mean_mm = _ensure_mean_mm(measurements)
    categories = _apply_rules(mean_mm)

    if HAS_PANDAS:
        measurements["treatment_category"] = categories
        measurements.to_csv(output_path, index=False)
    else:
        headers = list(measurements[0].keys()) + ["treatment_category"]
        with output_path.open("w") as csvfile:
            csvfile.write(",".join(headers) + "\n")
            for row, category in zip(measurements, categories):
                row["treatment_category"] = category
                csvfile.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    return output_path


if __name__ == "__main__":
    try:
        decisions_path = run_rule_based_decisions()
        print(f"Clinical decisions saved to: {decisions_path}")
        print("Sample usage: python rule_based_cds.py")
    except Exception as exc:  # noqa: BLE001 - exposed for CLI visibility
        print(f"Rule-based decision support failed: {exc}")
