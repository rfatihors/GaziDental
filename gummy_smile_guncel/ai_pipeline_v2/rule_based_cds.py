"""
Rule-based clinical decision support for gummy smile management (v2).

This module applies simple thresholds on mean gingival display measurements to
assign treatment categories. All inputs and outputs are confined to the
ai_pipeline_v2 workspace.
"""

from pathlib import Path
from typing import List

import pandas as pd


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


def _load_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Auto measurements not found at {path}. Run auto_measurement.py first."
        )
    return pd.read_csv(path)


def _ensure_mean_mm(df: pd.DataFrame) -> pd.Series:
    if "mean_mm" in df.columns:
        return df["mean_mm"]

    mm_cols = [col for col in df.columns if col.startswith("mm_model_")]
    if not mm_cols:
        raise KeyError(
            "mean_mm column is missing and no mm_model_* columns are available to compute it."
        )
    return df[mm_cols].mean(axis=1)


def _apply_rules(mean_mm_series: pd.Series) -> pd.Series:
    categories = []
    for value in mean_mm_series:
        category = None
        for label, rule_fn in TREATMENT_RULES:
            if pd.notna(value) and rule_fn(value):
                category = label
                break
        categories.append(category)
    return pd.Series(categories, index=mean_mm_series.index, name="treatment_category")


def run_rule_based_decisions() -> Path:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    input_path = base_dir / "data" / "auto_measurements.csv"
    output_path = base_dir / "data" / "clinical_decisions.csv"

    measurements = _load_measurements(input_path)
    mean_mm = _ensure_mean_mm(measurements)
    measurements["treatment_category"] = _apply_rules(mean_mm)

    measurements.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    try:
        decisions_path = run_rule_based_decisions()
        print(f"Clinical decisions saved to: {decisions_path}")
        print("Sample usage: python rule_based_cds.py")
    except Exception as exc:  # noqa: BLE001 - exposed for CLI visibility
        print(f"Rule-based decision support failed: {exc}")
