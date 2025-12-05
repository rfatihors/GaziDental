"""Unified orchestrator for ai_pipeline_v2.

Runs all pipeline modules in the prescribed order, stopping on first failure
and reporting the status of each stage. All modules must live within
``ai_pipeline_v2`` and their outputs remain in this workspace.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


BASE_DIR = Path(__file__).resolve().parent


def _module_commands() -> List[Tuple[str, List[str]]]:
    python_bin = sys.executable or "python"
    return [
        ("data_prep", [python_bin, str(BASE_DIR / "data_prep.py")]),
        ("manual_measurements", [python_bin, str(BASE_DIR / "manual_measurements.py")]),
        ("auto_measurement", [python_bin, str(BASE_DIR / "auto_measurement.py")]),
        ("xgboost_measurements", [python_bin, str(BASE_DIR / "xgboost_measurements.py")]),
        ("severity_model", [python_bin, str(BASE_DIR / "severity_model.py")]),
        ("rule_based_cds", [python_bin, str(BASE_DIR / "rule_based_cds.py")]),
        ("evaluation", [python_bin, str(BASE_DIR / "evaluation.py")]),
        ("xai_analysis", [python_bin, str(BASE_DIR / "xai_analysis.py")]),
    ]


def run_pipeline() -> None:
    for name, command in _module_commands():
        print(f"\n▶️ Running {name}...")
        result = subprocess.run(command, cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"❌ {name} failed with exit code {result.returncode}. Aborting pipeline.")
            sys.exit(result.returncode)
        print(f"✅ {name} completed successfully")

    print("\n✅ All pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
