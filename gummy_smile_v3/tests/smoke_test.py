from __future__ import annotations

import json
from pathlib import Path

from gummy_smile_v3.master_pipeline_v3 import run_pipeline


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    image_candidates = list((repo_root / "gummy_smile_v3" / "data" / "raw_images").glob("**/*.jpg"))
    if not image_candidates:
        raise FileNotFoundError("No sample images found for smoke test.")
    image_path = image_candidates[0]
    config_path = repo_root / "gummy_smile_v3" / "configs" / "config.yaml"
    output_root = repo_root / "gummy_smile_v3" / "results" / "smoke_test"

    run_dir = run_pipeline(
        image_path=image_path,
        config_path=config_path,
        output_dir_override=output_root,
        use_stub=True,
    )

    v1_csv = run_dir / "v1_xgboost_predictions.csv"
    v3_csv = run_dir / "v3_yolo_predictions.csv"
    report_path = run_dir / "report.json"

    if not v1_csv.exists():
        raise AssertionError("v1_xgboost_predictions.csv not created.")
    if not v3_csv.exists():
        raise AssertionError("v3_yolo_predictions.csv not created.")
    if not report_path.exists():
        raise AssertionError("report.json not created.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    v3_result = report.get("v3_result", {})
    if not (v3_result.get("gum_visibility_mm") or v3_result.get("gum_visibility_px")):
        raise AssertionError("report.json missing gum visibility values.")
    if not v3_result.get("etiology_class") or not v3_result.get("treatment_class"):
        raise AssertionError("Etiology/treatment not populated in report.json.")

    v1_result = report.get("v1_result", {})
    required_v1_keys = {
        "method",
        "gum_visibility_px",
        "gum_visibility_mm",
        "etiology_class",
        "treatment_class",
        "notes",
    }
    missing_v1_keys = required_v1_keys - set(v1_result)
    if missing_v1_keys:
        raise AssertionError(f"report.json missing v1 fields: {sorted(missing_v1_keys)}")


if __name__ == "__main__":
    main()
