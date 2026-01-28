from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from gummy_smile_v3.evaluation import (
    compare_methods,
    compute_severity_metrics,
    derive_severity_labels,
    run_intra_observer,
)
from gummy_smile_v3.methods.v3 import generate_diagnosis
from gummy_smile_v3.yolo import predict_and_measure


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_path(root: Path, value: str) -> Path:
    return (root / value).resolve()


def run_pipeline(config_path: Path) -> None:
    repo_root = config_path.parent.parent
    config = _load_config(config_path)

    paths = config["paths"]
    yolo_cfg = config.get("yolo", {})
    measurement_cfg = config.get("measurement", {})

    weights_path = _resolve_path(repo_root, paths["weights"])
    images_dir = _resolve_path(repo_root, paths["images_dir"])
    output_dir = _resolve_path(repo_root, paths["inference_output_dir"])
    measurement_output = _resolve_path(repo_root, paths["measurement_output"])

    print("[Pipeline] Running YOLOv11x-seg inference + measurements...")
    predict_and_measure(
        weights_path=weights_path,
        images_dir=images_dir,
        output_dir=output_dir,
        measurement_output=measurement_output,
        mm_per_pixel=measurement_cfg.get("mm_per_pixel", 1.0),
        regions=measurement_cfg.get("regions", 6),
        conf=yolo_cfg.get("conf", 0.25),
        iou=yolo_cfg.get("iou", 0.5),
        imgsz=yolo_cfg.get("imgsz", 1024),
        max_det=yolo_cfg.get("max_det", 5),
    )

    diagnosis_output = _resolve_path(repo_root, paths["diagnosis_output"])
    manual_path = _resolve_path(repo_root, paths["manual_measurements"])
    v1_path = _resolve_path(repo_root, paths["v1_predictions"])
    v3_path = measurement_output
    summary_output = _resolve_path(repo_root, paths["comparison_summary"])
    by_smileline_output = _resolve_path(repo_root, paths["comparison_by_smileline"])
    smileline_labels = _resolve_path(repo_root, paths["smileline_labels"])
    severity_output = _resolve_path(repo_root, paths["severity_metrics"])

    print("[Pipeline] Comparing V1 (XGBoost) vs V3 (YOLOv11x-seg) vs manual...")
    compare_methods(
        manual_path=manual_path,
        v1_path=v1_path,
        v3_path=v3_path,
        summary_output=summary_output,
        by_smileline_output=by_smileline_output,
        smileline_labels=smileline_labels,
    )

    print("[Pipeline] Generating etiology + treatment recommendations...")
    generate_diagnosis(measurement_output, diagnosis_output)

    if manual_path.exists() and diagnosis_output.exists():
        manual_df = pd.read_csv(manual_path)
        if "case_id" in manual_df.columns and "patient_id" not in manual_df.columns:
            manual_df = manual_df.rename(columns={"case_id": "patient_id"})
        if "patient_id" in manual_df.columns and "mean_mm" in manual_df.columns:
            diagnosis_df = pd.read_csv(diagnosis_output)
            if "patient_id" in diagnosis_df.columns and "etiology_code" in diagnosis_df.columns:
                manual_df["severity_label"] = derive_severity_labels(manual_df["mean_mm"])
                merged = manual_df.merge(
                    diagnosis_df[["patient_id", "etiology_code"]],
                    on="patient_id",
                    how="inner",
                )
                compute_severity_metrics(
                    merged["severity_label"].tolist(),
                    merged["etiology_code"].tolist(),
                    severity_output,
                )
            else:
                print("[Pipeline] Skipping severity metrics (missing diagnosis columns).")
        else:
            print("[Pipeline] Skipping severity metrics (missing manual measurement columns).")
    else:
        print("[Pipeline] Skipping severity metrics (files not found).")

    intra_first = _resolve_path(repo_root, paths["intra_observer_first"])
    intra_last = _resolve_path(repo_root, paths["intra_observer_last"])
    intra_report = _resolve_path(repo_root, paths["intra_observer_report"])

    if intra_first.exists() and intra_last.exists():
        print("[Pipeline] Running intra-observer evaluation...")
        run_intra_observer(intra_first, intra_last, intra_report)
    else:
        print("[Pipeline] Skipping intra-observer evaluation (files not found).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GummySmile v3 master pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
