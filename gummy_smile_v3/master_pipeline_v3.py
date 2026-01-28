from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from gummy_smile_v3.evaluation import evaluate_if_available
from gummy_smile_v3.measurement import measure_gum_visibility
from gummy_smile_v3.methods.v1 import run_xgboost
from gummy_smile_v3.methods.v3 import EtiologyResult, assign_etiology
from gummy_smile_v3.yolo import run_yolo_segmentation


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_optional_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return (root / value).resolve()


def _parse_px_per_mm(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_metadata_lookup(config: Dict[str, Any], repo_root: Path) -> Tuple[Dict[str, str], Optional[str]]:
    source = str(config.get("metadata_source", "filename")).lower()
    if source not in {"csv", "yaml"}:
        return {}, None
    metadata_path = _resolve_optional_path(repo_root, config.get("metadata_path"))
    if metadata_path is None or not metadata_path.exists():
        warning = "Metadata source set to csv/yaml but metadata_path is missing."
        if metadata_path is not None:
            warning = f"Metadata source set to csv/yaml but file not found: {metadata_path}"
        return {}, warning
    if source == "csv":
        df = pd.read_csv(metadata_path)
        if "case_id" in df.columns and "patient_id" not in df.columns:
            df = df.rename(columns={"case_id": "patient_id"})
        label_col = None
        for candidate in ("severity", "severity_label", "label", "class"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            return {}, None
        return dict(zip(df["patient_id"].astype(str), df[label_col].astype(str))), None
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in (data or {}).items()}, None


def _metadata_from_filename(image_path: Path) -> Optional[str]:
    parts = [image_path.stem.lower()] + [part.lower() for part in image_path.parts]
    for candidate in ("low", "normal", "high"):
        if any(candidate in part for part in parts):
            return candidate
    return None


def _get_metadata(
    image_path: Path,
    config: Dict[str, Any],
    metadata_lookup: Dict[str, str],
) -> Optional[str]:
    source = str(config.get("metadata_source", "filename")).lower()
    if source == "filename":
        return _metadata_from_filename(image_path)
    if source in {"csv", "yaml"}:
        return metadata_lookup.get(image_path.stem)
    return None


def _build_output_row(
    image_path: Path,
    method: str,
    gum_visibility_px: Optional[float],
    gum_visibility_mm: Optional[float],
    etiology: Any,
    notes: str,
) -> Dict[str, object]:
    return {
        "image_path": str(image_path),
        "method": method,
        "gum_visibility_px": gum_visibility_px,
        "gum_visibility_mm": gum_visibility_mm,
        "etiology_class": etiology.etiology_class,
        "treatment_class": etiology.treatment_class,
        "etiology_candidates": json.dumps(etiology.etiology_candidates, ensure_ascii=False),
        "treatment_recommendations": json.dumps(etiology.treatment_recommendations, ensure_ascii=False),
        "ambiguous": etiology.ambiguous,
        "notes": notes,
    }


def _etiology_without_calibration(note: str) -> EtiologyResult:
    return EtiologyResult(
        etiology_class="UNCLASSIFIED",
        treatment_class="UNCLASSIFIED",
        etiology_candidates=[],
        treatment_recommendations=[],
        ambiguous=True,
        notes=note,
    )


def run_pipeline(
    image_path: Path,
    config_path: Path,
    weights_override: Optional[Path] = None,
    px_per_mm_override: Optional[float] = None,
    output_dir_override: Optional[Path] = None,
    use_stub: bool = False,
) -> Path:
    repo_root = config_path.parent.parent.resolve()
    config = _load_config(config_path)

    weights_path = weights_override or _resolve_optional_path(repo_root, config.get("weights_path"))
    if not use_stub:
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError("Weights path not found. Please provide --weights path/to/best.pt.")

    px_per_mm = px_per_mm_override if px_per_mm_override is not None else _parse_px_per_mm(config.get("px_per_mm"))
    output_root = output_dir_override or _resolve_optional_path(repo_root, config.get("output_dir")) or repo_root / "results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    yolo_cfg = config.get("yolo", {})
    overlay_cfg = config.get("overlay", {})

    yolo_result = run_yolo_segmentation(
        image_path=image_path,
        weights_path=weights_path,
        output_dir=run_dir,
        conf=float(yolo_cfg.get("conf", 0.25)),
        iou=float(yolo_cfg.get("iou", 0.5)),
        imgsz=int(yolo_cfg.get("imgsz", 1024)),
        max_det=int(yolo_cfg.get("max_det", 5)),
        mask_color=tuple(overlay_cfg.get("mask_color", [0, 255, 0])),
        alpha=float(overlay_cfg.get("alpha", 0.4)),
        use_stub=use_stub,
    )

    if yolo_result["status"] == "ok" and yolo_result["mask_path"]:
        measurement = measure_gum_visibility(
            image_path=image_path,
            mask_path=Path(yolo_result["mask_path"]),
            regions=6,
            px_per_mm=px_per_mm,
        )
        gum_visibility_px = measurement.gum_visibility_px
        gum_visibility_mm = measurement.gum_visibility_mm
    else:
        gum_visibility_px = None
        gum_visibility_mm = None

    metadata_lookup, metadata_warning = _load_metadata_lookup(config, repo_root)
    metadata = _get_metadata(image_path, config, metadata_lookup)
    ambiguous_policy = config.get("ambiguous_policy", {})

    value_unit = "mm" if gum_visibility_mm is not None else "px"
    value_for_rule = gum_visibility_mm if gum_visibility_mm is not None else gum_visibility_px
    if px_per_mm is None:
        v3_etiology = _etiology_without_calibration("px_per_mm not set; E/T classification skipped.")
    else:
        v3_etiology = assign_etiology(value_for_rule, metadata, ambiguous_policy, value_unit=value_unit)
    v3_notes = v3_etiology.notes
    if metadata:
        v3_notes = "; ".join([note for note in [v3_notes, f"metadata={metadata}"] if note])

    v3_row = _build_output_row(
        image_path=image_path,
        method="v3_yolo",
        gum_visibility_px=gum_visibility_px,
        gum_visibility_mm=gum_visibility_mm,
        etiology=v3_etiology,
        notes=v3_notes,
    )
    v3_df = pd.DataFrame([v3_row])
    v3_csv = run_dir / "v3_yolo_predictions.csv"
    v3_df.to_csv(v3_csv, index=False)

    v1_model_path = _resolve_optional_path(repo_root, config.get("v1_model_path"))
    if v1_model_path is None or not v1_model_path.exists():
        raise FileNotFoundError("V1 XGBoost model not found. Please configure v1_model_path.")

    if yolo_result["status"] == "ok" and yolo_result["mask_path"]:
        v1_prediction = run_xgboost(
            mask_path=Path(yolo_result["mask_path"]),
            model_path=v1_model_path,
            px_per_mm=px_per_mm,
        )
        v1_gum_visibility_px = v1_prediction.gum_visibility_px
        v1_gum_visibility_mm = v1_prediction.predicted_mean_mm
    else:
        v1_gum_visibility_px = None
        v1_gum_visibility_mm = None

    if px_per_mm is None:
        v1_etiology = _etiology_without_calibration("px_per_mm not set; E/T classification skipped.")
    else:
        v1_etiology = assign_etiology(
            v1_gum_visibility_mm,
            metadata,
            ambiguous_policy,
            value_unit="mm" if v1_gum_visibility_mm is not None else "px",
        )
    v1_notes = "; ".join(
        note for note in [v1_etiology.notes, "XGBoost prediction used for mm estimate."] if note
    )

    v1_row = _build_output_row(
        image_path=image_path,
        method="v1_xgboost",
        gum_visibility_px=v1_gum_visibility_px,
        gum_visibility_mm=v1_gum_visibility_mm,
        etiology=v1_etiology,
        notes=v1_notes,
    )
    v1_df = pd.DataFrame([v1_row])
    v1_csv = run_dir / "v1_xgboost_predictions.csv"
    v1_df.to_csv(v1_csv, index=False)

    evaluation_output = run_dir / "evaluation.json"
    manual_path = _resolve_optional_path(repo_root, config.get("manual_measurements_path"))
    evaluation = {"status": "SKIP", "reason": "manual_measurements_path not set."}
    if manual_path:
        evaluation = evaluate_if_available(manual_path, v1_df, v3_df, image_path.stem, evaluation_output)
        if evaluation.get("status") == "SKIP":
            evaluation_output.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        evaluation_output.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "image_path": str(image_path),
        "weights_path": str(weights_path) if weights_path else None,
        "px_per_mm": px_per_mm,
        "measurement_unit": "mm" if px_per_mm is not None else "px",
        "output_dir": str(run_dir),
        "yolo": yolo_result,
        "v3_result": v3_row,
        "v1_result": v1_row,
        "evaluation": evaluation,
        "metadata_warning": metadata_warning,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GummySmile v3 single-image pipeline")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image to analyze.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to YOLOv11x-seg weights (best.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--px-per-mm", type=float, default=None, help="Override px_per_mm calibration.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument(
        "--stub-model",
        action="store_true",
        help="Use a stub mask generator instead of YOLO weights (for smoke tests).",
    )
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        config_path=args.config,
        weights_override=args.weights,
        px_per_mm_override=args.px_per_mm,
        output_dir_override=args.output_dir,
        use_stub=args.stub_model,
    )
