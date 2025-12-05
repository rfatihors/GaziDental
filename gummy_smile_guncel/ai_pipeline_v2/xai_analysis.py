"""
Explainable AI module for severity classifier and geometric overlays (v2).

This module produces SHAP-based explanations for the trained severity model
and renders visual overlays that highlight gingival masks, zenith points, and
lip reference lines. All artifacts are stored inside the ai_pipeline_v2
workspace.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from auto_measurement import (
    _clean_mask,
    _estimate_lip_line_y,
    _find_main_contour,
    _find_zenith_points,
    _load_binary_mask,
    _split_regions,
)


FEATURE_COLUMNS: List[str] = [
    "mm_model_1",
    "mm_model_2",
    "mm_model_3",
    "mm_model_4",
    "mm_model_5",
    "mm_model_6",
    "mean_mm",
    "max_mm",
]


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)


def _load_inputs(base_dir: Path) -> Tuple[pd.DataFrame, object]:
    data_path = base_dir / "data" / "auto_measurements.csv"
    model_path = base_dir / "models" / "severity_model.pkl"

    if not data_path.exists():
        raise FileNotFoundError(
            "Automatic measurements not found. Run auto_measurement.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            "Severity model not found. Train via severity_model.py first."
        )

    df = pd.read_csv(data_path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required feature columns: {missing}")

    model = joblib.load(model_path)
    return df, model


def _select_class_specific_values(shap_values, expected_value, class_index: int):
    if isinstance(shap_values, list):
        values = shap_values[class_index]
        base = expected_value[class_index] if isinstance(expected_value, list) else expected_value
        return values, base
    return shap_values, expected_value


def compute_shap_values(model, features: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    expected_value = explainer.expected_value
    return explainer, shap_values, expected_value


def save_global_importance(
    shap_values,
    features: pd.DataFrame,
    output_path: Path,
    class_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values, _ = _select_class_specific_values(shap_values, 0.0, class_index)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_force_plot(
    explainer: shap.TreeExplainer,
    shap_values,
    expected_value,
    features: pd.DataFrame,
    class_index: int,
    sample_index: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values, base = _select_class_specific_values(shap_values, expected_value, class_index)
    force = shap.force_plot(base, values[sample_index], features.iloc[sample_index], matplotlib=False)
    shap.save_html(str(output_path), force)


def _choose_image_and_mask(base_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    images_dir = base_dir / "data" / "images"
    masks_dir = base_dir / "data" / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return None, None
    for image_path in images_dir.glob("*.png"):
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            return image_path, mask_path
    return None, None


def render_visual_overlay(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    binary_mask = _load_binary_mask(mask_path)
    cleaned_mask = _clean_mask(binary_mask)
    contour = _find_main_contour(cleaned_mask)

    x, y, w, h = cv2.boundingRect(contour)
    bounds = _split_regions((x, y, w, h), regions=6)
    zenith_points = _find_zenith_points(contour, bounds)
    lip_line_y = _estimate_lip_line_y(image, contour)

    mask_rgb = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    mask_overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    for idx, (zx, zy) in enumerate(zenith_points, start=1):
        cv2.circle(mask_overlay, (zx, zy), 4, (0, 0, 255), -1)
        cv2.putText(
            mask_overlay,
            f"Z{idx}",
            (zx + 3, zy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.line(
        mask_overlay,
        (x, int(lip_line_y)),
        (x + w, int(lip_line_y)),
        (255, 0, 0),
        2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask_overlay)


def run_xai_analysis(sample_index: int = 0) -> Path:
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    df, model = _load_inputs(base_dir)
    features = df[FEATURE_COLUMNS].copy()

    explainer, shap_values, expected_value = compute_shap_values(model, features)

    class_probs = model.predict_proba(features)
    target_class = int(np.argmax(class_probs[sample_index]))

    output_dir = base_dir / "results" / "xai"
    global_path = output_dir / "global_feature_importance.png"
    save_global_importance(shap_values, features, global_path, target_class)

    force_path = output_dir / f"patient_{sample_index}_force.html"
    save_force_plot(
        explainer,
        shap_values,
        expected_value,
        features,
        target_class,
        sample_index,
        force_path,
    )

    image_path, mask_path = _choose_image_and_mask(base_dir)
    if image_path and mask_path:
        overlay_path = output_dir / "visual_overlay.png"
        render_visual_overlay(image_path, mask_path, overlay_path)
    else:
        print("No matching image/mask pair found for visual overlay; skipping.")

    return output_dir


if __name__ == "__main__":
    try:
        output_folder = run_xai_analysis()
        print(f"XAI artifacts saved to: {output_folder}")
        print("Sample usage: python xai_analysis.py")
    except Exception as exc:  # noqa: BLE001 - surfaced for CLI visibility
        print(f"XAI analysis failed: {exc}")
