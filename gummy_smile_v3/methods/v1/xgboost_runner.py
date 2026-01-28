from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb

from .pixel_features import image_to_pixel_min


@dataclass
class XGBoostPrediction:
    gum_visibility_px: float
    gum_visibility_mm: Optional[float]
    per_region_px: Dict[str, float]
    predicted_mean_mm: float


def _prepare_features(mask_path: Path) -> pd.DataFrame:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask could not be read: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    df_pixels_tmp = image_to_pixel_min(binary)
    df_pixels_tmp = df_pixels_tmp[::-1]
    df_pixels_tmp = df_pixels_tmp.T
    df_pixels_tmp.reset_index(drop=True, inplace=True)
    expected_cols = ["1p", "2p", "3p", "6p", "5p", "4p"]
    if df_pixels_tmp.shape[1] < len(expected_cols):
        missing = len(expected_cols) - df_pixels_tmp.shape[1]
        for _ in range(missing):
            df_pixels_tmp[df_pixels_tmp.shape[1]] = np.nan
    df_pixels_tmp = df_pixels_tmp.iloc[:, : len(expected_cols)]
    df_pixels_tmp = df_pixels_tmp.set_axis(expected_cols, axis=1)
    df_pixels_tmp = df_pixels_tmp.round(0)
    return df_pixels_tmp.fillna(0)


def run_xgboost(
    mask_path: Path,
    model_path: Path,
    px_per_mm: Optional[float] = None,
) -> XGBoostPrediction:
    features = _prepare_features(mask_path)
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    predictions = model.predict(features)
    predicted_values: List[float]
    if predictions.ndim == 1:
        predicted_values = [float(predictions[0])]
    else:
        predicted_values = [float(value) for value in predictions[0]]

    predicted_mean_mm = float(np.mean(predicted_values))
    per_region_px = {column: float(features.iloc[0][column]) for column in features.columns}
    gum_visibility_px = float(np.nanmean(list(per_region_px.values())))
    gum_visibility_mm = float(gum_visibility_px / px_per_mm) if px_per_mm else None

    return XGBoostPrediction(
        gum_visibility_px=gum_visibility_px,
        gum_visibility_mm=gum_visibility_mm,
        per_region_px=per_region_px,
        predicted_mean_mm=predicted_mean_mm,
    )
