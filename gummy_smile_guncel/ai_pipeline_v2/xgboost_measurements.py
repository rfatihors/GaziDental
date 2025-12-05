"""XGBoost-based regression of gingival measurements (pipeline v2).

This script trains a regressor to predict clinician-derived mean measurements
from manual measurement features (mm1…mm6). Model artefacts, predictions, and
metrics are stored inside the ai_pipeline_v2 workspace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_training_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    manual_cleaned_path = DATA_DIR / "manual_cleaned.csv"
    df = pd.read_csv(manual_cleaned_path)

    feature_cols = [f"mm{i}" for i in range(1, 7) if f"mm{i}" in df.columns]
    if not feature_cols:
        raise ValueError("No measurement features (mm1…mm6) found in manual_cleaned.csv")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["mean_mm"].astype(float)
    ids = df.get("patient_id", pd.Series([f"patient_{idx}" for idx in range(len(df))]))
    return X, y, ids


def _train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[XGBRegressor, dict]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "r2": float(r2_score(y_test, preds)),
    }
    return model, metrics


def _save_predictions(model: XGBRegressor, X: pd.DataFrame, ids: pd.Series) -> Path:
    predictions = model.predict(X)
    pred_df = pd.DataFrame({
        "patient_id": ids,
        "predicted_mean_mm": predictions,
    })
    output_path = DATA_DIR / "xgboost_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    return output_path


def train_and_predict() -> None:
    _ensure_dirs()
    X, y, ids = _load_training_data()
    model, metrics = _train_model(X, y)

    model_path = MODELS_DIR / "xgboost_regressor.pkl"
    joblib.dump(model, model_path)

    predictions_path = _save_predictions(model, X, ids)

    metrics_path = RESULTS_DIR / "xgboost_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train_and_predict()
