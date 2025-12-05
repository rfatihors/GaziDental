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


def _train_model(X: pd.DataFrame, y: pd.Series, ids: pd.Series) -> Tuple[XGBRegressor, dict, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42
    )
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
    return model, metrics, (X_train, X_test, y_train, y_test, ids_train, ids_test)


def _save_predictions(patient_ids: pd.Series, predictions, path: Path) -> Path:
    pred_df = pd.DataFrame({
        "patient_id": patient_ids,
        "predicted_mean_mm": predictions,
    })
    pred_df.to_csv(path, index=False)
    return path


def train_and_predict() -> None:
    _ensure_dirs()
    X, y, ids = _load_training_data()
    model, metrics, splits = _train_model(X, y, ids)
    _, X_test, _, _, _, ids_test = splits

    model_path = MODELS_DIR / "xgboost_regressor.pkl"
    joblib.dump(model, model_path)

    test_predictions = model.predict(X_test)
    test_predictions_path = _save_predictions(ids_test, test_predictions, DATA_DIR / "xgboost_test_predictions.csv")

    full_predictions = model.predict(X)
    full_predictions_path = _save_predictions(ids, full_predictions, DATA_DIR / "xgboost_full_predictions.csv")

    metrics_path = RESULTS_DIR / "xgboost_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to: {model_path}")
    print(f"Test predictions saved to: {test_predictions_path}")
    print(f"Full predictions saved to: {full_predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train_and_predict()
