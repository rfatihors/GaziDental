"""
Severity classification module for gummy smile pipeline (v2).

Trains LightGBM and XGBoost classifiers on automatic gingival measurements
and selects the best-performing model via grid search. Outputs are stored
entirely inside the ai_pipeline_v2 workspace.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
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
TARGET_COLUMN = "severity"


def _ensure_workspace_dirs(base_dir: Path) -> None:
    required_dirs = [
        base_dir / "data",
        base_dir / "models",
        base_dir / "results",
        base_dir / "results" / "xai",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the prepared automatic measurements dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected auto measurements at {data_path}. Run auto_measurement.py first."
        )
    return pd.read_csv(data_path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract model features and target labels."""
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    features = df[FEATURE_COLUMNS].copy()
    labels = df[TARGET_COLUMN].astype(int)
    return features, labels


def build_model_search_spaces() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    """Define candidate models and their hyperparameter grids."""
    models = {
        "lightgbm": (
            LGBMClassifier(random_state=RANDOM_STATE),
            {
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [100, 200, 400],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
            ),
            {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [150, 300, 500],
                "subsample": [0.8, 1.0],
            },
        ),
    }
    return models


def run_grid_search(
    model, param_grid: Dict[str, List[object]], X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Execute a stratified grid search for the given model."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Compute evaluation metrics for the trained model."""
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return metrics


def select_best_model(X: pd.DataFrame, y: pd.Series) -> Tuple[object, Dict[str, object]]:
    """Train candidate models and return the best-performing estimator."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_model_search_spaces()
    best_model = None
    best_metrics: Dict[str, object] = {}
    best_score = -np.inf

    for name, (model, params) in models.items():
        print(f"Running grid search for {name} ...")
        search = run_grid_search(model, params, X_train, y_train)
        metrics = evaluate_model(search.best_estimator_, X_test, y_test)
        print(
            f"{name} best params: {search.best_params_} | "
            f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_model = search.best_estimator_
            best_metrics = {**metrics, "best_params": search.best_params_, "model": name}

    if best_model is None:
        raise RuntimeError("No model was selected during grid search.")

    return best_model, best_metrics


def save_model(model, output_path: Path) -> None:
    """Persist the trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def run_training() -> Tuple[object, Dict[str, object], Path]:
    """End-to-end severity model training pipeline."""
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    data_path = base_dir / "data" / "auto_measurements.csv"
    model_output_path = base_dir / "models" / "severity_model.pkl"

    dataset = load_dataset(data_path)
    features, labels = prepare_features(dataset)
    model, metrics = select_best_model(features, labels)
    save_model(model, model_output_path)
    return model, metrics, model_output_path


if __name__ == "__main__":
    trained_model, eval_metrics, model_path = run_training()
    print(f"Best model saved to: {model_path}")
    print(f"Evaluation metrics: {eval_metrics}")
    print("Sample usage: python severity_model.py")
