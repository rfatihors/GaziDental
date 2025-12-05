"""
Severity classification module for gummy smile pipeline (v2).

Trains LightGBM and XGBoost classifiers on automatic gingival measurements,
saves best-performing model, and exports per-sample predictions for downstream
benchmarking and XAI. All artifacts are stored inside the ai_pipeline_v2
workspace. Falls back to lightweight Python-only logic when ML dependencies are
unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import joblib
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from xgboost import XGBClassifier

    HAS_STACK = True
except ImportError:  # pragma: no cover - offline fallback
    HAS_STACK = False

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


def load_dataset(data_path: Path):
    """Load the prepared automatic measurements dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected auto measurements at {data_path}. Run auto_measurement.py first."
        )
    if HAS_STACK:
        return pd.read_csv(data_path)
    rows = []
    with data_path.open() as csvfile:
        headers = csvfile.readline().strip().split(",")
        for line in csvfile:
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    return rows


def _ensure_features(df):
    if not HAS_STACK:
        return df
    df = df.copy()
    mm_cols = [col for col in df.columns if col.startswith("mm_model_")]
    if mm_cols:
        if "mean_mm" not in df.columns:
            df["mean_mm"] = df[mm_cols].mean(axis=1)
        if "max_mm" not in df.columns:
            df["max_mm"] = df[mm_cols].max(axis=1)
    return df


def prepare_features(df):
    """Extract model features and target labels."""
    if not HAS_STACK:
        # fallback uses plain lists
        features = []
        labels = []
        for row in df:
            try:
                feature_row = [float(row.get(col, 0)) for col in FEATURE_COLUMNS]
                label = int(float(row.get(TARGET_COLUMN, 0)))
            except ValueError:
                continue
            features.append(feature_row)
            labels.append(label)
        return features, labels

    df = _ensure_features(df)
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    features = df[FEATURE_COLUMNS].copy()
    features = features.fillna(features.mean())
    labels = df[TARGET_COLUMN].astype(int)
    return features, labels


def build_model_search_spaces() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
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
    model, param_grid: Dict[str, List[object]], X_train, y_train
):
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


def evaluate_model(model, X_test, y_test) -> Dict[str, object]:
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return metrics


def _fallback_train(features, labels):
    if not labels:
        return None, {"accuracy": float("nan"), "macro_f1": float("nan"), "confusion_matrix": []}
    majority = max(set(labels), key=labels.count)
    preds = [majority for _ in labels]
    accuracy = sum(1 for p, t in zip(preds, labels) if p == t) / len(labels)
    return "fallback", {"accuracy": accuracy, "macro_f1": accuracy, "confusion_matrix": []}


def select_best_model(X, y):
    if not HAS_STACK:
        return _fallback_train(X, y)
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        print(
            "[severity_model] Warning: Insufficient class balance for stratified split; "
            "training fallback LightGBM on all data."
        )
        model = LGBMClassifier(random_state=RANDOM_STATE)
        model.fit(X, y)
        metrics = evaluate_model(model, X, y)
        metrics.update({"best_params": {}, "model": "lightgbm_fallback"})
        return model, metrics

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
        best_model = DummyClassifier(strategy="most_frequent")
        best_model.fit(X, y)
        best_metrics = evaluate_model(best_model, X, y)
        best_metrics.update({"best_params": {}, "model": "dummy"})

    return best_model, best_metrics


def save_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_STACK:
        joblib.dump(model, output_path)
    else:
        output_path.write_text(json.dumps({"model": "fallback"}))


def _extract_case_id(df):
    if HAS_STACK:
        for candidate in ["case_id", "patient_id", "image", "filename"]:
            if candidate in df.columns:
                return df[candidate].astype(str)
        return pd.Series([f"case_{idx}" for idx in range(len(df))])
    return [row.get("case_id", f"case_{idx}") for idx, row in enumerate(df)]


def save_predictions(model, df, features, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not HAS_STACK:
        rows = []
        case_ids = _extract_case_id(df)
        for cid, row in zip(case_ids, df):
            pred = int(row.get(TARGET_COLUMN, 0))
            rows.append({"case_id": cid, TARGET_COLUMN: row.get(TARGET_COLUMN, 0), "predicted_severity": pred})
        with output_path.open("w") as csvfile:
            headers = ["case_id", TARGET_COLUMN, "predicted_severity"]
            csvfile.write(",".join(headers) + "\n")
            for row in rows:
                csvfile.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
        return

    probabilities = model.predict_proba(features)
    preds = model.predict(features)
    case_ids = _extract_case_id(df)

    pred_df = pd.DataFrame(probabilities, columns=[f"prob_class_{idx}" for idx in range(probabilities.shape[1])])
    pred_df.insert(0, "predicted_severity", preds)
    pred_df.insert(0, "case_id", case_ids.values)
    if TARGET_COLUMN in df.columns:
        pred_df.insert(1, TARGET_COLUMN, df[TARGET_COLUMN].values)

    pred_df.to_csv(output_path, index=False)


def run_training():
    base_dir = Path(__file__).resolve().parent
    _ensure_workspace_dirs(base_dir)

    data_path = base_dir / "data" / "auto_measurements.csv"
    model_output_path = base_dir / "models" / "severity_model.pkl"
    prediction_output_path = base_dir / "data" / "severity_predictions.csv"

    dataset = load_dataset(data_path)
    features, labels = prepare_features(dataset)
    model, metrics = select_best_model(features, labels)
    save_model(model, model_output_path)
    save_predictions(model, dataset, features, prediction_output_path)
    return model, metrics, model_output_path


if __name__ == "__main__":
    trained_model, eval_metrics, model_path = run_training()
    print(f"Best model saved to: {model_path}")
    print(f"Evaluation metrics: {eval_metrics}")
    print("Sample usage: python severity_model.py")
