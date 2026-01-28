from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from gummy_smile_v3.measurement.measurement_metrics import bundle_to_dict, evaluate_measurements
from gummy_smile_v3.methods.v3.diagnosis import DIAGNOSIS_RULES


def _load_dataframe(path: Path, required: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if required:
        for src, dest in required.items():
            if src in df.columns and src != dest:
                df = df.rename(columns={src: dest})
    return df


def _validate_columns(df: pd.DataFrame, columns: Dict[str, str], label: str) -> pd.DataFrame:
    missing = [col for col in columns.values() if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {label}: {missing}")
    return df


def _prepare_smileline_labels(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    if "patient_id" not in labels.columns:
        return None
    if "smileline" in labels.columns:
        labels = labels.rename(columns={"smileline": "smileline_type"})
    if "smileline_type" not in labels.columns:
        return None
    return labels[["patient_id", "smileline_type"]]


def _metric_row(label: str, truth: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    bundle = evaluate_measurements(truth, pred)
    row = {"comparison": label}
    row.update(bundle_to_dict(bundle))
    return row


def _severity_labels_from_rules() -> List[str]:
    return [rule["label"] for rule in DIAGNOSIS_RULES]


def _severity_label_from_mm(mean_mm: float) -> Optional[str]:
    for rule in DIAGNOSIS_RULES:
        min_mm = rule["min_mm"]
        max_mm = rule["max_mm"]
        lower_ok = True if min_mm is None else mean_mm >= min_mm
        upper_ok = True if max_mm is None else mean_mm < max_mm
        if lower_ok and upper_ok:
            return rule["label"]
    return None


def derive_severity_labels(mean_mm_values: Sequence[float]) -> pd.Series:
    labels = []
    for value in mean_mm_values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            labels.append(None)
        else:
            labels.append(_severity_label_from_mm(float(value)))
    return pd.Series(labels)


def compute_severity_metrics(
    truth_labels: Sequence[Optional[str]],
    predicted_labels: Sequence[Optional[str]],
    output_path: Path,
    label_order: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if len(truth_labels) != len(predicted_labels):
        raise ValueError("truth_labels and predicted_labels must be the same length")

    metrics_df = pd.DataFrame({"truth": truth_labels, "pred": predicted_labels}).dropna()
    if metrics_df.empty:
        raise ValueError("No overlapping severity labels to evaluate")

    if label_order is None:
        ordered_labels = _severity_labels_from_rules()
    else:
        ordered_labels = list(label_order)

    present = set(metrics_df["truth"]).union(set(metrics_df["pred"]))
    labels = [label for label in ordered_labels if label in present]
    if not labels:
        labels = sorted(present)

    precision, recall, f1, support = precision_recall_fscore_support(
        metrics_df["truth"],
        metrics_df["pred"],
        labels=labels,
        zero_division=0,
    )

    rows = []
    for label, prec, rec, f1_score, count in zip(labels, precision, recall, f1, support):
        rows.append(
            {
                "label": label,
                "precision": prec,
                "recall": rec,
                "f1": f1_score,
                "support": int(count),
                "average": "class",
            }
        )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        metrics_df["truth"],
        metrics_df["pred"],
        labels=labels,
        average="macro",
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        metrics_df["truth"],
        metrics_df["pred"],
        labels=labels,
        average="micro",
        zero_division=0,
    )
    total_support = int(support.sum())
    rows.extend(
        [
            {
                "label": "macro_avg",
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "support": total_support,
                "average": "macro",
            },
            {
                "label": "micro_avg",
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "support": total_support,
                "average": "micro",
            },
        ]
    )

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df


def compare_methods(
    manual_path: Path,
    v1_path: Path,
    v3_path: Path,
    summary_output: Path,
    by_smileline_output: Path,
    smileline_labels: Optional[Path] = None,
) -> None:
    manual_df = _load_dataframe(manual_path, required={"case_id": "patient_id"})
    v1_df = _load_dataframe(v1_path, required={"case_id": "patient_id"})
    v3_df = _load_dataframe(v3_path, required={"case_id": "patient_id"})

    manual_df = _validate_columns(manual_df, {"patient_id": "patient_id", "mean_mm": "mean_mm"}, "manual")
    v1_df = _validate_columns(v1_df, {"patient_id": "patient_id", "predicted_mean_mm": "predicted_mean_mm"}, "v1")
    v3_df = _validate_columns(v3_df, {"patient_id": "patient_id", "mean_mm": "mean_mm"}, "v3")
    v3_df = v3_df.rename(columns={"mean_mm": "mean_mm_v3"})

    merged = manual_df.merge(v1_df[["patient_id", "predicted_mean_mm"]], on="patient_id", how="inner")
    merged = merged.merge(v3_df[["patient_id", "mean_mm_v3"]], on="patient_id", how="inner")

    rows = [
        _metric_row("v1_vs_manual", merged["mean_mm"].to_numpy(), merged["predicted_mean_mm"].to_numpy()),
        _metric_row("v3_vs_manual", merged["mean_mm"].to_numpy(), merged["mean_mm_v3"].to_numpy()),
        _metric_row("v1_vs_v3", merged["predicted_mean_mm"].to_numpy(), merged["mean_mm_v3"].to_numpy()),
    ]
    summary_df = pd.DataFrame(rows)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)

    if smileline_labels is None:
        return

    smileline_df = _prepare_smileline_labels(smileline_labels)
    if smileline_df is None:
        return

    merged_smile = merged.merge(smileline_df, on="patient_id", how="left")
    by_rows = []
    for smile_type, group in merged_smile.groupby("smileline_type"):
        by_rows.append(
            {
                "smileline_type": smile_type,
                **bundle_to_dict(evaluate_measurements(group["mean_mm"].to_numpy(), group["predicted_mean_mm"].to_numpy())),
                "comparison": "v1_vs_manual",
            }
        )
        by_rows.append(
            {
                "smileline_type": smile_type,
                **bundle_to_dict(evaluate_measurements(group["mean_mm"].to_numpy(), group["mean_mm_v3"].to_numpy())),
                "comparison": "v3_vs_manual",
            }
        )
        by_rows.append(
            {
                "smileline_type": smile_type,
                **bundle_to_dict(
                    evaluate_measurements(group["predicted_mean_mm"].to_numpy(), group["mean_mm_v3"].to_numpy())
                ),
                "comparison": "v1_vs_v3",
            }
        )

    by_df = pd.DataFrame(by_rows)
    by_smileline_output.parent.mkdir(parents=True, exist_ok=True)
    by_df.to_csv(by_smileline_output, index=False)
