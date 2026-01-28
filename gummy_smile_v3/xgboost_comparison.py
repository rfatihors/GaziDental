from __future__ import annotations

from pathlib import Path

from gummy_smile_v3.evaluation.method_comparison import compare_methods


def run_xgboost_comparison(
    manual_path: Path,
    v1_path: Path,
    v3_path: Path,
    summary_output: Path,
    by_smileline_output: Path,
    smileline_labels: Path | None = None,
) -> None:
    compare_methods(
        manual_path=manual_path,
        v1_path=v1_path,
        v3_path=v3_path,
        summary_output=summary_output,
        by_smileline_output=by_smileline_output,
        smileline_labels=smileline_labels,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare v1 (XGBoost) predictions with v3 (YOLO) measurements."
    )
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--by-smileline-output", type=Path, required=True)
    parser.add_argument("--smileline-labels", type=Path)
    args = parser.parse_args()

    run_xgboost_comparison(
        manual_path=args.manual,
        v1_path=args.v1,
        v3_path=args.v3,
        summary_output=args.summary_output,
        by_smileline_output=args.by_smileline_output,
        smileline_labels=args.smileline_labels,
    )
