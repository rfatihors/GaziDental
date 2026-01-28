from __future__ import annotations

from pathlib import Path

from gummy_smile_v3.methods.v3.diagnosis import generate_diagnosis


def run_treatment_prediction(measurements_path: Path, output_path: Path) -> Path:
    return generate_diagnosis(measurements_path=measurements_path, output_path=output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate etiology and treatment recommendations for gummy smile v3."
    )
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_treatment_prediction(measurements_path=args.measurements, output_path=args.output)
