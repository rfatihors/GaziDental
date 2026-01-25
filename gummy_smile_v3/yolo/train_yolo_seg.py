from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def train_yolo_seg(
    data_yaml: Path,
    weights_path: Path,
    output_dir: Path,
    imgsz: int = 1024,
    batch: int = 8,
    epochs: int = 150,
    device: str = "0",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
        project=str(output_dir),
        name="yolo_v11x_seg",
        exist_ok=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv11x-seg for gummy smile v3.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    best_path = train_yolo_seg(
        data_yaml=args.data,
        weights_path=args.weights,
        output_dir=args.output,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
    )
    print(f"Best weights saved to: {best_path}")
