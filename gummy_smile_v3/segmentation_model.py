from __future__ import annotations

from pathlib import Path

from gummy_smile_v3.yolo.infer_yolo_seg import predict_and_measure


def run_segmentation(
    weights_path: Path,
    images_dir: Path,
    output_dir: Path,
    measurement_output: Path,
    mm_per_pixel: float = 1.0,
    regions: int = 6,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    max_det: int = 5,
) -> Path:
    return predict_and_measure(
        weights_path=weights_path,
        images_dir=images_dir,
        output_dir=output_dir,
        measurement_output=measurement_output,
        mm_per_pixel=mm_per_pixel,
        regions=regions,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run YOLOv11x-seg inference and measurements for gummy smile v3."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-output", type=Path, required=True)
    parser.add_argument("--mm-per-pixel", type=float, default=1.0)
    parser.add_argument("--regions", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=5)
    args = parser.parse_args()

    run_segmentation(
        weights_path=args.weights,
        images_dir=args.images,
        output_dir=args.output,
        measurement_output=args.measurement_output,
        mm_per_pixel=args.mm_per_pixel,
        regions=args.regions,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )
