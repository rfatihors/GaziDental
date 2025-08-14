import argparse
import os
from typing import Optional, List

from ultralytics import SAM

from config import DATASET_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 ile segmentasyon çalıştır")
    default_image = os.path.join(DATASET_ROOT, "gum", "gum", "images", "IMG_2632.jpg")
    parser.add_argument("--image", default=default_image, help="Girdi görselinin yolu")
    parser.add_argument("--bbox", type=float, nargs=4, default=None, help="x1 y1 x2 y2 formatında bbox")
    parser.add_argument("--points", type=float, nargs="*", default=None, help="Nokta koordinatları")
    parser.add_argument("--labels", type=int, nargs="*", default=None, help="Nokta etiketleri")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SAM("sam2_l.pt")
    model.info()

    kwargs = {}
    if args.bbox is not None:
        kwargs["bboxes"] = args.bbox
    if args.points is not None and args.labels is not None:
        kwargs["points"] = args.points
        kwargs["labels"] = args.labels

    results = model(args.image, **kwargs)
    results[0].show()


if __name__ == "__main__":
    main()
