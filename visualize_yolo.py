import argparse
import os
from typing import List

import cv2
import numpy as np

from config import DATASET_ROOT


def draw_outline(image_path: str, coordinates: List[str]) -> None:
    image = cv2.imread(image_path)
    coords = [
        (int(float(coordinates[i]) * image.shape[1]), int(float(coordinates[i + 1]) * image.shape[0]))
        for i in range(1, len(coordinates), 2)
    ]
    mask = np.zeros_like(image)
    cv2.polylines(
        mask, [np.array(coords, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2
    )
    result = cv2.addWeighted(image, 1, mask, 1, 0)
    cv2.imshow("Image with Outline", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_on_image(image_path: str, coordinates: List[str]) -> None:
    image = cv2.imread(image_path)
    coords = [
        (int(float(coordinates[i]) * image.shape[1]), int(float(coordinates[i + 1]) * image.shape[0]))
        for i in range(1, len(coordinates), 2)
    ]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], color=255)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Image with Mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO txt annotations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_img = os.path.join(DATASET_ROOT, "images", "image.png")
    default_txt = os.path.join(DATASET_ROOT, "txt_files", "example.txt")

    outline_parser = subparsers.add_parser(
        "draw-outline", help="Draw polygon outline from YOLO txt file"
    )
    outline_parser.add_argument("--image-path", default=default_img, help="Path to the image")
    outline_parser.add_argument("--txt-path", default=default_txt, help="Path to the txt file")

    mask_parser = subparsers.add_parser(
        "draw-mask", help="Apply mask from YOLO txt file"
    )
    mask_parser.add_argument("--image-path", default=default_img, help="Path to the image")
    mask_parser.add_argument("--txt-path", default=default_txt, help="Path to the txt file")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.txt_path, "r") as txt_file:
        coordinates = txt_file.readlines()[0].split()
    if args.command == "draw-outline":
        draw_outline(args.image_path, coordinates)
    elif args.command == "draw-mask":
        draw_on_image(args.image_path, coordinates)


if __name__ == "__main__":
    main()
