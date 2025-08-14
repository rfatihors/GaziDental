import argparse
import json
import os
from typing import List

import cv2
import numpy as np

from config import DATASET_ROOT


# ---------------------- JSON -> YOLO ----------------------

def normalize_coordinates(coordinates: List[float], width: int, height: int) -> List[float]:
    normalized = []
    for i in range(0, len(coordinates), 2):
        x = coordinates[i] / width
        y = coordinates[i + 1] / height
        normalized.extend([x, y])
    return normalized


def create_txt_content(category_id: int, segmentation: List[float], width: int, height: int) -> str:
    coords = normalize_coordinates(segmentation, width, height)
    return f"{category_id} {' '.join(map(str, coords))}\n"


def process_json(json_path: str, output_folder: str) -> None:
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    images = data["images"]
    annotations = data["annotations"]

    os.makedirs(output_folder, exist_ok=True)
    for image in images:
        image_id = image["id"]
        image_filename = image["file_name"]
        width = image["width"]
        height = image["height"]

        txt_content = ""
        for ann in annotations:
            if ann["image_id"] == image_id:
                segmentation = ann["segmentation"][0]
                txt_content += create_txt_content(ann["category_id"], segmentation, width, height)

        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        with open(os.path.join(output_folder, txt_filename), "w") as txt_file:
            txt_file.write(txt_content)


# ---------------------- Drawing utilities ----------------------

def draw_outline(image_path: str, coordinates: List[str]) -> None:
    image = cv2.imread(image_path)
    coords = [
        (int(float(coordinates[i]) * image.shape[1]), int(float(coordinates[i + 1]) * image.shape[0]))
        for i in range(1, len(coordinates), 2)
    ]
    mask = np.zeros_like(image)
    cv2.polylines(mask, [np.array(coords, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
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


# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON'dan YOLO formatına dönüştürme ve görselleştirme")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_json = os.path.join(DATASET_ROOT, "annotation", "formatted_file.json")
    default_txt = os.path.join(DATASET_ROOT, "txt_files")
    default_img = os.path.join(DATASET_ROOT, "images", "image.png")

    convert_parser = subparsers.add_parser("convert", help="COCO JSON'u YOLO txt dosyalarına dönüştür")
    convert_parser.add_argument("--json-path", default=default_json, help="COCO JSON dosyasının yolu")
    convert_parser.add_argument("--output-folder", default=default_txt, help="TXT dosyalarının kaydedileceği klasör")

    outline_parser = subparsers.add_parser("draw-outline", help="YOLO txt dosyasından kontur çiz")
    outline_parser.add_argument("--image-path", default=default_img, help="Görselin yolu")
    outline_parser.add_argument("--txt-path", default=os.path.join(default_txt, "example.txt"), help="TXT dosyasının yolu")

    mask_parser = subparsers.add_parser("draw-mask", help="YOLO txt dosyasından maske uygula")
    mask_parser.add_argument("--image-path", default=default_img, help="Görselin yolu")
    mask_parser.add_argument("--txt-path", default=os.path.join(default_txt, "example.txt"), help="TXT dosyasının yolu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "convert":
        process_json(args.json_path, args.output_folder)
    else:
        with open(args.txt_path, "r") as txt_file:
            coordinates = txt_file.readlines()[0].split()
        if args.command == "draw-outline":
            draw_outline(args.image_path, coordinates)
        elif args.command == "draw-mask":
            draw_on_image(args.image_path, coordinates)


if __name__ == "__main__":
    main()
