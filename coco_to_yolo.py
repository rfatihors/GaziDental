import argparse
import json
import os
from typing import List

from config import DATASET_ROOT


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
                txt_content += create_txt_content(
                    ann["category_id"], segmentation, width, height
                )

        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        with open(os.path.join(output_folder, txt_filename), "w") as txt_file:
            txt_file.write(txt_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to YOLO txt format"
    )
    default_json = os.path.join(DATASET_ROOT, "annotation", "formatted_file.json")
    default_txt = os.path.join(DATASET_ROOT, "txt_files")
    parser.add_argument(
        "--json-path", default=default_json, help="Path to COCO JSON file"
    )
    parser.add_argument(
        "--output-folder",
        default=default_txt,
        help="Directory to save YOLO txt files",
    )
    args = parser.parse_args()
    process_json(args.json_path, args.output_folder)


if __name__ == "__main__":
    main()
