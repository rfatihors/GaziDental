import json
import os
from pathlib import Path
import cv2
import numpy as np

from config import ANNOTATION_DIR, TXT_FILES_DIR, IMAGES_DIR


def normalize_coordinates(coordinates, width, height):
    normalized = []
    for i in range(0, len(coordinates), 2):
        x = coordinates[i] / width
        y = coordinates[i + 1] / height
        normalized.extend([x, y])
    return normalized


def create_txt_content(category_id, segmentation, width, height):
    normalized_coords = normalize_coordinates(segmentation, width, height)
    return f"{category_id} {' '.join(map(str, normalized_coords))}\n"


def process_json(json_path: Path, output_folder: Path) -> None:
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    images = data["images"]
    annotations = data["annotations"]

    output_folder.mkdir(parents=True, exist_ok=True)

    for image in images:
        image_id = image["id"]
        image_filename = image["file_name"]
        image_width = image["width"]
        image_height = image["height"]

        txt_content = ""
        for annotation in annotations:
            if annotation["image_id"] == image_id:
                category_id = annotation["category_id"]
                segmentation = annotation["segmentation"][0]
                txt_content += create_txt_content(
                    category_id, segmentation, image_width, image_height
                )

        txt_filename = Path(image_filename).stem + ".txt"
        txt_path = output_folder / txt_filename
        with open(txt_path, "w") as txt_file:
            txt_file.write(txt_content)


def draw_outline(image_path: Path, coordinates, color=(0, 255, 0), thickness=2):
    image = cv2.imread(str(image_path))
    coords = [
        (
            int(float(coordinates[i]) * image.shape[1]),
            int(float(coordinates[i + 1]) * image.shape[0]),
        )
        for i in range(1, len(coordinates), 2)
    ]
    coords = np.array(coords, dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.polylines(mask, [coords], isClosed=True, color=color, thickness=thickness)
    result = cv2.addWeighted(image, 1, mask, 1, 0)
    cv2.imshow("Image with Outline", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_mask(image_shape, coordinates):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(coordinates[1:], dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(mask, [pts], color=255)
    return mask


def draw_on_image(image_path: Path, coordinates):
    image = cv2.imread(str(image_path))
    coords = [
        (
            int(float(coordinates[i]) * image.shape[1]),
            int(float(coordinates[i + 1]) * image.shape[0]),
        )
        for i in range(1, len(coordinates), 2)
    ]
    mask = create_mask(image.shape, coords)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Image with Mask", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    json_path = ANNOTATION_DIR / "formatted_file.json"
    output_folder = TXT_FILES_DIR
    process_json(json_path, output_folder)

    # Example usage for drawing functions
    example_image = IMAGES_DIR / "2024-02-20 231035.png"
    example_txt = TXT_FILES_DIR / "2024-02-20 231035.txt"
    if example_image.exists() and example_txt.exists():
        with open(example_txt, "r") as txt_file:
            lines = txt_file.readlines()
            coordinates = lines[0].split()
        draw_outline(example_image, coordinates)
        draw_on_image(example_image, coordinates)
