import argparse
import json
import os
from PIL import Image, ImageDraw

from config import DATASET_ROOT


def create_mask(json_path: str, image_id: int, output_dir: str) -> str:
    with open(json_path, "r") as json_file:
        coco_data = json.load(json_file)

    target_image_data = next(
        (img for img in coco_data["images"] if img["id"] == image_id), None
    )
    if not target_image_data:
        raise ValueError(
            f"Belirtilen image_id'ye sahip görsel bulunamadı: {image_id}"
        )

    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    mask = Image.new(
        "L", (target_image_data["width"], target_image_data["height"]), 0
    )
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        for segment in annotation["segmentation"]:
            draw.polygon(segment, outline=None, fill=255)

    os.makedirs(output_dir, exist_ok=True)
    mask_save_path = os.path.join(output_dir, f"{target_image_data['file_name']}_mask.bmp")
    mask.save(mask_save_path)
    return mask_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO'dan BMP maske üretimi")
    default_json = os.path.join(DATASET_ROOT, "annotation", "formatted_file.json")
    default_masks = os.path.join(DATASET_ROOT, "masks")
    parser.add_argument("--json-path", default=default_json, help="COCO JSON dosyasının yolu")
    parser.add_argument(
        "--image-id", type=int, default=1, help="Hedef görselin ID değeri"
    )
    parser.add_argument(
        "--output-dir", default=default_masks, help="Maskelerin kaydedileceği klasör"
    )
    args = parser.parse_args()

    mask_path = create_mask(args.json_path, args.image_id, args.output_dir)
    print(f"Maske kaydedildi: {mask_path}")
