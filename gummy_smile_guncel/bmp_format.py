import json
from PIL import Image, ImageDraw
import os

# COCO JSON dosyasını oku
coco_json_path = 'local/gummy_smile_guncel/json_folders/labels_my-project-name_2025-05-06-07-11-24.json'

with open(coco_json_path, 'r') as json_file:
    coco_data = json.load(json_file)

# Maske klasörünü oluştur
masks_folder_path = 'local/gummy_smile_guncel/masks/'
os.makedirs(masks_folder_path, exist_ok=True)

# Tüm görseller için döngü başlat
for image_info in coco_data['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']

    # Sadece son noktadan böl, uzantıyı at
    base_name = file_name.rsplit('.', 1)[0]

    # Bu görsele ait anotasyonları al
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Maskeyi oluştur
    mask = Image.new('L', (image_info['width'], image_info['height']), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        segmentation = annotation['segmentation']

        for segment in segmentation:
            draw.polygon(segment, outline=None, fill=255)

    # Maske dosyasını kaydet (base_name.bmp)
    mask_save_path = os.path.join(masks_folder_path, f'{base_name}.bmp')
    mask.save(mask_save_path)

    print(f"Maske kaydedildi: {mask_save_path}")
