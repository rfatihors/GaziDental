import json
from PIL import Image, ImageDraw
import os  # os modülünü ekleyin

# COCO JSON dosyasını oku (örneğin, 'coco_annotations.json' olarak adlandırılmış)
coco_json_path = '/local/annotation/formatted_file.json'  # COCO JSON dosyanızın yolu

with open(coco_json_path, 'r') as json_file:
    coco_data = json.load(json_file)

# İlgili görsel için bilgileri seçin (örneğin, image_id'yi belirtin)
target_image_id = 1  # İlgili görselin image_id'si

# İlgili görselin COCO verilerini bulun
target_image_data = next((img for img in coco_data['images'] if img['id'] == target_image_id), None)
if target_image_data:
    # İlgili görselin etiketli bilgilerini bulun
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]

    # Maskeyi oluşturun
    mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        segmentation = annotation['segmentation']
        category_id = annotation['category_id']

        for segment in segmentation:
            draw.polygon(segment, outline=None, fill=255)  # fill parametresini sabit bir değerle değiştirin (255 beyaz renktir)

    # Klasörün varlığını kontrol et ve yoksa oluştur
    masks_folder_path = '/local/masks/'
    if not os.path.exists(masks_folder_path):
        os.makedirs(masks_folder_path)

    # Maskeyi BMP olarak kaydedin
    mask_save_path = os.path.join(masks_folder_path, f'{target_image_data["file_name"]}_mask.bmp')  # Hedef görselin adını kullanarak maske dosyasının adını oluşturun
    mask.save(mask_save_path)

    print(f"Maske kaydedildi: {mask_save_path}")

else:
    print(f"Belirtilen image_id'ye sahip görsel bulunamadı: {target_image_id}")

