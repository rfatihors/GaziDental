#Json dosyasından 58 görsel için 58 adet label oluşturma

import json
import os

json_file_path = 'local/dentalModels/termal_labels_new.json'

output_folder = 'local/dentalModels/termal_labels'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(json_file_path, 'r') as file:
    data = json.load(file)

images = data['images']
annotations = data['annotations']

for image in images:
    image_id = image['id']
    image_file_name = image['file_name']

    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

    new_json_data = {
        'image': image,
        'annotations': image_annotations
    }

    output_file_name = os.path.splitext(image_file_name)[0] + '.json'
    output_file_path = os.path.join(output_folder, output_file_name)

    with open(output_file_path, 'w') as output_file:
        json.dump(new_json_data, output_file, indent=4)

print("İşlem tamamlandı. JSON dosyaları '{}' klasörüne kaydedildi.".format(output_folder))


###############################


#Oluşturulan 58 adet json için ayrı ayrı etiketleri elde etme

import json
import os

input_dir = 'local/dentalModels/termal_labels'

output_dir = 'local/dentalModels/termal_labels_split'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):
        input_file_path = os.path.join(input_dir, file_name)

        with open(input_file_path, 'r') as json_file:
            data = json.load(json_file)

        image_info = data['image']
        annotations = data['annotations']

        for idx, annotation in enumerate(annotations):
            new_data = {
                "image": image_info,
                "annotations": [annotation]
            }

            new_file_name = f"{os.path.splitext(file_name)[0]}_{idx + 1}.json"
            output_file_path = os.path.join(output_dir, new_file_name)

            with open(output_file_path, 'w') as output_file:
                json.dump(new_data, output_file, indent=4)

print("JSON dosyaları başarıyla oluşturuldu.")


#######################


#Elde edilen json etiketleri için görsellerin oluşturulması

import os
import json
from PIL import Image, ImageDraw

def create_colored_mask(image, annotations):
    annotated_image = Image.new('RGB', image.size, (0, 0, 0))
    draw = ImageDraw.Draw(annotated_image)

    original_image = image.convert('RGBA')
    original_pixels = original_image.load()

    for annotation in annotations:
        segmentation = annotation.get('segmentation', [])
        for seg in segmentation:
            if len(seg) > 4:
                points = list(zip(seg[0::2], seg[1::2]))

                colors = [original_pixels[x, y] for x, y in points if 0 <= x < image.width and 0 <= y < image.height]
                if colors:
                    avg_color = tuple(map(lambda x: sum(x) // len(x), zip(*colors)))
                else:
                    avg_color = (255, 255, 255)

                draw.polygon(points, outline=avg_color, fill=avg_color)

    return annotated_image

def process_json_and_save_images(images_dir, labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.json')]

    for json_filename in json_files:
        json_path = os.path.join(labels_dir, json_filename)

        base_name = json_filename.split('_')[0]
        image_filename = base_name + '.jpg'
        image_path = os.path.join(images_dir, image_filename)

        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        annotations = json_data.get('annotations', [])

                    annotated_image = create_colored_mask(img, annotations)

                    output_filename = os.path.splitext(json_filename)[0] + '.png'
                    output_path = os.path.join(output_dir, output_filename)
                    annotated_image.save(output_path)

                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Image not found: {image_path}")

    print("İslem tamam.")

images_dir = 'local/dentalModels/JPEG_ve_makesense_dişlerin_haritalaması'
labels_dir = 'local/dentalModels/termal_labels_split'
output_dir = 'local/dentalModels/termal_images_split'

process_json_and_save_images(images_dir, labels_dir, output_dir)
