"""""
import json
import os

def normalize_coordinates(coordinates, width, height):
    normalized_coords = []
    for i in range(0, len(coordinates), 2):
        x = coordinates[i] / width
        y = coordinates[i + 1] / height
        normalized_coords.extend([x, y])
    return normalized_coords

def create_txt_content(category_id, segmentation, bbox, area, width, height):
    normalized_coords = normalize_coordinates(segmentation, width, height)
    txt_content = f"{category_id} {' '.join(map(str, normalized_coords))}\n"
    return txt_content

def process_json(json_path, output_folder):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

        images = data['images']
        annotations = data['annotations']

        for image in images:
            image_id = image['id']
            image_filename = image['file_name']
            image_width = image['width']
            image_height = image['height']

            txt_content = ""

            for annotation in annotations:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']
                    segmentation = annotation['segmentation'][0]
                    bbox = annotation['bbox']
                    area = annotation['area']

                    txt_content += create_txt_content(category_id, segmentation, bbox, area, image_width, image_height)

            txt_filename = os.path.splitext(image_filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)

            with open(txt_path, 'w') as txt_file:
                txt_file.write(txt_content)

if __name__ == "__main__":
    json_path = "/home/ahmetko/Projects/yapay-zeka/local/annotation/formatted_file.json"
    output_folder = "/home/ahmetko/Projects/yapay-zeka/local/txt_files"

    process_json(json_path, output_folder)

"""""
#############################

"""""

### Tüm görsel üzerinde etiketi çizdirme

import cv2
import numpy as np

def draw_outline(image_path, coordinates, color=(0, 255, 0), thickness=2):
    # Load image
    image = cv2.imread(image_path)

    # Normalize coordinates to match the image size
    coordinates = [(int(float(coordinates[i]) * image.shape[1]), int(float(coordinates[i + 1]) * image.shape[0])) for i in range(1, len(coordinates), 2)]

    # Convert coordinates to numpy array
    coordinates = np.array(coordinates, dtype=np.int32)

    # Draw outline on a black image
    mask = np.zeros_like(image)
    cv2.polylines(mask, [coordinates], isClosed=True, color=color, thickness=thickness)

    # Add the outline to the original image
    result = cv2.addWeighted(image, 1, mask, 1, 0)

    # Display the result
    cv2.imshow('Image with Outline', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image and txt file paths
    image_path = "/home/ahmetko/Projects/yapay-zeka/local/images/2024-02-20 231035.png"
    txt_path = "/home/ahmetko/Projects/yapay-zeka/local/txt_files/2024-02-20 231035.txt"

    # Read coordinates from the txt file
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
        coordinates = lines[0].split()

    # Draw outline on the image
    draw_outline(image_path, coordinates)
    

"""""
###########################


# Sadece maskeyi görüntüleme

import cv2
import numpy as np

def create_mask(image_shape, coordinates):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(coordinates[1:], dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(mask, [pts], color=255)
    return mask

def draw_on_image(image_path, coordinates):
    # Load image
    image = cv2.imread(image_path)

    # Normalize coordinates to match the image size
    coordinates = [(int(float(coordinates[i]) * image.shape[1]), int(float(coordinates[i + 1]) * image.shape[0])) for i in range(1, len(coordinates), 2)]

    # Create mask
    mask = create_mask(image.shape, coordinates)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the result
    cv2.imshow('Image with Mask', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image and txt file paths
    image_path = "/local/images/2024-02-20 231035.png"
    txt_path = "/local/txt_files/2024-02-20 231035.txt"

    # Read coordinates from the txt file
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
        coordinates = lines[0].split()

    # Draw on the image
    draw_on_image(image_path, coordinates)


