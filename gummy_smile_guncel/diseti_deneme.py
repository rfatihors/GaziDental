import os
import cv2
from tqdm import tqdm

from gummy_smile_guncel.combine_gum_smile_guncel import test_data_pred

image_dir = "local/gummy_smile_guncel/images"
mask_dir = "local/gummy_smile_guncel/masks"
output_dir = "local/gummy_smile_guncel/output"
mismatch_dir = "local/gummy_smile_guncel/uyumsuz_maske"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mismatch_dir, exist_ok=True)

missing_masks = []
mismatched_masks = []


image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in tqdm(image_files, desc="Kontrol ediliyor"):
    image_path = os.path.join(image_dir, image_file)
    mask_name = os.path.splitext(image_file)[0] + ".bmp"
    mask_path = os.path.join(mask_dir, mask_name)

    image = cv2.imread(image_path)
    if image is None:
        continue

    if not os.path.exists(mask_path):
        missing_masks.append(image_file)
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        mismatched_masks.append(image_file)
        continue

    if mask.shape[:2] != image.shape[:2]:
        mismatched_masks.append(image_file)
        resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        if len(resized_mask.shape) == 2:
            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        blend = cv2.addWeighted(image, 0.6, resized_mask, 0.4, 0)
        combined = cv2.hconcat([image, blend])
        cv2.imwrite(os.path.join(mismatch_dir, image_file), combined)
        continue

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, blended)

with open("eksik_maskeler.txt", "w") as f:
    for name in missing_masks:
        f.write(name + "\n")

print(f"\nToplam görsel: {len(image_files)}")
print(f"Eksik maskeler: {len(missing_masks)} - kayıt: eksik_maskeler.txt")
print(f"Uyumsuz (boyutsal) maskeler: {len(mismatched_masks)} - klasör: uyumsuz_maskeler")


from PIL import Image
import numpy as np
im = Image.fromarray((test_data_pred[0].transpose() * 255).astype(np.uint8))
im = Image.fromarray(arr)
im.save("local/gummy_smile_guncel_dt/your_file.bmp")

import matplotlib.image

matplotlib.image.imsave('local/gummy_smile_guncel_dt/your_file.bmp', test_data_pred[0])

import numpy as np
import cv2
np.squeeze(test_data_pred[0], axis=2)
arr = test_data_pred[0].astype(np.uint8)
a = np.transpose(arr, (1, 2, 0))
im = Image.fromarray(np.squeeze(a, axis=2))

cv2.imwrite('output_image.png', arr)

# Show the image
cv2.imshow('Image', arr)
cv2.waitKey(0)
cv2.destroyAllWindows()