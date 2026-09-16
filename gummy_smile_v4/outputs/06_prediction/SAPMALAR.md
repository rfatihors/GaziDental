# Aşama 6 — sapmalar ve notlar

- `IMG_7289_jpg`: **segmentation failure** — the fold model predicted no gingiva instance above conf 0.25 (empty mask). Reported as its own category (`n_segmentation_failure`, 1 of 145 = 0.7 %), not as a 0 mm measurement: the image has no mm value and is excluded from the mm and label metrics (n = 144), flagged `empty_prediction` in per_image_results.csv.
- Test-set boundary table: gingiva IoU undefined on 25 images (both masks empty) and edge errors undefined on 40 images (no column with gingiva in both masks) — all low/normal; reported as presence categories, not as zeros.
- Corrected values clipped at 0 mm (flag `clipped`): 0 of 145 OOF images, 0 of 29 test images. None.
