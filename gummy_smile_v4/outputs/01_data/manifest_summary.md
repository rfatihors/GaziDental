# Stage 1 — manifest summary

Mode: `keep_unmeasured_high_in_train = False` (clinical decision: unmeasured high images are dropped).

| group | start | dropped | kept | kept with reference | train-only |
|---|---|---|---|---|---|
| high | 216 | 71 | 145 | 145 | 0 |
| low | 303 | 3 | 300 | 0 | 0 |
| normal | 796 | 11 | 785 | 0 | 0 |
| **total** | 1315 | 85 | 1230 | 145 | 0 |

Patients after cleaning: 1230 (image = patient).

## Drop reasons
| group | reason_class | n |
|---|---|---|
| high | duplicate_of | 4 |
| high | no_reference_measurement | 66 |
| high | row_ambiguous | 1 |
| low | duplicate_of | 3 |
| normal | duplicate_of | 11 |

## Same-patient pairs
55 pairs → 55 connected components. Resolved by an earlier drop: 37; decided by rule: 18.

| kept | dropped | rule |
|---|---|---|
| high/IMG_2456_jpeg | high/IMG_24555_jpg | expert_set |
| high/IMG_2550_jpeg | high/IMG_2544-_jpeg | expert_set |
| high/IMG_3684-_jpg | high/IMG_3682_jpg | expert_set |
| high/IMG_3858-_jpg | high/IMG_3854-_jpg | expert_set |
| low/IMG_3660_jpg | low/IMG_3659_jpg | demographics->image_a |
| normal/52-IMG_9695_jpg | low/IMG_9696_jpg | demographics |
| low/gummy42_JPG | low/gummy16_JPG | demographics->image_a |
| low/IMG_1207_JPG | normal/IMG_1178_JPG | demographics->image_a |
| normal/IMG_3755_jpg | normal/IMG_3753_jpg | demographics->image_a |
| normal/IMG_2722_jpg | normal/IMG_2723_jpg | demographics->image_a |
| normal/23-IMG_4080_JPG | normal/23-IMG_4094_JPG | demographics->image_a |
| normal/IMG_2923_jpg | normal/IMG_2924_jpg | demographics->image_a |
| normal/IMG_2729_jpg | normal/IMG_2730_jpg | demographics->image_a |
| normal/IMG_4290_jpg | normal/IMG_4292_jpg | demographics->image_a |
| normal/IMG_78701_jpg | normal/IMG_7870_jpg | demographics->image_a |
| normal/IMG_2790_jpg | normal/IMG_2791_jpg | demographics->image_a |
| normal/IMG_4280_jpg | normal/IMG_4282_jpg | demographics->image_a |
| normal/IMG_4278_jpg | normal/IMG_4276_jpg | demographics->image_a |

## Cross-check with the expert set (145)
Kept high images with a reference measurement vs `uzman_seti_145_goruntu.csv`: identical.
