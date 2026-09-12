# Stage 1 — parse report

Source: `Hasta_ID-_Ölçümler.xlsx`, sheet `Yüksek Gülme Hattı` (header row 1).

## Rows
| metric | value |
|---|---|
| rows with a name | 692 |
| rows with all six teeth readable | 692 |
| rows with six values > 0 | 508 |
| rows with at least one `-` (0 mm) | 184 |
| duplicate normalised names (rows) | 2 — img7366 |
| age-prefixed names | 161 (prefix equals YAŞ column in 155) |
| age available (sheet / prefix) | 630 (630 / 0) |
| sex available | 630 |

## Cell types (6 × 692 = 4152 cells)
| parse_kind | count |
|---|---|
| div1000 | 3504 |
| star_sub1mm | 353 |
| dash_zero | 289 |
| ambiguous_100_999 | 3 |
| plain_mm | 3 |
| negative | 0 |
| missing | 0 |
| unparsed | 0 |

## Distribution of the image-level mean (complete rows)
median 2.76 mm, 95th percentile 5.73 mm, max 11.24 mm.

## Consistency with the clinicians' E labels
4150 labelled cells; 2 disagree with Table 1 applied to the parsed value:

| excel_row | name | tooth_index | raw | mm | kind | excel_label | expected |
|---|---|---|---|---|---|---|---|
| 314 | IMG_9152 | 6 | 667 | 0.667 | ambiguous_100_999 | E4 | E1 |
| 486 | IMG_9206 | 6 | 728 | 0.728 | ambiguous_100_999 | E4 | E1 |

These cells are flagged `label_inconsistent` and excluded from the primary oracle analysis.

## COCO export
1315 images: high 216, low 303, normal 796; gingiva instances 3938, lip instances 1318.
Frame check (2698×1799 ± 2 px): 830 images inside, 485 outside (`frame_uncertain`). 390 distinct sizes; top: 2698×1799 (430), 2699×1799 (281), 2700×1800 (45), 2697×1798 (36), 2698×1798 (15).

## Name matching (high sheet ↔ COCO)
`high` group (216 images): dash_base_fallback 30, exact 119, row_ambiguous 1, unmatched 66 → **149 unique matches** (exact + dash_base_fallback).
Secondary (assumed name collisions, reported only): low {'unmatched': 299, 'exact': 3, 'dash_base_fallback': 1}, normal {'unmatched': 769, 'exact': 19, 'coco_key_collision': 8}.
Excel rows without any COCO match: 523 (`unmatched_manual.csv`) — photographs of the earlier study.
Name collisions: 8 images (`name_collisions_coco.csv`): IMG_1556_JPEG, IMG_1556_jpg, IMG_1713_JPEG, IMG_1713_jpg, IMG_3915_JPEG, IMG_3915_jpg, IMG_8885_JPEG, IMG_8885_jpg.
COCO-internal: 5 bases exist both plain and dashed (img2633, img3424, img3694, img3864, img4034); 8 stems repeat across groups (IMG_3068_jpg [high/normal], IMG_3110_jpg [high/normal], IMG_3129_jpg [high/normal], IMG_3647_jpg [low/normal], IMG_3660_jpg [low/normal], IMG_3677_jpg [low/normal], IMG_7350_jpg [high/normal], IMG_8648_jpg [high/normal]) — different photographs (audit §5.1), hence images are identified by `group/stem`.

## Calibration workbook
20 images × 2 sessions, 240 cells; parse kinds: {'div1000': 240}.
