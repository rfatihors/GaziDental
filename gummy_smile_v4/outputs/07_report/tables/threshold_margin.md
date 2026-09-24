# Class agreement by distance to a Table-1 boundary (Reviewer 3 Methods 8, Reviewer 4)

| distance to the nearest boundary, mm | n | share of images | MAE, mm | class agreement | disagreements |
|---|---|---|---|---|---|
| < 0.5 | 68 | 0.469 | 0.519 | 0.603 | 27 |
| 0.5–1.0 | 27 | 0.186 | 0.628 | 0.778 | 6 |
| 1.0–2.0 | 41 | 0.283 | 0.481 | 0.951 | 2 |
| > 2.0 | 9 | 0.062 | 0.397 | 1.000 | 0 |
| all | 145 | 1.000 | 0.521 | 0.759 | 35 |

- source: outputs/09_final_rfdetr/per_image_results.csv
- boundaries_mm: 3, 4, 6, 8
- n: 145
- share_within_0_5_mm_of_a_boundary: 47 %
- agreement_near_a_boundary: 60 %
- agreement_away_from_a_boundary: 100 %
- share_of_disagreements_within_1_mm_of_a_boundary: 94 %
- reference_below_4_mm: 113 of 145 images
- note: the strata are formed on the REFERENCE value, so the grouping does not depend on the model; agreement is between the class of the measured value and the class of the reference value
