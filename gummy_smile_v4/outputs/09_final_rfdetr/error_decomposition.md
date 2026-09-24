# Error decomposition (a: OOF masks, n = 145)

total = pipeline − reference = (pipeline − GT-mask measurement) + (GT-mask measurement − reference), with the same method (C_p25) and scale (16.84 px/mm) on both mask sources. The first term is what the segmentation adds; the second is the geometry/scale error already characterised in Stage 3.

| component | bias, mm | MAE, mm | SD, mm | RMSE, mm |
|---|---|---|---|---|
| total (pipeline − reference) | +0.270 | 0.521 | 0.681 | 0.731 |
| segmentation (pipeline − GT-mask) | +0.120 | 0.279 | 0.334 | 0.354 |
| geometry (GT-mask − reference) | +0.150 | 0.542 | 0.744 | 0.756 |

Variance of the total error: 24 % segmentation, 119 % geometry, -43 % covariance (r between the two components -0.40).

## Segmentation-induced error vs boundary metrics of the same image
Pixel metrics converted to mm; slope in mm of measurement error per mm (or per unit) of the metric.

| metric | n | r | r_p | slope | intercept |
|---|---|---|---|---|---|
| gingiva_mask_iou | 145 | -0.184 | 0.027 | -0.694 | 0.690 |
| gingiva_top_edge_bias_mm | 145 | -0.171 | 0.040 | -0.177 | 0.123 |
| gingiva_bottom_edge_bias_mm | 145 | 0.467 | 0.000 | 0.369 | 0.092 |
| gingiva_thickness_mae_mm | 145 | 0.177 | 0.033 | 0.338 | -0.063 |
| gingiva_columns_missed_frac | 145 | 0.112 | 0.181 | 0.650 | 0.095 |
| gingiva_columns_spurious_frac | 145 | -0.045 | 0.588 | -0.855 | 0.127 |
