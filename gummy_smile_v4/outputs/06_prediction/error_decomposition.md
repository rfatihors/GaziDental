# Error decomposition (a: OOF masks, n = 144)

total = pipeline − reference = (pipeline − GT-mask measurement) + (GT-mask measurement − reference), with the same method (C_p25) and scale (16.84 px/mm) on both mask sources. The first term is what the segmentation adds; the second is the geometry/scale error already characterised in Stage 3.

| component | bias, mm | MAE, mm | SD, mm | RMSE, mm |
|---|---|---|---|---|
| total (pipeline − reference) | +0.684 | 0.839 | 0.730 | 0.998 |
| segmentation (pipeline − GT-mask) | +0.530 | 0.577 | 0.406 | 0.667 |
| geometry (GT-mask − reference) | +0.154 | 0.549 | 0.750 | 0.763 |

Variance of the total error: 31 % segmentation, 106 % geometry, -37 % covariance (r between the two components -0.32).

## Segmentation-induced error vs boundary metrics of the same image
Pixel metrics converted to mm; slope in mm of measurement error per mm (or per unit) of the metric.

| metric | n | r | r_p | slope | intercept |
|---|---|---|---|---|---|
| gingiva_mask_iou | 144 | -0.184 | 0.027 | -0.812 | 1.150 |
| gingiva_top_edge_bias_mm | 144 | -0.313 | 0.000 | -0.346 | 0.586 |
| gingiva_bottom_edge_bias_mm | 144 | 0.438 | 0.000 | 0.405 | 0.262 |
| gingiva_thickness_mae_mm | 144 | 0.369 | 0.000 | 0.590 | 0.103 |
| gingiva_columns_missed_frac | 144 | 0.107 | 0.203 | 0.682 | 0.506 |
| gingiva_columns_spurious_frac | 144 | -0.063 | 0.451 | -1.033 | 0.545 |
