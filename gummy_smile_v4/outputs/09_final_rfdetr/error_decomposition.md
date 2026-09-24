# Error decomposition (a: OOF masks, n = 145)

total = pipeline − reference = (pipeline − GT-mask measurement) + (GT-mask measurement − reference), with the same method (C_p25) and scale (16.84 px/mm) on both mask sources. The first term is what the segmentation adds; the second is the geometry/scale error already characterised in Stage 3.

| component | bias, mm | MAE, mm | SD, mm | RMSE, mm |
|---|---|---|---|---|
| total (pipeline − reference) | +0.270 | 0.521 | 0.681 | 0.730 |
| segmentation (pipeline − GT-mask) | +0.116 | 0.276 | 0.331 | 0.350 |
| geometry (GT-mask − reference) | +0.154 | 0.546 | 0.748 | 0.761 |

Variance of the total error: 24 % segmentation, 120 % geometry, -44 % covariance (r between the two components -0.41).

## Segmentation-induced error vs boundary metrics of the same image
Pixel metrics converted to mm; slope in mm of measurement error per mm (or per unit) of the metric.

| metric | n | r | r_p | slope | intercept |
|---|---|---|---|---|---|
| gingiva_mask_iou | 145 | -0.195 | 0.019 | -0.732 | 0.717 |
| gingiva_top_edge_bias_mm | 145 | -0.177 | 0.033 | -0.182 | 0.119 |
| gingiva_bottom_edge_bias_mm | 145 | 0.470 | 0.000 | 0.369 | 0.088 |
| gingiva_thickness_mae_mm | 145 | 0.188 | 0.023 | 0.356 | -0.077 |
| gingiva_columns_missed_frac | 145 | 0.121 | 0.148 | 0.697 | 0.089 |
| gingiva_columns_spurious_frac | 145 | -0.039 | 0.640 | -0.735 | 0.122 |
