# Error decomposition (a: OOF masks, n = 144)

total = pipeline − reference = (pipeline − GT-mask measurement) + (GT-mask measurement − reference), with the same method (C_p25) and scale (16.84 px/mm) on both mask sources. The first term is what the segmentation adds; the second is the geometry/scale error already characterised in Stage 3.

| component | bias, mm | MAE, mm | SD, mm | RMSE, mm |
|---|---|---|---|---|
| total (pipeline − reference) | +0.684 | 0.839 | 0.730 | 0.999 |
| segmentation (pipeline − GT-mask) | +0.534 | 0.580 | 0.407 | 0.671 |
| geometry (GT-mask − reference) | +0.150 | 0.544 | 0.746 | 0.759 |

Variance of the total error: 31 % segmentation, 104 % geometry, -36 % covariance (r between the two components -0.31).

## Segmentation-induced error vs boundary metrics of the same image
Pixel metrics converted to mm; slope in mm of measurement error per mm (or per unit) of the metric.

| metric | n | r | r_p | slope | intercept |
|---|---|---|---|---|---|
| gingiva_mask_iou | 144 | -0.175 | 0.036 | -0.771 | 1.123 |
| gingiva_top_edge_bias_mm | 144 | -0.307 | 0.000 | -0.340 | 0.589 |
| gingiva_bottom_edge_bias_mm | 144 | 0.435 | 0.000 | 0.403 | 0.267 |
| gingiva_thickness_mae_mm | 144 | 0.355 | 0.000 | 0.570 | 0.122 |
| gingiva_columns_missed_frac | 144 | 0.103 | 0.218 | 0.661 | 0.511 |
| gingiva_columns_spurious_frac | 144 | -0.069 | 0.410 | -1.130 | 0.551 |
