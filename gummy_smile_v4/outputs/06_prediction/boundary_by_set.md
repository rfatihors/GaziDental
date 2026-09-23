# Boundary error and segmentation quality in three image sets

**Model: yolo11x-seg @640.** Out-of-fold masks `outputs/05_predictions/oof` (yolo11x-seg @640, 5 fold models); final-model masks `outputs/05_predictions/test`. Every table below carries the model and the mask directory of each row; both boundary tables were computed here from those masks, not read from another run's CSV.

Same function as on the workstation (`gsv4.eval.boundary.boundary_report`, boundary IoU with 5 px dilation; edge errors column-wise over columns where both masks have gingiva). Pixel values converted at the global scale 16.84 px/mm. `n_*` columns: `n_gt_gingiva` images whose ground truth contains gingiva; `n_both` both masks non-empty (IoU and edge errors defined); `n_missed` GT gingiva but empty prediction; `n_spurious` prediction without GT gingiva; `n_neither` both empty (correct absence — IoU undefined, not zero). Statistics are computed over the images where they are defined (`*_n`).

| set | model | n | n_gt_gingiva | n_both | n_missed | n_spurious | n_neither | gingiva_mask_iou_n | gingiva_mask_iou_mean | gingiva_mask_iou_median | gingiva_boundary_iou_mean | gingiva_top_edge_mae_mm_mean | gingiva_top_edge_mae_mm_median | gingiva_top_edge_bias_mm_mean | gingiva_bottom_edge_mae_mm_mean | gingiva_bottom_edge_mae_mm_median | gingiva_bottom_edge_bias_mm_mean | gingiva_thickness_mae_mm_mean | gingiva_columns_missed_frac_mean | gingiva_columns_spurious_frac_mean | gingiva_n_columns_gt_median | lip_mask_iou_mean | lip_mask_iou_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | yolo11x-seg @640, 5 fold models | 145 | 145 | 144 | 1 | 0 | 0 | 145 | 0.758 | 0.774 | 0.270 | 0.322 | 0.255 | 0.160 | 0.763 | 0.729 | 0.661 | 0.724 | 0.042 | 0.015 | 935.000 | 0.794 | 0.814 |
| (b) test high, final model | yolo11x-seg @640 | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.763 | 0.785 | 0.281 | 0.324 | 0.239 | 0.047 | 0.755 | 0.718 | 0.596 | 0.741 | 0.033 | 0.019 | 941.000 | 0.811 | 0.809 |
| (b') test high, OOF masks | yolo11x-seg @640, 5 fold models | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.767 | 0.787 | 0.267 | 0.307 | 0.276 | 0.151 | 0.750 | 0.680 | 0.652 | 0.720 | 0.038 | 0.015 | 941.000 | 0.808 | 0.814 |
| (c) test all | yolo11x-seg @640 | 192 | 161 | 152 | 9 | 6 | 25 | 167 | 0.508 | 0.548 | 0.275 | 0.301 | 0.220 | 0.016 | 0.661 | 0.600 | 0.575 | 0.714 | 0.166 | 0.148 | 279.500 | 0.786 | 0.816 |
| (c) test low | yolo11x-seg @640 | 45 | 24 | 17 | 7 | 1 | 20 | 25 | 0.280 | 0.344 | 0.229 | 0.169 | 0.160 | -0.033 | 0.527 | 0.522 | 0.484 | 0.562 | 0.300 | 0.052 | 16.000 | 0.762 | 0.798 |
| (c) test normal | yolo11x-seg @640 | 118 | 108 | 106 | 2 | 5 | 5 | 113 | 0.493 | 0.542 | 0.284 | 0.316 | 0.227 | 0.016 | 0.657 | 0.596 | 0.584 | 0.731 | 0.148 | 0.217 | 309.500 | 0.788 | 0.822 |

Pixel units of the edge metrics:

| set | model | gingiva_top_edge_mae_px_mean | gingiva_top_edge_mae_px_median | gingiva_top_edge_bias_px_mean | gingiva_bottom_edge_mae_px_mean | gingiva_bottom_edge_bias_px_mean | gingiva_thickness_mae_px_mean |
|---|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | yolo11x-seg @640, 5 fold models | 5.42 | 4.30 | 2.70 | 12.84 | 11.14 | 12.19 |
| (b) test high, final model | yolo11x-seg @640 | 5.46 | 4.03 | 0.79 | 12.71 | 10.03 | 12.48 |
| (b') test high, OOF masks | yolo11x-seg @640, 5 fold models | 5.16 | 4.65 | 2.55 | 12.63 | 10.98 | 12.12 |
| (c) test all | yolo11x-seg @640 | 5.07 | 3.70 | 0.27 | 11.14 | 9.69 | 12.03 |
| (c) test low | yolo11x-seg @640 | 2.85 | 2.70 | -0.56 | 8.88 | 8.16 | 9.46 |
| (c) test normal | yolo11x-seg @640 | 5.32 | 3.83 | 0.27 | 11.07 | 9.84 | 12.32 |

## Reading the three sets

* **(a) is the primary segmentation result** for the measurement task: 145 high-smile-line images with a clinical reference, every mask predicted by a fold model that never saw the image or its same-patient twin. Gingiva mask IoU 0.758 (median 0.774); upper edge MAE 0.32 mm, lower edge MAE 0.76 mm with a systematic lower-edge bias of +0.66 mm (predicted gingiva extends further down than the annotation); lip IoU 0.794.
* **(b)** the final model on the 29 test high images: IoU 0.763, upper edge MAE 0.32 mm, lower edge bias +0.60 mm — the same picture as (a) on an independent model and the fixed test split; (b') shows the fold models on the same 29 images (IoU 0.767).
* **(c)** the whole test set (n = 192) has a lower mean gingiva IoU (0.508) **by construction, not because the model is worse there**: in low and normal smile lines the gingiva is thin or not visible at all. In the test low subset the annotated gingiva spans a median of 16 image columns (vs 935 in the high set) and 28 of 45 images have an empty GT or predicted gingiva; a few pixels of edge disagreement on a sliver one or two pixels tall drive IoU towards 0 even though the edge errors themselves are *smaller* than in the high set (upper edge MAE 0.17 mm low, 0.32 mm normal). Presence agreement in (c): both 152, correct absence 25, missed 9, spurious 6. The pipeline is specified for the high smile line only (visible gingiva → mm; otherwise NO_VISIBLE_GINGIVA), so (c) is reported for completeness of the segmentation evaluation and is not the basis of any measurement claim.
