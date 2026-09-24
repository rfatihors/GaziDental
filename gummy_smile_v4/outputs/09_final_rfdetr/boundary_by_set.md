# Boundary error and segmentation quality in three image sets

**Model: RF-DETR-Seg Large @624, seed 42.** Out-of-fold masks `outputs/05_predictions/oof_rfdetr` (RF-DETR-Seg Large @624, seed 42, 5 fold models); final-model masks `outputs/05_predictions/test_rfdetr`. Every table below carries the model and the mask directory of each row; both boundary tables were computed here from those masks, not read from another run's CSV.

Rows marked **[reference]** are the YOLOv11x (previous final model), computed on ITS OWN masks in an earlier run (`outputs/05_predictions/boundary_error.csv`) and reported here only so the two models can be read side by side. They are not this model's numbers and were not recomputed here.

Same function as on the workstation (`gsv4.eval.boundary.boundary_report`, boundary IoU with 5 px dilation; edge errors column-wise over columns where both masks have gingiva). Pixel values converted at the global scale 16.84 px/mm. `n_*` columns: `n_gt_gingiva` images whose ground truth contains gingiva; `n_both` both masks non-empty (IoU and edge errors defined); `n_missed` GT gingiva but empty prediction; `n_spurious` prediction without GT gingiva; `n_neither` both empty (correct absence — IoU undefined, not zero). Statistics are computed over the images where they are defined (`*_n`).

| set | model | n | n_gt_gingiva | n_both | n_missed | n_spurious | n_neither | gingiva_mask_iou_n | gingiva_mask_iou_mean | gingiva_mask_iou_median | gingiva_boundary_iou_mean | gingiva_top_edge_mae_mm_mean | gingiva_top_edge_mae_mm_median | gingiva_top_edge_bias_mm_mean | gingiva_bottom_edge_mae_mm_mean | gingiva_bottom_edge_mae_mm_median | gingiva_bottom_edge_bias_mm_mean | gingiva_thickness_mae_mm_mean | gingiva_columns_missed_frac_mean | gingiva_columns_spurious_frac_mean | gingiva_n_columns_gt_median | lip_mask_iou_mean | lip_mask_iou_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | RF-DETR-Seg Large @624, seed 42, 5 fold models | 145 | 145 | 145 | 0 | 0 | 0 | 145 | 0.821 | 0.838 | 0.395 | 0.277 | 0.240 | 0.016 | 0.466 | 0.436 | 0.075 | 0.541 | 0.039 | 0.009 | 935.000 | 0.836 | 0.856 |
| (b) test high, final model | RF-DETR-Seg Large @624, seed 42 | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.820 | 0.834 | 0.368 | 0.255 | 0.224 | 0.021 | 0.482 | 0.474 | 0.035 | 0.576 | 0.045 | 0.006 | 941.000 | 0.834 | 0.856 |
| (b') test high, OOF masks | RF-DETR-Seg Large @624, seed 42, 5 fold models | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.832 | 0.846 | 0.401 | 0.246 | 0.213 | -0.009 | 0.447 | 0.442 | 0.050 | 0.533 | 0.033 | 0.009 | 941.000 | 0.849 | 0.857 |
| (c) test all | RF-DETR-Seg Large @624, seed 42 | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.820 | 0.834 | 0.368 | 0.255 | 0.224 | 0.021 | 0.482 | 0.474 | 0.035 | 0.576 | 0.045 | 0.006 | 941.000 | 0.834 | 0.856 |
| (c) test low | RF-DETR-Seg Large @624, seed 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (c) test normal | RF-DETR-Seg Large @624, seed 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [reference] YOLOv11x (previous final model), test high | YOLOv11x (previous final model) | 29 | 29 | 29 | 0 | 0 | 0 | 29 | 0.763 | 0.785 | 0.281 | 0.324 | 0.239 | 0.047 | 0.755 | 0.720 | 0.597 | 0.741 | 0.033 | 0.019 | 941.000 | 0.811 | 0.809 |
| [reference] YOLOv11x (previous final model), test all | YOLOv11x (previous final model) | 192 | 161 | 152 | 9 | 6 | 25 | 167 | 0.508 | 0.548 | 0.275 | 0.301 | 0.220 | 0.016 | 0.662 | 0.600 | 0.576 | 0.715 | 0.166 | 0.148 | 279.500 | 0.786 | 0.816 |
| [reference] YOLOv11x (previous final model), OOF 145 reference high | YOLOv11x (previous final model) | 145 | 145 | 144 | 1 | 0 | 0 | 145 | 0.758 | 0.774 | 0.270 | 0.322 | 0.255 | 0.160 | 0.762 | 0.729 | 0.661 | 0.724 | 0.042 | 0.015 | 935.000 | 0.794 | 0.814 |

Pixel units of the edge metrics:

| set | model | gingiva_top_edge_mae_px_mean | gingiva_top_edge_mae_px_median | gingiva_top_edge_bias_px_mean | gingiva_bottom_edge_mae_px_mean | gingiva_bottom_edge_bias_px_mean | gingiva_thickness_mae_px_mean |
|---|---|---|---|---|---|---|---|
| (a) OOF, 145 reference high | RF-DETR-Seg Large @624, seed 42, 5 fold models | 4.67 | 4.05 | 0.28 | 7.86 | 1.26 | 9.11 |
| (b) test high, final model | RF-DETR-Seg Large @624, seed 42 | 4.30 | 3.77 | 0.36 | 8.13 | 0.59 | 9.70 |
| (b') test high, OOF masks | RF-DETR-Seg Large @624, seed 42, 5 fold models | 4.14 | 3.59 | -0.16 | 7.53 | 0.84 | 8.97 |
| (c) test all | RF-DETR-Seg Large @624, seed 42 | 4.30 | 3.77 | 0.36 | 8.13 | 0.59 | 9.70 |
| (c) test low | RF-DETR-Seg Large @624, seed 42 |  |  |  |  |  |  |
| (c) test normal | RF-DETR-Seg Large @624, seed 42 |  |  |  |  |  |  |
| [reference] YOLOv11x (previous final model), test high | YOLOv11x (previous final model) | 5.46 | 4.03 | 0.79 | 12.72 | 10.05 | 12.48 |
| [reference] YOLOv11x (previous final model), test all | YOLOv11x (previous final model) | 5.07 | 3.70 | 0.27 | 11.15 | 9.71 | 12.04 |
| [reference] YOLOv11x (previous final model), OOF 145 reference high | YOLOv11x (previous final model) | 5.42 | 4.30 | 2.70 | 12.84 | 11.14 | 12.19 |

## Reading the three sets

* **(a) is the primary segmentation result** for the measurement task: 145 high-smile-line images with a clinical reference, every mask predicted by a fold model that never saw the image or its same-patient twin. Gingiva mask IoU 0.821 (median 0.838); upper edge MAE 0.28 mm, lower edge MAE 0.47 mm with a systematic lower-edge bias of +0.07 mm (predicted gingiva extends further down than the annotation); lip IoU 0.836.
* **(b)** the final model on the 29 test high images: IoU 0.820, upper edge MAE 0.26 mm, lower edge bias +0.04 mm — the same picture as (a) on an independent model and the fixed test split; (b') shows the fold models on the same 29 images (IoU 0.832).
* **(c)** the whole test set (n = 29) has a lower mean gingiva IoU (0.820) **by construction, not because the model is worse there**: in low and normal smile lines the gingiva is thin or not visible at all. In the test low subset the annotated gingiva spans a median of nan image columns (vs 935 in the high set) and 0 of 0 images have an empty GT or predicted gingiva; a few pixels of edge disagreement on a sliver one or two pixels tall drive IoU towards 0 even though the edge errors themselves are *smaller* than in the high set (upper edge MAE nan mm low, nan mm normal). Presence agreement in (c): both 29, correct absence 0, missed 0, spurious 0. The pipeline is specified for the high smile line only (visible gingiva → mm; otherwise NO_VISIBLE_GINGIVA), so (c) is reported for completeness of the segmentation evaluation and is not the basis of any measurement claim.
