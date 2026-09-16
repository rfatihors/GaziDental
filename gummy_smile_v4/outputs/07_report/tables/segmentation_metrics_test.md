# Segmentation metrics on the fixed test set (per class)

| settings | conf | max_det | class | box_precision | box_recall | box_f1 | box_map50 | box_map50_95 | seg_precision | seg_recall | seg_f1 | seg_map50 | seg_map50_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| operating point (legacy file, see note) |  |  | diseti | 0.667 | 0.685 | 0.676 | 0.572 | 0.257 | 0.530 | 0.545 | 0.537 | 0.415 | 0.159 |
| operating point (legacy file, see note) |  |  | dudak | 0.990 | 0.995 | 0.992 | 0.992 | 0.732 | 0.984 | 0.989 | 0.987 | 0.981 | 0.692 |
| operating point (legacy file, see note) |  |  | all | 0.828 | 0.840 |  | 0.782 | 0.494 | 0.757 | 0.767 |  | 0.698 | 0.425 |

- source: outputs/05_predictions/test_metrics.json
- n_test_images: 192
- weights: /home/fatihors/Projects/GaziDental/gummy_smile_v4/runs/final/weights/best.pt
- settings_of_the_reported_map: operating_point_legacy
- note: WARNING: this file predates the two-settings evaluation, so the numbers below were computed at the pipeline operating point (conf 0.25, max_det 20), which truncates the precision-recall curve and understates mAP — re-run gsv4.train.evaluate_test to get the standard-settings figures
- boundary_by_group: {'high': {'gingiva_mask_iou': 0.7634131680388595, 'gingiva_boundary_iou': 0.28118889849067286, 'gingiva_top_edge_mae_px': 5.458826124524501, 'gingiva_bottom_edge_mae_px': 12.719360789270844}, 'low': {'gingiva_mask_iou': 0.279535685568267, 'gingiva_boundary_iou': 0.22839912453544722, 'gingiva_top_edge_mae_px': 2.84621333974878, 'gingiva_bottom_edge_mae_px': 8.903837970143911}, 'normal': {'gingiva_mask_iou': 0.492330827592235, 'gingiva_boundary_iou': 0.2836455724842604, 'gingiva_top_edge_mae_px': 5.317406327184866, 'gingiva_bottom_edge_mae_px': 11.086726261557388}}
