# Detection / segmentation metrics of the final model

Model: **yolo11x-seg @640**; masks `outputs/05_predictions/test`; evaluator: **Ultralytics val() (box/mask P, R, F1, mAP@50, mAP@50-95)**; source: `outputs/05_predictions/test_metrics.json`.

These numbers come from one framework's evaluator and are not comparable, row by row, with another framework's mAP; architecture comparisons in this project use the framework-independent boundary and IoU metrics of `boundary_by_set.md`.

| class | box_precision | box_recall | box_f1 | box_map50 | box_map50_95 | seg_precision | seg_recall | seg_f1 | seg_map50 | seg_map50_95 |
|---|---|---|---|---|---|---|---|---|---|---|
| diseti | 0.667 | 0.685 | 0.676 | 0.572 | 0.257 | 0.530 | 0.545 | 0.537 | 0.415 | 0.159 |
| dudak | 0.990 | 0.995 | 0.992 | 0.992 | 0.732 | 0.984 | 0.989 | 0.987 | 0.981 | 0.692 |
| all | 0.828 | 0.840 |  | 0.782 | 0.494 | 0.757 | 0.767 |  | 0.698 | 0.425 |

Weights: `/home/fatihors/Projects/GaziDental/gummy_smile_v4/runs/final/weights/best.pt`; split `test`, 192 images.
