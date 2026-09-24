# Detection / segmentation metrics of the final model

Model: **RF-DETR-Seg Large @624, seed 42**; masks `outputs/05_predictions/test_rfdetr`; evaluator: **RF-DETR's own COCO evaluation (pycocotools, iouType='segm')**; source: `outputs/08_architecture/rfdetr_metrics_rfdetr-seg-large_s42.json`.

These numbers come from one framework's evaluator and are not comparable, row by row, with another framework's mAP; architecture comparisons in this project use the framework-independent boundary and IoU metrics of `boundary_by_set.md`.

| metric | value |
|---|---|
| seed | 42.0000 |
| resolution | 624.0000 |
| test/loss | 12.7365 |
| test/mAP_50_95 | 0.5272 |
| test/mAP_50 | 0.8383 |
| test/mAP_75 | 0.5738 |
| test/mAR | 0.6829 |
| test/segm_mAP_50_95 | 0.4659 |
| test/segm_mAP_50 | 0.8081 |
| test/F1 | 0.8298 |
| test/precision | 0.8463 |
| test/recall | 0.8149 |

Split: `test`; variant `main`, seed 42.
