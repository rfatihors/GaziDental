# Segmentation metrics on the fixed test set (pooled over the classes by this model's own evaluator)

| metric | value |
|---|---|
| seed | 42.000 |
| resolution | 624.000 |
| test/loss | 12.737 |
| test/mAP_50_95 | 0.527 |
| test/mAP_50 | 0.838 |
| test/mAP_75 | 0.574 |
| test/mAR | 0.683 |
| test/segm_mAP_50_95 | 0.466 |
| test/segm_mAP_50 | 0.808 |
| test/F1 | 0.830 |
| test/precision | 0.846 |
| test/recall | 0.815 |

- source: outputs/08_architecture/rfdetr_metrics_rfdetr-seg-large_s42.json
- model: RF-DETR-Seg Large @624, seed 42
- evaluator: RF-DETR's own COCO evaluation (pycocotools, iouType='segm')
- masks: outputs/05_predictions/test_rfdetr
- note: this framework's own COCO evaluation; not row-comparable with another framework's mAP
