# Learning curve points (Supplementary S1)

| fraction | run | n_train_images | split | evaluator | metric | source | val/segm_mAP_50 | val/mAP_50_95 | val/mAP_50 | val/mAP_75 | val/mAR | val/segm_mAP_50_95 | val/F1 | val/precision | val/recall | resolution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.250 | lc25 | 211 | val |  | val/segm_mAP_50 | outputs/08_architecture/rfdetr_metrics_lc25_s42.json | 0.760 | 0.507 | 0.809 | 0.526 | 0.657 | 0.442 | 0.805 | 0.843 | 0.775 |  |
| 0.500 | lc50 | 423 | val |  | val/segm_mAP_50 | outputs/08_architecture/rfdetr_metrics_lc50_s42.json | 0.787 | 0.525 | 0.829 | 0.544 | 0.656 | 0.461 | 0.822 | 0.850 | 0.797 |  |
| 0.750 | lc75 | 635 | val |  | val/segm_mAP_50 | outputs/08_architecture/rfdetr_metrics_lc75_s42.json | 0.776 | 0.518 | 0.827 | 0.552 | 0.669 | 0.456 | 0.813 | 0.838 | 0.791 |  |
| 1.000 | final (reused) | 846 | val | rfdetr model.evaluate (pycocotools, iouType=segm) | val/segm_mAP_50 | outputs/08_architecture/rfdetr_metrics_rfdetr-seg-large_s42_val.json | 0.804 | 0.548 | 0.848 | 0.580 | 0.694 | 0.479 | 0.826 | 0.842 | 0.810 | 624.000 |

- source: outputs/09_final_rfdetr/learning_curve.csv
- metric: val/segm_mAP_50
- split: val
- evaluator: rfdetr model.evaluate (pycocotools, iouType=segm)
- verdict: not plateaued (report as limitation)
- reading: outputs/09_final_rfdetr/learning_curve.md
