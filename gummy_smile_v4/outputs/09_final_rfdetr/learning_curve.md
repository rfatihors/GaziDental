# Learning curve — RF-DETR-Seg Large @624 (PLAN.md 3)

Model: **RF-DETR-Seg Large @624, seed 42**; evaluator: **rfdetr model.evaluate (pycocotools, iouType=segm)**; split: **val**; metric: **val/segm_mAP_50**.
Each point is that training run's own COCO evaluation (`outputs/08_architecture/rfdetr_metrics_*.json`); the 100 % point is the
final model of PLAN.md 1, reused rather than retrained and evaluated on the same split as the subsets. The YOLOv11x
learning curve is not mixed in here: it belongs to the previous final model and stays in the appendix.

val/segm_mAP_50 gain 25→50 %: +0.0269; 75→100 %: +0.0274 → **still rising** (rule: 75→100 gain < 1/4 of the 25→50 gain).

| fraction | run | n_train_images | val/segm_mAP_50 |
|---|---|---|---|
| 0.25 | lc25 | 211 | 0.7599 |
| 0.50 | lc50 | 423 | 0.7868 |
| 0.75 | lc75 | 635 | 0.7763 |
| 1.00 | final (reused) | 846 | 0.8037 |
