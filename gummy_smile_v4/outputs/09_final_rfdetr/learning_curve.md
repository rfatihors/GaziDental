# Learning curve — RF-DETR-Seg Large @624 (PLAN.md 3)

Model: **RF-DETR-Seg Large @624, seed 42**; evaluator: **rfdetr model.evaluate (pycocotools, iouType=segm)**; split: **val**; metric: **val/segm_mAP_50**. Recorded by 1 of the 4 points; the others were written before the metrics file carried the field, so for them one evaluator is an assumption, not a check.
Each point is that training run's own COCO evaluation (`outputs/08_architecture/rfdetr_metrics_*.json`); the 100 % point is the
final model of PLAN.md 1, reused rather than retrained and evaluated on the same split as the subsets. The YOLOv11x
learning curve is not mixed in here: it belongs to the previous final model and stays in the appendix.

val/segm_mAP_50 gain 25→50 %: +0.0269; 75→100 %: +0.0274 → **not plateaued (report as limitation)** (rule: 75→100 gain < 1/4 of the 25→50 gain).

Increments between adjacent points: 25→50 %: +0.0269; 50→75 %: -0.0105; 75→100 %: +0.0274. The curve is not monotonic: the 75 % point falls below the one before it, so the steps between neighbouring points are of the same order as the noise between runs. Each point is one training run with one
seed (seed 42); the run-to-run spread was not measured, so the curve carries no error bars and the order
of two adjacent points is not evidence on its own.

| fraction | run | n_train_images | val/segm_mAP_50 |
|---|---|---|---|
| 0.25 | lc25 | 211 | 0.7599 |
| 0.50 | lc50 | 423 | 0.7868 |
| 0.75 | lc75 | 635 | 0.7763 |
| 1.00 | final (reused) | 846 | 0.8037 |
