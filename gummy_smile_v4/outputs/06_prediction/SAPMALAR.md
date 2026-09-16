# Aşama 6 — sapmalar ve notlar

- `IMG_7289_jpg`: the fold model predicted no gingiva instance (empty mask) → measurement 0 mm / NO_VISIBLE_GINGIVA, kept in the primary set (counted as an error, listed as `empty_prediction`).
- `outputs/05_predictions/test_metrics.json` has no `per_class` block (box/mask mAP, P, R, F1 of the final model on the test set): the workstation eval ran the boundary stage only (`--metrics-only`) and `runs/eval/DONE` was not written. Re-run `python -m gsv4.train.evaluate_test` on the workstation (validation runs, predictions are reused) and pull; the detection metrics table of Stage 7 stays pending until then.
- Test-set boundary table: gingiva IoU undefined on 25 images (both masks empty) and edge errors undefined on 40 images (no column with gingiva in both masks) — all low/normal; reported as presence categories, not as zeros.
