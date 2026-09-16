# Report build status

| item | status | needs | source | path |
|---|---|---|---|---|
| Figure: system block diagram (Reviewer 1) | done |  | gsv4 code | outputs/07_report/figures/pipeline_block_diagram.png |
| Figure: segmentation examples GT vs prediction (Reviewer 1) | done |  | outputs/05_predictions/test | outputs/07_report/figures/segmentation_examples.png |
| Figure: measurement vs clinical reference, GT masks (replaces Figure 6) | done |  |  | outputs/07_report/figures/measurement_gt_masks.png |
| Figure: measurement vs clinical reference, predicted masks (OOF) | done |  |  | outputs/07_report/figures/measurement_predicted_masks.png |
| Figure: learning curve (Reviewers 2 & 4, Supplementary S1) | done |  | outputs/05_predictions/learning_curve.png | outputs/07_report/figures/learning_curve.png |
| Figure: boundary error, upper/lower gingiva edge (Reviewer 2 #10) | done |  | outputs/05_predictions/boundary_error.csv | outputs/07_report/figures/boundary_error.png |
| Figure: GT overlay examples (Stage 2) | done |  | outputs/02_measure/gt_overlay_examples.png | outputs/07_report/figures/gt_overlay_examples.png |
| Table: Dataset before/after cleaning and per split (Reviewers 2 #5/#6, 4) | done |  | data/manifest | outputs/07_report/tables/dataset_counts.md |
| Table: Demographic coverage (Reviewer 3) | done |  | data/manifest/dataset_manifest.csv | outputs/07_report/tables/demographics.md |
| Table: Millimetre accuracy vs clinical reference (Reviewers 2, 4; Figure 6 replacement) | done |  | outputs/03_oracle | outputs/07_report/tables/measurement_accuracy.md |
| Table: Segmentation metrics on the fixed test set (per class) | pending | outputs/05_predictions/test_metrics.json without a per_class block — re-run gsv4.train.evaluate_test on the workstation (validation pass) |  | outputs/07_report/tables/segmentation_metrics_test.md |
| Table: Learning curve points (Supplementary S1) | done |  | outputs/05_predictions/learning_curve.csv | outputs/07_report/tables/learning_curve.md |
| Table: Model vs expert agreement (Reviewer 4: clinical validity) | pending | real expert forms (current outputs are a synthetic dry run) |  | outputs/07_report/tables/expert_agreement.md |
| Table: intra-observer reliability of the reference | done |  |  | outputs/03_oracle/intra_observer.md |
