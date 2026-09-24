# Architecture comparison — results


Protocol: `PROTOCOL.md`, written and committed before any run. Comparison set: 29 high-smile-line
test images with a clinical reference measurement. Measurement method **C_p25** at **16.84 px/mm**, both fixed
in configs/config.yaml and unchanged here. Every architecture ran at its published defaults with the shared budget
{"epochs": 100, "patience": 20, "imgsz": 640, "batch": 16, "deterministic": true} and seeds [42, 43, 44].

## Primary outcome: millimetre error against the clinical reference

Per seed:

| model | seed | n | mae_mm | bias_mm | rmse_mm |
|---|---|---|---|---|---|
| rfdetr-seg-large | 42 | 29 | 0.671 | 0.284 | 0.941 |
| rfdetr-seg-large | 43 | 29 | 0.651 | 0.255 | 0.926 |
| rfdetr-seg-large | 44 | 29 | 0.671 | 0.279 | 0.961 |
| rfdetr-seg-large@432 | 42 | 29 | 0.645 | 0.273 | 0.921 |
| rfdetr-seg-large@432 | 43 | 29 | 0.705 | 0.322 | 0.981 |
| rfdetr-seg-large@432 | 44 | 29 | 0.672 | 0.302 | 0.943 |
| yolo11x-seg | 42 | 29 | 0.857 | 0.571 | 1.036 |
| yolo11x-seg | 43 | 29 | 1.026 | 0.731 | 1.180 |
| yolo11x-seg | 44 | 29 | 0.989 | 0.702 | 1.132 |
| yolo11x-seg@1024 | 42 | 29 | 0.962 | 0.667 | 1.115 |
| yolo11x-seg@1024 | 43 | 29 | 0.821 | 0.513 | 1.018 |
| yolo11x-seg@1024 | 44 | 29 | 0.935 | 0.659 | 1.103 |
| yolo26x-seg | 42 | 29 | 0.983 | 0.700 | 1.115 |
| yolo26x-seg | 43 | 29 | 1.009 | 0.710 | 1.148 |
| yolo26x-seg | 44 | 29 | 1.012 | 0.735 | 1.138 |

Per model (mean ± SD over seeds):

| model | n_seeds | mae_mm_mean | mae_mm_sd | bias_mm_mean | bias_mm_sd | rmse_mm_mean | rmse_mm_sd |
|---|---|---|---|---|---|---|---|
| rfdetr-seg-large | 3 | 0.664 | 0.011 | 0.273 | 0.016 | 0.943 | 0.017 |
| rfdetr-seg-large@432 | 3 | 0.674 | 0.030 | 0.299 | 0.024 | 0.948 | 0.030 |
| yolo11x-seg | 3 | 0.957 | 0.089 | 0.668 | 0.085 | 1.116 | 0.073 |
| yolo11x-seg@1024 | 3 | 0.906 | 0.075 | 0.613 | 0.086 | 1.079 | 0.053 |
| yolo26x-seg | 3 | 1.001 | 0.016 | 0.715 | 0.018 | 1.134 | 0.017 |

## Paired difference (seed-averaged, bootstrap over the shared images)

Positive `diff_mae_mm` means the first model has the larger error, i.e. the second is better.

| n | mae_a | mae_b | diff_mae_mm | ci_low | ci_high | sd_paired_mm | excludes_zero | model_a | model_b |
|---|---|---|---|---|---|---|---|---|---|
| 29 | 0.955 | 0.662 | 0.293 | 0.150 | 0.439 | 0.391 | True | yolo11x-seg | rfdetr-seg-large |
| 29 | 0.955 | 1.001 | -0.046 | -0.097 | 0.005 | 0.144 | False | yolo11x-seg | yolo26x-seg |

Comparison members: rfdetr-seg-large, yolo11x-seg, yolo26x-seg. Resolution controls, reported separately: rfdetr-seg-large@432, yolo11x-seg@1024.

## Gingival edge errors, mm at the global scale

| model | seed | n | gingiva_top_edge_mae_mm | gingiva_top_edge_bias_mm | gingiva_bottom_edge_mae_mm | gingiva_bottom_edge_bias_mm |
|---|---|---|---|---|---|---|
| rfdetr-seg-large | 42 | 29 | 0.255 | 0.021 | 0.482 | 0.036 |
| rfdetr-seg-large | 43 | 29 | 0.259 | 0.027 | 0.473 | 0.039 |
| rfdetr-seg-large | 44 | 29 | 0.250 | -0.010 | 0.474 | 0.056 |
| rfdetr-seg-large@432 | 42 | 29 | 0.248 | 0.010 | 0.475 | 0.095 |
| rfdetr-seg-large@432 | 43 | 29 | 0.249 | -0.010 | 0.466 | 0.083 |
| rfdetr-seg-large@432 | 44 | 29 | 0.238 | 0.011 | 0.470 | 0.071 |
| yolo11x-seg | 42 | 29 | 0.277 | 0.097 | 0.728 | 0.619 |
| yolo11x-seg | 43 | 29 | 0.315 | 0.133 | 0.776 | 0.693 |
| yolo11x-seg | 44 | 29 | 0.273 | 0.123 | 0.729 | 0.617 |
| yolo11x-seg@1024 | 42 | 29 | 0.289 | -0.004 | 0.660 | 0.520 |
| yolo11x-seg@1024 | 43 | 29 | 0.281 | 0.090 | 0.664 | 0.536 |
| yolo11x-seg@1024 | 44 | 29 | 0.265 | 0.046 | 0.683 | 0.568 |
| yolo26x-seg | 42 | 29 | 0.255 | 0.093 | 0.726 | 0.625 |
| yolo26x-seg | 43 | 29 | 0.260 | 0.119 | 0.720 | 0.618 |
| yolo26x-seg | 44 | 29 | 0.264 | 0.035 | 0.748 | 0.657 |

## Is the difference a shift or is it scatter?

A lead in mean absolute error can be a constant offset, which calibration removes, or scatter, which
it does not. The seed-averaged per-image error is split below; `removable_by_calibration_mm` is the
most any post-hoc bias correction could take off that model's MAE.

| model | n | mae_mm | bias_mm | sd_of_error_mm | mae_without_own_bias_mm | removable_by_calibration_mm | within_0_5_mm | within_1_mm |
|---|---|---|---|---|---|---|---|---|
| rfdetr-seg-large | 29 | 0.662 | 0.273 | 0.915 | 0.517 | 0.145 | 0.414 | 0.862 |
| rfdetr-seg-large@432 | 29 | 0.671 | 0.299 | 0.905 | 0.510 | 0.160 | 0.448 | 0.828 |
| yolo11x-seg | 29 | 0.955 | 0.668 | 0.894 | 0.502 | 0.453 | 0.138 | 0.655 |
| yolo11x-seg@1024 | 29 | 0.905 | 0.613 | 0.892 | 0.523 | 0.382 | 0.207 | 0.586 |
| yolo26x-seg | 29 | 1.001 | 0.715 | 0.888 | 0.487 | 0.514 | 0.069 | 0.552 |

Correlation of the per-image error between architectures (seed-averaged):

| model | rfdetr-seg-large | rfdetr-seg-large@432 | yolo11x-seg | yolo11x-seg@1024 | yolo26x-seg |
|---|---|---|---|---|---|
| rfdetr-seg-large | 1.000 | 0.975 | 0.947 | 0.962 | 0.969 |
| rfdetr-seg-large@432 | 0.975 | 1.000 | 0.972 | 0.974 | 0.978 |
| yolo11x-seg | 0.947 | 0.972 | 1.000 | 0.987 | 0.987 |
| yolo11x-seg@1024 | 0.962 | 0.974 | 0.987 | 1.000 | 0.982 |
| yolo26x-seg | 0.969 | 0.978 | 0.987 | 0.982 | 1.000 |

## Secondary: per-class segmentation metrics on the fixed test set, standard settings

| model | seed | settings | conf | max_det | imgsz | class | box_precision | box_recall | box_f1 | box_map50 | box_map50_95 | seg_precision | seg_recall | seg_f1 | seg_map50 | seg_map50_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo11x-seg@1024 | 42 | standard | 0.001 | 300 | 1024 | diseti | 0.689 | 0.667 | 0.678 | 0.662 | 0.297 | 0.672 | 0.637 | 0.654 | 0.635 | 0.269 |
| yolo11x-seg@1024 | 42 | standard | 0.001 | 300 | 1024 | dudak | 0.960 | 0.995 | 0.977 | 0.990 | 0.727 | 0.961 | 0.995 | 0.977 | 0.990 | 0.692 |
| yolo11x-seg@1024 | 43 | standard | 0.001 | 300 | 1024 | diseti | 0.691 | 0.666 | 0.678 | 0.671 | 0.294 | 0.729 | 0.596 | 0.656 | 0.650 | 0.267 |
| yolo11x-seg@1024 | 43 | standard | 0.001 | 300 | 1024 | dudak | 0.930 | 0.989 | 0.959 | 0.989 | 0.720 | 0.938 | 0.984 | 0.961 | 0.979 | 0.684 |
| yolo11x-seg@1024 | 44 | standard | 0.001 | 300 | 1024 | diseti | 0.649 | 0.707 | 0.677 | 0.659 | 0.288 | 0.642 | 0.666 | 0.654 | 0.645 | 0.266 |
| yolo11x-seg@1024 | 44 | standard | 0.001 | 300 | 1024 | dudak | 0.966 | 0.995 | 0.980 | 0.991 | 0.723 | 0.956 | 0.984 | 0.970 | 0.981 | 0.686 |
| yolo11x-seg | 42 | standard | 0.001 | 300 |  | diseti | 0.672 | 0.601 | 0.635 | 0.607 | 0.259 | 0.581 | 0.520 | 0.549 | 0.507 | 0.202 |
| yolo11x-seg | 42 | standard | 0.001 | 300 |  | dudak | 0.978 | 0.989 | 0.983 | 0.992 | 0.734 | 0.972 | 0.984 | 0.978 | 0.982 | 0.683 |
| yolo11x-seg | 43 | standard | 0.001 | 300 |  | diseti | 0.650 | 0.606 | 0.628 | 0.601 | 0.252 | 0.598 | 0.490 | 0.538 | 0.518 | 0.206 |
| yolo11x-seg | 43 | standard | 0.001 | 300 |  | dudak | 0.953 | 1.000 | 0.976 | 0.992 | 0.720 | 0.943 | 0.979 | 0.961 | 0.980 | 0.685 |
| yolo11x-seg | 44 | standard | 0.001 | 300 |  | diseti | 0.653 | 0.618 | 0.635 | 0.602 | 0.266 | 0.575 | 0.479 | 0.523 | 0.479 | 0.185 |
| yolo11x-seg | 44 | standard | 0.001 | 300 |  | dudak | 0.967 | 0.995 | 0.981 | 0.991 | 0.730 | 0.959 | 0.984 | 0.971 | 0.980 | 0.695 |
| yolo26x-seg | 42 | standard | 0.001 | 300 |  | diseti | 0.666 | 0.647 | 0.656 | 0.636 | 0.279 | 0.576 | 0.521 | 0.547 | 0.503 | 0.204 |
| yolo26x-seg | 42 | standard | 0.001 | 300 |  | dudak | 0.981 | 0.989 | 0.985 | 0.990 | 0.738 | 0.976 | 0.984 | 0.980 | 0.981 | 0.689 |
| yolo26x-seg | 43 | standard | 0.001 | 300 |  | diseti | 0.690 | 0.634 | 0.661 | 0.626 | 0.276 | 0.627 | 0.478 | 0.542 | 0.480 | 0.191 |
| yolo26x-seg | 43 | standard | 0.001 | 300 |  | dudak | 0.965 | 0.995 | 0.980 | 0.991 | 0.732 | 0.962 | 0.984 | 0.973 | 0.981 | 0.685 |
| yolo26x-seg | 44 | standard | 0.001 | 300 |  | diseti | 0.629 | 0.626 | 0.627 | 0.609 | 0.258 | 0.640 | 0.496 | 0.559 | 0.543 | 0.215 |
| yolo26x-seg | 44 | standard | 0.001 | 300 |  | dudak | 0.943 | 0.995 | 0.968 | 0.992 | 0.724 | 0.967 | 0.995 | 0.981 | 0.993 | 0.698 |

Models evaluated here: yolo11x-seg, yolo11x-seg@1024, yolo26x-seg. A
predictor outside the Ultralytics framework is not evaluated through `model.val()` and therefore has
no row; its own trainer's mask AP is reported in its run record instead, and the two are not
interchangeable.

## Resolution controls (PROTOCOL_ADDENDUM_resolution.md)

Sensitivity analyses, not members of the comparison: §2 held the input resolution fixed and these
runs deliberately break that. Vertical mask-pixel size is the one that matters, because the measured
quantity is a vertical thickness.

| configuration | grid | horizontal_mm | vertical_mm |
|---|---|---|---|
| yolo11x-seg @ 640 | 160 | 1.001 | 1.001 |
| yolo11x-seg@1024 | 256 | 0.626 | 0.626 |
| rfdetr-seg-large @ 624 | 156 | 1.027 | 0.685 |
| rfdetr-seg-large@432 | 108 | 1.483 | 0.989 |

| n | mae_a | mae_b | diff_mae_mm | ci_low | ci_high | sd_paired_mm | excludes_zero | model_a | model_b |
|---|---|---|---|---|---|---|---|---|---|
| 29 | 0.955 | 0.671 | 0.285 | 0.167 | 0.396 | 0.320 | True | yolo11x-seg | rfdetr-seg-large@432 |
| 29 | 0.905 | 0.662 | 0.243 | 0.120 | 0.366 | 0.340 | True | yolo11x-seg@1024 | rfdetr-seg-large |

Pre-registered reading: control B points to architecture,
control A points to architecture.

**Verdict (rule 4): the lead survives at matched resolution: PROTOCOL.md §8 applies as written, the final model becomes RF-DETR-Seg Large and Stage 6 is repeated with it.**

## Pre-registered decision (PROTOCOL.md §8)

Rule: a challenger replaces yolo11x-seg only when it leads by more than 0.15 mm in mean absolute error and the paired 95 % bootstrap interval of that lead excludes zero (both conditions).

**Outcome: rfdetr-seg-large leads yolo11x-seg by 0.293 mm [0.150, 0.439]: Stage 6 is repeated with it (PROTOCOL.md §8).**

## Integrity check

A run that produced no lip mask anywhere, dropped instances for having no role, or whose gingival
edge error is entirely systematic is flagged here and must not be reported until it is repeated.

| model | n_rows | images_without_lip | instances_ignored | edge_error_fully_systematic | suspect | problems |
|---|---|---|---|---|---|---|
| rfdetr-seg-large | 87 | 0 | 0 | False | False |  |
| rfdetr-seg-large@432 | 87 | 0 | 0 | False | False |  |
| yolo11x-seg | 87 | 0 | 0 | False | False |  |
| yolo11x-seg@1024 | 87 | 0 | 0 | False | False |  |
| yolo26x-seg | 87 | 0 | 0 | False | False |  |

## What the difference means for the measurement (Stage 6 of each final model)

The comparison above is 29 test images. The clinically meaningful statement is what each architecture
adds to the measurement error on the full out-of-fold set, against the same method and scale on the
ground-truth masks — i.e. how much of the pipeline error is the segmentation's.

| Stage 6 run | model | pipeline MAE, mm | geometry part (GT masks), mm | what the model adds, mm | segmentation error component, mm |
|---|---|---|---|---|---|
| current final model | RF-DETR-Seg Large @624, seed 42, 5 fold models | 0.52 | 0.55 | -0.02 | 0.28 (bias +0.12) |
| previous final model | yolo11x-seg @640, 5 fold models | 0.84 | 0.55 | +0.29 | 0.58 (bias +0.53) |

The geometry part is the same measurement on the annotated masks (Stage 3) and is a property of the method, not of the model; the segmentation part is what the predicted masks add to it. A model whose segmentation part is near zero measures the gingival display as well as the annotation allows.

## Declared limits

* The architectures do not share a native mask resolution; part of any difference may be mask-head
  resolution rather than the ability to find the gingival margin (PROTOCOL.md §6).
* Class agreement and kappa are deliberately not reported per architecture: at n = 29 the interval
  is too wide to separate architectures (PROTOCOL.md §3).
* The rows above are published-default runs. The final model of the manuscript is the tuned
  YOLOv11x-seg of the main study and is not a member of this comparison.
