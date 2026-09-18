# Architecture comparison — results

Protocol: `PROTOCOL.md`, written and committed before any run. Comparison set: 29 high-smile-line
test images with a clinical reference measurement. Measurement method **C_p25** at **16.84 px/mm**, both fixed
in configs/config.yaml and unchanged here. Every architecture ran at its published defaults with the shared budget
{"epochs": 100, "patience": 20, "imgsz": 640, "batch": 16, "deterministic": true} and seeds [np.int64(42), np.int64(43), np.int64(44)].

## Primary outcome: millimetre error against the clinical reference

Per seed:

| model | seed | n | mae_mm | bias_mm | rmse_mm |
|---|---|---|---|---|---|
| rfdetr-seg-large | 42 | 29 | 2.282 | 1.822 | 2.761 |
| rfdetr-seg-large | 43 | 29 | 2.280 | 1.784 | 2.785 |
| rfdetr-seg-large | 44 | 29 | 2.291 | 1.759 | 2.785 |
| yolo11x-seg | 42 | 29 | 0.857 | 0.571 | 1.036 |
| yolo11x-seg | 43 | 29 | 1.026 | 0.731 | 1.180 |
| yolo11x-seg | 44 | 29 | 0.989 | 0.702 | 1.132 |
| yolo26x-seg | 42 | 29 | 0.983 | 0.700 | 1.115 |
| yolo26x-seg | 43 | 29 | 1.009 | 0.710 | 1.148 |
| yolo26x-seg | 44 | 29 | 1.012 | 0.735 | 1.138 |

Per model (mean ± SD over seeds):

| model | n_seeds | mae_mm_mean | mae_mm_sd | bias_mm_mean | bias_mm_sd | rmse_mm_mean | rmse_mm_sd |
|---|---|---|---|---|---|---|---|
| rfdetr-seg-large | 3 | 2.284 | 0.006 | 1.788 | 0.032 | 2.777 | 0.014 |
| yolo11x-seg | 3 | 0.957 | 0.089 | 0.668 | 0.085 | 1.116 | 0.073 |
| yolo26x-seg | 3 | 1.001 | 0.016 | 0.715 | 0.018 | 1.134 | 0.017 |

## Paired difference (seed-averaged, bootstrap over the shared images)

Positive `diff_mae_mm` means the first model has the larger error, i.e. the second is better.

| n | mae_a | mae_b | diff_mae_mm | ci_low | ci_high | sd_paired_mm | excludes_zero | model_a | model_b |
|---|---|---|---|---|---|---|---|---|---|
| 29 | 0.955 | 2.282 | -1.326 | -1.916 | -0.739 | 1.717 | True | yolo11x-seg | rfdetr-seg-large |
| 29 | 0.955 | 1.001 | -0.046 | -0.097 | 0.005 | 0.144 | False | yolo11x-seg | yolo26x-seg |

## Gingival edge errors, mm at the global scale

| model | seed | n | gingiva_top_edge_mae_mm | gingiva_top_edge_bias_mm | gingiva_bottom_edge_mae_mm | gingiva_bottom_edge_bias_mm |
|---|---|---|---|---|---|---|
| rfdetr-seg-large | 42 | 29 | 6.183 | -6.183 | 4.595 | -4.595 |
| rfdetr-seg-large | 43 | 29 | 6.137 | -6.137 | 4.578 | -4.578 |
| rfdetr-seg-large | 44 | 29 | 6.139 | -6.139 | 4.597 | -4.596 |
| yolo11x-seg | 42 | 29 | 0.277 | 0.097 | 0.728 | 0.619 |
| yolo11x-seg | 43 | 29 | 0.315 | 0.133 | 0.776 | 0.693 |
| yolo11x-seg | 44 | 29 | 0.273 | 0.123 | 0.729 | 0.617 |
| yolo26x-seg | 42 | 29 | 0.255 | 0.093 | 0.726 | 0.625 |
| yolo26x-seg | 43 | 29 | 0.260 | 0.119 | 0.721 | 0.618 |
| yolo26x-seg | 44 | 29 | 0.264 | 0.035 | 0.748 | 0.657 |

## Secondary: per-class segmentation metrics on the fixed test set, standard settings

| model | seed | settings | conf | max_det | class | box_precision | box_recall | box_f1 | box_map50 | box_map50_95 | seg_precision | seg_recall | seg_f1 | seg_map50 | seg_map50_95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo11x-seg | 42 | standard | 0.001 | 300 | diseti | 0.672 | 0.601 | 0.635 | 0.607 | 0.259 | 0.581 | 0.520 | 0.549 | 0.507 | 0.202 |
| yolo11x-seg | 42 | standard | 0.001 | 300 | dudak | 0.978 | 0.989 | 0.983 | 0.992 | 0.734 | 0.972 | 0.984 | 0.978 | 0.982 | 0.683 |
| yolo11x-seg | 43 | standard | 0.001 | 300 | diseti | 0.650 | 0.606 | 0.628 | 0.601 | 0.252 | 0.598 | 0.490 | 0.538 | 0.518 | 0.206 |
| yolo11x-seg | 43 | standard | 0.001 | 300 | dudak | 0.953 | 1.000 | 0.976 | 0.992 | 0.720 | 0.943 | 0.979 | 0.961 | 0.980 | 0.685 |
| yolo11x-seg | 44 | standard | 0.001 | 300 | diseti | 0.653 | 0.618 | 0.635 | 0.602 | 0.266 | 0.575 | 0.479 | 0.523 | 0.479 | 0.185 |
| yolo11x-seg | 44 | standard | 0.001 | 300 | dudak | 0.967 | 0.995 | 0.981 | 0.991 | 0.730 | 0.959 | 0.984 | 0.971 | 0.980 | 0.695 |
| yolo26x-seg | 42 | standard | 0.001 | 300 | diseti | 0.666 | 0.647 | 0.656 | 0.636 | 0.279 | 0.576 | 0.521 | 0.547 | 0.503 | 0.204 |
| yolo26x-seg | 42 | standard | 0.001 | 300 | dudak | 0.981 | 0.989 | 0.985 | 0.990 | 0.738 | 0.976 | 0.984 | 0.980 | 0.981 | 0.689 |
| yolo26x-seg | 43 | standard | 0.001 | 300 | diseti | 0.690 | 0.634 | 0.661 | 0.626 | 0.276 | 0.627 | 0.478 | 0.542 | 0.480 | 0.191 |
| yolo26x-seg | 43 | standard | 0.001 | 300 | dudak | 0.965 | 0.995 | 0.980 | 0.991 | 0.732 | 0.962 | 0.984 | 0.973 | 0.981 | 0.685 |
| yolo26x-seg | 44 | standard | 0.001 | 300 | diseti | 0.629 | 0.626 | 0.627 | 0.609 | 0.258 | 0.640 | 0.496 | 0.559 | 0.543 | 0.215 |
| yolo26x-seg | 44 | standard | 0.001 | 300 | dudak | 0.943 | 0.995 | 0.968 | 0.992 | 0.724 | 0.967 | 0.995 | 0.981 | 0.993 | 0.698 |

## Pre-registered decision (PROTOCOL.md §8)

Rule: a challenger replaces yolo11x-seg only when it leads by more than 0.15 mm in mean absolute error and the paired 95 % bootstrap interval of that lead excludes zero (both conditions).

**Outcome: no challenger meets both conditions: the final model stays yolo11x-seg.**

## Declared limits

* The architectures do not share a native mask resolution; part of any difference may be mask-head
  resolution rather than the ability to find the gingival margin (PROTOCOL.md §6).
* Class agreement and kappa are deliberately not reported per architecture: at n = 29 the interval
  is too wide to separate architectures (PROTOCOL.md §3).
* The rows above are published-default runs. The final model of the manuscript is the tuned
  YOLOv11x-seg of the main study and is not a member of this comparison.
