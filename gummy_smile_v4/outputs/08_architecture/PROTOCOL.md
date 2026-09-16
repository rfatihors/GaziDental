# Architecture comparison — pre-registration

**Written and committed before any run of this comparison.** Commit this file first; the runs
that follow refer to it. Nothing in it may be changed after the first training starts. If a
change proves unavoidable, it is appended as a dated amendment with its reason, and the original
text stays.

## 1. Purpose

Reviewer 4 objects that the architecture comparison in the submitted manuscript is not fair:

> The fifth issue is that the architecture comparison is not entirely fair: RF-DETR-Seg is
> evaluated using dataset v1, YOLO26 using v2, and YOLO11 using v3. Although the authors indicate
> they start from the same baseline, a comparison intended to conclude that one architecture is
> superior should use the same training, validation, and test sets, as well as the same
> preprocessing, so that the architecture is the only variable.

The objection is correct. This analysis answers it by repeating the comparison with the data, the
partition, the preprocessing and the evaluation held identical across architectures. It is a side
analysis of the architecture choice. It is not a new model-development stage.

## 2. What is held fixed and what is left free

Held identical across all architectures:

* the images and the annotations (the cleaned dataset, 1,230 images);
* the participant-level partition and the fixed test set (846 / 192 / 192), unchanged from the main study;
* the epoch budget (100) and the early-stopping rule (patience 20 on the validation metric);
* the seed set (3 seeds, listed in §5);
* the evaluation protocol, the metric implementation and the thresholds (§4);
* the measurement pipeline applied to the predicted masks (`gsv4.masks.extract` → `gsv4.measure`), unchanged.

Left to each architecture's published defaults:

* optimiser, learning rate and schedule, weight decay, EMA;
* augmentation policy;
* loss formulation and matching (dense anchor-free for YOLO, Hungarian set prediction for RF-DETR);
* batch size where the architecture's own default differs (YOLO 16, RF-DETR 4).

**Reason.** A single shared hyperparameter set cannot be fair here. RF-DETR is a DETR: set
prediction, Hungarian matching, no NMS, a separate encoder learning rate, EMA. YOLO is dense
prediction with mosaic augmentation. Imposing the YOLO settings on RF-DETR, or the reverse, would
handicap one family by construction. Holding the data, the budget and the evaluation identical
isolates the architecture as far as two different training regimes allow, and the remaining
difference is declared rather than hidden.

**Tuning symmetry.** No architecture is tuned for this comparison. All three run at published
defaults. The tuned YOLOv11x-seg of the main study is reported as a separate row, labelled as the
final model with additional tuning, so that no reader mistakes it for a member of the comparison.

**Resolution.** YOLO runs at 640. RF-DETR requires a multiple of 24 and each variant carries its
own native resolution; the comparison uses 624, the multiple of 24 closest to 640. The residual
difference (640 vs 624, 2.5 %) is stated in the results.

## 3. Primary and secondary outcomes

**Primary (paired, on the 29 high-smile-line test images with a clinical reference measurement):**

* mean absolute error of gingival display in millimetres against the clinical reference;
* mean difference (bias) in millimetres;
* upper gingiva edge error and lower gingiva edge error against the annotated masks (MAE and bias).

The comparison is paired: every architecture predicts the same images, so the relevant variance is
that of the per-image difference between two architectures, not that of the difference between a
model and the reference. Measured on our own data (fold models against the final model on these 29
images), the paired standard deviation is 0.362 mm, which resolves a difference of 0.132 mm at
n = 29 (95 %, paired); the unpaired equivalent would be 0.471 mm. The analysis must therefore be
paired, and it is sized for the differences that matter here (the segmentation-induced bias is
0.53 mm and the lower-edge bias 0.66 mm).

**Secondary:** per-class mask and box mAP on the fixed test set, computed at standard evaluation
settings (§4).

**Not reported per architecture:** agreement with the Table 1 class and the linear-weighted kappa.
At n = 29 the bootstrap confidence interval of kappa in this dataset spans roughly 0.20 to 0.59,
which cannot separate architectures. Reporting it per architecture would invite conclusions the
sample cannot support.

## 4. Evaluation settings

Two evaluations, both reported, never mixed:

| purpose | conf | NMS IoU | max detections |
|---|---|---|---|
| reported mAP (standard) | 0.001 | 0.7 | 300 |
| operating point (the pipeline's own) | 0.25 | 0.5 | 20 |

mAP at a confidence threshold of 0.25 truncates the precision-recall curve and understates average
precision; it is not the standard convention and is not comparable with a COCO-style evaluation.
The mAP reported in the manuscript and in this comparison is therefore computed at the standard
settings. The operating-point numbers (precision, recall, F1, confusion matrix, false positives and
false negatives) describe the configuration the measurement pipeline actually runs at, and are
reported separately and labelled as such. The masks used for the measurement come from the
operating point, unchanged.

RF-DETR reports COCO mask AP through pycocotools with `iouType="segm"`. Ultralytics integrates
average precision with 101-point interpolation over the same IoU grid, so the definitions agree;
any residual difference in matching is stated with the results.

## 5. Seeds and statistics

Three seeds per architecture: 42, 43, 44. Reported as mean ± standard deviation over seeds.

For the difference between two architectures, a paired bootstrap over the test images (2000
resamples, seed 42) gives the 95 % confidence interval of the difference in the primary outcome.
Seeds give the spread; the bootstrap gives the interval. A single seed cannot support a claim about
an architecture, and no such claim will be made from one.

## 6. Known bias, declared in advance

The native mask resolution of the architectures is not equal.

| model | native mask grid | one mask pixel in the original image |
|---|---|---|
| yolo11x-seg at 640 | 160 × 160 | ≈ 17 px ≈ 1.00 mm |
| RF-DETR-Seg at 624, downsample 4 | 156 × 156 | ≈ 17 px horizontally, ≈ 12 px vertically |

RF-DETR resizes to a square without letterboxing, so the effective mask resolution of a 3:2
photograph is anisotropic. Masks are upsampled back to the original image size in both families, so
the geometry is preserved; what differs is the resolution of the fine structure.

This matters because the primary outcome is an edge error of about 0.66 mm, which is below the size
of a single native mask pixel. Part of any difference between architectures may therefore reflect
mask-head resolution rather than the architecture's ability to find the gingival margin. This is
stated in the results and is not corrected for.

## 7. Use of the test set

The fixed test set has already been used once, for the final model. This comparison is its second
use. It is declared here in advance, and **no model selection follows from it**: the final model is
fixed before the comparison runs (§8). The test set is not used to choose hyperparameters, epochs,
seeds or architectures.

## 8. Pre-registered decision rule

**The result will be reported whatever it is**, including a result unfavourable to the architecture
used in the manuscript.

The final model stays **YOLOv11x-seg** (the tuned version of the main study). Stage 6 and the expert
agreement analysis stay on it.

One exception, fixed now: if another architecture leads on the primary outcome by **more than
0.15 mm in mean absolute error** and the paired 95 % confidence interval of that difference **does
not contain zero**, then Stage 6 is repeated with the winning architecture and the manuscript is
updated accordingly. Both conditions must hold. A lead that is smaller than 0.15 mm, or whose
interval contains zero, changes nothing.

The threshold is set at 0.15 mm because the paired design resolves 0.132 mm at n = 29, so a smaller
threshold would be within the noise of the comparison, and because the intra-observer repeatability
of the clinical reference is 0.17 mm per tooth, so a smaller difference is not measurable against
the reference either.

## 9. Reporting

The results table carries one row per architecture and seed, plus aggregate rows, plus a separate
row for the tuned final model, labelled:

| row | tuning | role |
|---|---|---|
| YOLOv11x-seg, published defaults | none | comparison member |
| YOLO26x-seg, published defaults | none | comparison member |
| RF-DETR-Seg, published defaults | none | comparison member |
| YOLOv11x-seg, tuned | 24-combination grid (main study) | final model, not a comparison member |

If RF-DETR cannot be installed or trained under conditions comparable to the others, that outcome
is written into this file as an amendment and reported as "could not be evaluated under controlled
conditions for reasons of installation or compatibility", with the specific failure named. That is
an honest result and it will not be presented as anything else.
