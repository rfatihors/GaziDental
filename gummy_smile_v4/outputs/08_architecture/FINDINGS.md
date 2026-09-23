# What the architecture comparison found, beyond which model won

Recorded for the Discussion. The mechanism is not identified and no causal claim is made.

## The lower gingival margin is over-included by the YOLO family, and it is not resolution

The pipeline measures a vertical thickness, so the vertical size of a mask pixel is what limits it.
The comparison and its two controls span that quantity by a factor of 1.6 in each family:

| configuration | vertical mask pixel | lower gingival edge bias | upper edge bias |
|---|---|---|---|
| YOLOv11x-seg at 640 | 1.00 mm | +0.643 mm | +0.118 mm |
| YOLOv11x-seg at 1024 | 0.63 mm | +0.541 mm | +0.044 mm |
| YOLO26x-seg at 640 | 1.00 mm | +0.633 mm | +0.082 mm |
| RF-DETR-Seg at 624 | 0.68 mm | +0.044 mm | +0.013 mm |
| RF-DETR-Seg at 432 | 0.99 mm | +0.083 mm | +0.004 mm |

Refining YOLO's mask grid by 1.6 times removed about a sixth of the bias (0.643 to 0.541 mm), and
RF-DETR shows almost none even on a grid coarser than YOLO's at 640. A resolution limit would have
behaved the other way round: the bias would track the pixel size across families, and it does not.

The upper, lip-side edge is accurate in every configuration. Whatever produces the bias acts on one
boundary, the thin festooned gingival margin, and not on the mask as a whole.

## What differs between the architectures is a constant, not precision

| configuration | MAE, mm | bias, mm | SD of error, mm | MAE with its own bias removed, mm |
|---|---|---|---|---|
| RF-DETR-Seg at 624 | 0.662 | +0.273 | 0.915 | 0.517 |
| RF-DETR-Seg at 432 | 0.671 | +0.299 | 0.905 | 0.510 |
| YOLOv11x-seg at 640 | 0.955 | +0.668 | 0.894 | 0.502 |
| YOLOv11x-seg at 1024 | 0.905 | +0.613 | 0.892 | 0.523 |
| YOLO26x-seg at 640 | 1.001 | +0.715 | 0.888 | 0.487 |

The scatter is the same across all five configurations (SD 0.888 to 0.915 mm) and the per-image
errors correlate at 0.947 to 0.987 between architectures: they succeed and fail on the same images.
Once each model's own bias is removed the ordering reverses, and the two families are
indistinguishable. The difference that the decision rule acted on is a shift.

## What it means clinically: the segmentation stops adding error

The comparison decided on 29 test images. Stage 6 then measured each final model on all 145 reference
images with out-of-fold masks, against the same method (`C_p25`) and the same scale (16.84 px/mm) that
Stage 3 fixed on the annotated masks. That splits the pipeline error into a part the measurement
geometry already had and a part the segmentation adds:

| final model | pipeline MAE vs clinical reference | same method on GT masks | what the model adds | lower gingival edge bias |
|---|---|---|---|---|
| RF-DETR-Seg Large @624 (current) | 0.52 mm | 0.54 mm | ≈ 0 | +0.08 mm |
| YOLOv11x-seg @640 (previous) | 0.84 mm | 0.54 mm | +0.30 mm | +0.66 mm |

This is the result that gives the architecture comparison its clinical meaning. With RF-DETR the
pipeline measures gingival display as accurately as the annotation itself allows: the remaining error
is the measurement geometry and the reference's own noise, not the model. The 0.29 mm lead measured on
29 images is therefore not a leaderboard difference but the removal of the segmentation as an error
source, and with it of the post-hoc calibration step the YOLO pipeline needed
(`outputs/09_final_rfdetr/PLAN.md` Amendment 3).

Numbers: `outputs/09_final_rfdetr/error_decomposition.md` and `prediction_summary.md` (current model),
`outputs/06_prediction/` (previous model), summarised in `RESULTS.md`.

## How this will be written

As an observation about two mask-head designs, with the mechanism unidentified. A dense
prototype-and-coefficient head and a query-attention head with learned upsampling differ in many
respects at once, and this study varied none of them in a controlled way: it varied the input
resolution, which is now excluded, and the architecture, which is a bundle. The honest statement is
that the over-inclusion is reproducible, specific to one family here, unrelated to mask resolution
over the range tested, and removable post hoc by a constant. Why one head produces it and the other
does not is a question for a study designed to answer it.

Practical consequence for this manuscript: the post-hoc offset calibration, which the Stage 6
addendum introduced for the YOLO pipeline, is a property of that pipeline and not of the measurement
method. With RF-DETR it is expected to be unnecessary, and `outputs/09_final_rfdetr/PLAN.md` §6 fixes
in advance the threshold below which it is dropped. **Outcome (23 Sep 2026): it was dropped.** The
correction now exists only as this appendix finding about the YOLO family; the reported pipeline has
no post-hoc calibration step at all (`configs/config.yaml`, `measurement.bottom_edge_offset_px: 0`).
