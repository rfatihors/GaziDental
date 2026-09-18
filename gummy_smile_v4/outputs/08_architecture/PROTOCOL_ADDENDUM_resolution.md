# Addendum 2 — resolution controls

**Written and committed before either control runs.** It fixes how their results will be read, so
that the reading cannot be chosen after the numbers are known. `PROTOCOL.md` and its Amendment 1 are
unchanged; nothing here alters the rule in §8.

## 1. Why

The pre-registered decision rule of `PROTOCOL.md` §8 has triggered. On the 29 high-smile-line test
images, RF-DETR-Seg Large leads YOLOv11x-seg by **0.293 mm** in mean absolute error, paired 95 %
bootstrap interval **[0.150, 0.439]**, which excludes zero. Both conditions of §8 hold, so the rule
as written says Stage 6 is to be repeated with RF-DETR.

But §6 of the same protocol declared, in advance, that the architectures do not share a native mask
resolution and that part of any difference could be that rather than the ability to find the
gingival margin. That confounder is now measurable, and it points the right way:

| configuration | mask grid | one mask pixel, vertically |
|---|---|---|
| YOLOv11x-seg at imgsz 640 | 160 × 160 | 1.00 mm |
| RF-DETR-Seg at resolution 624 | 156 × 156 | 0.68 mm |

The measured quantity is a vertical thickness. RF-DETR resizes to a square instead of letterboxing,
so on a 3:2 photograph it spends proportionally more of its mask grid on the vertical axis and ends
up 1.46 times finer exactly where the measurement is taken. A coarse grid cannot represent a thin
festooned margin and rounds outward, which appears as systematic over-inclusion at the lower edge.

**The decision is therefore not applied before the confounder is tested.** The rule is not being
changed, reinterpreted or suspended: whatever the controls show, both the triggering of §8 and the
control results will be reported.

## 2. The further fact that makes this worth testing

The lead is a constant shift, not better precision:

| | MAE, mm | bias, mm | SD of error, mm | MAE with its own bias removed, mm |
|---|---|---|---|---|
| RF-DETR-Seg Large at 624 | 0.662 | +0.273 | 0.915 | 0.517 |
| YOLOv11x-seg at 640 | 0.955 | +0.668 | 0.894 | 0.502 |
| YOLO26x-seg at 640 | 1.001 | +0.715 | 0.888 | 0.487 |

The scatter is equal or slightly better for YOLO, the per-image errors correlate at 0.947 to 0.987
between architectures, and once each model's own bias is removed the ordering reverses. The lower
gingival edge bias is +0.044 mm for RF-DETR against +0.643 and +0.633 for the two YOLO families, so
the systematic over-inclusion the pipeline already corrects post hoc is specific to YOLO at this
resolution. Whether it is specific to YOLO or to the resolution is exactly what the controls decide.

## 3. The two controls

Both are sensitivity analyses, **not members of the comparison**: `PROTOCOL.md` §2 held the input
resolution fixed at each architecture's own, and these runs deliberately break that. They are
reported in their own section and never merged into the comparison tables.

**Control B — bring RF-DETR down.** The same variant and the same weights, only the input resolution
changed to 432, giving a vertical mask pixel of 0.99 mm, matched to YOLOv11x at 640 (1.00 mm). This
isolates resolution with the architecture held constant.

**Control A — bring YOLO up.** YOLOv11x-seg at imgsz 1024, giving a vertical mask pixel of 0.63 mm,
slightly finer than RF-DETR at 624 (0.68 mm). This tests whether YOLO's disadvantage is removable.

Three seeds each (42, 43, 44), the same participant-level partition, the same 29-image comparison
set, the same measurement method and scale, the same paired bootstrap. Everything else follows
`PROTOCOL.md` §2 and §5.

## 4. Pre-registered interpretation

Fixed now, before either control runs. "Resolution explains it" is written R below.

1. **Control B.** Paired difference between RF-DETR at 432 and YOLOv11x at 640. If the RF-DETR lead
   falls **below 0.15 mm**, or the paired 95 % interval **contains zero**, then R.
2. **Control A.** Paired difference between YOLOv11x at 1024 and RF-DETR at 624. If the paired 95 %
   interval **contains zero**, or the difference **favours YOLO**, then R.
3. **Both controls give R.** The final model becomes **YOLOv11x-seg at imgsz 1024**. Stage 6 is
   repeated with that configuration. The architecture does not change; the resolution does.
4. **Neither control gives R**, i.e. the lead survives at matched resolution. §8 applies as written:
   the final model becomes **RF-DETR-Seg Large** and Stage 6 is repeated with it.
5. **The controls disagree.** Both are reported. The configuration with the **lowest uncorrected mean
   absolute error** is chosen, and the disagreement is stated in the manuscript as an unresolved
   uncertainty rather than settled by preference.
6. **Whichever configuration is chosen**, the post-hoc offset correction is re-estimated for it from
   scratch, on the Stage-3 development subset, exactly as in the Stage 6 addendum. It is not carried
   over. If the chosen configuration has no lower-edge bias worth correcting, the correction is
   dropped and the manuscript says so; RF-DETR's lower-edge bias of +0.044 mm suggests that is the
   likely outcome there.

## 5. What is reported either way

The triggering of §8, the control results, the bias-and-scatter decomposition, and the mask-pixel
geometry of every configuration. A configuration that wins because it resolves the gingival margin
more finely is a legitimate finding and will be described as such, not as an architectural
superiority. The reverse holds too: if the lead survives at matched resolution, it is an
architectural result and will be reported as one.
