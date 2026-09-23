# Stage 6 with RF-DETR-Seg Large — plan and pre-registration

**Written and committed before any run.** It fixes the configuration, the analyses and the primary
outcome before the numbers are known. Amendments are appended with a date and a reason; the original
text stays.

## 0. How we got here

`PROTOCOL.md` §8 triggered in favour of RF-DETR (0.293 mm [0.150, 0.439] over YOLOv11x-seg). Before
applying it, `PROTOCOL_ADDENDUM_resolution.md` tested the confounder declared in §6, that the
architectures do not share a vertical mask resolution. Both controls came back pointing to the
architecture, not the resolution:

| control | paired difference, mm | 95 % CI | reading |
|---|---|---|---|
| B: YOLOv11x at 640 against RF-DETR at 432 (matched grid) | 0.285 | [0.167, 0.396] | lead survives |
| A: YOLOv11x at 1024 against RF-DETR at 624 (YOLO finer) | 0.243 | [0.120, 0.366] | lead survives |

Rule 4 of the addendum therefore applies: the final model becomes RF-DETR-Seg Large and Stage 6 is
repeated with it. This plan says how.

## 1. The final model

The seed-42 run of RF-DETR-Seg Large at resolution 624 from the architecture comparison, unchanged.
Not retrained, not tuned: the three seeds gave 0.671, 0.651 and 0.671 mm mean absolute error, so the
choice among them is immaterial and seed 42 is the protocol's first seed.

Published defaults throughout, as in the comparison. This also removes an asymmetry that the previous
manuscript carried: the YOLOv11x of record had been selected through a 24-combination hyperparameter
grid, while the architectures it was compared against had not. Nothing here is tuned, and the
manuscript will say so.

## 2. Out-of-fold predictions

Five folds, the existing assignments in `data/manifest/splits.json`, unchanged. Each fold model is
trained on everything except its held-out fold and the same-patient twins of that fold, exactly as
the YOLO folds were, and predicts only its own held-out images. RF-DETR at resolution 624, published
defaults, early stopping with patience 20 on `val/segm_mAP_50_95`, seed 42.

Masks go to `outputs/05_predictions/oof_rfdetr/`. The YOLO out-of-fold masks stay where they are; the
two sets are never mixed, and the YOLO ones remain available for the architecture appendix.

## 3. Learning curve

Stratified 25 %, 50 % and 75 % subsets of the training partition, the same subsets the YOLO learning
curve used (`lists/lc25_train.txt` and the others), the same validation set, RF-DETR at 624. The
100 % point is the final model of §1. Reported as the segmentation model's data-adequacy evidence,
replacing the YOLO curve in the manuscript; the YOLO curve moves to the architecture appendix.

## 4. Test-set segmentation metrics

Two things are reported, and they are **not** put in one table with the YOLO numbers:

* RF-DETR's own COCO segmentation evaluation on the fixed test set (`iouType="segm"`, standard
  settings), which is what its framework computes;
* our own boundary and IoU metrics on the test masks, which are framework-independent and are the
  ones that can be compared with the YOLO runs.

Ultralytics mAP and COCO mAP agree in definition but not in matching detail, and mixing them in one
row would invite a comparison the numbers do not support. The architecture comparison's own
framework-independent metrics remain the basis for any statement about which architecture is better.

## 5. Primary outcome and its sensitivity analysis

**Primary:** millimetre accuracy on the 145 reference images with out-of-fold RF-DETR masks, against
the clinical reference. Method **C_p25** and scale **16.84 px/mm**, both unchanged: they were selected
in Stage 3 on ground-truth masks and are a property of the measurement geometry, not of the
segmentation model. Nothing is re-selected or re-fitted.

**Pre-registered sensitivity:** the same analysis on the **116 images that are not** in the 29-image
test set used to choose the architecture. The architecture was selected partly on those 29 images, so
including them in the primary result is a mild optimism; the sensitivity analysis removes it. Both are
reported, the primary first. If the two differ by more than 0.10 mm in mean absolute error, the
sensitivity figure becomes the headline number and the reason is stated.

## 6. Offset correction

Rule 6 of the addendum: the offset is re-estimated from scratch on the Stage-3 development subset for
this configuration. It is not carried over from YOLO.

RF-DETR's lower gingival edge bias is +0.044 mm at 624 and +0.083 mm at 432, against +0.643 mm for
YOLOv11x, so the correction is expected to be unnecessary. **If the estimated offset is smaller than
one native mask pixel of this configuration (0.68 mm vertically) and the corrected holdout mean
absolute error improves by less than 0.05 mm, the correction is dropped**: a single uncorrected result
is reported, and the offset analysis stays in the appendix as a finding about the YOLO family. That
threshold is fixed now.

## 7. Regioning fallback

The zenith regioning of method C falls back to equal splits when it cannot place six zeniths. The rate
is reported on the RF-DETR out-of-fold masks, and the pre-registered rule of Stage 3 still stands: above
30 %, the method is re-evaluated against A_p25 and both are reported.

## 8. Expert analysis

The model table for Stage 4 comes from the RF-DETR out-of-fold per-image results. The expert protocol,
the reference standard and the primary statistic are unchanged.

## 9. What is not decided here

The manuscript text, the rebuttal and the edit list are not updated until these results exist. The
architecture comparison, its controls and the finding of §10 are reported whatever Stage 6 shows.

## 10. Finding to carry into the Discussion

Recorded now so that it is not written after the fact:

| configuration | vertical mask pixel | lower gingival edge bias |
|---|---|---|
| YOLOv11x-seg at 640 | 1.00 mm | +0.643 mm |
| YOLOv11x-seg at 1024 | 0.63 mm | +0.541 mm |
| YOLO26x-seg at 640 | 1.00 mm | +0.633 mm |
| RF-DETR-Seg at 624 | 0.68 mm | +0.044 mm |
| RF-DETR-Seg at 432 | 0.99 mm | +0.083 mm |

The systematic over-inclusion of the lower gingival margin is a property of the YOLO family here and
not of the mask resolution: refining YOLO's grid by 1.6 times removed only a sixth of it, while
RF-DETR shows almost none even on a coarser grid than YOLO's. Once each model's own bias is removed the
three architectures are indistinguishable in scatter (SD of the error 0.888 to 0.915, per-image error
correlation 0.947 to 0.987), so what differs between them is a constant, not precision.

This will be described as an observation about the two mask-head designs, with the mechanism
unidentified. No causal claim is made: a dense prototype-and-coefficient head and a
query-attention-with-upsampling head differ in many ways at once, and this study varied neither in a
controlled fashion.

---

## Amendment 1 — 23 Sep 2026: a first run of §5–§6 was invalid and has been discarded

**What happened.** The first workstation pass ran
`scripts/run_oracle.py --masks outputs/05_predictions/oof_rfdetr --out 09_final_rfdetr/oracle`. That
script still carried its Stage-3 behaviour: it re-ran the *whole* method and scale selection, on the
predicted masks and against the same clinical reference, and reported `B_p05` at 13.55 px/mm as the
"selected" method. §5 of this plan says the opposite — method `C_p25` and scale 16.84 px/mm are fixed,
selected in Stage 3 on ground-truth masks, applied unchanged here. The re-selection was a second look
at the reference and it landed on the wrong side of a 0.004 mm difference in dev mean absolute error:
in its own table `C_p25` had holdout MAE 0.451 mm against 0.543 mm for `B_p05`.

**What was affected.** Only that run's outputs under `outputs/09_final_rfdetr/oracle/`, which are
discarded. `configs/config.yaml` was not touched (method C/p25, `px_per_mm` 16.8397 and
`bottom_edge_offset_px` -13 were all still in place, git status clean) and `outputs/03_oracle/` was
untouched. No result of this plan had been reported from that run.

**Fix.** `run_oracle.py` now reads the method and the scale from `configs/config.yaml` whenever
`--masks` is given and selects nothing; re-selection needs the explicit `--reselect` flag, which is
off by default and is a sensitivity analysis only. A run on predicted masks refuses `--write-config`
outright, so that path can no longer write to the config. §6's offset re-estimation runs with the
method fixed, through
`scripts/run_offset_correction.py --out 09_final_rfdetr --oof-masks … --test-masks … --adopted-offset-px 0`.

**Unchanged.** Every pre-registered decision above stands: the primary outcome of §5, its sensitivity
analysis, the offset rule of §6 and the fallback rule of §7 are as they were written before any run.
