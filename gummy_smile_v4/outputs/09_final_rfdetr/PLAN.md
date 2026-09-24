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

## Amendment 2 — 23 Sep 2026: every table names its masks, its model and its evaluator

§4 requires RF-DETR's segmentation metrics to come from its own COCO evaluation and to stay out of any
table with Ultralytics numbers. Two paths in the code did not yet honour that: `run_prediction_eval.py`
read the boundary/IoU table from `outputs/05_predictions/boundary_error.csv` and the detection metrics
from `outputs/05_predictions/test_metrics.json`, both of them YOLOv11x artefacts, whatever masks it was
given. A Stage-6 run on RF-DETR masks would therefore have reported YOLO's segmentation quality as its
own.

From now on:

* the boundary and IoU tables are computed from the mask directories the run was given
  (`--oof-masks`, `--test-masks`) and written with the model and the directory in every row;
* the model behind each directory is read from the prediction table itself, and anything ambiguous
  stops the run — no `mask_source` column, two predictor families in one table, an RF-DETR table
  without its `model` column, or several models where the final model must be one;
* detection and segmentation metrics come only from the evaluator that belongs to those masks
  (Ultralytics `val()` for YOLO, RF-DETR's own COCO evaluation for RF-DETR). When that evaluator has
  not run, `segmentation_metrics.md` stays empty and names the command; another model's file is never
  substituted;
* YOLOv11x stays visible as rows labelled `[reference] YOLOv11x (previous final model)`, computed on
  its own masks in its own run, never merged into this model's rows.

The final model of §1 is reused with `--predict-only`, which does not evaluate, so it had no COCO
metrics at all; `scripts/rfdetr_train_predict.py --predict-only --evaluate` now produces them as
`outputs/08_architecture/rfdetr_metrics_rfdetr-seg-large_s42.json`, and that is the file Stage 6 quotes.

Side effect on the YOLO Stage 6 already reported in `outputs/06_prediction/`: its test-set boundary rows
are now recomputed locally from `outputs/05_predictions/test` instead of being read from the CSV the
workstation wrote before the shared instance-mask core of 17 Sep. The (a) out-of-fold rows are unchanged
to the last digit; the test-set rows move in the third decimal (mean gingiva IoU by ≤ 0.001, every edge
statistic by ≤ 0.1 px ≈ 0.006 mm). Nothing in the measurement results, the offset analysis or any
conclusion depends on that difference, and the recomputed numbers are the ones the current code
reproduces from the masks on disk.

## Amendment 3 — 23 Sep 2026: the post-hoc offset calibration is dropped

**This decision was taken after seeing the Stage-6 results, and is recorded as such.** §6 fixed the
threshold in advance; what follows says exactly where the outcome fell against it and why the step is
being removed anyway.

**The YOLO offset is not carried over.** `measurement.bottom_edge_offset_px` was -13 px, estimated on
the YOLOv11x out-of-fold predictions, whose lower gingival edge was drawn +0.66 mm too low. RF-DETR's
is +0.08 mm, so the same shift over-corrects: applying it raises the mean absolute error from 0.52 mm
to 0.59 mm. Keeping it would have made the measurement worse. It is set to **0**.

**The re-estimated constant is not adopted either.** Re-fitting on the Stage-3 development subset for
this configuration gave 0.254 mm, with a holdout improvement of 0.063 mm. Against §6 that is a split
verdict, and both halves are stated: the improvement is 0.013 mm above the 0.05 mm threshold, so the
rule did **not** formally trigger the drop, while the estimated offset is far below one native mask
pixel of this configuration (0.254 mm against 0.68 mm), which is the half of the rule that did. The
step is dropped, for reasons that go beyond the rule:

* the gain is smaller than the reference's own repeatability (intra-observer SD 0.17 mm per tooth
  site), i.e. below the noise floor of the standard it is calibrated against;
* a post-hoc calibration estimated on the same clinical reference the pipeline is evaluated against
  is the weakest element of the analysis, and every reviewer answer has to carry that caveat;
* removing it leaves **one** result set instead of a primary/secondary pair, so the reported accuracy
  is what the pipeline does with no step fitted on the reference at all.

Trading 0.063 mm for that is the right trade, and it is a judgement, not a rule: the rule's threshold
was 0.05 mm and the improvement was 0.063 mm.

**Consequences.** Stage 6 and Stage 7 are regenerated with no offset: the uncorrected/corrected pair
disappears from every table, figure and summary, and `run_prediction_eval.py` reports one result set.
The offset analysis itself is kept and stays reproducible — `run_prediction_eval.py --offset-px -13`
restores the YOLO appendix — and the finding it belongs to is described in
`outputs/08_architecture/FINDINGS.md` as a property of the YOLO mask head, with
`outputs/06_prediction/offset_correction.md` and `offset_checks.md` as its record.

**What this does not change.** The primary outcome of §5 and its pre-registered sensitivity analysis,
the method and the scale of Stage 3, and the fallback rule of §7 are untouched. The 0.52 mm primary
result is the uncorrected number that was pre-registered as primary; dropping the correction removes a
secondary column, not the headline.

## Amendment 4 — 24 Sep 2026: the learning curve of §3 has not plateaued

**The result, before its reading.** RF-DETR-Seg Large @624, seed 42, each point evaluated by RF-DETR's
own COCO evaluation on the same `val` split, metric `val/segm_mAP_50`:

| subset | n_train images | val/segm_mAP_50 |
|---|---|---|
| 25 % | 211 | 0.7599 |
| 50 % | 423 | 0.7868 |
| 75 % | 635 | 0.7763 |
| 100 % (final model, reused) | 846 | 0.8037 |

Gain 25→50 %: +0.0269; 50→75 %: −0.0105; 75→100 %: +0.0274. The rule fixed with the YOLO curve —
plateau when the 75→100 gain is under a quarter of the 25→50 gain — does not trigger: the late gain is
the larger of the two. The curve is also not monotonic; the 75 % point falls below the 50 % point.

**How it is reported.** "Performance had not plateaued within the available training-set size; the
increments between adjacent points are of the same order as run-to-run variation, so the curve
indicates that additional data could still improve segmentation performance. This is stated as a
limitation." Each point is a single training run with a single seed, so the run-to-run spread was not
measured: the curve carries no error bars, and the order of two adjacent points is not a result on its
own. That caveat is stated wherever the curve is quoted.

**What it changes.** The manuscript can no longer say the dataset is at the plateau of its learning
curve. The three reviewer answers that leaned on a plateau — R2-9 (no meaningful gain), R2-sample-size
and R4-external-validity — now carry the sentence above; `scripts/build_rebuttal.py` and
`scripts/build_manuscript_edits.py` take the wording from the verdict in
`outputs/07_report/tables/learning_curve.md`, so neither can report a plateau the curve does not show.
The Limitations gain one sentence: additional training data could still improve segmentation.

**This is consistent with Reviewer 4, not a concession against us.** Reviewer 4 argued that the cohort
size does not establish external validity and that more, and more varied, data are needed. A curve that
has not plateaued says the same thing from the model's side. The claim that is withdrawn is the one that
was never supported — that this dataset is sufficient — and nothing in the measurement results, the
architecture comparison or the primary outcome of §5 depends on it.

**Unchanged.** §3 itself: the same subsets, the same validation set, the same 100 % point reused rather
than retrained, one evaluator and one split for all four points. Reporting a curve that does not
plateau is what §3 pre-registered as one of its two possible outcomes.

## Amendment 5 — 24 Sep 2026: seed spread of the reported measurement (pre-registration)

**Written and committed before any of these runs.** Nothing below was known when it was written.

### A5.1 The gap this closes

Every measurement number in the manuscript comes from one seed. The final model is the seed-42 run
of the architecture comparison (§1) and the five fold models that produce the out-of-fold masks were
all trained with seed 42. The architecture comparison did measure a between-seed spread — three
seeds per architecture, SD of the mean absolute error 0.011 mm for RF-DETR-Seg Large and 0.089 mm
for YOLOv11x-seg — but on 29 images, on the comparison's own single-model predictions, not on the
five-fold out-of-fold pipeline the manuscript reports. The reported ICC, Bland–Altman bias and class
agreement have no measured run-to-run spread at all. A reviewer is entitled to ask what happens if
the seed changes, and at the moment we cannot answer.

### A5.2 What is run

The five-fold out-of-fold pipeline of §2, unchanged in every respect except the seed, repeated for
**seed 43** and **seed 44**: ten trainings, about 30 hours. The same fold assignments
(`data/manifest/splits.json`), the same datasets, the same resolution 624, the same published
defaults, the same early stopping, the same measurement method `C_p25` at 16.84 px/mm, the same 145
reference images.

Each seed writes to its own directory — `outputs/05_predictions/oof_rfdetr_s43` and `…_s44` — so no
run can accumulate into another's table. Seed 42's existing masks in `outputs/05_predictions/oof_rfdetr`
are not touched, not re-run and not regenerated.

### A5.3 What this analysis may and may not do

Fixed now, because this is the point at which a seed study turns into seed shopping:

* **The purpose is to report the variability of the measurement result from seed to seed. Nothing else.**
* **The final model stays seed 42.** It is the protocol's first seed and it was fixed before the
  architecture comparison ran. No model selection, no configuration change, no change of measurement
  method, scale or offset follows from this analysis, whatever it shows. If seed 43 or 44 turns out
  to give a lower error, that is a number in the spread table and nothing more — it does not become
  the reported model, and the reported model is not re-chosen on this evidence.
* **The result is reported whichever way it comes out**, including a spread large enough to weaken
  the architecture conclusion.
* No analysis in this amendment touches the fixed test set. The 29-image test subset appears only
  where it already appears in §5, as a labelled subset of the out-of-fold table.

### A5.4 What is reported

**Primary.** For each of the three seeds, on the same 145 out-of-fold reference images: mean absolute
error, RMSE, ICC(2,1), Bland–Altman bias and limits of agreement, threshold-label agreement and
linear-weighted κ. Reported per seed and as **mean ± SD over the three seeds**, which is the number
that answers the "single seed" objection.

**Secondary.** The paired difference between each pair of seeds on the shared images, with its 95 %
bootstrap confidence interval, computed by the same `paired_difference` used for the architecture
comparison, so that the seed spread and the architecture difference are measured the same way and
can be put side by side.

**The comparison that gives it meaning.** The between-seed spread is reported against the
architecture lead of **0.293 mm [0.150, 0.439]** that changed the final model. Fixed now, before the
numbers exist:

* if the largest paired between-seed difference is small relative to 0.293 mm and its interval
  contains zero, the architecture conclusion stands as written and the seed spread is reported as
  evidence that it does;
* if the between-seed spread is **of the same order as 0.293 mm**, that is written into the
  Limitations as a limitation of the architecture comparison — that a difference of this size cannot
  be cleanly separated from run-to-run variation in the five-fold pipeline — and the architecture
  result is reported with that caveat attached wherever it is quoted. The architecture comparison is
  not re-run and its conclusion is not reversed on this evidence: it was decided under its own
  pre-registered rule, on its own three seeds per architecture, and this is a different measurement.

**Where it is used.** As the answer to a reviewer's "all your results come from one seed" objection,
in the Limitations, and beside the architecture comparison wherever the 0.293 mm lead is quoted.

### A5.5 Integrity

The class check (`scripts/check_mask_classes.py`) runs after every prediction and a mismatch aborts
the run, as in `train_final_rfdetr.sh`. This is not optional here: the label-space fault of
`outputs/08_architecture/PROTOCOL.md` Amendment 2 was invisible in the segmentation metrics and
visible only in the measurement, which is exactly the quantity this amendment reports.

### A5.6 What is unchanged

§1 to §9 of this plan, Amendments 1 to 4, the primary outcome, the sensitivity analysis, the method
and the scale of Stage 3, and every number already reported. This amendment adds a table; it removes
and replaces nothing.

## Amendment 6 — 24 Sep 2026: the measurement was not reproducible across machines, and why

**This was found while validating the seed-spread script of Amendment 5 against the existing seed-42
masks, before any seed-spread run.** It is a defect in our own code, it is fixed, and everything it
touched has been regenerated. Nothing here was chosen after seeing which way it would move a result.

### A6.1 The symptom

Re-measuring `outputs/05_predictions/oof_rfdetr` with the committed code gave a `C_p25` value for
`high/IMG_6724_jpg` of 26.083 px, while the committed Stage-6 table beside those masks said 26.667 px
— 0.035 mm apart. The mask had not changed (same commit), the measurement code had not changed
(`git log` over `gsv4/measure`, `gsv4/masks` and `gsv4/eval/oracle.py` since the Stage-6 run is empty),
and three repeated runs on this machine gave the same answer every time. Same code, same input,
different answer — on a different machine.

### A6.2 The cause, located exactly

Method C takes the local minima of the smoothed thickness profile as zenith candidates and enforces a
minimum separation through `scipy.signal.find_peaks(distance=…)`. For this image the candidates at
columns 943 and 992 are both zero-thickness plateaus, so their depth is *exactly* equal — 130.0 and
130.0, not nearly equal — and they are 49 columns apart against a minimum separation of 81. Exactly
one of the two can survive, and `find_peaks` decides which by ordering the candidates with
`np.argsort(priority)`, whose default sort is **not stable**. With an exact tie the survivor is
therefore chosen by the sort implementation, not by the data:

| priority | survivors |
|---|---|
| 943 and 992 exactly tied | 992 |
| 943 deeper by 1e-9 | 943 |
| 992 deeper by 1e-9 | 992 |

The workstation runs numpy 2.3.5 (`outputs/08_architecture/rfdetr_environment.json`) and this machine
numpy 1.26.4; the two sort implementations break the tie the other way round. Regioning B has the same
defect in two places: the same `distance` filter, and `np.argsort(prom)[::-1]` when more papillae are
found than needed. Regioning A (equal splits) is unaffected and never differed.

Exact ties are not rare here: both regionings key on plateaus of a profile quantised to whole pixels,
and any two stretches with no gingiva are *exactly* equally deep.

### A6.3 How wide it was

`scripts/verify_measurement_reproducible.py` re-measures a stored table from the masks beside it and
reports every difference. Against the pre-fix code:

| stored table | masks | images differing (of 145) | of them, the reported method `C_p25` | largest difference |
|---|---|---|---|---|
| `outputs/03_oracle/` (Stage 3, GT masks) | COCO annotations | **0** | 0 | — |
| `outputs/09_final_rfdetr/` (Stage 6) | `oof_rfdetr` | 10 | 1 | 0.31 mm (`B_median`), 0.035 mm in `C_p25` |

That split is itself the proof: Stage 3 was run on this machine and reproduces bit for bit, Stage 6
was run on the workstation and does not. It is an environment difference, not staleness and not
non-determinism within one machine.

### A6.4 The fix

`gsv4.measure.regions.select_by_distance` now performs the minimum-separation filter. The greedy is
scipy's own, step for step, so nothing but the tie can move; what is new is that the order among
*equal* priorities is stated instead of inherited from a sort:

* **zeniths (C):** on an exact tie the candidate **nearer the dental midline** is kept — the criterion
  method C already uses to choose its zeniths, and symmetric about the midline, so it favours neither
  side of the arch;
* **papillae (B):** on an exact tie the **leftmost** is kept, left to right across the arch. The rest
  of `regions_festoon` now reproduces `find_peaks`' own order explicitly (all maxima → separation on
  the profile height → prominence threshold on the survivors), because `find_peaks` applies `distance`
  before `prominence` and a re-ordering there would have changed more than the tie.

Three tests hold this: the same mask measured four times must give identical bits, not close numbers;
`select_by_distance` must match `find_peaks(distance=…)` exactly on 50 random signals with no ties;
and the tie rules must pick the documented candidate and mirror when the profile is mirrored.

### A6.5 What it changed, and what it did not

Stage 3 was re-run with `--write-config`, because the selection rule of Stage 3 has to be executed by
the code that measures. **The pre-registered method is unchanged: `C_p25`**, still the lowest dev MAE,
still the only combination within the 0.02 mm simplicity tolerance. The scale moved from **16.8397 to
16.8422 px/mm** (0.015 %). Stage 6 was then re-run for both final models, and every Stage-7 table,
figure, rebuttal answer and manuscript edit regenerated from them.

The reported millimetre results are unchanged at the precision they are reported in:

| quantity | before | after |
|---|---|---|
| primary, out-of-fold, MAE / ICC(2,1) / bias / κ | 0.521 / 0.877 / +0.270 / 0.72 | 0.521 / 0.877 / +0.270 / 0.72 |
| test set with the final model, MAE | 0.671 | 0.671 |
| pre-registered sensitivity (116 images), MAE | 0.506 | 0.506 |

What did move, all of it on the **annotated** masks, where exact ties are commoner:

| quantity | before | after |
|---|---|---|
| GT masks, all 145 reference images, MAE | 0.542 | 0.546 |
| GT masks, Stage-3 holdout, MAE | 0.518 | 0.527 |
| GT masks, ICC(2,1) all / holdout | 0.868 / 0.858 | 0.867 / 0.857 |
| what the segmentation adds (current model) | −0.02 mm | −0.03 mm |
| what the segmentation adds (previous model) | +0.30 mm | +0.29 mm |
| lower gingival edge bias, current model | +0.08 mm | +0.07 mm |
| zenith fallback rate, GT masks | 19 % (28 of 145) | 19 % (27 of 145) |

None of it changes a conclusion: the segmentation still adds essentially nothing with RF-DETR and
about 0.3 mm with YOLOv11x, and the decision rule of `outputs/08_architecture/PROTOCOL.md` §8 was
triggered by a 0.293 mm lead that these shifts are two orders of magnitude below.

### A6.6 One thing still to redo, and it is named here rather than left implicit

The architecture comparison measured its five configurations on the workstation, from masks that are
not versioned (`outputs/08_architecture/masks/`, git-ignored), so it cannot be re-measured here. Its
`selected_mm` columns therefore still carry the old tie order and the old scale, and
`run_architecture_comparison.py --measure-only` has to be re-run on the workstation. Two reasons this
does not put the comparison's conclusion in question, stated before that run rather than after:

* the scale changed by 0.015 %, which multiplies every configuration's error by the same factor and
  leaves a paired difference of 0.293 mm at 0.293 mm;
* the tie affected 1 of 145 predicted-mask images in `C_p25`, by 0.035 mm; even two affected images
  among the 29 would move a configuration's mean absolute error by about 0.005 mm, against a lead of
  0.293 mm with a 95 % interval of [0.150, 0.439]. The comparison is also paired and was measured for
  all five configurations by one machine, so a tie that flips affects every configuration alike.

The re-measured tables replace the current ones whatever they show, and `RESULTS.md` is regenerated
from them.

### A6.7 What is unchanged

Every pre-registered decision of this plan and of `PROTOCOL.md`: the primary outcome of §5 and its
sensitivity analysis, the offset rule of §6 and its outcome in Amendment 3, the fallback rule of §7,
the learning-curve reading of Amendment 4, and the seed-spread pre-registration of Amendment 5, whose
runs have not started.
