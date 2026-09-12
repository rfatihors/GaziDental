# Expert per-image scale (Ölçek, pixel/mm) — agreement and use

Experts' mean per-image scale: n = 134, mean 16.89 px/mm (SD between images 0.94); global oracle scale 16.8397 px/mm.
Agreement among the three experts on the scale: ICC(2,1) 0.356 [0.249, 0.465]; within-image CV median 0.048 (mean 0.051). This directly measures the precision of probe-based calibration in these photographs.
Intra-expert scale repeatability (20 repeats): expert 1: ICC 0.604, CV 0.060, expert 2: ICC 0.304, CV 0.060, expert 3: ICC 0.222, CV 0.084.

Model mm was computed twice: global scale (primary) and expert per-image scale (secondary); both appear in class_agreement.csv and mm_agreement.csv.

## Frame-outside images: does the per-image expert scale close the gap to the clinical reference?
| frame | n | mae_global_scale | mae_expert_scale | mean_expert_px_per_mm |
|---|---|---|---|---|
| inside 2698x1799 | 123 | 0.531 | 0.543 | 16.841 |
| outside 2698x1799 | 22 | 0.603 | 0.670 | 17.129 |
