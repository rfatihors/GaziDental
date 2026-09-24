# Scale estimation (B_p05)

Global scale: regression through the origin on the **dev subset only** (n = 87): **13.55 px/mm** (R² 0.745; residual SD 0.71 mm). Applied unchanged to holdout.
Leave-one-out on dev: mean 13.55, SD 0.054, range 13.44–13.93 px/mm; LOO MAE 0.531 mm.
Expected order of magnitude 15–20 px/mm (2698 px ≈ 15–16 cm field of view): OUTSIDE — check.

## Per-image ratio px / reference mm (all 145 images; mixes scale variation with measurement noise, reported without interpretation)
overall: mean 15.30, SD 4.72, CV 0.31, median 15.08.

By frame size (the reference was measured on 2698×1799 copies; other sizes may carry a different scale):

| frame | count | mean | std | median | cv |
|---|---|---|---|---|---|
| other sizes | 22 | 16.514 | 8.310 | 15.550 | 0.503 |
| 2698x1799 ±2 px | 123 | 15.088 | 3.756 | 14.974 | 0.249 |

Images outside the 2698×1799 ±2 px frame: 22 of 145 reference images (15.2 %); in the whole COCO export 485 of 1315.
Mean per-image ratio differs by +9.4 % between the two groups. Dev-only through-origin scale by group: inside 13.70 px/mm, outside 12.60 px/mm.
The per-image ratio differs by +9.4 % but the dev regression scale by -8.1 % — opposite directions, and the outside-frame group is small (n = 22) and much noisier (per-image CV 0.50 vs 0.25). This points to frame *uncertainty* (which copy ImageJ used) rather than a consistent scale shift; a dual global scale is therefore NOT proposed. Recommendation: keep the single scale and report the outside-frame subset as a sensitivity row (done in oracle_summary.md).

Reference calibration note: ImageJ used a single 1 mm probe interval per image (≈ 14 px); a 1 px marking error is ≈ 7.4 % of scale, so part of the per-image CV is the reference's own calibration noise.
