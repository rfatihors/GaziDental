# Scale estimation (C_p25)

Global scale: regression through the origin on the **dev subset only** (n = 87): **16.84 px/mm** (R² 0.717; residual SD 0.77 mm). Applied unchanged to holdout.
Leave-one-out on dev: mean 16.84, SD 0.069, range 16.71–17.32 px/mm; LOO MAE 0.565 mm.
Expected order of magnitude 15–20 px/mm (2698 px ≈ 15–16 cm field of view): OK.

## Per-image ratio px / reference mm (all 145 images; mixes scale variation with measurement noise, reported without interpretation)
overall: mean 18.86, SD 6.30, CV 0.33, median 18.15.

By frame size (the reference was measured on 2698×1799 copies; other sizes may carry a different scale):

| frame | count | mean | std | median | cv |
|---|---|---|---|---|---|
| other sizes | 22 | 20.025 | 10.958 | 17.854 | 0.547 |
| 2698x1799 ±2 px | 123 | 18.655 | 5.094 | 18.148 | 0.273 |

Images outside the 2698×1799 ±2 px frame: 22 of 145 reference images (15.2 %); in the whole COCO export 485 of 1315.
Mean per-image ratio differs by +7.3 % between the two groups. Dev-only through-origin scale by group: inside 16.96 px/mm, outside 16.09 px/mm.
The per-image ratio differs by +7.3 % but the dev regression scale by -5.1 % — opposite directions, and the outside-frame group is small (n = 22) and much noisier (per-image CV 0.55 vs 0.27). This points to frame *uncertainty* (which copy ImageJ used) rather than a consistent scale shift; a dual global scale is therefore NOT proposed. Recommendation: keep the single scale and report the outside-frame subset as a sensitivity row (done in oracle_summary.md).

Reference calibration note: ImageJ used a single 1 mm probe interval per image (≈ 17 px); a 1 px marking error is ≈ 5.9 % of scale, so part of the per-image CV is the reference's own calibration noise.
