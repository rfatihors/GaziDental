# Scale estimation (C_p25)

Global scale: **16.84 px/mm, FIXED** — taken from `configs/config.yaml` (Stage 3, fitted on the dev subset of the **ground-truth** masks) and applied unchanged to every image here. Nothing was fitted on these masks.
For information only, never used: a through-origin fit of the same method on the dev subset of *these* masks would give 17.21 px/mm (+2.2 %); leave-one-out on that fit: mean 17.21, SD 0.071, range 17.06–17.71 px/mm.
Expected order of magnitude 15–20 px/mm (2698 px ≈ 15–16 cm field of view): OK.

## Per-image ratio px / reference mm (all 145 images; mixes scale variation with measurement noise, reported without interpretation)
overall: mean 19.72, SD 5.62, CV 0.28, median 19.07.

By frame size (the reference was measured on 2698×1799 copies; other sizes may carry a different scale):

| frame | count | mean | std | median | cv |
|---|---|---|---|---|---|
| other sizes | 22 | 20.748 | 10.040 | 18.397 | 0.484 |
| 2698x1799 ±2 px | 123 | 19.536 | 4.435 | 19.086 | 0.227 |

Images outside the 2698×1799 ±2 px frame: 22 of 145 reference images (15.2 %); in the whole COCO export 485 of 1315.
Mean per-image ratio differs by +6.2 % between the two groups. Dev-only through-origin scale by group: inside 17.26 px/mm, outside 16.83 px/mm.
The per-image ratio differs by +6.2 % but the dev regression scale by -2.5 % — opposite directions, and the outside-frame group is small (n = 22) and much noisier (per-image CV 0.48 vs 0.23). This points to frame *uncertainty* (which copy ImageJ used) rather than a consistent scale shift; a dual global scale is therefore NOT proposed. Recommendation: keep the single scale and report the outside-frame subset as a sensitivity row (done in oracle_summary.md).

Reference calibration note: ImageJ used a single 1 mm probe interval per image (≈ 17 px); a 1 px marking error is ≈ 5.9 % of scale, so part of the per-image CV is the reference's own calibration noise.
