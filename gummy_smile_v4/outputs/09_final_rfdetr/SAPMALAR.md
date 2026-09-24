# Aşama 6 — sapmalar ve notlar

- Test-set boundary table: gingiva IoU undefined on 0 images (both masks empty) and edge errors undefined on 0 images (no column with gingiva in both masks) — all low/normal; reported as presence categories, not as zeros.
- No post-hoc offset calibration: `measurement.bottom_edge_offset_px = 0`, so this report has ONE result set and no uncorrected/corrected pair. The decision and its reasons are in `outputs/09_final_rfdetr/PLAN.md` (Amendment 3); the offset analysis itself stays in the appendix as a finding about the YOLO family (`outputs/06_prediction/offset_correction.md`, `offset_checks.md`, `outputs/08_architecture/FINDINGS.md`).
