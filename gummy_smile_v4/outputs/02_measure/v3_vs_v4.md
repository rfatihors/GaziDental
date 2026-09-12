# v3 vs v4 on the same ground-truth masks

v3 = legacy `measure_gum_visibility` (largest contour, top-edge deviation) on the merged lip+gingiva mask exactly as its pipeline ran; v4 = gingiva-mask vertical thickness (method A, p25). Units: px at original resolution.

| uid | group | width | height | gingiva_instances_merged | lip_instances | v3_value_px | v3_gingiva_only_px | v4_value_px | v4_method | v4_A_median_px | v4_C_median_px | v4_A_p25_lipanchored_px | gap_median_px | reference_mean_mm | qc_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high/111 (1)_jpg | high | 2699 | 1799 | 1 | 1 | 18.33 | 44.83 | 47.5 | A_p25 | 56.08 | 49.33 | 49.67 | 0.0 | 2.4516666666666667 |  |
| high/IMG_2456_jpeg | high | 2698 | 1799 | 1 | 1 | 36.5 | 6.67 | 48.25 | A_p25 | 62.92 | 49.0 | 50.88 | 0.0 | 2.718 |  |
| high/IMG_2522-_jpg | high | 2698 | 1799 | 1 | 1 | 14.17 | 37.33 | 116.33 | A_p25 | 125.5 | 118.67 | 120.0 | 5.0 | 6.7106666666666674 |  |
| normal/19-IMG_5863_JPG | normal | 2285 | 1524 | 3 | 1 | 23.17 | 5.17 | 6.33 | A_p25 | 14.0 | 14.0 | 6.92 | 1.0 | nan | gingiva_multi_component,region_zero,zenith_detection_failed |
| normal/20IMG_3908_JPEG | normal | 2526 | 1684 | 9 | 1 | 14.17 | 5.0 | 0.0 | A_p25 | 7.33 | 0.0 | 0.0 | 5.0 | nan | gingiva_multi_component,region_zero |
