# Rebuttal durumu — hakem maddeleri (klinik ekibe)

Üretim: `scripts/build_rebuttal.py`; sayılar `outputs/` altındaki dosyalardan okunur. Durum: READY = analiz çıktılarından cevaplandı; PENDING = uzman formlarını bekliyor; CLINICAL = metni klinik ekip yazacak. Toplam 15 madde: READY 12, PENDING 1, CLINICAL 2.

| madde | hakem | konu | durum | sorumlu | cevaplayan çıktı |
|---|---|---|---|---|---|
| R1-1 | Reviewer 1 | System diagram missing | **READY** | teknik | outputs/07_report/figures/pipeline_block_diagram.png |
| R1-2 | Reviewer 1 | Segmentation outputs not shown | **READY** | teknik | outputs/07_report/figures/segmentation_examples.png; outputs/07_report/tables/segmentation_metrics_test.md |
| R2-5 | Reviewer 2 | Image counts (1,315 → 2,235 / 3,403) | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R2-6 | Reviewer 2 | Data leakage between splits | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R2-10 | Reviewer 2 | Boundary accuracy; lip vs gingiva performance gap | **READY** | teknik | outputs/06_prediction/boundary_by_set.md; outputs/06_prediction/error_decomposition.md; outputs/07_report/figures/boundary_error.png |
| R2-power | Reviewer 2 | G*Power calculation inappropriate | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md |
| R2-fig6 | Reviewer 2 | Figure 6 inconsistency (measurement vs reference) | **READY** | teknik | outputs/02_measure/v3_vs_v4.md; outputs/07_report/figures/measurement_gt_masks.png; outputs/07_report/figures/measurement_predicted_masks.png; outputs/07_report/tables/measurement_accuracy.md |
| R3-1 | Reviewer 3 | Demographics / sex distribution | **READY** | teknik | outputs/07_report/tables/demographics.md |
| R4-1 | Reviewer 4 | Sample size and external validity | **READY** | teknik | outputs/07_report/figures/learning_curve.png |
| R4-2 | Reviewer 4 | Calibration (pixel to mm) | **READY** | teknik | outputs/03_oracle/scale_estimation.md; outputs/06_prediction/offset_correction.md; outputs/06_prediction/offset_checks.md |
| R4-3 | Reviewer 4 | Circular validation of the E/T classification | **PENDING** | teknik | outputs/04_expert/expert_summary.md; outputs/07_report/tables/expert_agreement.md; docs/Uzman_degerlendirme_protokolu.md |
| R4-4 | Reviewer 4 | Overlap with the earlier (J Dent 2026) cohort | **CLINICAL** | klinik | docs/Klinik_ekip_kararlari_15Eylul.md; outputs/01_data/OZET.md |
| R4-5 | Reviewer 4 | Data leakage (CLAIM 2024) | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R4-6 | Reviewer 4 | Intra-observer reliability of the reference | **READY** | teknik | outputs/03_oracle/intra_observer.md |
| R4-7 | Reviewer 4 | Terminology, rare classes and limitations | **CLINICAL** | klinik | docs/Klinik_ekip_kararlari_15Eylul.md; outputs/06_prediction/prediction_summary.md |

Not: hakemlerin orijinal metni depoda yok (`docs/Hakem_revizyonları.docx`); madde numaraları ve alıntılar belge geldiğinde `docs/hakem_maddeleri.yaml` üzerinden tamamlanacak. Listedeki maddeler teknik denetim raporu ve görev belgesinden derlendi; belgede başka maddeler varsa eklenecek.
