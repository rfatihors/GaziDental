# Rebuttal durumu — hakem maddeleri (klinik ekibe)

Üretim: `scripts/build_rebuttal.py`; sayılar `outputs/` altındaki dosyalardan okunur. Durum: READY = analiz çıktılarından cevaplandı; PENDING = uzman formlarını bekliyor; CLINICAL = metni klinik ekip yazacak. Toplam 31 madde: READY 22, PENDING 3 (uzman formları), PENDING_RUN 1 (iş istasyonu koşusu), CLINICAL 3, MISSING 2 (hakem belgesinde bulunmayan maddeler; metni sorumlu yazardan istenecek).

Makale metninde değişmesi gereken yerler ayrı bir dosyada: `MANUSCRIPT_EDITS.md` (gönderilen makale ve Appendix B–F üzerinden, her madde için mevcut cümle / önerilen cümle / gerekçe / hakem maddesi).

| madde | hakem | konu | durum | sorumlu | cevaplayan çıktı |
|---|---|---|---|---|---|
| R1-1 | Reviewer 1 | No system block diagram; how Fig. 6 treatment suggestions are reached | **READY** | teknik | outputs/07_report/figures/pipeline_block_diagram.png |
| R1-2 | Reviewer 1 | No segmentation result images | **READY** | teknik | outputs/07_report/figures/segmentation_examples.png; outputs/07_report/tables/segmentation_metrics_test.md |
| R2-1 | Reviewer 2 | Abstract implies all 1,315 images were used for the quantitative analysis | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R2-2 | Reviewer 2 | Box metrics vs Mask metrics not identified | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md |
| R2-3 | Reviewer 2 | Not in the document — text to be requested from the corresponding author | **MISSING** | klinik | — text to be requested from the corresponding author |
| R2-4 | Reviewer 2 | "YOLOv8 and YOLOv11" inconsistent with the rest of the manuscript | **READY** | teknik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R2-5 | Reviewer 2 | 1,315 vs 3,403 images and the instance counts in Figure 4 | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R2-6 | Reviewer 2 | Image- or patient-level split; data leakage; small test set | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R2-7 | Reviewer 2 | Not in the document — text to be requested from the corresponding author | **MISSING** | klinik | — text to be requested from the corresponding author |
| R2-8 | Reviewer 2 | Gingiva mAP@50 of 0.587 not reported in the text; should be discussed as a limitation | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/09_final_rfdetr/boundary_by_set.md; outputs/08_architecture/PROTOCOL.md |
| R2-9 | Reviewer 2 | No meaningful performance gain despite tuning, expansion and scaling | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; outputs/07_report/tables/segmentation_metrics_test.md |
| R2-10 | Reviewer 2 | Lip vs gingiva mAP gap should be investigated quantitatively | **READY** | teknik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/09_final_rfdetr/error_decomposition.md; outputs/07_report/figures/boundary_error.png |
| R2-sample-size | Reviewer 2 | Whether a G*Power chi-square calculation is appropriate for a deep-learning segmentation model | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; docs/Istatistik_analiz_plani.md |
| R3-General-3 | Reviewer 3 | Narrow the manuscript to a validated segmentation/measurement study, or compare with clinical assessment | **PENDING** | teknik | outputs/09_final_rfdetr/prediction_summary.md; outputs/04_expert/expert_summary.md |
| R3-Abstract-1 | Reviewer 3 | Abstract reports no millimetric accuracy and no validation of the etiological/treatment categories | **PENDING** | teknik | outputs/07_report/MANUSCRIPT_EDITS.md; outputs/09_final_rfdetr/prediction_summary.md |
| R3-Methods-5 | Reviewer 3 | Unequal numbers of patients per smile-line group; equal gender distribution requested | **READY** | teknik | outputs/07_report/tables/demographics.md; outputs/07_report/tables/dataset_counts.md |
| R3-Methods-7 | Reviewer 3 | Why low and average smile-line images were needed | **READY** | teknik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/07_report/tables/dataset_counts.md |
| R3-Results-1 | Reviewer 3 | Ethnicity and pigmentation of skin and gingiva | **CLINICAL** | klinik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/09_final_rfdetr/error_decomposition.md |
| R3-Results-2 | Reviewer 3 | Exact number of participants and images for training, validation and testing | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R3-Results-4 | Reviewer 3 | What "performance" means; sensitivity/specificity; false positives and false negatives | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/05_predictions/confusion_matrix.png |
| R3-Results-5 | Reviewer 3 | Distinguish segmentation performance from millimetre measurement ability | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/09_final_rfdetr/prediction_summary.md; outputs/09_final_rfdetr/error_decomposition.md |
| R3-Discussion-5 | Reviewer 3 | Conclusion must not state the model is clinically validated | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R4-1 | Reviewer 4 | The etiology-treatment module is not validated; no comparison with clinicians | **PENDING** | teknik | outputs/04_expert/expert_summary.md; outputs/07_report/tables/expert_agreement.md; docs/Uzman_degerlendirme_protokolu.md |
| R4-2 | Reviewer 4 | Overlap with the authors' J Dent 2026 study; redundant publication | **CLINICAL** | klinik | docs/Klinik_ekip_kararlari_15Eylul.md; outputs/01_data/OZET.md |
| R4-3 | Reviewer 4 | Figure 6 — v3_yolo vs v1_xgboost give radically different measurements | **READY** | teknik | outputs/02_measure/v3_vs_v4.md; outputs/07_report/figures/measurement_gt_masks.png; outputs/07_report/figures/measurement_predicted_masks.png; outputs/07_report/tables/measurement_accuracy.md |
| R4-4 | Reviewer 4 | Millimetre-level accuracy not demonstrated (MAE, RMSE, Bland-Altman, ICC) | **READY** | teknik | outputs/09_final_rfdetr/prediction_summary.md; outputs/07_report/tables/measurement_accuracy.md; outputs/07_report/figures/measurement_predicted_masks.png; outputs/03_oracle/intra_observer.md |
| R4-5 | Reviewer 4 | Pixel-to-millimetre conversion not described | **READY** | teknik | outputs/03_oracle/scale_estimation.md; outputs/09_final_rfdetr/offset_correction.md; outputs/06_prediction/offset_checks.md |
| R4-6 | Reviewer 4 | Validation-set metrics reported as final results instead of test-set metrics | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/07_report/tables/dataset_counts.md; outputs/09_final_rfdetr/measurement_accuracy.csv |
| R4-7 | Reviewer 4 | Unclear splitting scheme (70/15/15 then 92/4/4), 1,315 to 3,403, patient-level partitioning (CLAIM 2024) | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R4-external-validity | Reviewer 4 | Sample size does not establish external validity; no external testing | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; outputs/07_report/MANUSCRIPT_EDITS.md |
| R4-8 | Reviewer 4 | Architecture comparison is not fair (different dataset versions) | **PENDING_RUN** | teknik | outputs/08_architecture/PROTOCOL.md; outputs/07_report/tables/segmentation_metrics_test.md; outputs/07_report/MANUSCRIPT_EDITS.md |

Not: hakemlerin orijinal metni depoda yok (`docs/Hakem_revizyonları.docx`); madde numaraları ve alıntılar belge geldiğinde `docs/hakem_maddeleri.yaml` üzerinden tamamlanacak. Listedeki maddeler teknik denetim raporu ve görev belgesinden derlendi; belgede başka maddeler varsa eklenecek.
