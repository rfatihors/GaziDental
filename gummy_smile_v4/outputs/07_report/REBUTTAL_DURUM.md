# Rebuttal durumu — hakem maddeleri (klinik ekibe)

Üretim: `scripts/build_rebuttal.py`; sayılar `outputs/` altındaki dosyalardan okunur. Durum: READY = analiz çıktılarından cevaplandı; PENDING = uzman formlarını bekliyor; CLINICAL = metni klinik ekip yazacak. Toplam 60 madde: READY 32, PENDING 5 (uzman formları), PENDING_RUN 0 (iş istasyonu koşusu), CLINICAL 23, MISSING 0 (hakem belgesinde bulunmayan maddeler; metni sorumlu yazardan istenecek).

Makale metninde değişmesi gereken yerler ayrı bir dosyada: `MANUSCRIPT_EDITS.md` (gönderilen makale ve Appendix B–F üzerinden, her madde için mevcut cümle / önerilen cümle / gerekçe / hakem maddesi).

| madde | hakem | konu | durum | sorumlu | cevaplayan çıktı |
|---|---|---|---|---|---|
| R1-novelty | Reviewer 1 | Technical novelty is incremental and integrative rather than foundational | **CLINICAL** | klinik | outputs/08_architecture/RESULTS.md; outputs/07_report/MANUSCRIPT_EDITS.md |
| R1-1 | Reviewer 1 | No system block diagram; how Fig. 6 treatment suggestions are reached | **READY** | teknik | outputs/07_report/figures/pipeline_block_diagram.png |
| R1-2 | Reviewer 1 | No segmentation result images | **READY** | teknik | outputs/07_report/figures/segmentation_examples.png; outputs/07_report/tables/segmentation_metrics_test.md |
| R1-discussion-length | Reviewer 1 | The Discussion is too long and repetitive | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R2-1 | Reviewer 2 | Abstract implies all 1,315 images were used for the quantitative analysis | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R2-2 | Reviewer 2 | Box metrics vs Mask metrics not identified | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md |
| R2-3 | Reviewer 2 | Whether a G*Power chi-square calculation is appropriate for a deep-learning segmentation model | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; docs/Istatistik_analiz_plani.md |
| R2-4 | Reviewer 2 | "YOLOv8 and YOLOv11" inconsistent with the rest of the manuscript | **READY** | teknik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R2-5 | Reviewer 2 | 1,315 vs 3,403 images and the instance counts in Figure 4 | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R2-6 | Reviewer 2 | Image- or patient-level split; data leakage; small test set | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R2-7 | Reviewer 2 | Single examiner; no inter-rater reliability, so a possible ground-truth bias | **PENDING** | teknik | docs/Uzman_degerlendirme_protokolu.md; outputs/04_expert/expert_summary.md; outputs/03_oracle/intra_observer.md |
| R2-8 | Reviewer 2 | Gingiva mAP@50 of 0.587 not reported in the text; should be discussed as a limitation | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/09_final_rfdetr/boundary_by_set.md; outputs/08_architecture/PROTOCOL.md |
| R2-9 | Reviewer 2 | No meaningful performance gain despite tuning, expansion and scaling | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; outputs/07_report/tables/segmentation_metrics_test.md |
| R2-10 | Reviewer 2 | Lip vs gingiva mAP gap should be investigated quantitatively | **READY** | teknik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/09_final_rfdetr/error_decomposition.md; outputs/07_report/figures/boundary_error.png |
| R2-11 | Reviewer 2 | Difference and novelty against the authors' previous study (Ref. 30, J Dent 2026) | **CLINICAL** | klinik | docs/Klinik_ekip_kararlari_15Eylul.md; outputs/01_data/OZET.md |
| R2-12 | Reviewer 2 | Intra-observer ICC value, 95 % confidence interval and ICC type not given | **READY** | teknik | outputs/03_oracle/intra_observer.md |
| R3-General-1 | Reviewer 3 | Rewrite the introduction with a specific aim and references for all statements | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-General-2 | Reviewer 3 | Clarify the materials and methods | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-General-3 | Reviewer 3 | Narrow the manuscript to a validated segmentation/measurement study, or compare with clinical assessment | **PENDING** | teknik | outputs/09_final_rfdetr/prediction_summary.md; outputs/04_expert/expert_summary.md |
| R3-General-4 | Reviewer 3 | Rewrite the discussion on the study outcomes; draw relevant rather than general conclusions | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-General-5 | Reviewer 3 | Improve the language of the manuscript | **CLINICAL** | klinik |  |
| R3-Title-1 | Reviewer 3 | The title promises more than was validated | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Abstract-1 | Reviewer 3 | Abstract reports no millimetric accuracy and no validation of the etiological/treatment categories | **PENDING** | teknik | outputs/07_report/MANUSCRIPT_EDITS.md; outputs/09_final_rfdetr/prediction_summary.md |
| R3-Abstract-2 | Reviewer 3 | Abstract must state the limitations: single centre, one examiner, no clinical assessment | **READY** | teknik | outputs/03_oracle/intra_observer.md; outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Intro-1 | Reviewer 3 | A reference is missing from the paragraph describing the state of the literature | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Intro-2 | Reviewer 3 | The stated scientific gap is general and does not match what the study evaluates | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Intro-3 | Reviewer 3 | Several objectives are presented at once; identify one primary objective | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Methods-1 | Reviewer 3 | Study design, retrospective/prospective, recruitment process and dates | **CLINICAL** | klinik | outputs/07_report/tables/dataset_counts.md |
| R3-Methods-2 | Reviewer 3 | Inclusion criteria are not given, only exclusion criteria | **CLINICAL** | klinik | outputs/07_report/tables/dataset_counts.md |
| R3-Methods-3 | Reviewer 3 | Sample size belongs in Methods; the numbers of patients and images in Results | **READY** | teknik | outputs/07_report/tables/learning_curve.md; outputs/07_report/tables/dataset_counts.md |
| R3-Methods-4 | Reviewer 3 | Why 1,315 patients and not another number | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md |
| R3-Methods-5 | Reviewer 3 | Unequal numbers of patients per smile-line group; equal gender distribution requested | **READY** | teknik | outputs/07_report/tables/demographics.md; outputs/07_report/tables/dataset_counts.md |
| R3-Methods-6 | Reviewer 3 | Was the periodontal probe in Figure 1 used for calibration? The method is not described | **READY** | teknik | outputs/03_oracle/scale_estimation.md; outputs/04_expert/scale_agreement.md |
| R3-Methods-7 | Reviewer 3 | Why low and average smile-line images were needed | **READY** | teknik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/07_report/tables/dataset_counts.md |
| R3-Methods-8 | Reviewer 3 | Cut-off and clinically acceptable error; 0, 1 and 3 mm would not be called a gummy smile | **READY** | teknik | outputs/07_report/tables/threshold_margin.md; outputs/07_report/tables/measurement_accuracy.md |
| R3-Methods-9 | Reviewer 3 | Was a systematic search used to synthesise the evidence behind Table 1? | **CLINICAL** | klinik |  |
| R3-Methods-10 | Reviewer 3 | Table 1 has clinical deficiencies (a: is < 4 mm a problem; b: aetiology from millimetres alone; c: short upper lip needs a lip measurement) | **CLINICAL** | klinik | docs/Uzman_degerlendirme_protokolu.md; outputs/07_report/tables/expert_agreement.md |
| R3-Methods-11 | Reviewer 3 | Abbreviations such as YOLO must be expanded at first mention | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Results-1 | Reviewer 3 | Demographics not reported; ethnicity and pigmentation of skin and gingiva may affect segmentation | **CLINICAL** | klinik | outputs/09_final_rfdetr/boundary_by_set.md; outputs/09_final_rfdetr/error_decomposition.md |
| R3-Results-2 | Reviewer 3 | Exact number of participants and images for training, validation and testing | **READY** | teknik | outputs/07_report/tables/dataset_counts.md |
| R3-Results-3 | Reviewer 3 | Section 3.5 belongs in Methods; only the validation results belong in Results | **READY** | teknik | outputs/07_report/MANUSCRIPT_EDITS.md; outputs/07_report/tables/threshold_margin.md |
| R3-Results-4 | Reviewer 3 | What 'performance' means; sensitivity/specificity; false positives and false negatives | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/05_predictions/confusion_matrix.png |
| R3-Results-5 | Reviewer 3 | Distinguish segmentation performance from millimetre measurement ability | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/09_final_rfdetr/prediction_summary.md; outputs/09_final_rfdetr/error_decomposition.md |
| R3-Discussion-1 | Reviewer 3 | The discussion repeats literature that belongs in the introduction | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Discussion-2 | Reviewer 3 | The discussion should focus on this study's results and discuss the model's validity | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Discussion-3 | Reviewer 3 | Limitations must include selection bias, one examiner and demographic bias | **READY** | teknik | outputs/07_report/tables/demographics.md; outputs/03_oracle/intra_observer.md; outputs/07_report/tables/learning_curve.md |
| R3-Discussion-4 | Reviewer 3 | A conclusion appears twice: at the end of the discussion and as its own section | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-Discussion-5 | Reviewer 3 | Conclusion must not state the model is clinically validated | **CLINICAL** | klinik | outputs/07_report/MANUSCRIPT_EDITS.md |
| R3-References-1 | Reviewer 3 | Why reference 11 was used in the sample size calculation | **READY** | teknik | outputs/07_report/tables/learning_curve.md; docs/Istatistik_analiz_plani.md |
| R4-1 | Reviewer 4 | The etiology-treatment module is not validated; no clinical gold standard, no agreement with clinicians | **PENDING** | teknik | outputs/04_expert/expert_summary.md; outputs/07_report/tables/expert_agreement.md; docs/Uzman_degerlendirme_protokolu.md |
| R4-1b | Reviewer 4 | No cohort in which clinicians independently establish the cause and the plan; no sensitivity, specificity or kappa | **PENDING** | teknik | outputs/04_expert/expert_summary.md; outputs/07_report/tables/expert_agreement.md; docs/Uzman_degerlendirme_protokolu.md |
| R4-2 | Reviewer 4 | Overlap with the authors' J Dent 2026 study, same ethics number; originality and redundant publication | **CLINICAL** | klinik | docs/Klinik_ekip_kararlari_15Eylul.md; outputs/01_data/OZET.md |
| R4-3 | Reviewer 4 | Figure 6 — v3_yolo vs v1_xgboost give radically different measurements | **READY** | teknik | outputs/02_measure/v3_vs_v4.md; outputs/07_report/figures/measurement_gt_masks.png; outputs/07_report/figures/measurement_predicted_masks.png; outputs/07_report/tables/measurement_accuracy.md |
| R4-4 | Reviewer 4 | Millimetre-level accuracy not demonstrated (MAE, RMSE, Bland-Altman, ICC) | **READY** | teknik | outputs/09_final_rfdetr/prediction_summary.md; outputs/07_report/tables/measurement_accuracy.md; outputs/07_report/figures/measurement_predicted_masks.png; outputs/03_oracle/intra_observer.md |
| R4-5 | Reviewer 4 | Pixel-to-millimetre conversion not described | **READY** | teknik | outputs/03_oracle/scale_estimation.md; outputs/09_final_rfdetr/offset_correction.md; outputs/06_prediction/offset_checks.md |
| R4-6 | Reviewer 4 | Validation-set metrics reported as final results instead of test-set metrics | **READY** | teknik | outputs/07_report/tables/segmentation_metrics_test.md; outputs/07_report/tables/dataset_counts.md; outputs/09_final_rfdetr/measurement_accuracy.csv |
| R4-7 | Reviewer 4 | Unclear splitting scheme (70/15/15 then 92/4/4), 1,315 to 3,403, patient-level partitioning (CLAIM 2024) | **READY** | teknik | outputs/07_report/tables/dataset_counts.md; outputs/01_data/manifest_summary.md |
| R4-8 | Reviewer 4 | Architecture comparison is not fair (different dataset versions) | **READY** | teknik | outputs/08_architecture/PROTOCOL.md; outputs/08_architecture/PROTOCOL_ADDENDUM_resolution.md; outputs/08_architecture/RESULTS.md; outputs/08_architecture/FINDINGS.md; outputs/07_report/MANUSCRIPT_EDITS.md |
| R4-9 | Reviewer 4 | The etiology of a gummy smile cannot be derived from millimetres alone; this contradicts the logic of Table 1 | **CLINICAL** | klinik | docs/Uzman_degerlendirme_protokolu.md; outputs/07_report/tables/expert_agreement.md |
| R4-external-validity | Reviewer 4 | The G*Power calculation does not establish sufficiency or external validity | **READY** | teknik | outputs/07_report/figures/learning_curve.png; outputs/07_report/tables/learning_curve.md; outputs/07_report/MANUSCRIPT_EDITS.md |

Kaynak: `docs/Hakem_Yorumları.docx` — hakem mektubunun **tam** metni. Madde listesi ve alıntılar bu dosyadan `scripts/build_reviewer_items.py` ile üretiliyor; alıntılar mektubun satırlarından kesilerek alındığı için birebirdir. Daha önce kullanılan `docs/Hakem_revizyonları.docx` kısaltılmış bir özettir (31 madde) ve artık kaynak değildir; çeliştikleri yerde tam mektup esastır. Klinik ekibe gidecek maddeler ayrıca `KLINIK_EKIBE.md` dosyasında toplandı.
