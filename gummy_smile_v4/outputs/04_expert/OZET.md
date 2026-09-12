# Aşama 4 — Türkçe özet

- Veri: SYNTHETIC forms (p_agree ≈ 0.8, expert 2 biased upward, messy cells) written to /Users/fors/Projeler/GaziDental_calisma/GaziDental/gummy_smile_v4/outputs/04_expert/synthetic_forms; real model table /Users/fors/Projeler/GaziDental_calisma/GaziDental/gummy_smile_v4/outputs/03_oracle/per_image_results.csv.
- Formlar: satır düşürülmedi; uzman başına boş satır [0, 3, 0], eksik sınıf [0, 3, 0], eksik ölçek [6, 6, 2].
- Referans: çoğunluk; konsensüs bekleyen 12.
- Birincil (test alt kümesi, katı, global ölçek): doğrusal ağırlıklı κ 0.471 [-0.145, 0.836], n = 27. Tüm görüntüler: κ 0.549 [0.388, 0.688].
- Uzmanlar arası Fleiss κ 0.130 [0.036, 0.218]; mm ICC(2,1) 0.991; ölçek ICC(2,1) 0.356, görüntü içi CV medyan 0.048.
- Model–uzman ortalaması mm: ICC(2,1) 0.868, sapma +0.16 mm.
- Karma modeller: 3/3 kuruldu; 0 tanesi MixedLM, gerisi sınır durumu (hasta varyansı ≈ 0) → küme-dayanıklı OLS.
- Üretilen dosyalar: form_qc.csv, forms_long.csv, reference_standard.csv, consensus_pending.csv, class_agreement.csv, per_class.csv, strata.csv, intra_expert.csv, inter_expert.json, mm_agreement.csv, tooth_level_long.csv, mixed_models.md, scale_agreement.md, frame_scale_comparison.csv, expert_summary.md, manuscript_numbers.md.
