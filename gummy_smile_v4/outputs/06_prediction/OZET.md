# Aşama 6 — Türkçe özet (tahmin maskeleri üzerinde doğruluk)

- **Model ve kaynak:** OOF maskeleri yolo11x-seg @640, 5 fold models → `outputs/05_predictions/oof`; final model yolo11x-seg @640 → `outputs/05_predictions/test`. Sınır/IoU tabloları bu maskelerden burada hesaplandı; tespit/segmentasyon metrikleri yalnız bu modelin kendi değerlendiricisinden (Ultralytics val() (box/mask P, R, F1, mAP@50, mAP@50-95)) alınır, başka bir modelin dosyası kullanılmaz.
- OOF kontrolü: 145/145 görüntü, hepsi `yolo:masks.data`, fold başına [29, 29, 29, 29, 29]; **segmentasyon başarısızlığı n = 1** (IMG_7289_jpg: dişeti örneği yok → mm değeri yok; ayrı kategori olarak raporlanır, mm/sınıf metriklerine girmez).
- Yöntem ve ölçek sabit (Aşama 3, GT maske): **C_p25**, **16.84 px/mm**; tahmin maskelerinde yeniden seçim/yeniden uydurma yapılmadı. Post-hoc ofset düzeltmesi (maske düzeyi, -13 px = -0.77 mm) ikincil sonuç olarak her tabloda birlikte veriliyor.
- **Birincil (a) — 145 ölçümlü high, OOF maske (ölçülen n = 144):** MAE 0.84 mm, RMSE 1.00, r 0.875, ICC(2,1) 0.784 [0.288, 0.909]; sapma +0.68 mm, LoA -0.75…2.12; sınıf uyumu 62 %, doğrusal ağırlıklı κ 0.59 [0.50, 0.68].
  - Aynı yöntem GT maskede (Aşama 3): MAE 0.54, sapma +0.15 → segmentasyonun eklediği: MAE +0.30 mm, sapma +0.53 mm (alt dişeti kenarı GT'den 0.66 mm aşağıda çiziliyor; üst kenar +0.16 mm).
  - Aşama 3 holdout'u (ölçek hiç görmedi, n = 57): MAE 0.76, ICC 0.797.
  - **İkincil, düzeltilmiş (maske düzeyi, config `bottom_edge_offset_px` = -13 px = -0.77 mm):** MAE 0.48 mm, ICC 0.873, sapma -0.05, sınıf uyumu 81 %, κ 0.75 [0.66, 0.83]; holdout MAE 0.43; kaydırmayla sıfırlanan sütunu olan görüntü 78.
- **İkincil (b) — test high 29 görüntü, final model:** MAE 0.95 mm, r 0.804, ICC 0.731, sapma +0.65; κ 0.40 [0.20, 0.59] (n küçük, GA geniş). Düzeltilmiş: MAE 0.47, sapma -0.09, κ 0.64.
- **Segmentasyon kalitesi:** (a) dişeti IoU 0.758, üst kenar MAE 0.32 mm, alt kenar MAE 0.76 mm (sapma +0.66), dudak IoU 0.794; (b) IoU 0.763; (c) tüm test 192 görüntü IoU 0.508 — low 0.280, normal 0.493. Low/normal'da düşük IoU beklenen bir durum: dişeti ince ya da görünmüyor (low'da GT dişeti genişliği medyan 16 sütun, high'da 935); kenar hataları high'dan büyük değil. Birincil sonuç (a) üzerinden verilir.
- Zenith bölgeleme (C) geri düşme oranı OOF'ta 17 % (GT'de 19 %); %30 kuralı tetiklenmedi; A_p25 satırı tabloda.
- Diş düzeyi karma model (hasta rastgele kesişim): sapma +0.68 mm [+0.57, +0.80].
- Sapmalar: `SAPMALAR.md` (3 madde).
