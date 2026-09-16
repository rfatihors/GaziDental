# Aşama 6 — Türkçe özet (tahmin maskeleri üzerinde doğruluk)

- OOF kontrolü: 145/145 görüntü, hepsi `yolo:masks.data`, fold başına [29, 29, 29, 29, 29]; **segmentasyon başarısızlığı n = 1** (IMG_7289_jpg: dişeti örneği yok → mm değeri yok; ayrı kategori olarak raporlanır, mm/sınıf metriklerine girmez).
- Yöntem ve ölçek sabit (Aşama 3, GT maske): **C_p25**, **16.84 px/mm**; tahmin maskelerinde yeniden seçim/yeniden uydurma yapılmadı.
- **Birincil (a) — 145 ölçümlü high, OOF maske (ölçülen n = 144):** MAE 0.84 mm, RMSE 1.00, r 0.875, ICC(2,1) 0.784 [0.288, 0.909]; sapma +0.68 mm, LoA -0.75…2.12; sınıf uyumu 62 %, doğrusal ağırlıklı κ 0.59 [0.50, 0.68].
  - Aynı yöntem GT maskede (Aşama 3): MAE 0.54, sapma +0.15 → segmentasyonun eklediği: MAE +0.30 mm, sapma +0.53 mm (alt dişeti kenarı GT'den 0.66 mm aşağıda çiziliyor; üst kenar +0.16 mm).
  - Aşama 3 holdout'u (ölçek hiç görmedi, n = 57): MAE 0.76, ICC 0.797.
  - **İkincil, düzeltilmiş (config `offset_px` = -13 px = -0.77 mm):** MAE 0.49 mm, ICC 0.871, sapma -0.09, sınıf uyumu 80 %, κ 0.74 [0.65, 0.82]; holdout MAE 0.46; 0'a kırpılan 0.
- **İkincil (b) — test high 29 görüntü, final model:** MAE 0.95 mm, r 0.804, ICC 0.731, sapma +0.65; κ 0.40 [0.20, 0.59] (n küçük, GA geniş). Düzeltilmiş: MAE 0.49, sapma -0.13, κ 0.64.
- **Segmentasyon kalitesi:** (a) dişeti IoU 0.758, üst kenar MAE 0.32 mm, alt kenar MAE 0.76 mm (sapma +0.66), dudak IoU 0.794; (b) IoU 0.763; (c) tüm test 192 görüntü IoU 0.508 — low 0.280, normal 0.492. Low/normal'da düşük IoU beklenen bir durum: dişeti ince ya da görünmüyor (low'da GT dişeti genişliği medyan 16 sütun, high'da 935); kenar hataları high'dan büyük değil. Birincil sonuç (a) üzerinden verilir.
- Zenith bölgeleme (C) geri düşme oranı OOF'ta 17 % (GT'de 19 %); %30 kuralı tetiklenmedi; A_p25 satırı tabloda.
- Diş düzeyi karma model (hasta rastgele kesişim): sapma +0.68 mm [+0.57, +0.80].
- Sapmalar: `SAPMALAR.md` (3 madde).
