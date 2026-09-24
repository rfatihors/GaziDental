# Aşama 6 — Türkçe özet (tahmin maskeleri üzerinde doğruluk)

- **Model ve kaynak:** OOF maskeleri RF-DETR-Seg Large @624, seed 42, 5 fold models → `outputs/05_predictions/oof_rfdetr`; final model RF-DETR-Seg Large @624, seed 42 → `outputs/05_predictions/test_rfdetr`. Sınır/IoU tabloları bu maskelerden burada hesaplandı; tespit/segmentasyon metrikleri yalnız bu modelin kendi değerlendiricisinden (RF-DETR's own COCO evaluation (pycocotools, iouType='segm')) alınır, başka bir modelin dosyası kullanılmaz.
- OOF kontrolü: 145/145 görüntü, hepsi `rfdetr:masks`, fold başına [29, 29, 29, 29, 29]; **segmentasyon başarısızlığı n = 0** (yok: dişeti örneği yok → mm değeri yok; ayrı kategori olarak raporlanır, mm/sınıf metriklerine girmez).
- Yöntem ve ölçek sabit (Aşama 3, GT maske): **C_p25**, **16.84 px/mm**; tahmin maskelerinde yeniden seçim/yeniden uydurma yapılmadı. **Post-hoc ofset düzeltmesi yok** (`bottom_edge_offset_px = 0`): tek sonuç seti, düzeltilmiş/düzeltilmemiş ayrımı yok. Karar ve gerekçesi `outputs/09_final_rfdetr/PLAN.md` Amendment 3'te; ofset analizi YOLO ailesine özgü bir bulgu olarak ekte kalıyor.
- **Birincil (a) — 145 ölçümlü high, OOF maske (ölçülen n = 145):** MAE 0.52 mm, RMSE 0.73, r 0.892, ICC(2,1) 0.877 [0.804, 0.919]; sapma +0.27 mm, LoA -1.06…1.61; sınıf uyumu 76 %, doğrusal ağırlıklı κ 0.72 [0.63, 0.80].
  - Aynı yöntem GT maskede (Aşama 3): MAE 0.54, sapma +0.15 → segmentasyonun eklediği: MAE -0.02 mm, sapma +0.12 mm (alt dişeti kenarı GT'den 0.08 mm aşağıda çiziliyor; üst kenar +0.02 mm).
  - Aşama 3 holdout'u (ölçek hiç görmedi, n = 58): MAE 0.48, ICC 0.876.
- **İkincil (b) — test high 29 görüntü, final model:** MAE 0.67 mm, r 0.804, ICC 0.787, sapma +0.28; κ 0.62 [0.42, 0.79] (n küçük, GA geniş).
- **Segmentasyon kalitesi:** (a) dişeti IoU 0.821, üst kenar MAE 0.28 mm, alt kenar MAE 0.47 mm (sapma +0.08), dudak IoU 0.836; (b) IoU 0.821; (c) tüm test 29 görüntü IoU 0.821 — low nan, normal nan. Low/normal'da düşük IoU beklenen bir durum: dişeti ince ya da görünmüyor (low'da GT dişeti genişliği medyan nan sütun, high'da 935); kenar hataları high'dan büyük değil. Birincil sonuç (a) üzerinden verilir.
- Zenith bölgeleme (C) geri düşme oranı OOF'ta 19 % (GT'de 19 %); %30 kuralı tetiklenmedi; A_p25 satırı tabloda.
- Diş düzeyi karma model (hasta rastgele kesişim): sapma +0.27 mm [+0.16, +0.38].
- Sapmalar: `SAPMALAR.md` (2 madde).
