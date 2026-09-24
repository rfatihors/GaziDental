# Aşama 3 — Türkçe özet

- 145 ölçümlü high görüntü (GT maske), dev 87 / holdout 58; `label_inconsistent` dışlanan: 0.
- Seçilen yöntem **B_p05** (dev MAE 0.523 mm; basitlik kuralı ±0.02 mm). Global ölçek **13.55 px/mm** (yalnız dev'de kestirildi).
- Holdout: MAE 0.54 mm, RMSE 0.74, r 0.866, ICC(2,1) 0.857; sapma +0.18 mm, orantısal eğim +0.104 (p 0.150).
- Gözlemci içi (dişeti görünürlüğünün tekrar ölçümü): ICC(2,1) diş bölgesi düzeyi 0.995, görüntü ortalaması 0.998; SD 0.17 mm. Holdout MAE'nin gözlemci gürültüsü üstünde kalan kısmı ≈ 0.47 mm.
- Dudak-altı ↔ dişeti-üstü boşluk medyanı 7.0 px (IQR 5–9); beklenti ≈ 9 px → uyumlu.
- Çerçeve: 22/145 görüntü 2698×1799 dışında; görüntü bazlı oran farkı +9.4 %, dev regresyon ölçeği farkı -8.1 % → ikili ölçek önerilmez (yönler zıt / grup küçük ve gürültülü); tek ölçek + duyarlılık satırı.
- Seçilen bölgeleme B: 2 görüntüde başarısız → A'ya düşer (raporda açık).
- Ön analizle (MAE ≈ 0.63, ≈ 17 px/mm) karşılaştırma: FARKLI — kontrol.
