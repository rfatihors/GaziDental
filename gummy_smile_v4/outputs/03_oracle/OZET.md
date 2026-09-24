# Aşama 3 — Türkçe özet

- 145 ölçümlü high görüntü (GT maske), dev 87 / holdout 58; `label_inconsistent` dışlanan: 0.
- Seçilen yöntem **C_p25** (dev MAE 0.558 mm; basitlik kuralı ±0.02 mm). Global ölçek **16.84 px/mm** (yalnız dev'de kestirildi).
- Holdout: MAE 0.53 mm, RMSE 0.73, r 0.860, ICC(2,1) 0.857; sapma +0.13 mm, orantısal eğim +0.065 (p 0.378).
- Gözlemci içi (dişeti görünürlüğünün tekrar ölçümü): ICC(2,1) diş bölgesi düzeyi 0.995, görüntü ortalaması 0.998; SD 0.17 mm. Holdout MAE'nin gözlemci gürültüsü üstünde kalan kısmı ≈ 0.46 mm.
- Dudak-altı ↔ dişeti-üstü boşluk medyanı 8.0 px (IQR 5–12); beklenti ≈ 9 px → uyumlu.
- Çerçeve: 22/145 görüntü 2698×1799 dışında; görüntü bazlı oran farkı +7.2 %, dev regresyon ölçeği farkı -5.1 % → ikili ölçek önerilmez (yönler zıt / grup küçük ve gürültülü); tek ölçek + duyarlılık satırı.
- Seçilen bölgeleme C: 27 görüntüde başarısız → A'ya düşer (raporda açık).
- Ön analizle (MAE ≈ 0.63, ≈ 17 px/mm) karşılaştırma: makul.
