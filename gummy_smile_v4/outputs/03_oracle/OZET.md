# Aşama 3 — Türkçe özet

- 145 ölçümlü high görüntü (GT maske), dev 87 / holdout 58; `label_inconsistent` dışlanan: 0.
- Seçilen yöntem **C_p25** (dev MAE 0.557 mm; basitlik kuralı ±0.02 mm). Global ölçek **16.84 px/mm** (yalnız dev'de kestirildi).
- Holdout: MAE 0.52 mm, RMSE 0.72, r 0.861, ICC(2,1) 0.858; sapma +0.12 mm, orantısal eğim +0.054 (p 0.465).
- Gözlemci içi: ICC(2,1) diş 0.995, görüntü 0.998; SD 0.17 mm. Holdout MAE'nin gözlemci gürültüsü üstünde kalan kısmı ≈ 0.45 mm.
- Dudak-altı ↔ dişeti-üstü boşluk medyanı 8.0 px (IQR 5–12); beklenti ≈ 9 px → uyumlu.
- Çerçeve: 22/145 görüntü 2698×1799 dışında; görüntü bazlı oran farkı +7.3 %, dev regresyon ölçeği farkı -5.1 % → ikili ölçek önerilmez (yönler zıt / grup küçük ve gürültülü); tek ölçek + duyarlılık satırı.
- Seçilen bölgeleme C: 28 görüntüde başarısız → A'ya düşer (raporda açık).
- Ön analizle (MAE ≈ 0.63, ≈ 17 px/mm) karşılaştırma: makul.
