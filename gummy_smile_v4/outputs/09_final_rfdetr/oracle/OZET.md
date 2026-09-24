# Oracle ölçümü — tahmin maskeleri (outputs/05_predictions/oof_rfdetr)

- 145 ölçümlü high görüntü (tahmin maskesi: outputs/05_predictions/oof_rfdetr), dev 87 / holdout 58; `label_inconsistent` dışlanan: 0.
- Yöntem **C_p25** ve ölçek **16.84 px/mm** `configs/config.yaml`'den SABİT alındı (Aşama 3'te GT maskelerde seçildi); burada hiçbir şey yeniden seçilmedi, yeniden kestirilmedi ve config'e yazılmadı. Bu maskelerdeki dev MAE 0.549 mm. `estimator_comparison.csv` yalnız duyarlılık amaçlıdır; yeniden seçim ancak `--reselect` ile yapılır.
- Holdout: MAE 0.48 mm, RMSE 0.68, r 0.896, ICC(2,1) 0.876; sapma +0.29 mm, orantısal eğim +0.045 (p 0.478).
- Gözlemci içi (dişeti görünürlüğünün tekrar ölçümü): ICC(2,1) diş bölgesi düzeyi 0.995, görüntü ortalaması 0.998; SD 0.17 mm. Holdout MAE'nin gözlemci gürültüsü üstünde kalan kısmı ≈ 0.41 mm.
- Dudak-altı ↔ dişeti-üstü boşluk medyanı 7.0 px (IQR 5–9); beklenti ≈ 9 px → uyumlu.
- Çerçeve: 22/145 görüntü 2698×1799 dışında; görüntü bazlı oran farkı +6.2 %, dev regresyon ölçeği farkı -2.5 % → ikili ölçek önerilmez (yönler zıt / grup küçük ve gürültülü); tek ölçek + duyarlılık satırı.
- Bölgeleme C: 27 görüntüde başarısız → A'ya düşer (raporda açık); oran %19 (ön kayıtlı eşik %30).
- Ön analizle (MAE ≈ 0.63, ≈ 17 px/mm) karşılaştırma: makul.
