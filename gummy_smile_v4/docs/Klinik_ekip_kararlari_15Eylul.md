# Klinik ekip ve istatistikçi kararları — 15 Eylül eki

Bu belge `docs/` altındaki diğer belgelerden **daha yenidir**; çelişkide bu esas alınır.

## 1. J Dent 2026 örtüşmesi — kesinleşti

Klinik ekip: iki çalışma aynı kaynak görüntü havuzunu (başlangıçta 1.748 standart gülümseme fotoğrafı) kullandı; önceki çalışmada model geliştirmeye yalnızca yüksek gülme hattı görüntüleri (687) girdi. Elimizdeki ölçüm dosyası (`Hasta_ID-_Ölçümler.xlsx`, yüksek sayfası, 692 satır) o çalışmanın ölçüm setidir.

Doğrulanmış sayılar (Aşama 1 eşleştirmesinden):

| | Görüntü |
|---|---|
| Bu çalışma, temizlik öncesi | 1.315 (216 high / 303 low / 796 normal) |
| Önceki çalışmanın model setinde de bulunan | **149** (hepsi high; `matches_high.csv`) |
| Önceki çalışmada analize girmemiş | 1.166 (67 high + 1.099 low/normal) |
| Temizlik sonrası | 1.230; bunun 145'i önceki setle ortak, 1.085'i değil |
| Önceki train/valid/test setleri | **yeniden kullanılmadı** (yeni hasta düzeyi bölünme, farklı etiket şeması: 2 sınıf instance, farklı model) |

**Aşama 7 için:** `outputs/07_report/overlap_with_prior_study.md` üret: yukarıdaki tablo + `matches_high.csv`'den türeyen liste (hangi görüntüler ortak). Rebuttal paragrafı (İngilizce):

> Both studies draw on the same source pool of standardised smile photographs acquired at our centre. The previous study (J Dent 2026) used only high-smile-line images (n = 687) for model development, with a single "visible gingiva" class segmented by DeepLabV3+. Of the 1,315 images in the present study, 149 high-smile-line images were also part of the previous model-development set; the remaining 1,166 images (67 high, 303 low and 796 average smile line) were not analysed previously. After removal of duplicate photographs (Section 2.x), 145 of the 1,230 retained images overlap with the previous study. No training, validation or test partition from the previous study was reused: the present data were re-annotated with a two-class (lip, gingiva) instance-segmentation scheme, partitioned anew at the participant level, and used to train a different architecture. The two studies therefore share source images but differ in labelling scheme, model, task (lip–gingiva instance segmentation and threshold-based decision support versus gingiva-only segmentation and regression) and analytical outputs.

## 2. Uzman değerlendirmesi — tasarım netleştirmesi

Klinik ekibin metni ("her değerlendirici, ölçümün karşılık geldiği sınıfı atayacak") ile protokol (`Uzman_degerlendirme_protokolu.md`: klinik yargı, tablo gösterilmez) arasında fark var. **Geçerli protokol değişmedi:** uzmanlar mm ölçer **ve** sınıfı kendi klinik yargısıyla verir; eşik tablosunu görmez. Nedeni: uzman da tabloyu uygularsa karşılaştırma Reviewer 4'ün eleştirdiği döngüsel analize döner.

Buna ek olarak, klinik ekibin istediği "ölçüme dayalı sınıf" karşılaştırması **post-hoc** üretilir: uzmanın mm değerine kural motoru uygulanır → `expert_table_class`. Böylece üç karşılaştırma raporlanır:

| Karşılaştırma | Ne ölçer |
|---|---|
| Model sınıfı vs uzman **klinik** sınıfı (birincil) | Karar destek katmanının klinik yargıyla uyumu |
| Model sınıfı vs uzman mm'sine tablo uygulanmış sınıf | Ölçüm + eşik uygulamasının tekrarlanabilirliği |
| Uzman klinik sınıfı vs uzman mm'sine tablo uygulanmış sınıf | Klinisyenin tabloyu ne kadar izlediği (yorum için) |

**Aşama 4 için:** `run_expert_analysis.py`'ye `expert_table_class` türetimi ve yukarıdaki 2. ve 3. karşılaştırmalar eklenir; birincil analiz değişmez.

Sınıf etiketleri E1–E4 olarak kalır (klinik ekibin "Sınıf 1/2/3" ifadesi E4'ün veri setinde bulunmamasından; uzman E4 verebilir).

## 3. Birincil küme — teyit

Klinik ekip "yalnızca eğitimde kullanılmamış bağımsız test görüntüleri" diyor. 5 katlı OOF tahminlerde her görüntü, onu eğitimde görmemiş bir modelle tahmin edildiğinden bu tanımı sağlar; birincil küme = 145 OOF, ikincil = sabit test alt kümesi (Aşama 4 ön-belirleme notu geçerli). Makale metninde "each image was predicted by a model that had not seen it during training (5-fold cross-validation at the participant level)" ifadesi kullanılır.

## 4. İstatistikçinin ek cevapları

- **Referans gözlemci:** model karşılaştırmasında referans **tek klinik gözlemci** (mevcut ImageJ ölçümleri; gözlemci içi ICC 0.995). Uzman ortalaması ikincil. Uzmanlar arası uyum ICC(2,1) ile raporlanır; ICC(2,k) ek bilgi.
- **Sıralama etkisi:** uzmanın önce ölçüp sonra sınıflaması bağımsızlığı ihlal etmez; ek duyarlılık analizi gerekmez. Kappa "ölçüme dayalı kategorik sınıflandırmalar arasındaki uyum" olarak yorumlanır. (Bölüm 2'deki üçlü karşılaştırma bu yorumu destekler.)

## 5. Alt sınır (E0) — nihai klinik metin

Ayrı E0 bandı yok. Kural motorundaki `NO_VISIBLE_GINGIVA` (0 mm) ve `> 0 → E1` davranışı geçerli. Makale terminolojisi: "gummy smile / aşırı dişeti görünürlüğü" yerine **"high smile line / yüksek gülme hattı"**; tablo "olası etiyolojik durumlar ve tedavi alternatifleri", tanı veya endikasyon değil. Literatürde aşırı görünürlük için 1 mm alt eşiği önerilir (Bhola 2015); bu kural motoruna **eklenmez**, tartışmada anılır. Klinik ekibin kaynaklı metni (6 referans) tartışma/sınırlılık bölümüne girecek; `docs/` altında ham hâli mevcut.

## 6. Pigmentasyon / fenotip

Kayıtlı değil; alt grup analizi yapılamaz. Sınırlılık: "gingival and skin pigmentation were not recorded; the effect of phenotype on segmentation performance could not be assessed."

## 7. Kalibrasyon — nihai ifade

Görüntü bazlı; her fotoğraftaki periodontal prob üzerinde ardışık iki 1 mm işareti arası Straight Line ile seçilip Set Scale (known distance 1 mm); ölçek değerleri kaydedilmemiş; prob "Hu-Friedy UNC periodontal probe". Uzman formlarındaki 5 mm aralıkla belirlenen ölçekler, görüntü bazlı ölçek dağılımı ve 1 mm kalibrasyonun hassasiyeti için kullanılacak (Aşama 4, madde 1).
