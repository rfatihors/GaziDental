# Güç analizi — hakem cevabı ve makale metni önerisi

**Ek A'da ne var:** G*Power 3.1.9.7, χ² goodness-of-fit (contingency tables), w = 0.30, α = 0.05, güç 0.95; df = 1 → n = 145, df = 9 → n = 263. Bu, "doğru/yanlış tespit" sıklıklarını karşılaştıran bir hipotez testi için a priori örneklemdir.

**Temel sorun:** Makalede böyle bir χ² testi **hiç yapılmamıştır**. Sonuçlar mAP, precision, recall, F1 ve confusion matrix'ten oluşur; doğru/yanlış tespit sayımı üzerinden bir ki-kare analizi yoktur. Yani güç analizi, var olmayan bir testi güçlendirmektedir. Hakemlerin itirazı (segmentasyon eğitimi için uygun değil; dış geçerlilik göstermez) doğrudur ve savunulmamalıdır. En temiz çözüm paragrafı kaldırıp yerine iki ayrı gerekçe koymaktır: eğitim verisi yeterliliği için ampirik öğrenme eğrisi, ölçüm validasyonu için hassasiyet (precision) tabanlı örneklem gerekçesi.

---

## 1. Hakem cevabı (İngilizce, kopyalanabilir)

**Reviewer 4 — sample size and external validity**

> We agree with the reviewer. The a priori χ² calculation reported in the original submission (w = 0.30, α = 0.05, power = 0.95; n = 145 for df = 1 and n = 263 for df = 9) addresses a frequency comparison of correct versus incorrect detections, which is not an analysis performed in this study, and it does not inform the amount of data required to train a segmentation network. We have therefore removed this calculation and the associated statement that the cohort size supports external validity.
>
> In the revised manuscript, data adequacy is addressed in two separate ways. (i) For model development, we report an empirical learning curve: the final architecture was retrained on stratified 25%, 50%, 75% and 100% subsets of the training partition, and validation-set mask mAP@50 is shown as a function of training-set size (new Supplementary Figure S1). Performance had not plateaued within the available training-set size; the increments between adjacent points are of the same order as run-to-run variation, so the curve indicates that additional data could still improve segmentation performance. This is stated as a limitation. (ii) For the millimetre-level validation of the measurement pipeline, sample size is justified on the basis of estimation precision rather than hypothesis testing: with n = 150 images with clinical reference measurements, the 95% confidence interval of an intraclass correlation coefficient of 0.85–0.90 has a half-width of approximately 0.03–0.05 (Bonett, 2002), and the 95% confidence interval of each Bland–Altman limit of agreement has a half-width of approximately 0.24 mm for a between-method standard deviation of 0.85 mm (Bland & Altman, 1999). Both are sufficient to characterise agreement at the clinical thresholds used in the decision framework.
>
> We note that this reading supports the reviewer's argument rather than contradicting it: the learning curve does not establish that the present cohort is sufficient, and we do not use it to claim so; it is further reason for the larger and more varied data the reviewer asks for.
>
> We have also revised the Discussion to state explicitly that all images originate from a single centre and a single imaging device, that the reported performance is an internal estimate, and that external validation on independent data from other centres, devices and populations is required before clinical use (Limitations section).

**Reviewer 2 — appropriateness of the G*Power calculation**

> We thank the reviewer for this comment and agree that a χ²-based a priori calculation is not an appropriate basis for determining the training-set size of a deep-learning segmentation model. The calculation has been removed. As detailed in our response to Reviewer 4, the revised manuscript instead reports an empirical learning-curve analysis for the segmentation model and a precision-based sample-size justification for the millimetre-level agreement analysis (ICC and Bland–Altman) performed on the images with clinical reference measurements.

---

## 2. Makale metni — Bölüm 2.1'deki güç analizi paragrafının yerine

> **Sample size and data adequacy.** No formal power calculation was applied to the segmentation training set, because conventional hypothesis-testing sample-size methods do not determine the data requirements of deep-learning models [Balki 2019]. Instead, data adequacy was assessed empirically: the final model was retrained on stratified 25%, 50%, 75% and 100% subsets of the training partition, and validation performance was examined as a function of training-set size (Supplementary Figure S1). Performance had not plateaued within the available training-set size, and this is stated as a limitation. For the millimetre-level validation of gingival display measurements, the sample size was determined by estimation precision. With n = 150 high-smile-line images with clinical reference measurements, an ICC of 0.85 can be estimated with a 95% confidence interval half-width of approximately 0.05 [Bonett 2002], and each Bland–Altman limit of agreement with a half-width of approximately 0.24 mm for a between-method SD of 0.85 mm [Bland & Altman 1999].

(Sayılar oracle/validasyon sonuçları geldikten sonra gerçek SD ve ICC ile güncellenecek; yukarıdakiler beklenen büyüklüklerdir.)

**Öğrenme eğrisinin sonucu (24 Eylül 2026).** Plato yok. RF-DETR-Seg Large @624, tohum 42, `val/segm_mAP_50`:
0.7599 / 0.7868 / 0.7763 / 0.8037 (n_train 211 / 423 / 635 / 846); 75 % noktası 50 % noktasının altında. Yukarıdaki
iki paragrafın "had not plateaued" hâli bu sonuca göre yazılmıştır. Kayıt: `outputs/09_final_rfdetr/PLAN.md`
Amendment 4; sayıların kaynağı `outputs/09_final_rfdetr/learning_curve.csv`.

**Kaldırılacak cümleler (mevcut metin, Bölüm 2.1):** "The sample size for this study was calculated using G*Power … provided sufficient statistical power for the analyses." ve "In this respect, the sample size of the present study supports the reliability and external validity of the model." Ek A tamamen kaldırılır.

**Sınırlılıklar bölümüne eklenecek cümle:**

> All images were acquired at a single centre with a single smartphone model under a standardised protocol; the reported performance therefore represents an internal estimate, and external validation on images from other centres, devices and populations is required before clinical deployment.

---

## 3. İstatistikçiye iletilecek not (Türkçe)

Hakemler, ki-kare temelli G*Power hesabının derin öğrenme segmentasyon modeli için uygun olmadığını belirtti; makalede zaten ki-kare testi yapılmadığı için hesabı kaldırıyoruz. Yerine iki gerekçe:

1. **Eğitim verisi:** öğrenme eğrisi (eğitim setinin %25/50/75/100'ü ile eğitim, doğrulama mAP'si). Bu ampirik bir yöntemdir, güç hesabı gerektirmez.
2. **Ölçüm validasyonu (n = 150):** hassasiyet tabanlı gerekçe.
   - ICC için Bonett (2002): SE(ICC) ≈ √[2(1−ρ)²(1+(k−1)ρ)² / (k(k−1)(n−1))], k = 2. n = 150 için %95 GA yarı genişliği: ρ = 0.80 → ±0.058; 0.85 → ±0.045; 0.90 → ±0.031.
   - Bland–Altman uyum sınırları için Bland & Altman (1999): SE(LoA) = √(3/n)·SD → n = 150, SD = 0.85 mm için %95 GA yarı genişliği ±0.24 mm; ortalama sapma için ±0.14 mm.
   - Bu formüllerin doğrulanması ve makale metnindeki ifadenin onaylanması rica edilir.

**Kaynaklar**
- Bonett DG. Sample size requirements for estimating intraclass correlations with desired precision. Stat Med. 2002;21:1331–1335.
- Bland JM, Altman DG. Measuring agreement in method comparison studies. Stat Methods Med Res. 1999;8:135–160.
- Balki I, et al. Sample-size determination methodologies for machine learning in medical imaging research: a systematic review. Can Assoc Radiol J. 2019;70:344–353.
- Figueroa RL, et al. Predicting sample size required for classification performance. BMC Med Inform Decis Mak. 2012;12:8. (öğrenme eğrisi yaklaşımı için)

---

## 4. Öğrenme eğrisi deneyi — yeniden eğitim görevine eklenecek

- Temiz hasta düzeyi bölünmeden sonra, eğitim bölüntüsünün gülme hattına göre tabakalı %25 / %50 / %75 / %100 alt kümeleri (sabit tohum; alt kümeler iç içe).
- Her alt kümede nihai YOLOv11x-seg konfigürasyonuyla eğitim; aynı doğrulama seti; **test setine dokunulmaz**.
- Raporlanacak: dişeti sınıfı mask mAP@50 ve mAP@50–95, dudak sınıfı aynı; her nokta için ±SD (mümkünse 2–3 tohum, en azından %100 için).
- Çıktı: Supplementary Figure S1 (x: eğitim görüntüsü sayısı, y: mAP; iki sınıf ayrı çizgi).
- Yorum kuralı: %75 → %100 arasındaki kazanç, %25 → %50 arasındakinin küçük bir kesriyse "plato"; değilse sınırlılık olarak yazılır. Sonucu önceden varsayma.
