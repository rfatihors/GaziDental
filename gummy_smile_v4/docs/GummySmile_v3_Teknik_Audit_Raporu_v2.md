# GummySmile v3 — Teknik Audit Raporu (v2 — kod ve veri üzerinde doğrulanmış)

**Kapsam:** `rfatihors/GaziDental` deposu, `gummy_smile_v3` modülü (5 Eylül 2026 itibarıyla `main`)
**Amaç:** Hakem revizyonlarına cevap verilmeden önce ölçüm hattının, veri setinin ve karar mekanizmasının teknik doğruluğunun denetlenmesi
**Durum:** Salt okunur inceleme. Hiçbir kod veya veri değiştirilmemiştir.
**Sürüm:** v2.1 — klinik ekibin 5 Eylül cevapları işlendi (Bölüm A).

---

## A. Klinik ekipten gelen cevaplar ve etkileri (5 Eylül)

| Konu | Cevap | Etkisi |
|---|---|---|
| `-` işareti | "Görünen dişeti yok; dudak gingival marjinin üstünde, hatta dişi kapatıyor olabilir." | **0 mm** yorumu onaylandı. 290 hücre analize 0 olarak girer; ayrı bayrakla izlenir. |
| ImageJ hangi görüntüde | Telefon fotoğrafları PC'de **2698×1799**'a küçültülüp ImageJ'de ölçüldü (ekran görüntüsüyle). | COCO dışa aktarımıyla **aynı koordinat çerçevesi** — GT maske pikselleri ile ImageJ mm'leri doğrudan karşılaştırılabilir. 2698×1799 dışındaki azınlık görüntülerde çerçeve doğrulanmalı. |
| Kalibrasyon | Her görüntü için ayrı; probdaki **ardışık iki 1 mm işareti** Straight Line ile seçilip Set Scale = 1 mm. Değerler kaydedilmemiş. Her ölçülen fotoğrafta prob var. Farklı prob tipleri olabilir, hepsi 1 mm aralıklı; yayında "Hu-Friedy UNC". | Tek 1 mm aralık ≈ 17 px; 1 px işaretleme hatası ≈ %6 ölçek hatası. Bölüm 6'daki görüntü bazlı oran dağılımının (CV 0.38) önemli kısmı bu kalibrasyon gürültüsüdür. Prob her fotoğrafta olduğundan **daha uzun aralıkla (≥ 5 mm) bağımsız yeniden kalibrasyon** yapılabilir ve hakeme "kalibrasyon hassasiyeti" olarak raporlanabilir. |
| Tek görüntü / hasta | "Her hastadan tek görüntü elde edildi." | **Veriyle çelişiyor** — bkz. 5.2 (revize). Aynı hastanın problu ve probsuz iki çekimi veri setinde birlikte bulunuyor. Muhtemelen ölçüm için tek fotoğraf kullanıldı, ancak Roboflow'a her iki çekim de yüklendi. |
| Aynı isimli dosyalar | Farklı telefonlar; hepsi farklı hastalar. `IMG_2544` ile `IMG_2544.` farklı görseller. | Doğrulandı. Roboflow sondaki noktayı `-`'ye çevirmiş: Excel `IMG_2544.` ↔ COCO `IMG_2544-_jpeg`. Ad normalizasyonu bu eki **korumalı**, birleştirmemeli. Excel'deki 11 "yinelenen" grubun bir kısmı (ör. `ımg_2544`/`IMG_2544.`) aslında **iki farklı görüntüdür**, yinelenen değildir. |
| Alt sınır (E0) | Ayrı bir "endikasyon yok" bandı yok. Dişeti **> 0 mm görünüyorsa** hasta beklentisine göre gingivektomi "olası tedavi alternatifi". Sistem yalnızca **yüksek gülme hattı** (dişeti görünen) vakalara uygulanmalı. Zeynep hoca onayı bekleniyor. | Kural motoru: 0 mm → `NO_VISIBLE_GINGIVA` (sınıf yok, uygulanamaz); > 0 → E1…E4. Tedaviler "olası alternatif" olarak etiketlenir. E1 için `min_mm` = 0 (hariç). |
| Hasta–görüntü listesi | Excel hazırlanmış, ek olarak gönderilecek; yaş/cinsiyet kısmen eksik. | Henüz elimizde yok. Geldiğinde 5.2'deki çiftlerle çapraz kontrol edilecek. |
| Gözlemci içi | 20 görüntü, ayrı Excel gönderilecek. | Depodaki 15 görüntülük dosyalar eski; yeni dosya beklenmeli. |
| J Dent 2026 örtüşmesi | Henüz cevap yok. | 5.4'teki isim eşleşmesi (≈ 150 yüksek gülme hattı) hasta listesiyle doğrulanacak. |

## B. Klinik ekipten gelen ikinci cevap ve dosyalar (6 Eylül)

**Gelen dosyalar:** `Hasta_ID-_Ölçümler.xlsx` (3 sayfa: yüksek 692 satır ölçüm + yaş/cinsiyet; düşük 100 ve normal 246 satır yaş/cinsiyet), `calibration.xlsx` (20 görüntü, iki oturum), `Bakıldı-ayni_hasta_aday.csv` (CSV olarak kaydedildiğinden **sarı/mor işaretler kaybolmuş**; xlsx istendi).

| Konu | Cevap | Etkisi |
|---|---|---|
| Aynı hasta çiftleri | Görsellere bakıldı, aynı olanlar sarı işaretlendi (bize ulaşmadı). **Ölçümü olmayan yüksek gülme hattı görüntüleri (mor) veri setinden çıkarılsın** — muhtemelen probsuz oldukları için ölçülmemiş. | Ölçümsüz `high` görüntüleri: **66**. Çıkarılınca 55 çiftin 37'si kendiliğinden çözülür; **18 çift** kalır (14 normal/low, 4 high). Öneri: her çiftten bir görüntü çıkarılıp veri seti "her hastadan tek görüntü" hâline getirilir. High'daki 4 çiftte iki görüntünün de ölçümü var (`IMG_3684-`/`IMG_3682`, `IMG_3858-`/`IMG_3854-`, `IMG_2456`/`IMG_24555`, `IMG_2544-`/`IMG_2550`); hangi ölçümün geçerli olduğu soruldu. Yüksek grup 216 → 150'ye iner (eğitim verisi maliyeti hocaya bildirildi; alternatif: 66'yı yalnızca train'de tutmak). |
| Hakem cümlesi | "(acquired with and without the periodontal probe…)" ifadesi çıkarılsın; probsuz çekim yapılmaması gerekiyordu. | Cümle revize edildi: "…identified 55 participants for whom two photographs from the same session had been included. One photograph per participant was retained, the dataset was re-partitioned so that each participant appears in only one subset, and the final model was retrained…" |
| Hasta ID | Ayrı bir hasta-ID listesi yok; dosya, ölçüm tablosu + yaş/cinsiyet listeleridir. "Her hastadan tek ölçüm fotoğrafı." | Temizlik sonrası **görüntü = hasta**; `patient_id` = dosya adı. Yaş/cinsiyet kapsamı mevcut veri setinde: high 108/150 yaş, 104/150 cinsiyet; low 99/303; normal 181/796. Reviewer 3'e mevcut kapsamla raporlanacak. |
| Önekli adlar | `25-IMG_4552` gibi önekler **hastanın yaşıdır**; `IMG_4552` ile `25-IMG_4552` farklı görüntülerdir. | 154 önekli satırın 149'unda önek = YAŞ sütunu. Bu satırların yalnızca 10'u COCO ile taban ad üzerinden çakışır (8'i `normal`); yani önekli görüntüler esasen bu veri setinde yok. Eşleştirmede önek korunur, yaş olarak da çıkarılır. |
| Dot ↔ dash | (v2.1'deki varsayım kısmen düzeltildi) | COCO `high`'daki 87 `-` adının 8'i Excel'de noktalı, 30'u noktasız, 49'unun satırı yok. Kademeli eşleştirme kuralı tanımlandı (tam → taban geri düşme → belirsiz). Beklenen oracle seti ≈ 145. |
| Farklı boyutlu görüntüler | Muhtemelen orijinalde ölçüldü, emin değil. | Sorun değil: kalibrasyon her görüntüde kendi içinde yapıldığından mm değerleri çerçeveden bağımsız; yalnızca global ölçek varsayımı etkilenir → prob kalibrasyonu ile çözülür. |
| Gözlemci içi (20 görüntü) | `calibration.xlsx` | Hesaplandı: diş düzeyi n = 120, **ICC(2,1) = 0.995**, ICC(3,1) = 0.995; görüntü ortalaması n = 20, **ICC = 0.998**; ortalama fark −0.01 mm, SD 0.17 mm, %95 uyum sınırı ±0.33 mm; eşleşmiş t p = 0.56. Makaledeki "20 randomly selected images" ifadesi artık doğru. **SD 0.17 mm, referansın gürültü tabanıdır**: GT-maske MAE'sinin (0.63) bunun çok üstünde olması, farkın gözlemci gürültüsünden değil tahminci/ölçek/hizalamadan geldiğini gösterir. |

**6 Eylül akşamı gelen ek cevap:** morlar hariç 55 çiftin hepsi aynı hasta → her çiftten yaş/cinsiyet kaydı olan tutulacak (4 high çifti dâhil). **Power endişesi:** yüksek grup 150'ye inerse örneklem hesabının altında kalınacağı düşünülüyor (hoca "200" hatırlıyor; makalede 145 ve 263 yazıyor). Önerilen çözüm: 66 ölçümsüz görüntü **yalnızca train'de** tutulur (ikiziyle aynı hasta grubunda; val/test'e girmez) → segmentasyon örneklemi ≈ 1.295 / 216 high olarak kalır, mevcut power hesabı geçerli; milimetrik doğrulama ayrı bir uyum analizidir ve n = 150, ICC ≥ 0.85 için 0.10 GA genişliği hedefiyle gereken ≈ 120'nin üzerindedir (Bonett 2002). Tahmin maskeleri üzerindeki doğrulama, sabit test setinde ~20 ölçümlü görüntü kalacağından, ölçümlü 150 görüntüde hasta-gruplu 5-kat CV ile out-of-fold tahminlerle yapılmalıdır. Yaş/cinsiyet kayıtları sonradan tutulmaya başlandığı için eksik; makalede "kayıtlı olan n üzerinden" ifadesiyle raporlanacak.

**Bekleyenler:** train-only önerisine hocanın onayı, Zeynep hocanın E0 onayı, hakemin power yorumu ve istatistikçi görüşü, J Dent 2026 örtüşme sayısı (isim eşleşmesi ≈ 150 high; kesinleştirilecek).

### v2'de ne değişti?

v1 raporu kısmen belge okumasına dayanıyordu. v2'de depo klonlanmış, tüm iddialar doğrudan kod ve veri dosyaları üzerinde sınanmıştır. Sonuç:

| v1 bulgusu | v2 durumu |
|---|---|
| 1.1 Geometri yanlış | **Doğrulandı**, GT maske görselleriyle netleşti |
| 1.2 Sınıf ayrımı yok, en büyük kontur = dudak | **Doğrulandı** |
| 1.3 `imgsz` 1024 | Doğrulandı; ayrıca **maske çözünürlüğü sorunu** eklendi (yeni 1.3) |
| 1.5 Ölçüm yokken E1/T1 | **Doğrulandı** |
| 2.1 Metaveriyle belirsizlik çözme | Doğrulandı; ek olarak **E1–E2 çakışması kodda hiç yok** (yeni 2.3) |
| 3. Figure 6 sayısal açıklama | **Zayıflatıldı**: dosya adı çakışmaları nedeniyle manuel değerle birebir eşleme güvenilir değil; mekanik açıklama geçerli. v1 değerleri de güvenilir değil |
| 4. Excel değerleri mm×1000; 387 tam satır | **Kısmen yanlıştı**: Excel'in kendi notları dört farklı kodlama kuralı tanımlıyor; doğru okunduğunda **692/693 satır tam** (yeni Bölüm 4) |
| 5.1 Altı görüntü birden fazla bölüntüde | **Yanlıştı**: aynı isimli dosyalar piksel düzeyinde farklı fotoğraflardır (iPhone `IMG_xxxx` çakışması) |
| 5.2 On iki görüntü iki gülme hattı sınıfında | **Yanlıştı**: aynı sebeple |
| 5.3 COCO'da görüntü başına 1 instance, Figure 4 ile uyumsuz | **Yanlıştı**: yalnızca `high` grubu için doğru; toplam 3.938 dişeti / 1.318 dudak instance (≈3:1), Figure 4 ile **uyumlu** |
| 5.4 693 ölçüm vs 216 yüksek gülme hattı | **Netleşti**: Excel önceki (J Dent 2026) çalışmanın ölçüm setidir; mevcut veri setiyle isim eşleşmesi 179 görüntü |
| — | **Yeni**: gerçek sızıntı riski = aynı hastanın ardışık fotoğrafları (en az 9 çapraz-bölüntü çift) |
| — | **Yeni**: `max_det: 5` normal/low görüntülerde dişeti parçalarını kesiyor |
| — | **Yeni**: iki paralel giriş noktası ve iki farklı kural motoru |
| — | **Yeni**: GT maskeler üzerinde referans doğrulama ön sonucu (MAE ≈ 0.63 mm) |

---

## 0. Yönetici özeti

Denetim, hakemlerin işaret ettiği sorunların büyük kısmının **metodolojik değil, yazılımsal** olduğunu doğrulamıştır. İki kritik bulgu değişmemiştir:

> **(1) Mevcut ölçüm hattı dişeti görünürlüğünü ölçmemektedir.** Dişeti ve dudak maskeleri tek bir ikili maskede birleştirilmekte, en büyük kontur seçilmekte (pratikte dudak) ve o konturun *üst kenarının bölgeler arası dalgalanması* hesaplanmaktadır. Gingival margin hiçbir yerde kullanılmamaktadır.

> **(2) Elde bulunan E1–E4 / T1–T4 etiketleri bağımsız klinik değerlendirme değildir.** 3.868 etiketli hücrenin 3.865'i literatür tablosunun milimetrik değere birebir uygulanmasıyla açıklanmaktadır. Bu etiketlerle model çıktısını karşılaştırmak döngüsel bir analizdir.

v2'nin en önemli **olumlu** bulgusu: ground-truth dişeti maskeleri üzerinden doğru geometriyle yapılan ölçüm, mevcut ImageJ değerleriyle **tek bir global ölçek kullanılarak** MAE ≈ 0.63 mm, RMSE ≈ 0.85 mm düzeyinde uyuşmaktadır (n = 148, ön analiz, Bölüm 6). Bu, J Dent 2026 makalesinde raporlanan aralıkla (MAE 0.48–0.64) aynı büyüklük sırasındadır ve **yeni veri toplanmadan** milimetrik validasyonun yapılabileceğini göstermektedir.

v2'nin en önemli **düzeltmesi**: v1'deki veri seti sızıntısı ve instance sayısı bulguları hatalıydı. Gerçek sızıntı riski dosya adı çakışması değil, **aynı hastaya ait ardışık fotoğrafların farklı bölüntülere düşmesidir**; bunun çözümü hasta kimliği eşlemesi gerektirir.

---

## 1. Ölçüm hattındaki hatalar

### 1.1 Geometri tanımı yanlış (kritik — doğrulandı)

`measurement/yolo_measurements.py` ve `measurement/measure_gum_visibility.py` birebir aynı mantığı paylaşır:

```python
contour    = max(contours, key=cv2.contourArea)     # en büyük kontur
lip_line_y = float(contour[:, :, 1].min())          # o konturun kendi en üst noktası
...
zenith_y   = bölgedeki kontur noktalarının en üst y'si
gummy_px   = float(zenith_y - lip_line_y)
```

Hesaplanan büyüklük `(bölgedeki üst sınır) − (global üst sınır)`; yani maskenin **üst** kenarının bölgeler arası dikey sapması. Klinik dişeti görünürlüğü ise üst dudağın **alt** kenarı ile gingival margin (dişeti maskesinin **alt** kenarı) arasındaki mesafedir. Alt sınır hesaba hiç girmemektedir.

GT anotasyonları görselleştirildiğinde (`high/train`, iki örnek) beklenen anatomi net görülür: `dudak` sınıfı yalnızca **üst dudak vermilyonunu** kapsayan ince bir bant; `diseti` sınıfı üstte dudak alt kenarına dayanan, altta **festonlu** (papillalar aşağı sarkan, diş zenitlerinde incelen) bir banttır. Ölçmek istediğimiz şey tam olarak bu kırmızı bandın diş zenitlerindeki dikey kalınlığıdır.

Pratik sonuç: en büyük kontur dudak olduğunda (neredeyse her görüntüde) ölçülen şey **üst dudağın üst kenarının (Cupid yayının) eğriliğidir** — dişeti görünürlüğüyle ilgisi yoktur.

### 1.2 Sınıf ayrımı yapılmıyor (kritik — doğrulandı)

Her iki çıkarım dosyasında (`yolo/infer_yolo.py`, `yolo/infer_yolo_seg.py`):

```python
mask = result.masks.data.max(dim=0).values.cpu().numpy()   # tüm sınıflar tek maske
```

Sınıf kimlikleri (`result.boxes.cls`) hiç okunmamaktadır. Ölçümde `max(contours, key=cv2.contourArea)` ile en büyük kontur seçilir; dudak alanı dişetinden belirgin biçimde büyük olduğundan seçilen kontur dudaktır. Ayrıca dişeti maskesinin parçalandığı görüntülerde (normal/low gülme hattında dişeti **birden fazla papilla parçası** olarak etiketlenmiştir, bkz. 5.3) diğer parçalar sessizce atılır.

### 1.3 Çıkarım çözünürlüğü ve maske koordinat çerçevesi (yeni ayrıntı)

- `configs/config.yaml`: `yolo.imgsz: 1024`; model 640×640 ile eğitilmiştir.
- Daha önemlisi: Ultralytics'te `result.masks.data` varsayılan olarak **letterbox'lanmış çıkarım çözünürlüğündedir**, orijinal görüntü çözünürlüğünde değildir (`retina_masks=True` verilmediği sürece). Kod, bu maskeyi doğrudan orijinal görüntünün üzerine bindirmeye çalışır (`color_mask[mask > 0] = ...`); boyutlar uyuşmadığında bu satır hata verir, uyuştuğu (ör. kare görüntü) nadir durumlarda ise piksel ölçümleri orijinal değil çıkarım çerçevesinde olur.
- Kalibrasyon (`px_per_mm`) hangi çerçevede belirlendiyse ölçüm de **aynı çerçevede** yapılmalıdır. COCO dışa aktarımındaki görüntüler ~2698×1799 (orijinal ölçüye yakın, Roboflow yeniden boyutlandırması **yok**); ImageJ ölçümlerinin de bu dosyalar üzerinde yapıldığı varsayılmaktadır (doğrulanmalı, bkz. Bölüm 9).

### 1.4 Birim karışıklığı (doğrulandı)

`config.yaml`'de `px_per_mm: null`; `infer_yolo_seg.predict_and_measure` içinde `mm_per_pixel: float = 1.0` varsayılanı. Kalibrasyon verilmediğinde `mean_mm` kolonu **piksel** taşır. `px_per_mm` ve `mm_per_pixel` (birbirinin tersi) kod tabanında birlikte kullanılmaktadır.

### 1.5 Ölçüm başarısızlığında güvensiz varsayılan (doğrulandı)

`methods/v3/rule_engine.py`, ölçüm `None`/`NaN` iken `E1/T1` (gingivektomi) döndürür. Ayrıca fonksiyonun sonundaki "hiçbir kural eşleşmedi" dalı da E1'e düşer.

### 1.6 `max_det: 5` (yeni)

Normal ve low gülme hattı görüntülerinde dişeti **görüntü başına ortalama 3–4, en fazla 7–8 ayrı instance** olarak etiketlenmiştir (papillalar). `max_det: 5` bu görüntülerde tespitleri keser. Değer en az 20'ye çıkarılmalıdır.

### 1.7 İki paralel giriş noktası, iki farklı kural motoru (yeni)

| Giriş noktası | Ölçüm | Kural motoru | Çakışma davranışı |
|---|---|---|---|
| `master_pipeline_v3.py` | `measure_gum_visibility` | `methods/v3/rule_engine.assign_etiology` | E2–E3 için aday listesi + varsayılan E2 |
| `segmentation_model.py` → `treatment_predictor.py` | `infer_yolo_seg.predict_and_measure` | `methods/v3/diagnosis._match_rule` | Öncelik sırasıyla **tek sınıf**: E1 (1) > **E3 (2)** > E2 (3) > E4 (4) |

Aynı 5 mm değeri bir motorda "E2 (aday: E2, E3)", diğerinde "E3" üretir. Hangi giriş noktasının Figure 6'yı ürettiği izlenememektedir (`results/` boş). Tek bir kural motoru kalmalı, diğeri ona delege etmelidir.

---

## 2. Karar mekanizmasındaki sorunlar

### 2.1 Belirsizlik, klinik olarak ilgisiz bir değişkenle çözülüyor (doğrulandı)

```yaml
ambiguous_policy:
  use_metadata: true
  metadata_map: {high: E3, normal: E2, low: E2}
```

E2–E3 bandında karar gülme hattı sınıfına göre verilir; gülme hattı ile etiyoloji (dudak hiperaktivitesi vs. dentoalveolar ekstrüzyon) arasında nedensellik yoktur. Metaveri varken `ambiguous=False` yapılıp tek sınıfa indirgenir — makalenin ve Reviewer 4'ün olumlu bulduğu "çakışan aralıkta tek sınıfa zorlamama" iddiasıyla çelişir.

### 2.2 Eşik tablosunun alt sınırı yok (doğrulandı)

`RULES["E1"]["min_mm"] = None`: 0 mm görünürlük de E1/T1 (gingivektomi) ile eşleşir.

**Klinik karar alındı (Bölüm A):** ayrı bir E0 bandı tanımlanmayacak; dişeti görünmüyorsa (0 mm) sistem uygulanamaz, görünüyorsa (> 0 mm) E1'den itibaren sınıflar geçerlidir ve tedaviler "olası tedavi alternatifi" olarak sunulur; sistem yüksek gülme hattı vakalarıyla sınırlıdır. Kodda: `value <= 0` → `NO_VISIBLE_GINGIVA` (sınıf ve tedavi yok), E1 alt sınırı 0 (hariç). Zeynep hocanın onayı bekleniyor; makale metnindeki ifade ona göre yazılacak.

### 2.3 E1–E2 çakışması (3–4 mm) kodda hiç yok (yeni — makale iddiasıyla doğrudan çelişir)

`assign_etiology` sırayla `E1`'i (≤ 4 mm) kontrol edip **hemen döner**:

```python
if _matches(gum_visibility_value, RULES["E1"]):
    return _build_result("E1", False, notes)     # 3.5 mm → E1, ambiguous=False
```

Oysa Tablo 1'e göre 3–4 mm hem E1 (< 4) hem E2 (3–6) aralığındadır. Klinisyenlerin kendi etiketleri de bunu yansıtır: Excel'de **793 hücre "E1-E2"**, 829 hücre "E2-E3". Kod yalnızca E2–E3 çakışmasını ele alır. Makaledeki "measurements falling within overlapping threshold ranges were assigned to all matching categories" cümlesi mevcut kodla **kısmen yanlıştır**.

### 2.4 Sınır değerlerinde dahil/hariç tutarsızlığı (yeni)

`rule_engine.py`: E1 `≤ 4.0` (tabloda `< 4`), E4 `≥ 8.0` (tabloda `> 8`); 4.0 mm → E1, 8.0 mm → E4. `diagnosis.py` ise E1 için `< 4` hariç, E4 için `> 8` hariç kullanır. Sınırlar tek yerde, açıkça tanımlanmalıdır.

---

## 3. Figure 6'nın açıklaması (revize)

**Mekanik açıklama (kesin):** v3 kolonundaki değerler, Bölüm 1.1–1.2'deki hata nedeniyle dişeti görünürlüğü değil, dudak konturunun üst kenar eğriliğidir. Kalibrasyon verilmediyse (`px_per_mm: null`, `mm_per_pixel=1.0`) bu değer **piksel** olarak "mm" kolonuna yazılmıştır. 8–9 "mm" büyüklüğü, 1024-ölçekli bir dudak maskesinde altı bölgenin ortalama üst-kenar sapmasıyla tutarlıdır.

**v1 (XGBoost) değerleri de güvenilir değildir (yeni):** `methods/v1` regresörü, önceki çalışmada **512×512 DeepLabV3+ dişeti-yalnız maskelerinin** sütun bazlı piksel sayılarıyla eğitilmiştir. v3 pipeline'ında ise aynı regresöre **dudak+dişeti birleşik maske, 1024 ölçeğinde** verilmektedir (`run_xgboost(mask_path=yolo_result["mask_path"])`). Özellikler eğitim dağılımının tamamen dışındadır; ağaç tabanlı bir regresör bu durumda uç yapraklara doygunlaşır. Figure 6'daki 4.36 / 4.91 değerlerinin manuel ölçüme yakın görünmesi tesadüf olarak değerlendirilmelidir.

**Sayısal eşleme düzeltildi:** v1 raporunda Figure 6'daki `IMG_2544` için manuel değer 2.63 mm alınmıştı. Oysa mevcut veri setindeki tek `IMG_2544`, COCO'da `IMG_2544-` (= Excel `IMG_2544.`) olan görüntüdür; manuel ortalaması **6.94 mm**'dir. Buna göre v3 = 8.13 (yanlış geometri), v1 = 4.36 (dağılım dışı girdi) — **ikisi de** manuel değerden uzaktır; v1'in "manuele yakın" görünmesi iddiası geçerli değildir. `IMG_2633` ise COCO'da iki farklı fotoğraf (`IMG_2633` ve `IMG_2633-`) olarak bulunduğundan hangisinin Figure 6'daki olduğu belirsizdir. Rebuttal'da Figure 6, "tespit edilmiş ve düzeltilmiş bir yazılım hatasının çıktısı" olarak açıklanmalı; yerine düzeltilmiş ölçüm hattıyla üretilmiş, manuel ölçümle yan yana konmuş yeni bir figür sunulmalıdır.

---

## 4. Manuel ölçüm dosyasının doğru okunması (yeni — v1'deki varsayımları düzeltir)

`data/manual_measurements/ölçümler ai guncel.xlsx` (693 satır, 6 bölgesel ölçüm + 6 E + 6 T etiketi + yaş/cinsiyet). Dosyanın kendi içinde iki gözlemci notu vardır:

> **Not 1:** "1 mm'den küçük olan sayıların başına sıfır yazdığımda sıfırı dahil etmiyor. o yüzden başına `*` işareti koydum"
> **Not 2:** "`-` işareti dişeti dişin üzerine geliyor. ölçülecek dişeti mesafesi yok demek."

Bu notlar ve hücre dağılımı birlikte değerlendirildiğinde kodlama kuralları:

| Hücre biçimi | Adet | Anlamı | Doğru okuma |
|---|---|---|---|
| Tam sayı ≥ 1000 (ör. `3207`) | 3.505 | Türkçe Excel'de `3.207` → nokta binlik ayracı sayıldı | ÷ 1000 → 3.207 mm |
| Tam sayı ≥ 10000 (ör. `13768`) | 16 | 13.768 mm (şiddetli vakalar) | ÷ 1000 |
| `*0.718` metin | 353 | < 1 mm gerçek ölçüm (Not 1) | `*` at, 0.718 mm — **NaN değil** |
| `-`, `-(mesafe yok)` | 290 | O dişte görünen dişeti yok (Not 2; klinisyen 5 Eylül'de onayladı: "dudak gingival marjinin üstünde") | **0 mm**, bayrakla |
| Tam sayı 100–999 (ör. `712`) | 5 | Muhtemelen `0.712` (sıfır düştü, `*` konmadı) | ÷ 1000, **belirsiz** bayrağı |
| `3`, `4`, `'3.00'`, `'3.44'` | 4 | 3 mm, 4 mm, 3.00 mm, 3.44 mm (E etiketleriyle tutarlı) | olduğu gibi mm |
| `-684` | 1 | anlamsız | dışarıda bırak |

**Bu kurallarla 693 satırın 692'sinde altı ölçümün tamamı okunabilmektedir** (v1'deki "387 tam satır" tahmini, `*0.xxx` ve `-` hücrelerinin NaN sayılmasından kaynaklanıyordu). 508 satırda altı değer de > 0; 185 satırda en az bir `-` (sıfır) vardır.

Okunan değerlerin dağılımı klinik olarak makuldür: medyan 2.81 mm, %95 persentil 6.37 mm, maksimum 14.1 mm.

**E etiketleri tablonun birebir uygulamasıdır (doğrulandı):** 3.868 etiketli hücreden yalnızca 3'ü, okunan mm değerine tablo uygulandığında çıkan etiketle uyuşmaz; bu 3'ün ikisi yukarıdaki 100–999 belirsizliğine aittir (`667` ve `728` değerleri "E4" etiketli — yani gözlemci bunları > 8 mm olarak düşünmüş olabilir; hücre belirsizdir). Diğer bir deyişle, **E etiketi, hücrenin nasıl okunması gerektiği için bağımsız bir doğrulama kanalıdır** ve ayrıştırıcıda tutarlılık denetimi olarak kullanılmalıdır.

**Yinelenen adlar (revize):** Sondaki nokta ayrı görüntü olduğundan (`ımg_2544` ≠ `IMG_2544.`; Bölüm A) bu tür çiftler yinelenen **değildir** ve nokta eki korunarak eşleştirilir. Geriye kalan gerçek belirsiz gruplar: `IMG_4552`/`25-IMG_4552`, `IMG_7366`×2, `IMG_1471`/`39-IMG_1471`, `IMG_8645`/`27-IMG_8645`, `IMG_8669`/`21-IMG_8669`, `IMG_1195`/`55-IMG_1195`, `IMG_1212`/`39-IMG_1212`, `30-IMG_0586`/`35-IMG_0586`, `42-IMG_0590`/`55-IMG_0590`, `30-IMG_0647`/`31-IMG_0647` (10 grup; sıra numarası önekli olanlar farklı telefonlardan farklı hastalar olabilir). Değerleri farklıdır; klinisyene sorulmalı, o zamana kadar analizden çıkarılmalıdır.

---

## 5. Veri seti bulguları (büyük ölçüde revize)

### 5.1 Dosya adı çakışmaları ≠ aynı görüntü (v1 bulgusu geri alındı)

COCO dizinlerinde normalize adı çakışan 18 grup vardır (v1'de 6 + 12 olarak raporlanan bulgular bunların alt kümesidir). **Her grup piksel düzeyinde karşılaştırıldığında hepsi farklı fotoğraftır** (32×32 gri tonlama farkı 27–60/255; boyutlar da farklı, ör. 2698×1799 vs 2593×1729). Sebep: iPhone `IMG_xxxx` numaralandırması cihazlar/oturumlar arasında tekrarlar (klinik ekip doğruladı: farklı telefonlar, farklı hastalar). Dolayısıyla:

- Bölüntüler arası "yinelenme" ve "gülme hattı tutarsızlığı" bulguları **geçerli değildir**.
- Ancak `IMG_xxxx` adı **benzersiz kimlik olarak kullanılamaz**; ölçüm/COCO eşleştirmesi ve rebuttal sayıları bunu dikkate almalıdır.
- **Sondaki nokta ayrı bir görüntüdür:** klinik ekip `IMG_2544` ile `IMG_2544.` ifadelerinin farklı fotoğraflar olduğunu belirtti. Roboflow, dosya adındaki fazladan noktayı `-`'ye çevirmiştir (`IMG_2544..jpg` → `IMG_2544-_jpeg.rf.….jpeg`); COCO'da 180 böyle ad vardır (87 high, 88 low, 5 normal). 18 çakışma grubunun 5'i tam olarak "taban ad + `-` varyantı" çiftidir (`IMG_2633`, `IMG_3864`, `IMG_3694`, `IMG_3424`, `IMG_4034`). Eşleştirmede bu ek **korunmalıdır**.

### 5.2 Gerçek sızıntı: aynı hastanın iki fotoğrafı (yeni — görsel olarak doğrulandı)

Klinik ekip "her hastadan tek görüntü" demektedir; veri seti bunu doğrulamamaktadır. İki aşamalı tarama (32×32 korelasyon ile aday, ardından ORB özellik eşleme + RANSAC homografi ile doğrulama; ≥ 50 inlier eşiği — dağılım keskin biçimde iki modludur, eşik belirsiz değildir):

| Ölçüt | Çift sayısı |
|---|---|
| Aynı sahne (≥ 50 RANSAC inlier) | **55** |
| Farklı bölüntülerde (train/valid/test) | **22** |
| **Test setini** içeren | **15** |

Dört çift görsel olarak incelendi (`ayni_hasta_ornek_ciftler.jpg`): hepsi aynı kişi, **biri periodontal problu, diğeri probsuz** çekim. Örnekler: `IMG_2923` (normal/train) ↔ `IMG_2924` (normal/valid); `IMG_3079` (high/train) ↔ `IMG_3082-` (high/valid); `IMG_3160` (high/**test**) ↔ `IMG_3163-` (high/train); `IMG_2821` (high/**test**) ↔ `IMG_2819-` (high/train). `IMG_7870` / `IMG_78701` aynı dosyanın iki kopyasıdır. Tam liste: `ayni_hasta_aday_ciftler.csv`.

Muhtemel açıklama: klinik protokolde her hastadan problu (ölçüm için) ve probsuz bir çekim yapılmış; ImageJ ölçümü tek fotoğrafta yapılmış, ancak Roboflow'a her ikisi de yüklenmiş. 55 sayısı **alt sınırdır** (aday aşaması poz farkı büyük çiftleri eleyebilir).

Sonuçlar:
- Reviewer 2 (#6) ve Reviewer 4'ün (CLAIM 2024) sızıntı itirazı **haklıdır**; test setinin en az 15 görüntüsünün eğitim setinde ikizi vardır. Rebuttal'da "her hastadan tek görüntü" yazılamaz.
- Hasta düzeyinde bölünme zorunludur. Klinikten gelecek hasta–görüntü Excel'i bu çiftleri içermiyorsa (yalnızca ölçülen fotoğraf eşlenmişse), `ayni_hasta_aday_ciftler.csv` ile birleştirilmelidir.
- Bu çiftler klinik ekibe gösterilmelidir (Bölüm 9).

### 5.3 Instance sayıları Figure 4 ile uyumludur (v1 bulgusu geri alındı)

COCO dışa aktarımı (ham, augmentasyonsuz):

| Grup | Görüntü | Dişeti instance | Dudak instance | Görüntü başına dişeti |
|---|---|---|---|---|
| high | 216 | 216 | 216 | 1.0 |
| low | 303 | 519 | 303 | 1.7 |
| normal | 796 | 3.203 | 799 | 4.0 |
| **Toplam** | **1.315** | **3.938** | **1.318** | 3.0 |

Yüksek gülme hattında dişeti tek sürekli bant; normal/low'da yalnızca papillalar görünür ve **her papilla ayrı instance** olarak etiketlenmiştir. 3:1 oranı buradan gelir.

Train bölüntüsü: 2.839 dişeti / 932 dudak. Raporda v7 için verilen 2.814 / 923 bu sayılarla örtüşür; Figure 4'teki 5.628 / 1.846 ise **tam olarak 2 × (2.814 / 923)**, yani train setinin 2× augmentasyonlu hâlidir. Reviewer 2 (#5) ve Reviewer 4'ün "1.315 → 3.403 nasıl oldu" sorusuna verilecek cevabın iskeleti budur; kesin görüntü sayıları Roboflow sürüm metaverisinden alınmalıdır.

### 5.4 Manuel ölçüm dosyası önceki çalışmanın setidir (netleşti)

Excel'deki 682 benzersiz addan yalnızca **179'u** mevcut COCO setinde vardır (150 high, 22 normal, 7 low; normal/low eşleşmeleri ad çakışması olabilir). Kalan **503 ad bu veri setinde yoktur** → dosya, J Dent 2026 çalışmasının (1.748 fotoğraf, 687 yüksek gülme hattı) ölçüm setidir. Mevcut 216 yüksek gülme hattı görüntüsünün 150'sinin manuel ölçümü vardır (147'si tekil eşleşme); 65'inin **yoktur** (yeni çekimler olabilir).

Bu, Reviewer 4'ün overlap sorusuna kısmi bir cevaptır (≥ 147–150 ortak yüksek gülme hattı görüntüsü) ama kesin sayı hasta kimliğiyle doğrulanmalıdır.

### 5.5 Gözlemci içi güvenilirlik (doğrulandı, ek ayrıntı)

Depodaki `calibration-first/last.xlsx` her biri **15 görüntü** (makale: 20). Klinik ekip 20 görüntülük ayrı bir dosya göndereceğini bildirdi; geldiğinde depodakiler değiştirilmeli. Depodaki 15 görüntünün yalnızca **2'si** mevcut COCO setindedir; yani ICC önceki çalışmanın görüntüleri üzerinde hesaplanmıştır (yeni dosyada da kontrol edilmeli). `evaluation/intra_observer.py` ICC(3,1) hesaplar ve eşleşmiş t-testi p > 0.05'i "tutarlı" olarak raporlar; anlamsızlık uyum kanıtı değildir, ICC ve %95 GA esas alınmalıdır.

### 5.6 Görüntü çözünürlüğü

COCO görüntüleri ağırlıklı olarak 2698×1799 / 2699×1799; dağılım 1292×861 ile 4006×2671 arasında. Klinik ekip, ImageJ ölçümlerini PC'de **2698×1799**'a küçültülmüş kopyalarda yaptığını doğruladı (ekran görüntüsü). Yani COCO çerçevesi = ImageJ çerçevesi; GT maske pikselleri ile mm değerleri doğrudan ilişkilendirilebilir. Farklı boyutlu azınlık görüntülerde (yaklaşık %20) ImageJ'nin hangi kopyada yapıldığı belirsizdir; bunlar duyarlılık analizinde ayrı tutulmalıdır. Boyutların kümelenmesi ve standart çekim mesafesi, **tek bir global `px_per_mm`** varsayımını destekler.

---

## 6. Referans (oracle) doğrulaması — ön sonuç (yeni)

Doğru geometrinin işe yarayıp yaramadığını görmek için, GT dişeti poligonları rasterleştirilerek sütun bazlı dikey kalınlık profili çıkarılmış, dişeti x-uzanımı 6 eşit bölgeye bölünmüş ve bölge içi persentiller manuel ImageJ ortalamasıyla karşılaştırılmıştır (`high` grubu, tekil eşleşen 148 görüntü; ölçek sıfırdan geçen regresyonla **aynı veride** kestirilmiştir — iyimser olabilir).

| Tahminci | Pearson r | Global px/mm | MAE (mm) | RMSE (mm) |
|---|---|---|---|---|
| bölge p10 | 0.81 | 14.6 | 0.66 | 0.89 |
| **bölge p25** | **0.83** | **17.1** | **0.63** | **0.85** |
| bölge medyan | 0.83 | 20.7 | 0.65 | 0.85 |
| bölge min | 0.78 | 10.8 | 0.80 | 1.05 |
| bölge max (papilla) | 0.74 | 36.0 | 0.89 | 1.10 |

Ek gözlem: GT'de dudak alt kenarı ile dişeti üst kenarı arasında **medyan 9 px (IQR 6–12 px, ≈ 0.5 mm)** sistematik bir boşluk vardır. ImageJ ölçümü dudak kenarından yapıldığından, "dudak alt kenarı → gingival margin" mesafesini ölçen bir tahminci de sınanmalıdır.

Bu ön sonuç, (i) geometrinin doğru olduğunu, (ii) ~17 px/mm global ölçeğin makul olduğunu (2698 px genişlik ≈ 16 cm görüş alanı), (iii) milimetrik validasyonun mevcut veriyle yapılabileceğini gösterir. Nihai analiz prompt'taki dev/holdout ayrımıyla, tahmin maskeleri üzerinde (best.pt gerektirir) tekrarlanmalıdır.

**Kalibrasyon hassasiyeti (Bölüm A ile birlikte okunmalı):** ImageJ ölçeği her görüntüde **tek bir 1 mm aralık** (~17 px) işaretlenerek belirlenmiş; işaretlemede 1 px sapma ≈ %6, 2 px ≈ %12 ölçek hatası demektir. Görüntü bazlı px/mm oranının CV'si (0.38) bu gürültüyü içerir; dolayısıyla global ölçekle elde edilen 0.63 mm MAE'nin bir kısmı referansın kendi kalibrasyon belirsizliğidir. Prob her ölçülen fotoğrafta bulunduğundan, **≥ 5 mm'lik bir aralık** işaretlenerek görüntü bazlı bağımsız `px_per_mm` elde edilebilir; bu hem gerçek ölçek dağılımını (SD, CV) hem de ImageJ kalibrasyonunun hassasiyetini sayısal olarak hakeme sunma imkânı verir. Bu iş, iki tıklamalı basit bir araçla ~150 görüntüde 1–2 saatte yapılabilir.

---

## 7. Yeniden üretilebilirlik

- `data/labels_smileline.csv`, `data/splits.json`, `results/*.csv` **boştur**.
- `yolo/weights/best.pt` depoda yoktur.
- Figure 6'nın hangi giriş noktası (1.7), hangi `px_per_mm` ve hangi ağırlıkla üretildiği izlenemez.
- `evaluation/evaluate.py` tek görüntü için MAE/RMSE hesaplar (n = 1); toplu değerlendirme aracı yoktur.

---

## 8. Öncelik sırası ve bağımlılıklar

| # | Adım | Bağımlılık |
|---|---|---|
| 1 | Ölçüm modülünün yeniden yazımı (sınıf ayrımı, dişeti kalınlığı, birim, tek kural motoru, E1–E2 çakışması) | Yok |
| 2 | Excel ayrıştırıcısının Bölüm 4 kurallarıyla yazılması (`-` = 0 mm onaylandı; nokta eki korunur) | Yok |
| 3 | Oracle doğrulaması: GT maskeler ↔ ImageJ, dev/holdout, tahminci seçimi, global vs görüntü bazlı ölçek | 1, 2 |
| 4 | İki tıklamalı prob kalibrasyon aracı; ölçülen ~150 yüksek gülme hattı görüntüsünde ≥ 5 mm aralıkla `px_per_mm`; global ölçek ve ImageJ hassasiyetiyle karşılaştırma | 3 (araç hemen yazılabilir) |
| 5 | Hasta–görüntü Excel'i + `ayni_hasta_aday_ciftler.csv` ile hasta düzeyinde temiz bölüntü; sabit test seti | Klinikten Excel |
| 6 | Temiz bölüntüyle bir kez yeniden eğitim; sabit test seti; **test seti** metrikleri | 5 |
| 7 | Tahmin maskeleri üzerinde milimetrik doğruluk (MAE, RMSE, ICC, Bland–Altman) + sınır hatası ayrıştırması | 3, 6, `best.pt` |
| 8 | Klinik uyum analizi — yalnızca kör klinisyen değerlendirmesi elde edilirse | Dış veri |
| 9 | Rebuttal ve makale düzeltmeleri | 1–8 |

1–3 için gereken tüm veri depodadır.

---

## 9. Klinik ekiple durum: cevaplananlar ve açık kalanlar

**Cevaplandı (5 Eylül):** `-` = 0 mm; ImageJ çerçevesi 2698×1799; kalibrasyon görüntü bazlı, tek 1 mm aralık, prob her ölçülen fotoğrafta; aynı isimli dosyalar farklı hastalar; sondaki nokta farklı görüntü; E0 yok, > 0 mm → olası tedavi alternatifi, sistem yüksek gülme hattına özgü; ham 6-diş ölçümleri elimizdeki Excel.

**Bekleniyor:**
1. Hasta–görüntü eşleme Excel'i (hazır, gönderilecek).
2. 20 görüntülük gözlemci içi tekrar ölçüm dosyası.
3. Zeynep hocanın E0 / "olası tedavi alternatifi" ifadesi onayı.
4. J Dent 2026 ile hasta/görüntü örtüşme sayısı.

**Yeni sorulması gerekenler:**
5. **Aynı hastanın iki fotoğrafı (kritik):** Veri setinde en az 55 hastanın problu ve probsuz iki çekimi birlikte bulunuyor (`ayni_hasta_ornek_ciftler.jpg`, `ayni_hasta_aday_ciftler.csv`). Nazikçe sorulmalı: "Ölçüm tek fotoğrafta yapılmış olsa da Roboflow'a her iki çekim de yüklenmiş görünüyor; hasta listesinde her iki dosya adı da aynı hastaya bağlanabilir mi?" Hasta listesi bu çiftleri içermiyorsa listemizle birleştirilecek.
6. **Önekli yinelenen adlar:** `IMG_4552`/`25-IMG_4552`, `IMG_7366`×2, `IMG_1471`/`39-IMG_1471`, `IMG_8645`/`27-IMG_8645`, `IMG_8669`/`21-IMG_8669`, `IMG_1195`/`55-IMG_1195`, `IMG_1212`/`39-IMG_1212`, `30-IMG_0586`/`35-IMG_0586`, `42-IMG_0590`/`55-IMG_0590`, `30-IMG_0647`/`31-IMG_0647` — bunlar da farklı hastalar mı; hangisi mevcut veri setindeki fotoğraf?
7. **Farklı boyutlu görüntüler:** 2698×1799 olmayan ~%20 görüntüde ImageJ ölçümü hangi kopyada yapıldı? (Yalnızca duyarlılık analizi için.)

---

## 10. Rebuttal açısından değerlendirme

- Figure 6 tutarsızlığı artık spekülatif bir klinik risk değil, **tespit edilmiş, mekanizması kod satırıyla gösterilebilen ve düzeltilmiş** bir yazılım hatasıdır.
- Milimetrik doğruluk analizi için referans ölçümler mevcuttur ve **doğru okunduğunda 692 tam kayıt** içerir; mevcut veri setiyle ≈ 147 yüksek gülme hattı görüntüsü eşleşir. GT maskeler üzerinde ön sonuç MAE ≈ 0.63 mm'dir.
- v1'deki "dosya adı yinelenmesi" ve "instance tutarsızlığı" bulguları rebuttal'a **yazılmamalıdır**; yanlıştı. Doğru ifade: bölünme görüntü düzeyinde yapılmıştır; içerik tabanlı denetimde aynı hastaya ait (problu/probsuz) en az 55 çift tespit edilmiş, 22'si farklı bölüntülerde, 15'i test setinde bulunmuştur; veri seti hasta düzeyinde yeniden bölünmüş ve model yeniden eğitilmiştir. "Her hastadan tek görüntü" ifadesi **kullanılmamalıdır**; "her hastadan tek ölçüm fotoğrafı" doğrudur.
- Kalibrasyon sorusuna (Reviewer 4) artık kesin cevap verilebilir: görüntü bazlı, prob üzerindeki 1 mm aralık, ImageJ Set Scale, 2698×1799 çerçevesi. Ek olarak bağımsız ≥ 5 mm prob kalibrasyonuyla ölçek dağılımı (SD/CV) raporlanabilir.
- Figure 4 sayıları açıklanabilir: instance ≠ görüntü (papillalar), ×2 augmentasyon.
- Makaledeki "çakışan aralıklarda tüm adaylar raporlanır" iddiası, E1–E2 için kodda **karşılığı olmadığından** düzeltilmeli; düzeltilen kod bu iddiayı gerçekten sağlamalıdır.
- Etiyoloji/tedavi katmanının klinik validasyonu mevcut verilerle **yapılamaz**; makalenin ana iddiası segmentasyon + milimetrik ölçüm validasyonuna daraltılmalı, E/T katmanı valide edilmemiş yorumlama çerçevesi olarak sunulmalıdır.
