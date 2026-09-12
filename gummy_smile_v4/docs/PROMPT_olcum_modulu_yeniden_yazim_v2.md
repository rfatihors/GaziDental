# Görev: `gummy_smile_v3` ölçüm modülünün yeniden yazımı ve referans doğrulaması (v2)

Sen bu depoda (`rfatihors/GaziDental`, `gummy_smile_v3/`) çalışan bir yazılım mühendisisin. Görevin, dişeti görünürlüğü (gingival display) ölçüm hattını **doğru geometriyle** yeniden yazmak ve elde bulunan manuel ImageJ ölçümlerine karşı doğrulamaktır.

Bu bir araştırma projesidir; çıktılar hakem revizyonunda kullanılacaktır. **Doğruluk, hızdan ve zariflikten önce gelir.** Emin olmadığın hiçbir şeyi varsayma; ölç, doğrula, raporla. Aşağıdaki sayıların çoğu depo üzerinde önceden doğrulanmıştır; sen yine de kendi çıktılarınla teyit et ve uyuşmazlık varsa raporla.

---

## 0. Temel kavram — bunu yanlış anlama

Ölçmek istediğimiz büyüklük **dişeti görünürlüğüdür**: gülümseme sırasında görünen dişeti bandının dikey yüksekliği.

Anatomik olarak bu bant:
- **üstten** üst dudağın alt kenarı ile,
- **alttan** gingival margin (dişetinin diş üzerindeki kenarı) ile

sınırlanır. Görünen dişeti bölgesinin kendisi tam olarak ölçmek istediğimiz banttır.

> **Ölçtüğümüz şey `diseti` sınıfına ait maskenin dikey kalınlığıdır.**
> **`dudak` maskesini ÖLÇMÜYORUZ.** Dudak maskesi yalnızca (a) dişeti üst sınırının dudak alt kenarıyla çakıştığını denetlemek ve (b) alternatif "dudak-ankraj" tahmincisi için kullanılır.

GT anotasyonları görselleştirildiğinde durum şudur (bunu ilk iş olarak kendin de çiz ve `results/oracle_validation/gt_overlay_examples.png` olarak kaydet):
- `dudak` = yalnızca **üst dudak vermilyonu**, ince bir bant. Üst kenarı Cupid yayını izler.
- `diseti` = üstte dudak alt kenarına dayanan, altta **festonlu** bant: her dişin orta hattında (zenith) **en ince**, iki diş arasında (papilla) **en kalın**.
- **Yüksek gülme hattında** dişeti tek sürekli banttır (görüntü başına 1 instance).
- **Normal/low gülme hattında** yalnızca papillalar görünür; dişeti görüntü başına ortalama 3–4, en fazla 7–8 **ayrı parça** olarak etiketlenmiştir. Diş zenitlerinin üzerinde **hiç dişeti pikseli yoktur** ve o dişteki görünürlük gerçekten **0 mm**'dir.

Mevcut kod bunun tam tersini yapmaktadır (Bölüm 1). Yeni kodda bu ayrım mutlak olmalıdır.

### Diş bazlı ölçüm ve feston

Klinik ölçüm (ImageJ) her diş için o dişin zenith noktasından yapılmıştır; 6 ön diş (13-12-11-21-22-23), Excel'de "soldan sağa diş numarası". Dolayısıyla bir diş bölgesi içinde bandın **kalın değil, ince** kısmı hedeftir. Bölge içinde `max` almak papillayı ölçer ve sistematik olarak yüksek değer üretir — bu hatayı yapma.

Dikkat: yüksek gülme hattında dişeti bandı premolarlara kadar uzanabilir; dişeti x-uzanımını 6'ya eşit bölmek 6 ön dişle hizalanmayabilir. Bu yüzden birden fazla bölgeleme yöntemi denenecek (3.2).

---

## 1. Mevcut durumdaki hatalar (referans)

`measurement/yolo_measurements.py` ve `measurement/measure_gum_visibility.py` (aynı mantık):

```python
contour    = max(contours, key=cv2.contourArea)   # en büyük kontur = pratikte DUDAK
lip_line_y = float(contour[:, :, 1].min())        # o konturun KENDİ en üst noktası
zenith_y   = bölgedeki maskenin en üst noktası
gummy_px   = zenith_y - lip_line_y                # üst kenarın kendi içindeki dalgalanması
```

- Alt sınır (gingival margin) hiç kullanılmıyor. Ölçülen şey dudağın üst kenarının (Cupid yayının) eğriliğidir.

`yolo/infer_yolo.py` ve `yolo/infer_yolo_seg.py`:

```python
mask = result.masks.data.max(dim=0).values   # sınıflar birleştiriliyor; boxes.cls hiç okunmuyor
```

- `masks.data` Ultralytics'te varsayılan olarak **letterbox'lanmış çıkarım çözünürlüğündedir**, orijinal görüntü değil. Kod bunu doğrudan orijinal görüntüye bindirmeye çalışıyor.
- `configs/config.yaml`: `imgsz: 1024` (model 640'ta eğitildi), `max_det: 5` (normal/low görüntülerde dişeti parçalarını keser), `px_per_mm: null` iken çıktı kolonu `mean_mm`; `mm_per_pixel` ve `px_per_mm` birlikte kullanılıyor.
- İki paralel giriş noktası var: `master_pipeline_v3.py` (→ `methods/v3/rule_engine.py`) ve `segmentation_model.py`/`treatment_predictor.py` (→ `methods/v3/diagnosis.py`). İki kural motoru aynı değere farklı sınıf verebiliyor (`diagnosis.py` 4–6 mm'yi öncelikle **E3**'e atıyor; `rule_engine.py` E2).
- `rule_engine.assign_etiology` E1'i (≤ 4 mm) önce kontrol edip hemen döndüğünden **3–4 mm'deki E1–E2 çakışması hiç raporlanmıyor**; ölçüm `NaN` iken E1/T1 dönüyor.

---

## 2. Kapsam

### Yapılacaklar
- Sınıf-duyarlı maske çıkarımı ve yeni ölçüm modülü
- Excel ayrıştırıcısı (Bölüm 4.2 kuralları)
- Ground-truth (COCO) maskeler üzerinden referans doğrulama harness'ı — **tahmin maskeleriyle de çalışacak şekilde** (ileride `best.pt` gelince aynı harness kullanılacak)
- Tek kural motoru; E1–E2 ve E2–E3 çakışmalarının ikisini de raporlayan
- Birim testleri
- Doğrulama raporu + pipeline blok diyagramı (Mermaid; Reviewer 1 talebi)

### Yapılmayacaklar
- Model yeniden eğitimi yok; mimari arayışı yok
- Yeni bağımlılık yok (`requirements.txt` dışına çıkma; gerekirse gerekçesini raporla)
- `methods/v1/*` (XGBoost) kodunu değiştirme — karşılaştırma için olduğu gibi kalsın (ama 4.7'deki notu oku)
- Eşik tablosuna E0 bandı ekleme — klinik karar: E0 yok; 0 mm "uygulanamaz", > 0 mm E1 (bkz. 3.5)

---

## 3. Uygulanacak tasarım

### 3.1 Sınıf-duyarlı maske çıkarımı

Yeni modül: `gummy_smile_v3/measurement/masks.py`

- Sınıf kimliklerini `result.boxes.cls` üzerinden al, isimleri `model.names` sözlüğünden çöz. Sınıf isimleri bu projede `diseti` ve `dudak`. İsimleri koda gömme; `configs/config.yaml` içinde `class_names: {gingiva: diseti, lip: dudak}` tanımla ve oradan oku. Model içindeki isimler beklenenle uyuşmuyorsa açık hata fırlat.
- Maskeleri **orijinal görüntü çözünürlüğünde** üret: `model.predict(..., retina_masks=True)` kullan **veya** `result.masks.xy` poligonlarını orijinal boyutta rasterleştir. Her durumda `assert mask.shape == image.shape[:2]`. Bu assert'i sessizce geçme.
- Her sınıf için **tüm instance'ların birleşimini (union)** al. "En büyüğünü seç" davranışı yok.
- Dişeti ve dudak maskelerini **ayrı PNG** olarak kaydet (`masks/{stem}_gingiva.png`, `masks/{stem}_lip.png`). Birleşik maske üretme.
- `max_det` en az 20; `conf` mevcut değerde (yorum: operasyon noktası doğrulama setinde seçilmeli).
- Aynı modül, COCO poligonlarından ve diskteki hazır PNG'lerden de aynı formatta maske üretebilmeli (`from_coco`, `from_png`) — oracle ve gelecekteki tahmin analizleri aynı ölçüm kodunu kullanacak.

### 3.2 Ölçüm

Yeni modül: `gummy_smile_v3/measurement/gingival_display.py`

Girdi: dişeti ikili maskesi `G`, (opsiyonel) dudak ikili maskesi `L`, `px_per_mm` (opsiyonel), yapılandırma.

1. **x-uzanımı.** Ölçüm penceresi `[x0, x1]`: dişeti maskesinin x-uzanımı. Dudak maskesi varsa dudağın x-uzanımıyla kesiştir (dişeti dudağın dışına taşamaz). Uzanım boşsa `no_gingiva_mask`.

2. **Sütun bazlı kalınlık profili.** `[x0, x1]` içindeki **her** sütun için, o sütundaki **en uzun kesintisiz dikey doluluk aralığını** bul (delik/gürültü nedeniyle `y_max − y_min` kullanma). `t(x) = run_bottom − run_top + 1`. Dişeti pikseli olmayan sütunlarda **`t(x) = 0`** (NaN değil, atlama değil — normal/low görüntülerde ve `-` işaretli dişlerde bu gerçek sıfırdır). `run_top(x)` ve `run_bottom(x)` sakla (sırasıyla dudak-altı sınırı ve gingival margin).

3. **Profil temizliği.** `t(x)` üzerinde küçük medyan filtre (pencere ≈ genişliğin %1'i, tek sayı, yapılandırılabilir).

4. **Bölgeleme.** Üçünü de uygula, üçünü de raporla:
   - **A (eşit bölme):** `[x0, x1]` 6 eşit sütuna bölünür. (Yüksek gülme hattında premolarlar dâhil olabilir; hizalama sorunlu.)
   - **B (feston tabanlı):** `t(x)` profilinde papillalara karşılık gelen yerel maksimumları bul; ardışık maksimumlar arası = diş bölgesi. 6 bölge çıkmazsa `festoon_detection_failed` bayrağı, A'ya düş.
   - **C (orta hat ankrajlı zenith):** Orta hattı belirle (dudak maskesinin x-medyanı; yoksa görüntü ortası). `t(x)` profilinin yerel **minimumlarını** (zenith adayları) bul; orta hattın her iki yanından en yakın 3'er tanesini al → 6 zenith. Bu, önceki çalışmada valide edilmiş `methods/v1/pixel_features.image_to_pixel_min` mantığının temizlenmiş hâlidir; oradaki 30 px sabit aralığını çözünürlüğe göre ölçekle (görüntü genişliğinin ~%1'i). 6 minimum bulunamazsa bayrak, A'ya düş.

5. **Bölge değeri.** Her bölge için birden fazla tahminci, hepsini kaydet: `p10`, `p25`, `median`, `min`. (C yönteminde bölge = zenith etrafında dar pencere, ör. ±%1 genişlik; değer = penceredeki medyan.)

6. **Dudak-ankrajlı tahminci (ek).** `L` varsa her sütun için `lip_bottom(x)` = dişeti üst kenarının hemen üstündeki en alt dudak pikseli. Mesafe `d(x) = run_bottom(x) − lip_bottom(x)`. Bu, GT'de gözlenen ~9 px (≈ 0.5 mm) anotasyon boşluğunu kapsar ve ImageJ'nin ölçtüğü şeye (dudak kenarı → gingival margin) daha yakın olabilir. Aynı bölgeleme ve tahmincilerle `d(x)` için de bölge değerleri üret ve ayrı kolonlarda raporla (`*_lipanchored`).

7. **Görüntü değeri.** Altı bölge değerinin ortalaması → `gingival_display_px` (birincil), medyanı da kaydet. Bölge değerlerinin sırasını da kaydet (soldan sağa 1–6).

8. **Birim.** Tek isim: `px_per_mm`; `mm = px / px_per_mm`. `mm_per_pixel` ismini kod tabanından kaldır. `px_per_mm` yoksa `gingival_display_mm = None`, `unit = "px"`; **asla `mean_mm` adlı kolonda piksel döndürme.**

Fonksiyon saf olsun (maske dizileri girer, dataclass çıkar); dosya G/Ç ayrı katmanda.

### 3.3 Dudak maskesiyle sınır denetimi

`L` varsa her sütun için `gap(x) = run_top(x) − lip_bottom(x)`. Görüntü başına medyan ve IQR raporla. GT'de beklenen büyüklük sırası **medyan ≈ 9 px, IQR 6–12 px** (2698 px genişlikte). Eşik: `gap` medyanı görüntü yüksekliğinin %2'sini (veya ~40 px) aşarsa `lip_gingiva_boundary_mismatch`. Sistematik büyük fark bulursan **dur ve raporla** — anotasyon tanımının varsayımdan farklı olduğu anlamına gelir.

### 3.4 QC bayrakları (CSV'ye yaz, hiçbirini yutma)

`no_gingiva_mask`, `no_lip_mask`, `gingiva_multi_component` (bileşen sayısıyla), `region_zero` (bölge değeri 0 — normal/low'da beklenen), `region_empty_window` (C yönteminde pencere boş), `festoon_detection_failed`, `zenith_detection_failed`, `lip_gingiva_boundary_mismatch`, `mask_shape_mismatch`, `implausible_value` (negatif veya > 20 mm eşdeğeri).

### 3.5 Kural motoru

Tek motor: `methods/v3/rule_engine.py`. `methods/v3/diagnosis.py` bu motora delege etsin veya `DEPRECATED` işaretlensin; iki farklı davranış kalmasın.

- Ölçüm `None`/`NaN` → `UNCLASSIFIED`, gerekçe `notes`'ta. "Hiçbir kural eşleşmedi" dalı da `UNCLASSIFIED`.
- **Genel çakışma mantığı:** değerin eşleştiği **tüm** sınıfları hesapla. Birden fazlaysa `ambiguous=True`, `etiology_class = "E1-E2"` / `"E2-E3"` gibi birleşik etiket, adaylar ve tedaviler birlikte. Bu, klinisyenlerin Excel'de kullandığı etiket biçimiyle aynı (793 "E1-E2", 829 "E2-E3" hücresi var).
- Sınırları tek yerde, açıkça tanımla ve testle sabitle: E1 `< 4.0`; E2 `[3.0, 6.0]`; E3 `[4.0, 8.0]`; E4 `> 8.0`. 4.0 → E2-E3; 8.0 → E3; 3.0 → E1-E2.
- `ambiguous_policy.use_metadata` varsayılanı `false`; `metadata_map` yorum satırına. Koda yorum: gülme hattı sınıfı ile etiyoloji arasında nedensel ilişki yoktur.
- **Alt sınır (klinik karar alındı):** ayrı bir E0 bandı yoktur. `value <= 0` (dişeti görünmüyor) → `NO_VISIBLE_GINGIVA`: etiyoloji/tedavi sınıfı yok, `notes` = "no visible gingiva; framework applies to high smile line only". `0 < value < 4` → E1. E1 için `min_mm = 0.0` (hariç) yaz; `None` bırakma. Çıktı alanlarında tedavi listelerini "possible treatment alternatives" olarak adlandır (`treatment_alternatives`), "recommendation" değil — klinik ekibin isteği.
- Sistem yalnızca dişetinin göründüğü (yüksek gülme hattı) görüntüler için tasarlanmıştır; bunu modül docstring'ine ve `report.json`'a `applicability` alanı olarak yaz.

### 3.6 Yapılandırma

`configs/config.yaml`: `yolo.imgsz: 640`, `yolo.max_det: 20`, `yolo.retina_masks: true`, `class_names` bloğu, `px_per_mm` korunur, `mm_per_pixel` varsa kaldırılır, `ambiguous_policy.use_metadata: false`.

---

## 4. Referans (oracle) doğrulaması — görevin en önemli parçası

**Amaç:** Ölçüm geometrisinin doğruluğunu, model hatasından bağımsız sınamak. Tahmin maskeleri değil, **ground-truth anotasyon maskeleri** kullanılacak.

Yeni betik: `gummy_smile_v3/evaluation/oracle_validation.py` (CLI: `--masks gt|png_dir` — ileride tahmin maskeleriyle aynı betik çalışacak).

### 4.1 Ground-truth maskeler

- Kaynak: `data/coco_dataset/{high,low,normal}/{train,valid,test}/_annotations.coco.json`
- Kategoriler: `1 = diseti`, `2 = dudak`; `0 = dudak-diseti` supercategory'dir, anotasyonlarda kullanılmaz.
- Beklenen sayılar (kendin teyit et): 1.315 görüntü (216 high / 303 low / 796 normal); 3.938 dişeti, 1.318 dudak instance. Poligonlar `list` biçimindedir (RLE yok).
- Görüntüler ağırlıklı olarak **2698×1799** (1292×861 – 4006×2671 arası), Roboflow yeniden boyutlandırması yok. Klinik ekip, ImageJ ölçümlerini fotoğrafları PC'de 2698×1799'a küçülterek yaptığını doğruladı: **COCO çerçevesi = ImageJ çerçevesi.** Boyut dağılımını raporla; 2698×1799 (±2 px) olmayan görüntüleri `frame_uncertain` bayrağıyla işaretle ve duyarlılık analizinde ayrı tut.
- Aynı görüntünün tüm dişeti instance'larını birleştir (normal/low'da çok parçalı).

### 4.2 Referans ölçümlerin yüklenmesi — kodlama kuralları Excel'in kendi notlarından gelir

Kaynak: `data/manual_measurements/Hasta_ID-_Ölçümler.xlsx` (klinikten 6 Eylül'de geldi; eski `ölçümler ai guncel.xlsx` dosyasının yerine geçer). Üç sayfa:
- `Yüksek Gülme Hattı`: 692 satır — ad, 6 bölgesel ölçüm (diş 13-12-11-21-22-23), boş, 6 E, 6 T, boş kolonlar, **YAŞ (sütun X), CİNSİYET (sütun Y)**. Başlık 2 satır (`header=1`). Yaş 634, cinsiyet 630 satırda dolu.
- `Düşük Gülme Hattı`: 100 satır — ad, boş, yaş, cinsiyet (hepsi COCO `low` grubunda).
- `Normal Gülme Hattı`: 246 satır — ad, boş, yaş, cinsiyet (181'i COCO'da).

Dosyanın içinde iki gözlemci notu var (eski dosyadan); ayrıştırıcı bunlara göre yazılacak:

> Not 1: "1 mm'den küçük olan sayıların başına sıfır yazdığımda sıfırı dahil etmiyor. o yüzden başına `*` işareti koydum"
> Not 2: "`-` işareti dişeti dişin üzerine geliyor. ölçülecek dişeti mesafesi yok demek."

Kurallar (her hücre için hem değer hem `parse_kind` sakla):

| Hücre | Beklenen adet | Okuma | `parse_kind` |
|---|---|---|---|
| tam sayı ≥ 1000 | ~3.505 | `/1000` (Türkçe Excel'de `3.207` nokta binlik ayracı sayılmış) | `div1000` |
| tam sayı ≥ 10000 | 16 | `/1000` (10–14 mm, şiddetli vaka) | `div1000` |
| `*0.xxx` metin | 353 | `float(s[1:])` — **gerçek < 1 mm ölçüm, NaN DEĞİL** | `star_sub1mm` |
| `-`, `-(mesafe yok)` (boşluklu varyantlar dâhil) | 290 | **0.0 mm** — Not 2; klinisyen onayladı: "görünen dişeti yok, dudak gingival marjinin üstünde" + `zero_by_convention=True` | `dash_zero` |
| tam sayı 100–999 | 5 | `/1000`, ama **belirsiz** bayrağı (aşağıdaki E denetimine tabi) | `ambiguous_100_999` |
| küçük tam sayı (< 10) veya `'3.00'` gibi sayısal metin | 4 | olduğu gibi mm | `plain_mm` |
| negatif | 1 | NaN | `negative` |

**E etiketiyle tutarlılık denetimi (zorunlu):** Aynı hücrenin E etiketi (E1 / E1-E2 / E2-E3 / E3 / E4 / `-`), gözlemcinin değeri nasıl anladığını gösterir. Okunan mm'ye Tablo 1'i uygula; çıkan etiket Excel'deki etiketle uyuşmuyorsa `label_inconsistent` bayrağı ve o hücreyi birincil analizden çıkar. Beklenen: 3.868 etiketli hücrede ≤ 5 uyuşmazlık (bilinen ikisi: `IMG_9152` bölge 6 = `667` ve `IMG_9206` bölge 6 = `728`, ikisi de "E4" etiketli).

Dağılım denetimi: doğru okunduğunda medyan ≈ 2.8 mm, %95 ≈ 6.4 mm, maks 14.1 mm, **691/692 satırda altı değer de okunabilir** olmalı (yeni dosyada 692 satır). 387 gibi bir sayı çıkıyorsa `*` ve `-` kurallarını yanlış uyguluyorsun.

**Ad normalizasyonu — üç kural, hepsi klinik ekiple doğrulandı:**

1. **Sondaki nokta ayrı görüntüdür.** Excel'de `IMG_2544` ile `IMG_2544.` farklı fotoğraflardır. Roboflow bu noktayı `-`'ye çevirir (`IMG_6849.` ↔ COCO `IMG_6849-_jpg`). Ancak COCO'daki her `-` eki Excel'de noktalı değildir: `high` grubundaki 87 `-` adının yalnızca 8'i Excel'de noktalı, 30'u noktasız, 49'unun Excel satırı yoktur. Bu yüzden eşleştirme **kademeli** yapılır:
   - (a) Tam eşleşme: nokta/`-` eki `~` işaretine çevrilir ve korunur (`IMG_2544.` → `img2544~`, `IMG_2544-` → `img2544~`, `IMG_2544` → `img2544`).
   - (b) Geri düşme: COCO adı `~` ile bitiyor, Excel'de `~`'lı karşılığı yok ama `~`'sız taban ad var **ve** COCO'da `~`'sız taban görüntü yoksa → tabana eşle, `match_kind = dash_base_fallback`. (Beklenen: 30 görüntü.)
   - (c) Hem `IMG_x` hem `IMG_x-` COCO'da varsa ve Excel'de yalnızca `IMG_x` varsa → belirsiz, `name_ambiguous`, eşleştirme (`IMG_2633`, `IMG_3864`, `IMG_3694`, `IMG_3424`, `IMG_4034`).
2. **Baştaki sayı öneki hastanın yaşıdır**, ayrı görüntüdür. `25-IMG_4552` = 25 yaşındaki hastanın `IMG_4552` adlı fotoğrafı; `IMG_4552` ise başka bir hasta (22 yaş). Excel yüksek sayfasında 154 önekli ad var; 149'unda önek YAŞ sütunuyla aynı. COCO'da yalnızca `normal` grubunda 106 önekli ad var (`23-IMG_4080_JPG` gibi), `high`'da yok. Kural: öneki `age_from_prefix` olarak çıkar, YAŞ sütunu boşsa doldur; eşleştirmede öneki **koru** (önekli Excel adı ↔ önekli COCO adı). Önek atılarak tabana eşleme **yapılmaz** — bu 154 satırın neredeyse tamamı bu veri setinde olmayan (önceki çalışma) fotoğraflardır; atılırsa yanlış eşleşir.
3. Küçük harf, `ı→i`, `.rf.<hash>` ve uzantı (`_jpg`, `_jpeg`, `.jpg`, `.JPG`) atılır, kalan alfanümerik olmayanlar silinir.

Birim testleri: `('IMG_2544.','IMG_2544-_jpeg.rf.x.jpeg')` → eşleşir (a); `('IMG_6545','IMG_6545-_jpg.rf.x.jpg')` → COCO'da `IMG_6545` yoksa eşleşir (b), varsa eşleşmez; `('ımg_2544','IMG_2544-_jpeg…')` → eşleşmez; `('25-IMG_4552','IMG_4552_jpg…')` → eşleşmez.

Excel'de aynı taban ada sahip 10 grup (`IMG_4552`/`25-IMG_4552`, `IMG_7366`×2, `IMG_1471`/`39-IMG_1471`, …) **farklı hastalardır** (yaş/cinsiyet farklı, klinik ekip doğruladı). Bunlardan yalnızca `IMG_7366` (31 E / 51 K) ve `IMG_8645` (53 K / 27 K) COCO `high`'da vardır ve hangi satırın o fotoğrafa ait olduğu bilinmediğinden **ikisi de oracle setinden çıkarılır** (`row_ambiguous`).

Yaş/cinsiyet kolonlarını da oku ve doluluk oranını raporla (Reviewer 3'ün cinsiyet dağılımı sorusu için).

### 4.3 Eşleştirme

Normalize adlarla Excel ↔ COCO. **Bilinen gerçek:** Excel önceki çalışmanın (J Dent 2026) ölçüm setidir; 682 benzersiz adın yalnızca **~179'u** mevcut COCO setinde var (150 high, 22 normal, 7 low). `high` grubunda 216 görüntüden ~150'sinin ölçümü var, **~147'si tekil** eşleşme. Bu sayıları raporla; 503 eşleşmeyen adı `unmatched_manual.csv`'ye yaz, panik yapma — bunlar bu veri setinde olmayan fotoğraflardır.

**Uyarı 1 — ad çakışması:** iPhone `IMG_xxxx` numaraları cihazlar arasında çakışır (klinik ekip doğruladı: farklı telefonlar, farklı hastalar). Kural (c) ile belirsiz kalanları `name_collisions_coco.csv`'ye yaz. Normal/low gruplarındaki Excel yüksek-sayfası eşleşmelerinin (~17) ad çakışması olduğunu varsay; oracle setine alma.

**Uyarı 2 — aynı hastanın iki fotoğrafı (doğrulandı):** 55 çift (`data/manual_measurements/ayni_hasta_aday_ciftler.csv`; klinik ekip görsel olarak inceledi, "aynı değil" dediği çiftlerin listesi gelince `same_patient=False` sütunu eklenecek). Bunların 37'si `high` grubunda ve bir üyesi ölçümsüzdür; 4.8'deki temizlik adımıyla çözülür. Oracle analizini etkilemez.

**Hasta kimliği:** Klinik ekip "her hastadan tek ölçüm fotoğrafı" diyor; ayrı bir hasta ID listesi **yok**. Temizlik (4.8) sonrasında **her görüntü = bir hasta** kabul edilir; `patient_id` = tutulan görüntünün normalize adı; `train_only` ikizler aynı `patient_id`'yi alır. Bunu `dataset_manifest.csv`'ye yaz.

Birincil analiz seti: `high` grubu, tekil eşleşen (a veya b), `label_inconsistent`/`row_ambiguous`/`name_ambiguous` içermeyen görüntüler (**≈ 145 bekle**; kural (a) 112 + 8, kural (b) 30, eksi belirsizler). İkincil: normal/low eşleşmeleri yalnızca raporla.

### 4.4 Ölçek belirleme

`px_per_mm` bilinmiyor. Referans mm cinsinden olduğuna göre:

- **Tek global ölçek:** sıfırdan geçen regresyonla `px_per_mm` kestir; R², artık dağılımı. Beklenen büyüklük sırası **15–20 px/mm** (2698 px genişlik ≈ 15–16 cm görüş alanı, 15 cm çekim mesafesi). Çok farklı bir değer çıkarsa geometriyi veya ayrıştırmayı yeniden kontrol et.
- **Görüntü bazlı ölçek:** her görüntü için oran; ortalama, SD, CV. **Not:** görüntü bazlı oran ölçek değişimini ölçüm gürültüsüyle karıştırır; CV'yi yorumlamadan raporla. Ayrıca oranı görüntü boyutuna karşı çiz — boyutu 2698×1799'dan farklı olanlar farklı ölçek taşıyabilir.
- Global ölçeği **leave-one-out** ile doğrula.

**Referansın kendi kalibrasyonu (klinik ekipten öğrenildi):** ImageJ ölçeği her görüntüde ayrı ayrı, prob üzerindeki **ardışık iki 1 mm işareti** Straight Line ile seçilip Set Scale (known distance 1 mm) ile belirlenmiş; ölçek değerleri kaydedilmemiş. 1 mm ≈ 17 px olduğundan 1 px işaretleme sapması ≈ %6 ölçek hatası demektir — görüntü bazlı oran dağılımının bir kısmı bu gürültüdür; raporda böyle yorumla (sayıyla). Prob, ölçülen her fotoğrafta vardır (farklı prob tipleri olabilir; işaret aralığı hep 1 mm).

**Ek görev — prob kalibrasyon aracı:** `gummy_smile_v3/tools/probe_click_calibration.py`. OpenCV penceresi; görüntü listesini sırayla açar, kullanıcı probun üzerinde **bilinen uzunluktaki en uzun düz aralığın** iki ucuna tıklar (varsayılan 5 mm; komut satırından `--span-mm`), `Enter` kaydeder, `s` atlar, `u` geri alır. Çıktı `probe_calibration.csv`: `image, x1, y1, x2, y2, span_mm, px_per_mm, note`. Zoom (fare tekerleği) ve tıklanan noktayı büyüteçte gösterme şart — 5 mm ≈ 85 px, 1 px hassasiyet gerekiyor. Harness bu dosyayı `--probe-calibration` ile alır ve raporlar: prob-türetimli `px_per_mm` dağılımı (ortalama, SD, CV, görüntü boyutuna göre); global ölçekle fark; prob ölçeğiyle hesaplanan mm'nin ImageJ ile MAE/RMSE'si; ImageJ'nin 1 mm'lik kalibrasyonunun ima ettiği hassasiyet. Aracı yaz ve sentetik bir görüntüde test et; tıklamaları kullanıcı yapacak.

### 4.5 Tahminci seçimi

Kombinasyonlar: bölgeleme {A, B, C} × tahminci {p10, p25, median, min} × {dişeti kalınlığı, dudak-ankrajlı}. Her biri için:

- MAE, RMSE (mm); Pearson r; ICC(2,1) — pingouin mevcut
- Bland–Altman: bias, %95 uyum sınırları
- Klinik eşik uyumu: her görüntünün mm değerine Tablo 1 uygulanınca çıkan E etiketi ile Excel'deki görüntü-ortalaması etiketi arasındaki uyum (bu bir *ölçüm* doğrulamasıdır, klinik validasyon değildir — raporda böyle adlandır)

**Birincil sonuç noktası: görüntü düzeyi ortalama** (E/T sınıflaması bunu kullanır). Diş bazlı (bölge 1–6) hata ikincil; A/B/C hizalama farkları yüzünden diş bazlı eşleşme belirsiz olabilir, bunu da raporla.

**Duyarlılık analizleri:** (i) `dash_zero` içeren satırlar dâhil / hariç; (ii) yalnızca 2698×1799 boyutlu görüntüler; (iii) `ambiguous_100_999` hücreleri hariç.

Seçim: görüntülerin rastgele **%60 geliştirme** alt kümesinde yap, **%40 holdout**'ta raporla; `seed=42`; listeleri diske yaz. Seçim ile raporlamayı aynı veride yapma.

Ön analizde (A yöntemi, dişeti kalınlığı, aynı-veri ölçek, n = 148) p25 için r ≈ 0.83, MAE ≈ 0.63 mm, RMSE ≈ 0.85 mm, ölçek ≈ 17 px/mm bulunmuştur. Bu **hedef değil, makullük kontrolüdür**; kendi sonuçlarını bağımsız üret.

### 4.6 Çıktılar — `results/oracle_validation/`

- `gt_overlay_examples.png` — 4–6 GT örneği (dişeti kırmızı, dudak mavi; high ve normal'den)
- `parse_report.md` — hücre türü sayıları, satır sayıları, yinelenenler, `label_inconsistent`, eşleşme oranı, yaş/cinsiyet doluluğu
- `duplicates.csv`, `unmatched_manual.csv`, `name_collisions_coco.csv`
- `boundary_check.csv` — görüntü bazlı `gap` medyan/IQR
- `estimator_comparison.csv` — tüm kombinasyonlar, dev ve holdout ayrı
- `scale_estimation.md`
- `bland_altman.png`, `scatter_gt_vs_manual.png`
- `qc_flags.csv`
- `pipeline_block_diagram.md` — Mermaid; görüntü → segmentasyon → sınıf-ayrımlı maskeler → kalınlık profili → bölgeleme → mm → kural motoru → rapor
- `intra_observer.md` — `data/manual_measurements/calibration.xlsx` (sayfalar `İlk Ölçümler` / `İkinci Ölçümler`, 20 görüntü × 6 diş, aynı ×1000 kodlaması) üzerinden ICC(2,1) ve ICC(3,1) + %95 GA (diş düzeyi n = 120 ve görüntü ortalaması n = 20), ortalama fark, SD, %95 uyum sınırları, Bland–Altman. Beklenen büyüklük: ICC ≈ 0.995 / 0.998, SD ≈ 0.17 mm. Eski `calibration-first/last.xlsx` (15 görüntü) artık kullanılmaz; `evaluation/intra_observer.py` bu dosyaya göre yeniden yazılır ve t-testi "uyum ölçütü" olarak sunulmaz.
- `dataset_manifest.csv`, `manifest_summary.md` (4.8)
- `oracle_summary.md` — bulgular, **yapılan tüm varsayımlar**, en iyi tahmincinin holdout MAE/RMSE/ICC değerleri; ayrıca referansın kendi tekrarlanabilirliği (SD ≈ 0.17 mm) ile karşılaştırma — GT-maske MAE'sinin gözlemci gürültüsünün çok üstünde olan kısmı tahminci/ölçek/bölge hizalama kaynaklıdır

### 4.7 v1 (XGBoost) hakkında not — dokunma ama raporla

`methods/v1` regresörü, önceki çalışmada 512×512 DeepLabV3+ **dişeti-yalnız** maskelerinin sütun bazlı piksel sayılarıyla eğitilmiştir. `master_pipeline_v3.py` ona **dudak+dişeti birleşik maskeyi 1024 ölçeğinde** veriyor; özellikler eğitim dağılımının tamamen dışında. Kodu değiştirme, ama `oracle_summary.md`'de bunu belirt ve v1 kolonunun Figure 6'da anlamlı olmadığını not et.

---

### 4.8 Veri seti temizleme spesifikasyonu (klinik ekibin 6 Eylül kararları)

Yeni betik: `gummy_smile_v3/data/build_clean_manifest.py`. **Hiçbir dosyayı silme veya taşıma**; yalnızca `data/dataset_manifest.csv` üret (`image, group, orig_split, keep, drop_reason, patient_id, has_reference_measurement, age, sex, source_sheet`). Yeniden bölme ve eğitim ayrı görevdir.

1. **Ölçümsüz yüksek gülme hattı görüntüleri (66; liste `olcumsuz_yuksek_gulme_hatti_66.csv`)** — iki mod, `--unmeasured-high {train_only,drop}`; varsayılan **`train_only`** (power/örneklem gerekçesiyle klinik ekibe önerildi, onay bekleniyor):
   - `train_only`: `keep=True, split_constraint=train_only`. 37'sinin ölçümlü ikizi vardır; ikiz **aynı `patient_id`'ye** bağlanır ve bölmede birlikte hareket eder (ölçümlü ikiz de zorunlu olarak train'e düşer; sızıntı yok). `has_reference_measurement=False`.
   - `drop`: `keep=False, drop_reason=no_reference_measurement` (klinik ekibin ilk tercihi).
2. **Kalan 18 çiftte tek görüntü tut** (klinik ekip doğruladı: morlar hariç 55 çiftin hepsi aynı hasta): 14 normal/low + 4 high. Tutulacak: yaş/cinsiyet kaydı olan; ikisinde de varsa/yoksa `image_a`. High'daki 4 çift (`IMG_3684-`/`IMG_3682`, `IMG_3858-`/`IMG_3854-`, `IMG_2456`/`IMG_24555`, `IMG_2544-`/`IMG_2550`): yaş/cinsiyeti dolu olan satırın görüntüsü tutulur (klinik ekip onayladı). Çıkarılan → `keep=False, drop_reason=duplicate_of:<image>`.
3. `IMG_78701` = `IMG_7870` birebir kopya → birini çıkar.
4. Demografi: `high` için yüksek sayfadaki YAŞ/CİNSİYET (+ önekten yaş), `low`/`normal` için ilgili sayfalar. Kapsamı raporla (beklenen: high 108/150 yaş, 104/150 cinsiyet; low 99/303; normal 181/796).
5. Sonuç sayılarını `manifest_summary.md`'ye yaz: grup bazında başlangıç / çıkarılan / kalan / train_only (beklenen `train_only` modunda: ≈ 216 high [150 ölçümlü + 66 train-only], ≈ 300 low, ≈ 785 normal; toplam ≈ 1.295). Bu sayılar makaledeki 1.315'in yerine geçecek; kesin rakamı sen üreteceksin.

Yeniden bölme ve eğitim (ayrı görev, burada yapma; tasarımı şimdiden not et):
- Segmentasyon metrikleri: hasta düzeyi (= temizlik sonrası görüntü düzeyi, ikizler tek grup) tabakalı bölünme; `train_only` kısıtı; **sabit test seti, bir kez** raporlanır.
- Milimetrik doğrulama (tahmin maskeleri üzerinde): sabit test setinde yalnızca ~20 ölçümlü high görüntü kalır, MAE/ICC için az. Bu yüzden ölçümlü 150 görüntü için **hasta-gruplu 5-kat çapraz doğrulama** ile out-of-fold tahmin maskeleri üretilir (her katta, held-out görüntünün train-only ikizi eğitimden çıkarılır) → n = 150 üzerinde MAE/RMSE/ICC/Bland–Altman. Makalede iki analiz ayrı ayrı tanımlanır.

## 5. Testler — `gummy_smile_v3/tests/test_gingival_display.py`

Cevabı elle bilinen sentetik maskelerle:

1. Sabit yükseklikte dikdörtgen bant → ölçüm tam olarak o yükseklik.
2. Festonlu bant (bilinen zenith ve papilla yükseklikleri) → `p10`/`min` zenith'e yakın, `max` papillaya yakın; C yöntemi 6 zenith'i doğru x'lerde bulmalı.
3. İki parçaya bölünmüş dişeti → her iki parça dâhil (eski `max(contour)` davranışı başarısız olur).
4. Dişeti + dudak birlikte verildiğinde `gingival_display_px` **yalnızca dişetinden** gelmeli; dudak maskesinin varlığı değeri değiştirmemeli. **Regresyonun kalbi.**
5. Delikli maske → en uzun kesintisiz aralık, `y_max − y_min` değil.
6. **Boş sütunlar:** yalnızca papilla üçgenleri içeren maske (normal gülme hattı taklidi) → zenith bölgelerinde değer 0, görüntü ortalaması papilla yüksekliğinin çok altında.
7. Dudak-ankrajlı tahminci: dişeti üst kenarı ile dudak alt kenarı arasında bilinen `g` px boşluk → `d = t + g`.
8. Birim: `px_per_mm = 10`, 50 px → 5.0 mm; `px_per_mm = None` → `mm is None`, `unit == "px"`.
9. Maske/görüntü boyut uyuşmazlığı → açık hata.
10. Kural motoru: `NaN` → `UNCLASSIFIED`; `0.0` → `NO_VISIBLE_GINGIVA` (sınıf yok); `0.5` → `E1`; 3.5 → `E1-E2` (ambiguous); 5.0 → `E2-E3`; 4.0 → `E2-E3`; 8.0 → `E3`; 8.1 → `E4`; 2.0 → `E1`; `use_metadata=false` iken metaveri sonucu değiştirmemeli.
11. Excel ayrıştırıcı: `3207 → 3.207`, `'*0.718' → 0.718`, `'-' → 0.0 (dash_zero)`, `'-(mesafe yok)' → 0.0`, `'3.00' → 3.0`, `-684 → NaN`, `712 → 0.712 (ambiguous)`.
12. Ad normalizasyonu: `'IMG_2544.'` ↔ `'IMG_2544-_jpeg.rf.abc.jpeg'` eşleşir; `'ımg_2544'` ↔ aynı COCO adı **eşleşmez**; `'25-IMG_4552'` → `img4552` + `prefix_stripped=True`.

---

## 6. Çalışma yöntemi

1. Önce **hiçbir şey değiştirmeden** mevcut davranışı belgele: 5 COCO görüntüsünde (high ve normal'den) GT maskeleri eski `measure_gum_visibility` ile ölç, çıktıyı kaydet. Yeni kodla farkı göstereceğiz.
2. GT overlay görsellerini üret ve bak; Bölüm 0'daki anatomiyle uyuşmuyorsa **dur ve raporla**.
3. Yeni modülleri **yanına** yaz; eski dosyaları silme, `DEPRECATED` notu ekle.
4. Küçük ve sık commit; her mesajda ne ve neden.
5. Varsayımları kodda değil `oracle_summary.md`'de listele.
6. Beklenmedik bir şey bulursan (sınır boşluğu büyük, ölçek 15–20 px/mm'den çok uzak, 692 yerine çok daha az tam satır, eşleşme ~147'den çok düşük) **devam etme, dur ve raporla.**

---

## 7. Kabul ölçütleri

- [ ] Dişeti ve dudak maskeleri ayrı üretiliyor; hiçbir yerde sınıflar birleştirilmiyor
- [ ] Maskeler orijinal görüntü çözünürlüğünde; boyut assert'i var
- [ ] Ölçüm dişeti maskesinin dikey kalınlığından geliyor; boş sütunlar 0; dudak yalnızca denetim ve dudak-ankrajlı tahminci için
- [ ] Aynı sınıfın tüm instance'ları birleştiriliyor; "en büyük kontur" davranışı kaldırıldı
- [ ] `mm_per_pixel` kaldırıldı; yalnızca `px_per_mm`; kalibrasyon yokken çıktı `px` etiketli
- [ ] Tek kural motoru; E1–E2 ve E2–E3 çakışmaları birleşik etiketle raporlanıyor; `NaN` → `UNCLASSIFIED`; `0` → `NO_VISIBLE_GINGIVA`; metaveri devre dışı; tedaviler "alternatives" olarak adlandırılmış
- [ ] Ad normalizasyonu sondaki nokta/`-` ekini koruyor
- [ ] Prob kalibrasyon aracı çalışıyor ve `--probe-calibration` harness'ta işleniyor
- [ ] Bölüm 5'teki on iki test geçiyor
- [ ] Excel ayrıştırıcı ≥ 690 tam satır buluyor; hücre türü sayıları `parse_report.md`'de
- [ ] `dataset_manifest.csv` ve `manifest_summary.md` üretilmiş; 66 ölçümsüz high için `train_only` kısıtı ve ikiz gruplaması, 18 çift için tek görüntü kuralı uygulanmış
- [ ] Gözlemci içi analiz `calibration.xlsx` (20 görüntü) ile yeniden hesaplanmış
- [ ] Oracle doğrulaması `high` grubunda ≥ 140 görüntüde çalışıyor; dev/holdout ayrımı diskte
- [ ] `oracle_summary.md` en iyi tahmincinin holdout MAE/RMSE/ICC değerlerini mm cinsinden veriyor
- [ ] Sınır denetimi (dudak altı ↔ dişeti üstü) sayısal olarak raporlanmış
- [ ] Blok diyagramı üretilmiş
- [ ] `imgsz` 640, `max_det` ≥ 20, `retina_masks` açık

---

## 8. Bilinen tuzaklar

- `data/labels_smileline.csv`, `data/splits.json`, `results/*.csv` **boştur**; gülme hattı sınıfını COCO dizin yapısından al.
- `yolo/weights/best.pt` depoda **yoktur**. Bu görev tamamen GT maskelerle yapılır; ağırlık gerektiren adım kapsam dışıdır.
- `IMG_xxxx` adları benzersiz değildir (cihazlar arası çakışma). Dosya adı üzerinden "aynı görüntü" iddiası yapma; gerekirse piksel karşılaştır.
- `IMG_6849.jpg` / `IMG_6849..jpg` **farklı görüntülerdir** (klinik ekip doğruladı). Excel'de sondaki `.`, COCO'da sondaki `-` olarak görünür; normalizasyon bunları **birleştirmemeli**, `~` işaretiyle ayrı tutmalıdır (4.2).
- Excel: Türkçe `ı`; `.lower()` tek başına yetmez. `-` hücrelerinin boşluklu varyantları (`'               -'`) vardır; `strip()` sonrası karşılaştır.
- T kolonunda bir hücre yanlışlıkla `E1` yazılmış; T etiketlerini analizde kullanmıyorsan sadece raporla.
- COCO kategori `0` supercategory'dir.
- Ultralytics `masks.data` çıkarım çözünürlüğündedir; `retina_masks=True` olmadan orijinale bindirme.
- Aynı hastanın problu/probsuz iki fotoğrafı farklı bölüntülerde olabilir (doğrulanmış: 55 çift, 22'si çapraz bölüntü, 15'i test setinde; `ayni_hasta_aday_ciftler.csv`). Oracle analizini etkilemez ama raporda not düş; hasta düzeyinde bölünme ve yeniden eğitim ayrı görevdir.
