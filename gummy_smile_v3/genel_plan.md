# 1) GummySmile v3 – Detaylı Uçtan Uca İş Planı

## 1.1 Hedef ve Kabul Kriterleri

### Nihai hedef

* **V1 (heuristic)**, **V2 (geometry + tabular ML)**, **V3 (YOLOv8-seg + measurement)** yaklaşımlarını **tek bir master pipeline** altında çalıştırmak.
* V3 ile:

  * makesense JSON → YOLOv8-seg dataset
  * YOLO eğitimi + inference
  * YOLO maskeleri üzerinden otomatik ölçüm
  * Manuel ölçümlerle karşılaştırma (MAE/RMSE/ICC, Bland–Altman)
  * V1/V2/V3 performans karşılaştırma tabloları
* **Intra-observer** tutarlılığı (15–20 gün arayla iki ölçüm) analizi v3’e entegre ve raporlanır.

### Kabul kriterleri (hakeme gösterilecek çıktılar)

1. **Intra-observer**: ICC (absolute agreement) + Bland–Altman grafiği + tablo (mean diff, limits).
2. **Segmentation**: YOLOv8-seg için IoU/Dice (GT mask varsa).
3. **Measurement**: Manuel ölçümlere göre MAE/RMSE + ICC (method vs manual).
4. **Method comparison**: V1 vs V2 vs V3 tek tabloda.
5. Tek komutla çalışabilen master:

   * `python gummy_smile_v3/master_pipeline_v3.py --config gummy_smile_v3/configs/config.yaml`
6. Sonuçların kaydı:

   * `gummy_smile_v3/results/` altında CSV + görsel overlay + log.

---

## 1.2 Repo Referansları (Codex’in “gerçek kodu” okuyacağı yerler)

Codex’in incelemesi gereken kaynak kodlar (senin verdiğin path’ler):

* **V1 kaynakları**:
  `gummy_smile_guncel/`
  (repo: `https://github.com/rfatihors/GaziDental/tree/master/gummy_smile_guncel`)

* **V2 kaynakları**:
  `gummy_smile_guncel/ai_pipeline_v2/`
  (repo: `https://github.com/rfatihors/GaziDental/tree/master/gummy_smile_guncel/ai_pipeline_v2`)

* **Intra-observer kaynakları**:
  `intra_observer/`
  (repo: `https://github.com/rfatihors/GaziDental/tree/master/intra_observer`)

V3’te bu kodları **kopyala-yapıştır değil**, “referans alıp yeniden organize ederek” kullanılacak şekilde kuracağız (temiz mimari).

---

## 1.3 V3 Klasör Yapısı ve Modüller (Detay)

`gummy_smile_v3/` altında:

### A) Konfigürasyon

* `configs/config.yaml`

  * data paths (raw_images, raw_annotations, excel paths)
  * split oranları (train/val/test)
  * yolo parametreleri (model size, epochs, imgsz, batch)
  * measurement parametreleri (ön diş sayısı=6 üst/6 alt, pixel→mm kalibrasyon vb.)
  * evaluation seçenekleri (hangi metrikler, bootstrap CI, vs.)

### B) Veri ve anotasyon

* `data/raw_images/` : tüm fotoğraflar (orijinal)
* `data/raw_annotations/` : makesense JSON’lar
* `data/yolo_dataset/` : YOLO formatlı dataset
* `annotation_tools/makesense_to_yolo_seg.py` : JSON→YOLO-seg converter
* `annotation_tools/dataset_audit.py` : veri bütünlüğü kontrolü (image var mı, label var mı, duplicate var mı)

### C) Yöntemler (v1/v2/v3 ölçüm)

* `methods/v1/`:

  * `measurement_v1.py` : v1’deki heuristik ölçüm fonksiyonları (v1 kaynaktan refactor)
  * `runner_v1.py` : batch çalıştırıp sonuç csv üretir
* `methods/v2/`:

  * `measurement_v2.py` : v2’deki auto_measurement ve feature çıkarımı
  * `ml_v2.py` : xgboost/lightgbm eğitimi (mevcut v2 kodundan birebir davranışı koruyarak)
  * `runner_v2.py`
* `methods/v3/`:

  * `yolo_infer.py` : YOLO mask üretimi (batch)
  * `measurement_v3.py` : YOLO mask üzerinden ölçüm (v2 ölçüm mantığını reuse/refactor)
  * `runner_v3.py`

### D) Eğitim / inference (YOLO)

* `yolo/train_yolo_seg.py` : YOLOv8-seg eğitim script’i
* `yolo/infer_yolo_seg.py` : batch inference + mask export
* `yolo/yolo_data.yaml` : ultralytics dataset yaml (names: {0: diseti})

### E) Manuel ölçüm okuma + kalibrasyon

* `measurement/manual_loader.py` : Excel’den manuel ölçümleri standart dataframe’e çeker
* `measurement/calibration.py` : pixel→mm (metal prob boğumları 1mm; varsa otomatik tespit / yoksa semi-auto)

### F) Değerlendirme

* `evaluation/intra_observer.py` : mevcut intra_observer kodunu v3 standardına taşır (CLI + sonuç kaydı)
* `evaluation/segmentation_metrics.py` : Dice/IoU (GT varsa)
* `evaluation/measurement_metrics.py` : MAE/RMSE/ICC/Bland–Altman, bootstrap CI opsiyonel
* `evaluation/method_comparison.py` : v1/v2/v3 tablo ve özet

### G) Master orchestration

* `master_pipeline_v3.py`

  * adım adım:

    1. audit
    2. split
    3. (opsiyonel) yolo train
    4. yolo infer
    5. v1 runner
    6. v2 runner
    7. v3 runner
    8. intra-observer
    9. method comparison + rapor çıktıları

---

## 1.4 İş Kalemleri (Sıralı ve Detaylı)

### FAZ 1 — V3 “temiz başlangıç” ve veri standardizasyonu

1. `gummy_smile_v3/` klasörünü oluştur.
2. Config sistemi (yaml + argparse) kur.
3. `dataset_audit.py` yaz:

   * raw_images altında tüm image’ları listele
   * raw_annotations JSON’larını listele
   * JSON içinden `file_name`’leri çıkar
   * eksik image, duplicate image, boş annotation vb. raporla
   * `results/audit_report.csv` üret

**Çıktı:** audit raporu + log

---

### FAZ 2 — makesense JSON → YOLOv8-seg dönüşümü (tek sınıf: diseti)

1. `makesense_to_yolo_seg.py`:

   * Birden fazla JSON’u okuyacak
   * JSON içindeki her image için:

     * polygon segmentation → YOLO-seg satırı
     * class_id=0
     * normalize: x/W, y/H
   * Her image için tek `.txt` label
2. Converter, `yolo_dataset/images/all` ve `yolo_dataset/labels/all` üretreveya doğrudan train/val/test sonrası klasörlere koyacak (tercih: önce all, sonra split).

**Doğrulama:**

* label dosyası olan her image için annotation satırı var mı?
* polygon point sayıları çift mi? (x,y)
* normalize aralığı [0,1] mi?

**Çıktı:** YOLO label’ları + dönüştürme log’u

---

### FAZ 3 — Split (train/val/test)

1. `split_dataset.py` (veya master içinde fonksiyon):

   * 70/15/15 split
   * Stratified split: sınıf dağılımı (düşük/normal/yüksek) korunacaksa:

     * sınıf etiketleri nereden geliyor?

       * Eğer gülme hattı etiketi Excel/CSV’de varsa oradan,
       * yoksa dosya adlandırma/klasör bilgisi varsa oradan.
2. `splits.json` kaydet (reproducible)

**Çıktı:** train/val/test klasörleri dolu + splits.json

---

### FAZ 4 — YOLOv8-seg eğitimi (diseti)

1. `yolo_data.yaml` üret (paths + names)
2. `train_yolo_seg.py`:

   * ultralytics YOLO API
   * model: `yolov8n-seg` (başlangıç)
   * epochs, imgsz, batch config’ten
   * çıktıyı `models/trained_yolo/` altına koy
3. Eğitim raporu:

   * loss curve
   * mAP vs epoch (ultralytics verir)

**Çıktı:** best.pt + eğitim logları + sonuç grafikleri

---

### FAZ 5 — YOLO inference ve mask export

1. `infer_yolo_seg.py`:

   * test (ve/veya tüm dataset) üstünde çalıştır
   * her image için:

     * predicted mask (png veya numpy)
     * overlay görsel (image üstüne maske)
   * `results/v3/seg_predictions/` altına kaydet

**Çıktı:** predicted masks + overlay’ler

---

### FAZ 6 — Ölçüm: V1 / V2 / V3 koşuları

Bu faz, “v1/v2’nin aynı dataset üzerinde karşılaştırılabilir olması” için kritik.

#### 6.1 V1 runner

* V1 kodunu **repo’dan okuyup** refactor:

  * girdi: image (+ gerekiyorsa v1 maske)
  * çıktı: ölçüm değerleri (mm veya px→mm)
* `results/v1/measurements.csv`

#### 6.2 V2 runner

* V2 kodunu **repo’dan okuyup** refactor:

  * `auto_measurement` ve `xgboost_measurements` + `severity_model` akışını
  * aynı giriş/çıkış standardına sok
* `results/v2/measurements.csv`
* `results/v2/predictions.csv`

#### 6.3 V3 runner

* V3 ölçüm `measurement_v3.py`:

  * YOLO mask → contour → ölçüm
  * pixel→mm kalibrasyon:

    * metal prob boğumları referansı
    * mümkünse otomatik tespit, değilse semi-auto (manuel klikleme) opsiyonu
* `results/v3/measurements.csv`

**Önemli:** Üç runner da aynı kolon setini üretmeli:

* image_id / file_name
* tooth_index (üst 6 / alt 6)
* measurement_mm (veya px + mm)
* quality flags (mask missing, low conf vb.)

---

### FAZ 7 — Manuel ölçüm eşleştirme

* `manual_loader.py`:

  * “ölçümler ai guncel.xlsx” ve varsa calibration excel’lerini okuyup normalize et
  * `file_name` eşleşmesini standardize et (v1’deki dosya eşitsizlik script’lerinden ders al)
* Çıktı: `results/manual/manual_measurements.csv`

---

### FAZ 8 — Intra-observer (v3 içine entegre)

* `evaluation/intra_observer.py`:

  * repo’daki intra_observer kodunu baz al
  * v3 sonuç klasör yapısına uygun hale getir
  * output:

    * `results/intra_observer/icc.csv`
    * `results/intra_observer/bland_altman.png`
    * `results/intra_observer/summary.txt`

---

### FAZ 9 — Ölçüm metrikleri ve method comparison

* `measurement_metrics.py`:

  * v1 vs manual
  * v2 vs manual
  * v3 vs manual
  * MAE/RMSE/ICC + Bland–Altman (method-vs-manual)
* `method_comparison.py`:

  * tek tabloda v1/v2/v3 kıyas
  * ayrıca “fail case” listesi:

    * mask yok
    * düşük confidence
    * saç/sakal/ışık problemi (örnek görsel linkleri)

**Çıktı:**

* `results/comparison_tables/methods_summary.csv`
* `results/comparison_tables/methods_summary.xlsx`
* `results/comparison_tables/methods_summary.md`

---

### FAZ 10 — Master pipeline ve CLI

* `master_pipeline_v3.py`:

  * `--step` ile modüler koşabilsin (sadece convert, sadece train, sadece eval vb.)
  * default: hepsini koş
* Hata yakalama:

  * eksik dosya, boş label, inference fail vb. durumlarda log + graceful exit


---

