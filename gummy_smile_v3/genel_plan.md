+# 1) GummySmile v3 – Detaylı Uçtan Uca İş Planı (Güncel)
 
 ## 1.1 Hedef ve Kabul Kriterleri
 
 ### Nihai hedef
 
+* **V1 (XGBoost)** ve **V3 (YOLOv11x-seg + measurement)** yaklaşımlarını **tek bir master pipeline** altında çalıştırmak.
+* V3 ile (inference-only):
 

+  * YOLOv11x-seg inference (hazır `best.pt` ile)
   * YOLO maskeleri üzerinden otomatik ölçüm
   * Manuel ölçümlerle karşılaştırma (MAE/RMSE/ICC, Bland–Altman)
+  * V1/V3 performans karşılaştırma tabloları
+  * Dişeti görünürlüğüne göre etiyoloji + tedavi önerisi (E1–E4 / T1–T4)
 * **Intra-observer** tutarlılığı (15–20 gün arayla iki ölçüm) analizi v3’e entegre ve raporlanır.
 
 ### Kabul kriterleri (hakeme gösterilecek çıktılar)
 
 1. **Intra-observer**: ICC (absolute agreement) + Bland–Altman grafiği + tablo (mean diff, limits).
+2. **Segmentation**: YOLOv11x-seg için IoU/Dice (GT mask varsa).
 3. **Measurement**: Manuel ölçümlere göre MAE/RMSE + ICC (method vs manual).
+4. **Method comparison**: V1 vs V3 tek tabloda.
 5. Tek komutla çalışabilen master:
 
    * `python gummy_smile_v3/master_pipeline_v3.py --config gummy_smile_v3/configs/config.yaml`
 6. Sonuçların kaydı:
 
+   * `gummy_smile_v3/results/` altında CSV + log.
 
 ---
 
 ## 1.2 Repo Referansları (Codex’in “gerçek kodu” okuyacağı yerler)
 
 Codex’in incelemesi gereken kaynak kodlar (senin verdiğin path’ler):
 
 * **V1 kaynakları**:
   `gummy_smile_guncel/`
   (repo: `https://github.com/rfatihors/GaziDental/tree/master/gummy_smile_guncel`)
 
 * **Intra-observer kaynakları**:
   `intra_observer/`
   (repo: `https://github.com/rfatihors/GaziDental/tree/master/intra_observer`)
 
 V3’te bu kodları **kopyala-yapıştır değil**, “referans alıp yeniden organize ederek” kullanılacak şekilde kuracağız (temiz mimari).
 
 ---
 
 ## 1.3 V3 Klasör Yapısı ve Modüller (Detay)
 
 `gummy_smile_v3/` altında:
 
 ### A) Konfigürasyon
 
 * `configs/config.yaml`
 

+  * data paths (raw_images, excel paths)
+  * yolo inference parametreleri (imgsz, conf, iou, max_det)
+  * measurement parametreleri (ön diş sayısı=6, pixel→mm kalibrasyon vb.)
+  * comparison çıktıları ve diagnosis raporu
 
 ### B) Veri ve anotasyon
 
 * `data/raw_images/` : tüm fotoğraflar (orijinal)
+* `data/raw_annotations/` : (opsiyonel, mevcut değilse kullanılmaz)
+* `annotation_tools/` : (opsiyonel; v3 inference-only akışta zorunlu değil)
 
 ### C) Yöntemler (v1/v2/v3 ölçüm)
 
 * `methods/v1/`:
 

+  * `xgboost_predictions.csv` ile tahmin sonuçları okunur.
 * `methods/v3/`:
 

+  * `diagnosis.py` : E1–E4 / T1–T4 etiyoloji + tedavi önerisi
 
 ### D) Eğitim / inference (YOLO)
 

 * `yolo/infer_yolo_seg.py` : batch inference + mask export

+* `measurement/yolo_measurements.py` : YOLO mask üzerinden ölçüm
 
 ### E) Manuel ölçüm okuma + kalibrasyon
 

+* `measurement/manual_loader.py` : (opsiyonel; manuel CSV üzerinden okunuyor)
+* `measurement/calibration.py` : (opsiyonel; şimdilik sabit mm/px kullanılıyor)
 
 ### F) Değerlendirme
 
 * `evaluation/intra_observer.py` : mevcut intra_observer kodunu v3 standardına taşır (CLI + sonuç kaydı)

+* `evaluation/segmentation_metrics.py` : (opsiyonel) Dice/IoU (GT varsa)
+* `evaluation/measurement_metrics.py` : MAE/RMSE/ICC/Bland–Altman
+* `evaluation/method_comparison.py` : v1/v3 tablo ve özet
 
 ### G) Master orchestration
 
 * `master_pipeline_v3.py`
 
   * adım adım:
 

+    1. yolo infer
+    2. v3 measurement
+    3. v1 vs v3 comparison
+    4. diagnosis recommendation
+    5. intra-observer
 
 ---
 
 ## 1.4 İş Kalemleri (Sıralı ve Detaylı)
 

+### FAZ 1 — V3 “temiz başlangıç” ve veri standardizasyonu (opsiyonel)
 
 1. `gummy_smile_v3/` klasörünü oluştur.
 2. Config sistemi (yaml + argparse) kur.
+3. `dataset_audit.py` yaz (opsiyonel):
 
    * raw_images altında tüm image’ları listele
    * raw_annotations JSON’larını listele
    * JSON içinden `file_name`’leri çıkar
    * eksik image, duplicate image, boş annotation vb. raporla
    * `results/audit_report.csv` üret
 
 **Çıktı:** audit raporu + log
 
 ---
 
+### FAZ 2 — (Opsiyonel) makesense JSON → YOLO-seg dönüşümü (tek sınıf: diseti)
 
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
 
+### FAZ 3 — (Opsiyonel) Split (train/val/test)
 
 1. `split_dataset.py` (veya master içinde fonksiyon):
 
    * 70/15/15 split
    * Stratified split: sınıf dağılımı (düşük/normal/yüksek) korunacaksa:
 
      * sınıf etiketleri nereden geliyor?
 
        * Eğer gülme hattı etiketi Excel/CSV’de varsa oradan,
        * yoksa dosya adlandırma/klasör bilgisi varsa oradan.
 2. `splits.json` kaydet (reproducible)
 
 **Çıktı:** train/val/test klasörleri dolu + splits.json
 
 ---
 

+### FAZ 4 — (Kaldırıldı) YOLO eğitimi
 

+Eğitim yapılmayacak. `best.pt` hazır olarak sağlanacak ve sadece inference yapılacaktır.
 
 ---
 

+## 1.5 Güncel Durum Özeti (Yapılanlar)
 

+* Master pipeline ve config güncellendi (inference-only).
+* YOLO inference + measurement çıktıları üretiliyor.
+* V1 vs V3 karşılaştırma metrikleri hesaplanıyor (MAE/RMSE/ICC/Bland–Altman).
+* Intra-observer analizi entegre edildi.
+* Etiyoloji + tedavi önerisi (E1–E4 / T1–T4) CSV çıktısı eklenmiş durumda.
 

+## 1.6 Eksik / Opsiyonel Kalanlar
 

+* Segmentation metrikleri (IoU/Dice) için `evaluation/segmentation_metrics.py` henüz yok.
+* Görsel overlay çıktıları (mask overlay) henüz yazılmadı.
+* `measurement/manual_loader.py` ve `measurement/calibration.py` opsiyonel modülleri mevcut değil.
 
 ---
 

+### FAZ 5 — YOLO inference ve ölçüm (Mevcut akış)
 

+1. `yolo/infer_yolo_seg.py`:
 

+   * `data/raw_images/` üstünde çalıştırılır.
+   * her image için:
 

+     * predicted mask (png)
+     * `status=ok` veya `status=no_mask`
+   * `results/yolo_predictions/` altına kaydedilir.
+2. `measurement` çıktısı:
 
-**Önemli:** Üç runner da aynı kolon setini üretmeli:
+   * `results/yolo_measurements.csv` (mean_mm + status)
 

+**Çıktı:** predicted masks + ölçüm CSV.
 
 ---
 

+### FAZ 6 — V1/V3 comparison + diagnosis
 

+* `evaluation/method_comparison.py`:
 

+  * v1 vs manual
+  * v3 vs manual
+  * v1 vs v3
+* `methods/v3/diagnosis.py`:
+
+  * mean_mm → E1–E4 / T1–T4 önerisi
+  * `results/diagnosis_recommendations.csv`
 
 ---
 
+### FAZ 7 — Intra-observer (v3 içine entegre)
 
 * `evaluation/intra_observer.py`:
 
   * repo’daki intra_observer kodunu baz al

+    * `results/intra_observer_report.csv`
 
 ---
 

+* Segmentation metrikleri (IoU/Dice) için `evaluation/segmentation_metrics.py`
+* Mask overlay görselleri (predicted mask + image)
+* `measurement/manual_loader.py` ve `measurement/calibration.py` ile Excel tabanlı akış
 
 
 ---
