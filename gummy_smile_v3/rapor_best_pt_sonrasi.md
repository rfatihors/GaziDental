### X.X. Eğitilmiş YOLOv11x-seg Modelinin Proje Kapsamında Kullanımı

Aşağıda, YOLOv11x-seg modeli eğitilmiş ve `best.pt` elde edilmiştir noktasından sonra yürütülen süreç aktarılmıştır. Eğitilmiş ağırlık dosyası, tekil görüntü üzerinde çıkarım yapmak üzere çağrılmış ve eğitim sürecinden tamamen ayrıştırılarak karar destek hattının girişine alınmıştır. Bu kapsamda `master_pipeline_v3.py` içerisinde ağırlık yolu doğrulanmış, `run_yolo_segmentation` fonksiyonuna geçirilmiş ve model yalnızca inference amaçlı çalıştırılmıştır. Böylece `best.pt` ağırlıkları, ölçüm ve sınıflandırma zincirinin üretim aşamasında kullanılan sabit bir bileşen hâline getirilmiştir.【F:gummy_smile_v3/master_pipeline_v3.py†L92-L179】【F:gummy_smile_v3/yolo/infer_yolo.py†L44-L112】

### X.X. Segmentasyon Çıktılarının Üretilmesi

Tekil görüntü girdisiyle inference süreci başlatılmış; YOLOv11x-seg çıktıları üzerinden dişeti ve ilgili anatomik alanlara ait piksel düzeyinde segmentasyon maskesi elde edilmiştir. Üretilen maskeler, hem ham maske çıktısı olarak diskte saklanmış hem de özgün görüntü üzerine yarı saydam bindirme ile görselleştirilmiştir. Bu ara çıktılar, sonraki ölçüm modülüne doğrudan veri sağlayan temel katman olarak kullanılmıştır.【F:gummy_smile_v3/yolo/infer_yolo.py†L13-L112】【F:gummy_smile_v3/master_pipeline_v3.py†L148-L173】

### X.X. Segmentasyon Maskeleri Üzerinden Dişeti Görünürlüğü Ölçümü

Segmentasyon maskeleri üzerinden dişeti görünürlüğü ölçümü, maske konturunun çıkarılması ve üst dudak hattının en üst piksel koordinatına göre referans alınması ile gerçekleştirilmiştir. Maske alanı yatayda altı bölgeye ayrılmış, her bölgede gingival zenit noktaları belirlenmiş ve bu noktalardan dudak hattına dikey piksel mesafesi hesaplanmıştır. Bölgesel değerlerin ortalaması alınarak görüntüye ait ortalama dişeti görünürlüğü piksel cinsinden elde edilmiştir. Kalibrasyon parametresi (`px_per_mm`) sağlandığında bu değer mm birimine dönüştürülmüş, parametre olmadığında ölçüm yalnızca piksel cinsinden raporlanmıştır. Bu ölçüm yaklaşımı, önceki piksel tabanlı yöntemlere kıyasla maske temelli anatomik sınırların daha doğrudan kullanılmasına olanak tanımış ve ölçümün tutarlılığını artırmıştır.【F:gummy_smile_v3/measurement/measure_gum_visibility.py†L18-L92】【F:gummy_smile_v3/master_pipeline_v3.py†L164-L205】

### X.X. Klinik Eşiklere Dayalı Sınıflandırma (E1–E4)

Ölçülen dişeti görünürlüğü değeri mm cinsine çevrildikten sonra klinik eşiklere göre E1–E4 sınıflarına atanmıştır. Eşikler `E1: <4 mm`, `E2: 3–6 mm`, `E3: 4–8 mm`, `E4: >8 mm` olarak tanımlanmış ve bu aralıklar klinik referans tablosu ile uyumlu şekilde uygulanmıştır. E2 ve E3 aralıklarının 4–6 mm bandında çakışması durumunda, kural motoru önce olası her iki sınıfı aday olarak işaretlemiş; ardından konfigürasyondaki belirsizlik politikası ve varsa metaveri eşlemesi ile karar verilmiştir. Metaveri kullanılmadığında varsayılan seçim uygulanmış ve belirsizlik durumu raporlamaya yansıtılmıştır.【F:gummy_smile_v3/methods/v3/rule_engine.py†L21-L136】【F:gummy_smile_v3/master_pipeline_v3.py†L190-L212】【F:gummy_smile_v3/data/etiyoloji_araliklari.jpeg†L1】

### X.X. Tedavi Öneri Mekanizması

E1–E4 sınıfları, sistemde tanımlı etiyoloji ve tedavi önerileriyle eşleştirilmiş, bu eşleştirme makine öğrenmesi yerine kural tabanlı ve açıklanabilir bir yapı üzerinden gerçekleştirilmiştir. Her sınıf için etiyolojik adaylar ve tedavi listeleri sabit kurallar olarak tanımlanmış; sonuç raporuna tedavi sınıfı (T1–T4) ile birlikte öneri listeleri yazdırılmıştır. Böylece karar destek çıktısı, klinik açıdan yorumlanabilir ve izlenebilir biçimde raporlanmıştır.【F:gummy_smile_v3/methods/v3/rule_engine.py†L21-L149】【F:gummy_smile_v3/master_pipeline_v3.py†L63-L118】

### X.X. v1 ve v3 Yaklaşımlarının Karşılaştırmalı Kullanımı

v1 yaklaşımında, segmentasyon maskesi üzerinden elde edilen siyah–beyaz piksel özellikleri çıkarılmış ve XGBoost regresörü ile ortalama dişeti görünürlüğü mm cinsinden tahmin edilmiştir. v3 yaklaşımında ise YOLOv11x-seg maskeleri üzerinden doğrudan ölçüm yapılmış ve aynı rapor formatında etiyoloji ile tedavi sınıfı üretilmiştir. Her iki yöntem, aynı görüntü girdiği üzerinden çalıştırılmış; sonuçlar aynı rapor şeması altında birleştirilerek karşılaştırmaya uygun biçimde kaydedilmiştir. v3 yaklaşımı, maskeye dayalı doğrudan ölçüm ve görselleştirilebilir çıktı üretimi sayesinde daha izlenebilir bir teknik kazanım sunmuştur.【F:gummy_smile_v3/methods/v1/xgboost_runner.py†L22-L79】【F:gummy_smile_v3/measurement/measure_gum_visibility.py†L52-L92】【F:gummy_smile_v3/master_pipeline_v3.py†L164-L259】

### X.X. Genel Değerlendirme

Aşağıda sunulan yaklaşımda segmentasyon tabanlı ölçüm hattı, dişeti görünürlüğü hesaplamasını otomatikleştirmiş, ölçümün bölgesel dağılımını kontrol edilebilir hâle getirmiş ve klinik eşiklere dayalı karar destek çıktısının üretimini sistematikleştirmiştir. Maskelerin ara çıktı olarak kaydedilmesi, ölçüm adımlarının izlenmesini ve gerektiğinde görsel doğrulamayı mümkün kılmıştır. Bu nedenle GummySmile v3 içinde YOLOv11x-seg temelli v3 hattı, klinik karar destek açısından daha tutarlı ve açıklanabilir bir nihai çözüm olarak konumlandırılmıştır.【F:gummy_smile_v3/yolo/infer_yolo.py†L13-L112】【F:gummy_smile_v3/measurement/measure_gum_visibility.py†L52-L92】【F:gummy_smile_v3/master_pipeline_v3.py†L148-L259】
