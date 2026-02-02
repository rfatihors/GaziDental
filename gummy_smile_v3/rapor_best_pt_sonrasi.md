### X.X. Eğitilmiş YOLOv11x-seg Modelinin Proje Kapsamında Kullanımı

Aşağıda, YOLOv11x-seg modeli eğitilmiş ve `best.pt` elde edilmiştir noktasından sonra yürütülen süreç aktarılmıştır. Eğitilmiş ağırlık dosyası, tekil görüntü üzerinde çıkarım yapmak üzere çağrılmış ve eğitim sürecinden tamamen ayrıştırılarak karar destek hattının girişine alınmıştır. Bu kapsamda ağırlık yolu doğrulanmış, segmentasyon aşamasına geçirilmiş ve model yalnızca inference amaçlı çalıştırılmıştır. Böylece `best.pt` ağırlıkları, ölçüm ve sınıflandırma zincirinin üretim aşamasında kullanılan sabit bir bileşen hâline getirilmiştir.

### X.X. Segmentasyon Çıktılarının Üretilmesi

Tekil görüntü girdisiyle inference süreci başlatılmış; YOLOv11x-seg çıktıları üzerinden dişeti ve ilgili anatomik alanlara ait piksel düzeyinde segmentasyon maskesi elde edilmiştir. Üretilen maskeler, hem ham maske çıktısı olarak diskte saklanmış hem de özgün görüntü üzerine yarı saydam bindirme ile görselleştirilmiştir. Bu ara çıktılar, sonraki ölçüm modülüne doğrudan veri sağlayan temel katman olarak kullanılmıştır.

### X.X. Segmentasyon Maskeleri Üzerinden Dişeti Görünürlüğü Ölçümü

Segmentasyon maskeleri üzerinden dişeti görünürlüğü ölçümü, maske konturunun çıkarılması ve üst dudak hattının en üst piksel koordinatına göre referans alınması ile gerçekleştirilmiştir. Bu aşamada, önceden tanımlı bölgesel örnekleme yaklaşımı çerçevesinde anatomik sınırları temsil edecek şekilde referans noktaları belirlenmiş ve bu noktalardan dudak hattına dikey piksel mesafesi hesaplanmıştır. Bölgesel değerlerin ortalaması alınarak görüntüye ait ortalama dişeti görünürlüğü piksel cinsinden elde edilmiştir. Kalibrasyon parametresi (`px_per_mm`) sağlandığında bu değer mm birimine dönüştürülmüş; parametre bulunmadığında ölçüm yalnızca piksel cinsinden raporlanmış, klinik sınıflandırma ve tedavi önerisi adımı devre dışı bırakılmış ve sistem yalnızca ön değerlendirme çıktısı üretmiştir. Bu ölçüm yaklaşımı, önceki piksel tabanlı yöntemlere kıyasla maske temelli anatomik sınırların daha doğrudan kullanılmasına olanak tanımakta ve ölçüm kararlılığını artırması beklenmektedir.

### X.X. Klinik Eşiklere Dayalı Sınıflandırma (E1–E4)

Ölçülen dişeti görünürlüğü değeri mm cinsine çevrildikten sonra klinik eşiklere göre E1–E4 sınıflarına atanmıştır. Eşikler `E1: <4 mm`, `E2: 3–6 mm`, `E3: 4–8 mm`, `E4: >8 mm` olarak tanımlanmış ve bu aralıklar klinik referans tablosu ile uyumlu şekilde uygulanmıştır. E2 ve E3 aralıklarının 4–6 mm bandında çakışması durumunda, kural motoru önce olası her iki sınıfı aday olarak işaretlemiş; ardından konfigürasyondaki belirsizlik politikası ve varsa metaveri eşlemesi ile karar verilmiştir. Metaveri kullanılmadığında varsayılan seçim uygulanmış ve belirsizlik durumu raporlamaya yansıtılmıştır.

### X.X. Tedavi Öneri Mekanizması

E1–E4 sınıfları, sistemde tanımlı etiyoloji ve tedavi önerileriyle eşleştirilmiş, bu eşleştirme makine öğrenmesi yerine kural tabanlı ve açıklanabilir bir yapı üzerinden gerçekleştirilmiştir. Her sınıf için etiyolojik adaylar ve tedavi listeleri sabit kurallar olarak tanımlanmış; sonuç raporuna tedavi sınıfı (T1–T4) ile birlikte öneri listeleri yazdırılmıştır. Böylece karar destek çıktısı, klinik açıdan yorumlanabilir ve izlenebilir biçimde raporlanmıştır.

### X.X. v1 ve v3 Yaklaşımlarının Karşılaştırmalı Kullanımı

v1 yaklaşımında, segmentasyon maskesi üzerinden elde edilen siyah–beyaz piksel özellikleri çıkarılmış ve XGBoost regresörü ile ortalama dişeti görünürlüğü mm cinsinden tahmin edilmiştir. v3 yaklaşımında ise YOLOv11x-seg maskeleri üzerinden doğrudan ölçüm yapılmış ve aynı rapor formatında etiyoloji ile tedavi sınıfı üretilmiştir. Her iki yöntem, aynı görüntü girdiği üzerinden çalıştırılmış; sonuçlar aynı rapor şeması altında birleştirilerek karşılaştırmaya uygun biçimde kaydedilmiştir. v3 yaklaşımı, maskeye dayalı doğrudan ölçüm ve görselleştirilebilir çıktı üretimi sayesinde daha izlenebilir ve açıklanabilir bir ölçüm altyapısı sunmakta ve önceki yaklaşımlara kıyasla teknik avantaj sağlamaktadır.

### 5.X. Projenin Bu Döneminde Kullanılan Model ve Teknolojiler

Bu dönemde dişeti görünürlüğünün değerlendirilmesi için segmentasyon tabanlı yaklaşım tercih edilmiş; bunun nedeni, anatomik sınırların piksel düzeyinde ayrıştırılmasına olanak tanıması ve ölçümün maskeler üzerinden doğrudan geometrik olarak yapılabilmesidir. YOLOv11x-seg modeli, instance/semantic segmentation kabiliyetiyle dişeti ve ilgili bölgeleri tekil görüntüden ayrıştırmış; önceki dönemdeki piksel tabanlı yaklaşıma kıyasla maske konturu, referans hatlar ve bölgesel örnekleme üzerinden ölçümü mümkün kılan bir altyapı sağlamıştır. GPU hızlandırmalı inference ile tekil görüntüden hızlıca çıktı üretimi gerçekleştirilmiş ve ara çıktılar overlay maskeler olarak görselleştirilebilir şekilde saklanmıştır.

Mask tabanlı ölçüm katmanı, segmentasyon maskesi üzerinden kontur çıkarımı, bölgesel örnekleme ve `px_per_mm` kalibrasyon parametresi ile pikselden mm’ye dönüşüm yapılmasını mümkün kılmış; bu sayede karar destek için gerekli sayısal girdiler doğrudan ve izlenebilir biçimde üretilmiştir. Ölçüm sonucunun klinik eşiklere göre E1–E4 sınıflarına atanması, kural tabanlı bir mekanizma ile gerçekleştirilmiş; bu yaklaşım, belirsizlik durumlarının (ambiguity) yönetilmesini ve metaveri eşlemesi ile kararın iyileştirilmesini destekleyen açıklanabilir bir karar akışı sağlamıştır.

Yazılım mimarisi, eğitim ve inference süreçlerinin ayrıştırıldığı modüler bir pipeline olarak kurgulanmış; tekil görüntüden uçtan uca otomatik çıktı üretimi, ölçüm, sınıflandırma ve görselleştirme adımlarını ardışık şekilde birleştirmiştir. Modüler yapı sayesinde yeni sınıflar, ek ölçüm modülleri veya farklı segmentasyon modelleri ileride sisteme entegre edilebilir nitelikte tasarlanmış; bu da mimarinin genişletilebilir ve sürdürülebilir olmasını mümkün kılmıştır.

### X.X. Genel Değerlendirme

Aşağıda sunulan yaklaşımda segmentasyon tabanlı ölçüm hattı, dişeti görünürlüğü hesaplamasını otomatikleştirmiş, ölçümün bölgesel dağılımını kontrol edilebilir hâle getirmiş ve klinik eşiklere dayalı karar destek çıktısının üretimini sistematikleştirmiştir. Maskelerin ara çıktı olarak kaydedilmesi, ölçüm adımlarının izlenmesini ve gerektiğinde görsel doğrulamayı mümkün kılmıştır. Bu nedenle GummySmile v3 içinde YOLOv11x-seg temelli v3 hattı, klinik karar destek açısından daha tutarlı ve açıklanabilir bir nihai çözüm sunması beklenen bir yaklaşım olarak konumlandırılmıştır.
