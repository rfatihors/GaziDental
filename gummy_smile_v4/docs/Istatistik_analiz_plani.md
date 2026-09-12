# Nihai istatistiksel analiz planı

İstatistikçinin 10 Eylül tarihli cevaplarıyla sabitlendi. Analizler başlamadan önce bu plan geçerlidir; sonradan değiştirilirse makalede belirtilmelidir.

---

## İstatistikçinin cevapları — özet

| Soru | Cevap | Planda değişiklik |
|---|---|---|
| Öğrenme eğrisi gerekçesi uygun mu? | Uygun; makine öğreniminde kullanılan standart bir yöntem | Değişiklik yok |
| ICC formülleri ve n = 150 hassasiyeti doğru mu? | Doğru | Değişiklik yok |
| ICC(2,1) uygun mu? | Uygun | Değişiklik yok |
| Diş bazlı ölçümlerde küme etkisi | **Karma etkili model kullanılmalı**; hakemler buna dikkat ediyor | **DEĞİŞİKLİK** — aşağıda 2.2 |
| Kappa örneklemi | R `kappaSize` ile hesaplandı: min 110, kayıpla 123; n = 150 ≈ %90 güç | **YENİ** — hazır metin, aşağıda 3.1 |
| Ağırlıklı kappa ve ağırlık şeması | Ağırlıklı kappa uygun; **birincil analizde doğrusal ağırlık** (daha koruyucu, yorumu kolay) | **DEĞİŞİKLİK** — birincil ölçüt belirlendi |
| Referans standart | Çoğunluk kararı uygun; üçü de farklıysa **model sonucuna kör ortak değerlendirme (konsensüs)**; yöntem analizden önce tanımlanmalı | **NETLEŞTİ** — protokole işlendi |
| Nadir sınıf (E4) | Vaka sayıları ve %95 GA ile dikkatli raporlanmalı | Aşağıda 3.4 |

---

## 1. Eğitim verisi yeterliliği

Değişiklik yok. Temiz hasta düzeyi bölünme sonrası eğitim bölüntüsünün tabakalı %25 / %50 / %75 / %100 alt kümeleriyle yeniden eğitim; doğrulama setinde sınıf bazında mask mAP@50 grafiklenir. Test setine dokunulmaz.

## 2. Milimetrik ölçüm doğruluğu

### 2.1 Birincil analiz (görüntü düzeyi)

- Sonuç değişkeni: görüntü başına 6 diş ölçümünün ortalaması.
- ICC(2,1), iki yönlü mutlak uyum, %95 GA.
- Bland–Altman: ortalama sapma, %95 uyum sınırları ve bunların GA'ları.
- MAE, RMSE, dağılım grafiği.
- n ≈ 145–150 (yüksek gülme hattı, klinik referansı olan görüntüler).

### 2.2 Diş bazlı analiz — karma etkili model (DEĞİŞİKLİK)

Aynı hastaya ait 6 diş ölçümü bağımsız olmadığı için diş düzeyindeki analizler karma etkili modelle yapılacaktır:

- Model: `fark ~ 1 + (1 | hasta)` — sabit etki olarak ortalama sapma (bias), rastgele etki olarak hasta.
- Diş pozisyonunun etkisi ayrıca sınanır: `fark ~ dis_pozisyonu + (1 | hasta)`. Bu, modelin belirli dişlerde (örn. lateraller, kaninler) sistematik olarak sapıp sapmadığını gösterir — hakemlerin ilgisini çekecek bir analizdir.
- Varyans bileşenlerinden hastalar arası ve hasta içi varyans raporlanır.
- Diş düzeyinde ICC de karma modelden türetilir; bağımsız gözlem varsayan basit ICC raporlanmaz.
- Uygulama: Python `statsmodels.MixedLM` veya R `lme4::lmer`.

### 2.3 Referans ölçümün güvenilirliği

- **Gözlemci içi** (mevcut): 20 görüntü, iki oturum; ICC(2,1) = 0.995 (diş düzeyi, n = 120), 0.998 (görüntü ortalaması); ortalama fark −0.01 mm, SD 0.17 mm, %95 uyum sınırları ±0.33 mm.
- **Gözlemciler arası** (yeni, üç uzman mm ölçümü yapacak): ICC(2,1) tek ölçüm ve ICC(2,k) ortalama ölçüm için ayrı ayrı raporlanır; k = gözlemci sayısı.
- Model, birincil olarak klinik referans ölçüme, ikincil olarak üç uzmanın ortalamasına karşı karşılaştırılır.

## 3. Sınıf atamasının uzmanlarla uyumu

### 3.1 Örneklem gerekçesi — makale metnine hazır (istatistikçiden)

> Örneklem büyüklüğü, yapay zekâ modelinin E1–E4 sınıf ataması ile klinisyen arasındaki uyumun Cohen Kappa katsayısıyla değerlendirilmesi esas alınarak R 4.5.2 programında (R Core Team, 2025) `kappaSize` paketi kullanılarak hesaplanmıştır. Dört kategorili sınıflandırmada kabul edilebilir en düşük Kappa katsayısı 0.40, beklenen Kappa katsayısı 0.60, iki yönlü anlamlılık düzeyi 0.05 ve istatistiksel güç %80 olarak kabul edilmiştir. En nadir E4 sınıfının prevalansının %2 olduğu koruyucu senaryoda minimum örneklem büyüklüğü 110 görüntü olarak hesaplanmıştır. Yaklaşık %10 olası veri kaybı dikkate alındığında gerekli örneklem büyüklüğü 123 görüntüye yükselmiştir. Çalışmada yer alan 150 görüntünün yaklaşık %90 güç sağladığı ve genel sınıflandırma uyumunun değerlendirilmesi için yeterli olduğu belirlenmiştir.

İngilizce karşılığı (Methods):

> The sample size was determined for the assessment of agreement between the AI model's E1–E4 class assignment and the clinicians' assignment using Cohen's kappa, calculated in R 4.5.2 (R Core Team, 2025) with the `kappaSize` package (Rotondi, 2018). For a four-category classification, the minimum acceptable kappa was set at 0.40, the expected kappa at 0.60, the two-sided significance level at 0.05 and statistical power at 80%. Under a conservative scenario in which the rarest class (E4) has a prevalence of 2%, the minimum required sample size was 110 images, increasing to 123 images after allowing for approximately 10% data loss. The 150 images included in the study provide approximately 90% power and are therefore sufficient for the evaluation of overall classification agreement.

Kaynaklar: R Core Team (2025), *R: A language and environment for statistical computing* (v4.5.2); Rotondi MA (2018), *kappaSize: Sample size estimation functions for studies of interobserver agreement* (R package v1.2).

### 3.2 Birincil ölçüt

**Doğrusal ağırlıklı Cohen kappa**, %95 GA. E1–E4 sıralı kabul edilir; komşu sınıf hatası, uzak sınıf hatasından daha az cezalandırılır. Ağırlıksız Cohen kappa ve gözlenen uyum yüzdesi ikincil olarak raporlanır. Yaygınlık dengesizliği nedeniyle PABAK da verilir.

### 3.3 Referans standart (analizden önce sabitlendi)

1. Üç uzmanın **çoğunluk kararı** referans kabul edilir.
2. Üçü de farklı sınıf verirse, uzmanlar **model çıktısına kör biçimde** bir araya gelip ortak (konsensüs) karar verir.
3. Konsensüs sağlanamayan vaka kalırsa birincil analizden çıkarılır ve sayısı raporlanır.

Uzmanlar arası uyum **Fleiss kappa** ile raporlanır (modelden beklenebilecek üst sınır).

### 3.4 Sınıf bazında raporlama ve E4 sorunu

Sınıf bazında uyum yüzdesi, duyarlılık ve özgüllük, **vaka sayıları ve %95 GA ile birlikte** verilir.

**Önemli uyarı:** Klinik referans ölçümlere göre bu 145 görüntüde **hiç E4 vakası yoktur**; en yüksek ortalama dişeti görünürlüğü 7.53 mm'dir (eşik > 8 mm). Dolayısıyla:

- Model bu veri setinde E4 üretmeyecektir; örneklem hesabındaki %2 E4 prevalansı varsayımı koruyucu (conservative) kalmıştır, sonucu geçersiz kılmaz.
- Uzmanlar kendi klinik yargılarıyla E4 atayabilir; bu durumda uyuşmazlık "model E3 – uzman E4" biçiminde görünür ve ayrıca yorumlanır.
- **E4/T4 (Le Fort I) dalı bu çalışmada doğrulanamaz**; makalede açıkça belirtilmelidir.

### 3.5 Çakışan aralıklar ve uyuşmazlık analizi

- Modelin çift aday verdiği vakalar (örn. "E1–E2") iki kuralla puanlanır: **katı** (yalnızca birinci sınıf) ve **esnek** (uzmanın sınıfı adaylardan biriyse uyumlu). İkisi de raporlanır.
- Uyuşmazlıklar, ölçülen mm değerinin klinik eşiğe (3, 4, 6, 8 mm) uzaklığına göre tabakalandırılır: "model yanıldı" ile "değer zaten sınırda" ayrımı.

## 4. Dış geçerlilik

Tek merkez, tek cihaz. Raporlanan başarım iç geçerlilik tahminidir. "Örneklem büyüklüğü dış geçerliliği destekler" ifadesi makaleden çıkarılır; sınırlılıklarda bağımsız merkez/cihaz/popülasyon verisiyle dış doğrulama gereği belirtilir.

---

## İstatistikçiye sorulacak iki ek soru

Tasarım, ilk notu gönderdikten sonra bir noktada değişti: **üç uzman yalnızca sınıf atamayacak, aynı zamanda ImageJ ile 6 diş için mm ölçümü de yapacak.** Buna bağlı iki soru:

1. **Gözlemciler arası ICC modeli.** Üç uzman + mevcut klinik referans gözlemci için hangi ICC modelini önerirsiniz: ICC(2,1) tek ölçüm mü, ICC(2,k) ortalama ölçüm mü, yoksa ikisi birlikte mi raporlanmalı? Model karşılaştırmasında referans olarak tek bir gözlemcinin ölçümü mü yoksa gözlemci ortalaması mı alınmalı?

2. **Sıralama etkisi / bağımsızlık.** Uzmanlar önce ölçüm yapıp sonra sınıf atayacak (klinik pratiğe uygun olması için). Bu durumda uzmanın sınıf ataması kendi ölçümüne dayanmış oluyor. Kappa analizinin yorumunda bunun nasıl ifade edilmesi gerekir; ek bir duyarlılık analizi önerir misiniz?

Ayrıca bilgi olarak: veri setinde E4 (> 8 mm) vakası bulunmamaktadır (3.4). Örneklem hesabındaki %2 varsayımı koruyucu kalmıştır; sınıf bazında raporlamayı buna göre yapacağız.
