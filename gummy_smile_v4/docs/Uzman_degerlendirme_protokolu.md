# Uzman değerlendirme protokolü — dişeti görünürlüğü etiyoloji sınıflaması

## 1. Amaç

Yapay zekâ tabanlı iş akışının atadığı etiyoloji sınıfı (E1–E4) ile bağımsız uzmanların klinik değerlendirmesini karşılaştırmak. Bu analiz, hakemlerin "karar destek katmanının klinik geçerliliği gösterilmemiş" eleştirisine yanıt olarak yapılmaktadır.

## 2. Değerlendirilecek görüntüler

- **145 yüksek gülme hattı fotoğrafı** (liste: `uzman_seti_145_goruntu.csv`).
- Seçim ölçütü: yüksek gülme hattı grubunda olup klinik referans (ImageJ) ölçümü bulunan ve tek bir hastaya karşılık gelen görüntüler. Aynı hastanın ikinci çekimleri çıkarılmıştır (her hastadan tek görüntü).
- Her uzman aynı 145 görüntüyü değerlendirir; **sıralama uzmana özgü olarak karıştırılır**.
- Setin sonuna, uzmana söylenmeden **20 görüntü ikinci kez** (farklı ID ile) eklenir. Bu, uzmanın kendi içindeki tutarlılığının (gözlemci içi uyum) hesaplanmasını sağlar. Toplam 165 değerlendirme.

## 3. Körleme (kritik)

Uzmanlar **olağan klinik değerlendirme yaklaşımlarıyla** karar verir. Fotoğraf üzerinden kendileri ölçüm yapmakta serbesttir; fotoğrafların çoğunda periodontal prob görünmektedir ve klinik pratikte değerlendirme genellikle ölçüm içerir. Kısıtlanan şey ölçüm yapmaları değil, **çalışmada elde edilen sonuçları görmeleridir.**

Analizin geçerli olması için uzmanlar şunları **görmemelidir**:

- modelin çıktısı (atadığı sınıf veya tedavi önerisi),
- çalışmada elde edilen milimetrik ölçüm değerleri (ImageJ referansı veya model çıktısı),
- makaledeki E1–E4 / T1–T4 **eşik tablosu** (hangi mm aralığının hangi sınıfa karşılık geldiği),
- görüntülerin orijinal dosya adları veya gülme hattı grup bilgisi.

Eşik tablosunun gizlenmesi kritiktir: uzmanlar da aynı tabloyu mekanik olarak uygularsa iki taraf aynı kuralı çalıştırmış olur ve karşılaştırma, Reviewer 4'ün eleştirdiği döngüsel analize dönüşür. Sınıf **tanımları** (E1 = gecikmiş pasif erüpsiyon vb.) formda verilir; verilmeyen şey mm eşikleridir. Değerlendiriciler birbirlerinden bağımsız çalışır, birbirlerinin formlarını görmez ve vakaları aralarında tartışmaz. ImageJ ölçümlerini yapan araştırmacı bu üç uzmandan biri **olmamalıdır**.

Görüntüler `U1-001`, `U1-002` … biçiminde anonim ID'lerle, uzman başına ayrı klasörde iletilir. Anon ID ↔ orijinal görüntü eşlemesi (`ANAHTAR_uzman_N.csv`) yalnızca araştırma ekibinde kalır.

## 4. Uzmanlara verilecek talimat

> Aşağıdaki fotoğraflarda aşırı dişeti görünürlüğü (gummy smile) bulunan bireyler yer almaktadır. Her fotoğraf için, dişeti görünürlüğünün **temel etiyolojisini** kendi klinik değerlendirmenize göre belirleyip aşağıdaki sınıflardan birini seçiniz:
>
> - **E1** — Gecikmiş pasif erüpsiyon / dişeti büyümesi / kalın dişeti fenotipi
> - **E2** — Hipermobil (hiperaktif) üst dudak / kısa üst dudak
> - **E3** — Dentoalveolar ekstrüzyon / derin kapanış
> - **E4** — Vertikal maksiller fazlalık (VME)
>
> Birden fazla etiyolojinin olası olduğunu düşünüyorsanız, en olası sınıfı "Etiyoloji sınıfı" sütununa, ikinci adayı "İkinci aday" sütununa yazınız. Her değerlendirme için 1–5 arası bir güven düzeyi belirtiniz (1 = hiç emin değilim, 5 = çok eminim).
>
> Lütfen sırayla ilerleyiniz ve verdiğiniz cevapları geriye dönüp değiştirmeyiniz. Değerlendirmeyi tek oturumda tamamlamanız gerekmez; ancak vakaları başka meslektaşlarınızla tartışmayınız.
>
> Normalde nasıl değerlendiriyorsanız aynen öyle ilerleyiniz; fotoğraf üzerinden ölçüm yapmak isterseniz serbestsiniz (fotoğrafların çoğunda periodontal prob görünmektedir). Çalışmada elde edilen milimetrik ölçüm değerleri ve kullanılan eşik tablosu, değerlendirmenizin bağımsız olması için bilerek paylaşılmamıştır.
>
> Değerlendirme fotoğraf üzerinden yapılacaktır; hastanın klinik muayene bilgileri (üst dudak uzunluğu ve hareketliliği, klinik kron boyu, kapanış ilişkisi) sağlanmamaktadır. Bu bir kısıtlılıktır ve çalışmada açıkça belirtilecektir.

## 5. Analiz planı

**Referans standart (istatistikçi onayıyla, analizden önce sabitlendi):** üç uzmanın çoğunluk kararı. Üçü de farklı sınıf verirse, uzmanlar **model çıktısına kör biçimde** bir araya gelip ortak (konsensüs) karar verir. Konsensüs sağlanamayan vaka kalırsa birincil analizden çıkarılır ve sayısı raporlanır.

**Birincil sonuç:** model ile uzman çoğunluk kararı arasında **doğrusal ağırlıklı Cohen kappa** (%95 GA). E1–E4 sıralı kabul edilir; ağırlıksız kappa, gözlenen uyum yüzdesi ve PABAK ikincil olarak raporlanır. Örneklem gerekçesi: R `kappaSize` ile minimum 110 (kayıpla 123) görüntü; n = 150 yaklaşık %90 güç sağlar.

**İkincil sonuçlar:**
- Sınıf bazında uyum yüzdesi, duyarlılık, özgüllük — **vaka sayıları ve %95 GA ile birlikte** (nadir sınıflar için zorunlu).
- **mm ölçümleri:** uzmanlar 6 diş için ImageJ ölçümü de yapacaktır. Gözlemciler arası güvenilirlik ICC(2,1) ve ICC(2,k) ile raporlanır. Diş bazlı analizlerde, aynı hastanın 6 ölçümü bağımsız olmadığından **karma etkili model** kullanılır (istatistikçi önerisi).
- Uzmanlar arası uyum (Fleiss kappa) — modelden beklenebilecek üst sınırı gösterir.
- Uzman içi uyum (20 tekrar görüntü üzerinden, uzman başına Cohen kappa).
- Modelin çift aday verdiği vakalar (ör. "E1–E2") için iki puanlama: **katı** (yalnızca birinci sınıf sayılır) ve **esnek** (uzmanın sınıfı adaylardan biriyse uyumlu sayılır).
- Uyuşmazlıkların, ölçülen milimetre değerinin eşiğe (3, 4, 6, 8 mm) uzaklığına göre tabakalandırılması: "model yanıldı" ile "değer zaten sınırda" ayrımı.
- Yaygınlığa duyarlılık nedeniyle kappa ile birlikte gözlenen uyum yüzdesi ve PABAK da raporlanır (bkz. 6).

**Birincil/ikincil küme:** Model çıktısı, modelin eğitimde görmediği test alt kümesinde yansızdır. Bu nedenle birincil analiz test alt kümesinde, tüm 145 görüntüdeki sonuç ikincil olarak raporlanır.

## 6. Önceden bilinen kısıtlılıklar (makaleye yazılacak)

- **Sınıf dağılımı dengesizdir.** Klinik referans ölçümlere göre bu 145 görüntüde tablo sınıfları: E1 = 89, E1–E2 = 24, E2–E3 = 26, E3 = 6. Kappa yaygınlığa duyarlı olduğundan gözlenen uyum yüzdesi ve PABAK ile birlikte raporlanacaktır.
- **E4 sınıfı hiç bulunmamaktadır.** Bu görüntülerdeki en yüksek ortalama dişeti görünürlüğü 7.53 mm'dir; > 8 mm ölçüm yoktur. Dolayısıyla E4/T4 (Le Fort I) dalı bu veriyle doğrulanamaz ve makalede bu açıkça belirtilmelidir. (Not: ilk gönderimdeki Şekil 6'da görülen 8–9 mm'lik E4 çıktıları, düzeltilen ölçüm hatasından kaynaklanmıştır.)
- Değerlendirme iki boyutlu frontal fotoğraf üzerinden yapılmakta, klinik muayene ve sefalometrik veri içermemektedir. Reviewer 4'ün talep ettiği ideal tasarım, hastaların gerçek klinik muayenesine dayanan bağımsız tanılardır; mevcut koşullarda fotoğraf temelli uzman değerlendirmesi bu talebe verilebilecek en yakın yanıttır ve makalede bu şekilde ifade edilmelidir. Klinik kayıtlarda gerçek muayeneye dayalı tanı ve tedavi planı bulunan hastalar varsa, az sayıda olsalar bile ayrı bir alt analiz olarak eklenmeleri kanıt düzeyini belirgin biçimde yükseltir.
- Ölçüm, fotoğraftaki gülümseme şiddetine duyarlıdır: aynı hastanın iki çekimi arasında belirgin fark gözlenebilmektedir (örn. bir olguda 2.6 mm ve 6.9 mm). Bu, standart çekim protokolünün önemini gösterir ve kısıtlılık olarak belirtilmelidir.

## 7. Uygulama adımları

Proje klasöründe şunlar bulunmalıdır: `goruntuleri_hazirla.py`, `ANAHTAR_arastirmaci.csv`, `Uzman_1/2/3_form.xlsx`, `OKUYUNUZ.txt`.

1. Fotoğraflar getirilir (yalnızca `high` klasörü, ~300 MB):
   ```bash
   cd ~/Projects/GummySmile
   git clone --filter=blob:none --sparse --depth 1 https://github.com/rfatihors/GaziDental.git repo
   cd repo && git sparse-checkout set gummy_smile_v3/data/coco_dataset/high && cd ..
   ```
2. Görüntüler anonim adlarla kopyalanır (tek parametre; liste anahtar dosyasından okunur, çıktı `GORUNTULER/` klasörüdür):
   ```bash
   python3 goruntuleri_hazirla.py --kaynak repo/gummy_smile_v3/data/coco_dataset/high
   ```
   Beklenen çıktı: `Kopyalanan: 165 / 165`
3. Bulunamayan dosya olursa `BULUNAMAYAN.txt` incelenir, eksikler kaynak klasöre eklenir, betik tekrar çalıştırılır.
4. Her uzmana **yalnızca** `GORUNTULER` klasörü + kendi formu gönderilir. Uzmanın ihtiyaç duyduğu tüm talimatlar formun başında yazılıdır. `ANAHTAR_arastirmaci.csv` ve `OKUYUNUZ_SADECE_ARASTIRMACI.txt` **gönderilmez**; ikincisi tekrar edilen görüntülerin varlığını açıkladığı için uzmana ulaşırsa uzman içi tutarlılık ölçümü geçersiz olur.
5. Doldurulmuş formlar geri geldiğinde anahtar dosyasıyla birleştirilip analiz yapılır.
