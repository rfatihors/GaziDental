# Veri envanteri — hangi dosya ne işe yarıyor

Son durum, 9 Eylül 2026. Kaynak: `rfatihors/GaziDental` deposu + klinik ekipten gelen dosyalar.

---

## 1. Fotoğraflar

### 1.1 `gummy_smile_v3/data/coco_dataset/` — ÇALIŞMANIN ANA GÖRÜNTÜ SETİ

**1.315 fotoğraf**, gülme hattına göre üç klasörde, her biri Roboflow'un train/valid/test bölünmesiyle:

| Grup | Görüntü | train | valid | test |
|---|---|---|---|---|
| high | 216 | 158 | 28 | 30 |
| low | 303 | 214 | 45 | 44 |
| normal | 796 | 558 | 119 | 119 |
| **Toplam** | **1.315** | 930 | 192 | 193 |

- Yanında COCO formatında etiketler (`_annotations.coco.json`): `diseti` ve `dudak` maskeleri.
- Çözünürlük ağırlıklı olarak **2698×1799** — hocanın ImageJ ölçümü yaptığı çerçeveyle aynı. Yani bu dosyalar ölçüm için doğru kopyalardır.
- Dosya adları Roboflow tarafından değiştirilmiş: `IMG_2544-_jpeg.rf.<hash>.jpeg`. Sondaki `-`, orijinaldeki fazladan noktadır (`IMG_2544..jpeg`).
- Toplam 0.93 GB.

**Uzman paketi için gereken tüm görüntüler burada.** Doğruladım: 145 görüntünün + 20 tekrarın hepsi bu klasörden kopyalanabiliyor (165/165). Drive'a bakmaya gerek yok.

### 1.2 Drive'daki orijinal fotoğraflar

Bunlara **şu an ihtiyaç yok**. Tek fark dosya adları ve muhtemelen bazı görüntülerin ham hâli. ImageJ ölçümleri 2698×1799 kopyalarda yapıldığına göre (hoca doğruladı, ekran görüntüsüyle), ölçüm/analiz için depodaki kopyalar doğru olanlar.

---

## 2. Ölçüm ve demografi dosyaları

| Dosya | Nerede | İçerik | Kullanım |
|---|---|---|---|
| **`Hasta_ID-_Ölçümler.xlsx`** | Klinikten geldi (6 Eylül) | 3 sayfa: *Yüksek* 692 satır (6 diş ölçümü + E/T etiketleri + yaş/cinsiyet), *Düşük* 100 satır, *Normal* 246 satır (yalnız yaş/cinsiyet) | **GÜNCEL REFERANS.** Milimetrik validasyonun referansı ve demografi kaynağı |
| `ölçümler ai guncel.xlsx` | Depoda | 693 satır, aynı yapı, eski sürüm | Artık kullanılmıyor; yerini yukarıdaki aldı |
| **`calibration.xlsx`** | Klinikten geldi (6 Eylül) | 20 görüntü × 6 diş, iki ayrı oturum | **GÜNCEL.** Gözlemci içi güvenilirlik (ICC 0.995 hesaplandı) |
| `calibration-first.xlsx` / `calibration-last.xlsx` | Depoda | 15 görüntü, iki oturum | Eski; makaledeki "20 görüntü" ifadesiyle uyuşmuyordu, artık kullanılmıyor |

**Önemli:** Ölçüm dosyasındaki 692 satırın büyük kısmı **bu veri setinde olmayan** fotoğraflara aittir (önceki J Dent 2026 çalışmasının seti). Mevcut 216 yüksek gülme hattı görüntüsünün 149'unun ölçümü var, 66'sının yok.

---

## 3. Türetilmiş listeler (benim ürettiklerim)

| Dosya | İçerik |
|---|---|
| `uzman_seti_145_goruntu.csv` | Uzmanlara gidecek 145 görüntü + mm değeri + tablo sınıfı + yaş/cinsiyet |
| `ANAHTAR_arastirmaci.csv` | G001–G165 anonim ID ↔ orijinal görüntü eşlemesi, tekrar bayrağı |
| `olcumsuz_yuksek_gulme_hatti_66.csv` | Ölçümü olmayan, çıkarılan 66 yüksek gülme hattı görüntüsü |
| `ayni_hasta_aday_ciftler.csv` | Aynı hastanın iki çekimi olan 55 çift (ORB+RANSAC ile bulundu) |

### 145 sayısı nereden geliyor?

```
216   yüksek gülme hattı görüntüsü (coco_dataset/high)
 −66  klinik ölçümü olmayanlar (hocanın kararıyla çıkarıldı)
 −1   IMG_7366 — ölçüm dosyasında iki farklı satır var, hangisi olduğu belirsiz
 −4   aynı hastanın ikinci çekimi (her çiftten problu olan tutuldu)
─────
 145
```

---

## 4. HÂLÂ EKSİK OLANLAR — Drive'da aranacaklar

1. **`best.pt` (eğitilmiş model ağırlığı)** — depoda yok, sadece bir README var. Tahmin maskeleri üzerinde milimetrik doğruluk analizi (MAE/RMSE/ICC/Bland–Altman) bunsuz yapılamaz. Ancak modeli temiz bölünmeyle yeniden eğiteceğimiz için **eskisi zorunlu değil**; yeni eğitim yeni bir `best.pt` üretecek. Eskisi yalnızca "Figure 6 hangi ağırlıkla üretildi" sorusu için faydalı olurdu.
2. **Roboflow v7 / v8 sürüm dışa aktarımları** — makaledeki 2.235 ve 3.403 görüntü sayılarının nasıl oluştuğunu belgelemek için sürüm ekran görüntüleri veya dışa aktarımlar. Reviewer 2 (#5) ve Reviewer 4 bunu soruyor. Roboflow hesabından da alınabilir.
3. **Boş dosyalar** (`labels_smileline.csv`, `splits.json`, `results/*.csv`) — yeniden üretilecek, Drive'da aranmasına gerek yok.

Bunların dışında analiz için gereken her şey elimizde.

---

## 5. Şu an ne yapılabilir (hiçbir eksik olmadan)

1. Uzman paketini oluştur: `python goruntuleri_hazirla.py --kaynak <depo>/gummy_smile_v3/data/coco_dataset/high`
2. Ölçüm modülünü yeniden yaz (prompt hazır).
3. Excel ayrıştırıcı + oracle doğrulaması (GT maskeler ↔ ImageJ).
4. Hasta düzeyinde temiz bölünme.
5. Yeniden eğitim (yeni `best.pt` üretir) → sonra tahmin maskeleri üzerinde milimetrik analiz.
