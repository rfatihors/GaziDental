# gummy_smile_v4 — Claude Code görev tanımı

Sen `rfatihors/GaziDental` deposunda, `gummy_smile_v4/` klasöründe çalışan bir araştırma yazılım mühendisisin. Görevin, `gummy_smile_v3` altındaki dişeti görünürlüğü (gingival display) ölçüm ve karar destek hattını, tespit edilmiş hataları düzelterek ve hakem revizyonu için gereken analizleri ekleyerek **sıfırdan, temiz bir v4 olarak** yazmaktır.

Bu bir tıbbi araştırma projesidir; çıktılar bir makale revizyonunda kullanılacaktır. **Doğruluk hızdan önce gelir.** Emin olmadığın hiçbir şeyi varsayma: ölç, doğrula, raporla. Beklenmedik bir şey bulduğunda durup sor.

---

## 0. Başlamadan önce oku

Sırayla, tamamını:

1. `docs/Veri_envanteri.md` — hangi dosya ne, hangisi güncel, hangisi eski
2. `docs/GummySmile_v3_Teknik_Audit_Raporu_v2.md` — v3'teki hatalar, doğrulanmış sayılar, klinik ekibin kararları (Bölüm A ve B en kritik)
3. `docs/PROMPT_olcum_modulu_yeniden_yazim_v2.md` — ölçüm modülü, Excel ayrıştırma, eşleştirme, temizlik ve oracle doğrulama **spesifikasyonu**. Bu belge ayrıntılı kurallar için tek referanstır; burada tekrar edilmez.
4. `docs/Istatistik_analiz_plani.md` — istatistikçiyle sabitlenmiş analiz planı
5. `docs/Uzman_degerlendirme_protokolu.md` — üç uzmanın yapacağı değerlendirmenin tasarımı; formların yapısı
6. `docs/Guc_analizi_hakem_cevabi_ve_metin.md` — öğrenme eğrisi deneyi tanımı (Bölüm 4)

Sonra `../gummy_smile_v3/` altındaki kodu **salt okunur** olarak incele: `measurement/`, `methods/v3/`, `yolo/`, `master_pipeline_v3.py`, `configs/config.yaml`, `evaluation/`. Neyin neden yanlış olduğunu audit raporuyla eşleştir. v3'ten kopyala-yapıştır yapma; v3 referanstır, kaynak değil.

---

## 1. Kesin kurallar

- **`gummy_smile_v3/` altına hiçbir şey yazma.** Tüm kod `gummy_smile_v4/` altında.
- Görüntüleri kopyalama. v4, v3'ün verisine göreli yolla erişir: `../gummy_smile_v3/data/coco_dataset/`. Yol `configs/config.yaml` içinde tek yerde tanımlı.
- Git'e büyük dosya koyma: `*.pt`, `runs/`, `outputs/`, `GORUNTULER/`, `*.zip`, üretilmiş maskeler `.gitignore`'da. Yalnızca kod, küçük CSV/JSON sonuçlar, figürler (PNG < 2 MB) ve raporlar commit edilir.
- Her aşama sonunda: testler geçer → `git add -A && git commit` (mesaj: aşama adı + ne yapıldığı) → `git push origin master`. Push başarısız olursa dur ve söyle.
- Python 3.11, `venv`. İki ayrı bağımlılık dosyası: `requirements.txt` (ölçüm/analiz — GPU gerektirmez, Mac'te çalışır) ve `requirements-train.txt` (eğitim — Linux iş istasyonu, **RTX 5090**). Eğitim kodu burada (Mac) yazılır ve `--dry-run` ile doğrulanır; gerçek eğitim kullanıcı tarafından iş istasyonunda `git pull` sonrası çalıştırılır (Aşama 5).
- Kod ve docstring'ler İngilizce; üretilen raporlar İngilizce (makaleye girecek) + her rapor için kısa Türkçe `OZET.md`.
- `pytest` ile testler; her modülün testi var. Testler klinik veri gerektirmez (sentetik maskeler / sentetik formlar).
- Rastgelelik olan her yerde `seed=42`, `configs/config.yaml`'da tanımlı.
- `silinecek.md` dosyasını ilk commit'te sil.

---

## 2. Klasör yapısı

```
gummy_smile_v4/
├── README.md                     ne, neden, nasıl çalıştırılır
├── requirements.txt
├── .gitignore
├── configs/config.yaml           tüm yollar, sınıf adları, eşikler, seed, YOLO ayarları
├── docs/                         (kullanıcı koyar) yukarıdaki 6 belge
├── data/
│   ├── inputs/                   (kullanıcı koyar) Excel/CSV girdileri — Bölüm 2.1
│   ├── manifest/                 üretilen: dataset_manifest.csv, splits.json, manifest_summary.md
│   └── expert/                   (sonradan gelecek) doldurulmuş 3 uzman formu
├── gsv4/                         Python paketi
│   ├── io/        excel_parser.py, naming.py, coco.py, forms.py
│   ├── dataset/   manifest.py, split.py
│   ├── masks/     extract.py  (YOLO / COCO / PNG → sınıf ayrımlı ikili maskeler)
│   ├── measure/   gingival_display.py, profile.py, regions.py
│   ├── rules/     rule_engine.py
│   ├── eval/      oracle.py, agreement.py (ICC, Bland–Altman, mixed model), kappa.py, boundary.py
│   ├── train/     prepare_yolo_dataset.py, train.py, learning_curve.py, cv_predict.py, evaluate_test.py
│   └── report/    figures.py, tables.py
├── scripts/                      CLI giriş noktaları (her aşama için bir tane)
├── scripts/train_all.sh          iş istasyonunda tüm eğitimleri sırayla çalıştırır
├── tests/
└── outputs/                      (gitignore) raporlar, figürler, CSV'ler
```

### 2.1 Girdi dosyaları — `data/inputs/`

Kullanıcı bunları koyacak; başlamadan varlıklarını kontrol et, eksikse dur:

| Dosya | Durum | Kullanım |
|---|---|---|
| `Hasta_ID-_Ölçümler.xlsx` | var | Klinik referans mm ölçümleri (yüksek sayfası) + yaş/cinsiyet (3 sayfa) |
| `calibration.xlsx` | var | Gözlemci içi, 20 görüntü × 2 oturum |
| `ayni_hasta_aday_ciftler.csv` | var | 55 aynı-hasta çifti (hepsi klinik ekipçe doğrulandı) |
| `olcumsuz_yuksek_gulme_hatti_66.csv` | var | Çıkarılacak 66 yüksek gülme hattı görüntüsü |
| `uzman_seti_145_goruntu.csv` | var | Uzman setine giren 145 görüntü + referans mm + tablo sınıfı |
| `ANAHTAR_arastirmaci.csv` | var | G001–G165 anonim ID ↔ görüntü; `tekrar` sütunu |
| `data/expert/Uzman_{1,2,3}_form.xlsx` | **sonra gelecek** | Doldurulmuş formlar (ölçüm + sınıf) |
| `outputs/05_predictions/` | **sonra gelecek** | İş istasyonundaki eğitimden dönen tahmin maskeleri ve `results.csv` dosyaları (Bölüm 3, Aşama 5) |

---

## 3. Aşamalar

Her aşamanın sonunda kabul ölçütlerini kendin doğrula, `outputs/<asama>/` altına raporunu yaz, commit et. Bir aşama tamamlanmadan sonrakine geçme.

### Aşama 1 — Veri katmanı (yerel)

`gsv4/io/excel_parser.py`, `naming.py`, `coco.py`; `gsv4/dataset/manifest.py`, `split.py`.

- Excel ayrıştırma: spesifikasyon `docs/PROMPT…v2.md` §4.2. Beklenen: 692 satır, ≥ 690'ı tam; hücre türü sayıları raporlanır; E etiketiyle tutarlılık denetimi.
- Ad eşleştirme: §4.2 üç kural (nokta/`-` → `~`, yaş öneki korunur, kademeli eşleştirme). Beklenen: `high` grubunda 149 tekil eşleşme.
- Temizlik → `data/manifest/dataset_manifest.csv`: §4.8. Kararlar: 66 ölçümsüz high çıkar; 55 çiftten problu olanı tut (probsuz olan `ayni_hasta_aday_ciftler.csv`'de `image_b` değil — hangisinin problu olduğu `uzman_seti_145_goruntu.csv`'de tutulan görüntüden anlaşılır; high dışı çiftlerde `Düşük`/`Normal` sayfasında yaş/cinsiyeti olan tutulur, ikisinde de yoksa `image_a`); `IMG_7366` belirsiz → çıkar; `IMG_78701` = `IMG_7870` → birini çıkar. Beklenen kalan: high **145**, low ≈ 299, normal ≈ 786.
- Bölünme → `data/manifest/splits.json`: hasta düzeyi = görüntü düzeyi (temizlik sonrası); gülme hattına göre tabakalı; **sabit test seti**. Oran: low/normal 70/15/15; **high 60/20/20** (ölçüm ve uzman analizlerinin test alt kümesi yeterince büyük olsun; ≈ 87/29/29). Aynı zamanda 145 ölçümlü high görüntüsü için **5 katlı çapraz doğrulama fold'ları** üret (Aşama 5'te out-of-fold tahmin için); fold'lar `splits.json` içinde.
- Demografi kapsamı raporu (Reviewer 3).
- Çıktı: `outputs/01_data/parse_report.md`, `manifest_summary.md`, `demographics.md`, `split_summary.md`.

**Kabul:** sayılar yukarıdaki beklentilerle uyuşur veya sapma gerekçesiyle raporlanır; testler geçer.

### Aşama 2 — Ölçüm modülü ve kural motoru (yerel)

`gsv4/masks/extract.py`, `gsv4/measure/*`, `gsv4/rules/rule_engine.py`.

Spesifikasyon: `docs/PROMPT…v2.md` §0, §3 (tamamı). Özellikle:
- Sınıf ayrımlı maske (`retina_masks=True` veya `masks.xy` rasterleştirme; `assert mask.shape == image.shape[:2]`), aynı sınıfın tüm instance'larının birleşimi, **en büyük kontur davranışı yok**.
- Ölçüm = dişeti maskesinin dikey kalınlığı; boş sütunlar 0; A/B/C bölgeleme; p10/p25/median/min; dudak-ankrajlı tahminci; tek birim `px_per_mm`.
- Kural motoru: tek motor; E1–E2 ve E2–E3 çakışmaları birleşik etiket; `value <= 0 → NO_VISIBLE_GINGIVA`; `NaN → UNCLASSIFIED`; metaveri kapalı; sınırlar E1 `<4`, E2 `[3,6]`, E3 `[4,8]`, E4 `>8`; çıktı alanı `treatment_alternatives`.
- `configs/config.yaml`: `imgsz: 640`, `max_det: 20`, `retina_masks: true`, `class_names: {gingiva: diseti, lip: dudak}`.
- Testler: §5'teki 12 test.

**Kabul:** sentetik maske testleri geçer; dudak maskesinin varlığı dişeti ölçümünü değiştirmez (regresyon testi).

### Aşama 3 — Oracle doğrulaması (yerel)

`gsv4/eval/oracle.py`, `scripts/run_oracle.py`. Spesifikasyon §4.1–4.6.

- GT COCO maskeleri → ölçüm → klinik referans ile karşılaştırma; %60 geliştirme / %40 holdout; tahminci seçimi; global vs görüntü bazlı ölçek; duyarlılık analizleri (dash-zero satırlar, 2698×1799 dışı görüntüler).
- Ön analizle karşılaştırma: A yöntemi, p25, n = 148 için r ≈ 0.83, MAE ≈ 0.63 mm, ölçek ≈ 17 px/mm bulunmuştu. Bu hedef değil, makullük kontrolü.
- Dudak-altı ↔ dişeti-üstü boşluk raporu (beklenen medyan ≈ 9 px).
- Gözlemci içi analiz `calibration.xlsx` ile: ICC(2,1) diş düzeyi ve görüntü ortalaması, Bland–Altman. Beklenen ≈ 0.995 / 0.998, SD ≈ 0.17 mm.
- Çıktı: `outputs/03_oracle/oracle_summary.md`, `estimator_comparison.csv`, figürler, `intra_observer.md`.

**Kabul:** en iyi tahminci holdout'ta raporlanmış; seçilen yöntem `configs/config.yaml`'a `measurement.method` olarak yazılmış.

### Aşama 4 — Uzman analiz modülü (yerel, sentetik veriyle)

`gsv4/io/forms.py`, `gsv4/eval/kappa.py`, `gsv4/eval/agreement.py`, `scripts/run_expert_analysis.py`.

Formların yapısı: `docs/Uzman_degerlendirme_protokolu.md` + gerçek boş form `data/inputs/` altında yoksa şu sütun düzeni: `Sıra | Görüntü ID | Ölçek (pixel/mm) | 13 | 12 | 11 | 21 | 22 | 23 | Etiyoloji sınıfı | İkinci aday | Güven | Not`, başlık 4. satırda, veri 5. satırdan itibaren, 165 satır, sayfa adı `Degerlendirme`.

Analizler (`docs/Istatistik_analiz_plani.md` §2–3 ile birebir):
- Form okuma + `ANAHTAR_arastirmaci.csv` ile birleştirme; tekrar satırları ayrıştırılır.
- **Uzman içi:** 20 tekrar üzerinden kappa (sınıf) ve ICC (mm), uzman başına.
- **Uzmanlar arası:** Fleiss kappa (sınıf); ICC(2,1) ve ICC(2,k) (mm); üç uzman + klinik referans gözlemci dahil.
- **Model vs uzman (sınıf):** doğrusal ağırlıklı Cohen kappa (birincil), ağırlıksız kappa, gözlenen uyum, PABAK; sınıf bazında duyarlılık/özgüllük vaka sayısı ve %95 GA ile; katı/esnek puanlama; eşiğe uzaklığa göre tabakalama. Referans: çoğunluk kararı; üçü farklıysa `consensus_pending` bayrağı (konsensüs sonradan girilecek).
- **Model vs uzman (mm):** ICC, Bland–Altman; diş bazlı analiz **karma etkili model** (`statsmodels.MixedLM`: `diff ~ 1 + (1|patient)` ve `diff ~ tooth + (1|patient)`).
- **Kappa %95 GA:** bootstrap (2000 örnek, seed sabit).
- Model çıktısı henüz yokken **sentetik model çıktısı** ve **sentetik formlar** üreten bir yardımcı (`tests/synth.py`) ile tüm hat uçtan uca çalışır; gerçek veri gelince aynı komut kullanılır.
- Uyarı: veri setinde E4 yoktur; kod E4'ü sıfır vakayla düzgün ele almalı (bölme sıfır hatası, boş sınıf).

**Kabul:** sentetik verilerle uçtan uca rapor üretiliyor; bilinen kappa/ICC değerli sentetik senaryolarda doğru sonuç veriyor (test).

### Aşama 5 — Eğitim hattı (yazılır ve dry-run edilir; iş istasyonunda çalıştırılır)

`gsv4/train/*`, `scripts/train_all.sh`, `scripts/README_TRAINING.md`, `requirements-train.txt`.

Donanım: Linux iş istasyonu, **NVIDIA RTX 5090 (32 GB, Blackwell)**. Blackwell için PyTorch'un CUDA 12.8 derlemesi gerekir (`torch>=2.7`, `--index-url https://download.pytorch.org/whl/cu128`); `requirements-train.txt` buna göre. `README_TRAINING.md` içine doğrulama adımı koy: `python -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"` → capability `(12, 0)` görünmeli. Ultralytics güncel sürüm.

- `prepare_yolo_dataset.py`: manifest + splits → Ultralytics segment formatı (`data.yaml`); COCO poligonlarından YOLO-seg etiketleri; görüntüler kopyalanmaz, **symlink** ile bağlanır; augmentasyon yalnızca train (Ultralytics'in kendi augmentasyonu; Roboflow çoğaltması yok); test setine hiçbir augmentasyon.
- `train.py`: YOLOv11x-seg, v3'teki nihai konfigürasyon (100 epoch, batch 16, lr0 0.0005, lrf 0.02, AdamW, cosine, close_mosaic 10, patience 20, imgsz 640) — `configs/config.yaml`'dan okunur. 32 GB VRAM ile batch 16 rahat sığar; `cache=True` kullanılabilir.
- `learning_curve.py`: train bölüntüsünün tabakalı iç içe %25/50/75/100 alt kümeleri, aynı doğrulama seti, sınıf bazında mask mAP → CSV + figür.
- `cv_predict.py`: 145 ölçümlü high görüntüsü için 5 katlı out-of-fold tahmin (fold'lar `splits.json`'dan; low/normal görüntüler her fold'da eğitimde). Her görüntü için tahmin maskesi (dişeti/dudak ayrı, ikili PNG) `outputs/05_predictions/oof/` altına.
- `evaluate_test.py`: sabit test setinde sınıf bazında box/mask mAP@50, mAP@50–95, precision, recall, F1; confusion matrix; boundary IoU ve kenar mesafesi hatası (dişeti üst ve alt kenarı ayrı — Reviewer 2 #10). Test seti tahmin maskeleri `outputs/05_predictions/test/`.
- `scripts/train_all.sh`: sırayla (1) final model, (2) öğrenme eğrisi 3 ek eğitim, (3) 5 fold, (4) test değerlendirmesi, (5) tahmin maskelerini yazma. Her adım kendi log dosyasına; bir adım başarısız olursa dur. Toplam 9 eğitim; 5090'da YOLOv11x için tahminen 1–1.5 saat/eğitim, gece çalıştırılacak şekilde tasarla (`nohup`/`tmux` notu README'de).
- Git'e ne döner: `outputs/05_predictions/**/*.png` (ikili maskeler, küçük) ve her eğitimin `results.csv`, `args.yaml` dosyaları **commit edilir** (`.gitignore`'da istisna); `*.pt` ve `runs/` tam çıktısı edilmez. Böylece Mac'teki Aşama 6 yalnızca `git pull` ile devam eder.

**Kabul:** hazırlama betiği yerelde çalışıp `data.yaml` üretir (COCO → YOLO dönüşümü 5 görüntüde görsel olarak doğrulanır); tüm eğitim betikleri `--dry-run` ile parametre ve yol doğrulaması yapar; `README_TRAINING.md` iş istasyonunda sıfırdan kurulumu adım adım anlatır.

### Aşama 6 — Tahmin maskeleri üzerinde doğruluk (iş istasyonu çıktıları `git pull` ile geldikten sonra)

`scripts/run_prediction_eval.py`: Aşama 3'teki harness'ı `--masks outputs/05_predictions/oof` ile çalıştır; MAE/RMSE/ICC/Bland–Altman; karma model; eşik bazlı sınıf uyumu; sınır hatası ayrıştırması. Test alt kümesi ve out-of-fold tamamı ayrı raporlanır.

### Aşama 7 — Raporlama (yerel)

`gsv4/report/*`, `scripts/build_report.py`:
- Sistem blok diyagramı (Mermaid + PNG) — Reviewer 1.
- Segmentasyon örnek çıktıları (GT ve tahmin yan yana, 6 görüntü) — Reviewer 1.
- Bland–Altman, dağılım, öğrenme eğrisi, sınır hatası figürleri.
- Makale tabloları CSV/Markdown: veri seti sayıları (temizlik öncesi/sonrası, bölüntü başına), demografi, segmentasyon metrikleri (test), ölçüm doğruluğu, uzman uyumu.
- `outputs/07_report/REVIZYON_OZETI.md`: her hakem maddesi ↔ hangi çıktı cevaplıyor.

---

## 4. Doğrulanmış gerçekler (tekrar keşfetme)

- COCO kategori: `1 = diseti`, `2 = dudak`, `0` supercategory.
- Görüntü sayıları: high 216 / low 303 / normal 796 = 1.315; instance: dişeti 3.938, dudak 1.318.
- Görüntüler ~2698×1799; ImageJ ölçümleri bu çerçevede yapıldı.
- Excel kodlaması: `3207` → 3.207 mm; `*0.718` → 0.718 mm; `-` → 0 mm (klinik ekip onayladı); `25-IMG_4552` öneki = yaş.
- `IMG_2544` ≠ `IMG_2544.`; Roboflow noktayı `-` yapar; ama COCO'daki her `-` Excel'de noktalı değildir (kademeli eşleştirme).
- Aynı-hasta çiftleri: 55, hepsi doğrulandı; problu çekim ölçülen çekimdir.
- Veri setinde E4 (> 8 mm) **yoktur**; maks ortalama 7.53 mm.
- Referans gözlemci içi: ICC 0.995, SD 0.17 mm.
- İstatistikçi kararları: birincil kappa = doğrusal ağırlıklı; diş bazlı analizler karma model; referans = çoğunluk, kör konsensüs.
- v3 Figure 6'daki v3 ve v1 değerleri güvenilir değildir; yeniden üretilmeyecek, açıklanacaktır.

---

## 5. Çalışma yöntemi

1. Aşama 0'ı bitirmeden kod yazma. Okuduklarını 20 satırlık bir `docs/OKUMA_NOTU.md` ile özetle: en kritik 5 hata, en kritik 5 kural. Bunu ilk commit'e koy.
2. Her aşamada önce testleri yaz, sonra kodu.
3. Sayısal beklentiden sapma varsa (ör. 145 yerine 140 çıkıyor) **durma ama raporla**: `outputs/<asama>/SAPMALAR.md`. Sapma büyükse (> %5 veya mantıksal) dur ve sor.
4. Belgelerde çelişki bulursan (v2 prompt ile analiz planı arasında vb.) **analiz planı ve audit Bölüm A/B** daha yenidir, onları esas al; çelişkiyi `docs/OKUMA_NOTU.md`'ye yaz.
5. Aşama 5'in eğitimini burada (Mac) başlatma; betikler ve `README_TRAINING.md` hazır olunca kullanıcıya "iş istasyonunda çalıştırılmaya hazır" de ve Aşama 4 ile 7'nin yerel kısımlarına devam et.
6. Her commit push edilir. Depo dalı: `master`.

Başla: Aşama 0.
