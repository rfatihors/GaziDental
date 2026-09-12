# OKUMA_NOTU — Aşama 0 özeti (12 Eylül 2026)

Kaynaklar: docs/ altındaki 6 belge + `../gummy_smile_v3/` kodu (salt okunur). Audit'teki bulguların hepsi kod üzerinde birebir doğrulandı.

## En kritik 5 hata (v3)

1. **Yanlış geometri** — `measurement/*.py`: `max(contours, key=area)` (pratikte dudak) alınıp `zenith_y − lip_line_y` = aynı konturun **üst** kenarının bölgeler arası dalgalanması hesaplanıyor; gingival margin (alt kenar) hiç kullanılmıyor. Ölçülen şey Cupid yayının eğriliği.
2. **Sınıf ayrımı yok** — `yolo/infer_yolo*.py`: `masks.data.max(dim=0)` ile diseti+dudak tek maske; `boxes.cls` okunmuyor; letterbox çözünürlüğünde maske orijinal görüntüye bindiriliyor; `imgsz: 1024` (model 640), `max_det: 5` (normal/low'da papilla parçalarını keser).
3. **Birim karışıklığı** — `mm_per_pixel=1.0` varsayılanıyla piksel değerleri `mean_mm` kolonuna yazılıyor; `px_per_mm` ve `mm_per_pixel` birlikte kullanılıyor.
4. **Kural motoru** — `NaN → E1/T1`; "hiçbir kural eşleşmedi" → E1; E1–E2 (3–4 mm) çakışması hiç yok; sınırlar tutarsız (E1 `≤4` vs `<4`); alt sınır yok (0 mm → E1); E2–E3 belirsizliği `use_metadata: true` ile **dosya yolundaki klasör adından** çözülüyor.
5. **Üç farklı eşik uygulaması + iki giriş noktası** — `rule_engine.py` (5 mm → E2), `diagnosis.py` (5 mm → E3, öncelik E1>E3>E2>E4), `method_comparison._severity_label_from_mm` (ilk eşleşen, `<max`). `diagnosis._severity_from_mean` gülme hattını mm'den türetiyor (döngüsel). Ek: bölünme görüntü düzeyinde; aynı hastanın problu/probsuz çifti (55; 22 çapraz bölüntü, 15'i test) → sızıntı. `evaluate.py` n=1 metrik; `intra_observer.py` t-testi p>0.05'i "tutarlı" sayıyor.

## En kritik 5 kural (v4)

1. **Sınır**: `gummy_smile_v3/` altına hiçbir şey yazılmaz; görüntü kopyalanmaz; yol `configs/config.yaml`'da tek yerde (`../gummy_smile_v3/data/coco_dataset/`); `*.pt`, `runs/`, `outputs/`, maskeler git'e girmez; her aşama: testler → commit → push; seed 42; Python 3.11, sabit sürümler; kod/docstring İngilizce.
2. **Ölçüm** = `diseti` maskesinin sütun bazlı dikey kalınlığı (en uzun kesintisiz run; delik varsa `y_max−y_min` değil); boş sütun **0** (NaN değil); aynı sınıfın tüm instance'larının union'ı, "en büyük kontur" yok; maske orijinal çözünürlükte (`retina_masks=True` / `masks.xy`, `assert shape`); bölgeleme A/B/C, tahminci p10/p25/median/min; dudak yalnızca sınır denetimi (gap medyan ≈ 9 px) ve `*_lipanchored` tahminci için; tek birim `px_per_mm`, kalibrasyon yoksa `unit="px"`, mm `None`.
3. **Excel/eşleştirme**: `≥1000 → ÷1000`, `*0.718 → 0.718`, `-` → **0 mm** (`dash_zero` bayrağı), `100–999 → ÷1000 + ambiguous`, negatif → NaN, `'3.00'` → 3.0; E etiketiyle tutarlılık denetimi (≤5 uyumsuz beklenir); ad: nokta/`-` → `~` **korunur**, yaş öneki **korunur**, `ı→i`, `.rf.<hash>`/uzantı atılır; kademeli eşleştirme (tam → `dash_base_fallback` → `name_ambiguous`); `IMG_xxxx` benzersiz kimlik değildir.
4. **Kural motoru** tek: E1 `<4`, E2 `[3,6]`, E3 `[4,8]`, E4 `>8`; çakışma → `E1-E2` / `E2-E3` birleşik etiket + `ambiguous=True`; `value<=0 → NO_VISIBLE_GINGIVA`; `NaN → UNCLASSIFIED`; `use_metadata=false`; çıktı `treatment_alternatives`; `applicability`: yalnızca yüksek gülme hattı. Sabit noktalar: 3.0→E1-E2, 4.0→E2-E3, 8.0→E3, 8.1→E4.
5. **İstatistik**: birincil = doğrusal ağırlıklı Cohen kappa (+ ağırlıksız, gözlenen uyum, PABAK; bootstrap 2000, seed 42); referans = çoğunluk, üçü farklıysa kör konsensüs (`consensus_pending`); diş bazlı → `MixedLM` (`diff ~ 1 + (1|patient)`, `diff ~ tooth + (1|patient)`), bağımsız-gözlem ICC yok; ICC(2,1) ve ICC(2,k); E4 sıfır vaka → sıfıra bölme güvenli; oracle %60 dev / %40 holdout; sabit test seti; sapma raporla, >%5 ise dur.

## Belgeler arası çelişkiler / açık noktalar (esas: görev belgesi > analiz planı > audit A/B > PROMPT v2)

1. **66 ölçümsüz high**: PROMPT §4.8 varsayılanı `train_only` (≈216 high / 1.295 toplam); audit B "önerildi, onay bekleniyor"; **görev belgesi "çıkar" (high 145, low ≈299, normal ≈786)**. Görev belgesi esas; `--unmeasured-high {drop,train_only}` seçeneği kodda tutulur, varsayılan `drop`.
2. **Eğitim ortamı**: görev belgesi Colab (`notebooks/train_colab.ipynb`); kurulum belgesi RTX 5090 iş istasyonu (`scripts/train_all.sh`, `README_TRAINING.md`, torch cu128). Aşama 5'te ikisi de üretilecek; aynı `gsv4/train` betiklerini çağırır. Kullanıcıya sorulacak.
3. **`IMG_8645`**: PROMPT §4.2 hem `IMG_7366` hem `IMG_8645`'i `row_ambiguous` sayıp çıkarıyor; envanter/görev belgesi yalnızca `IMG_7366`'yı çıkarıyor (216−66−1−4=145). Önek korunduğunda `27-IMG_8645` ayrı ad olduğundan `IMG_8645` tekil eşleşir; görev belgesi esas, Aşama 1'de sayısal olarak doğrulanacak.
4. **Tekil eşleşme sayısı**: 147 (audit 5.4 / PROMPT), 149 (görev belgesi, envanter), 150 ölçümlü. Görev belgesi (149) beklenti; sapma `SAPMALAR.md`'ye.
5. **Çift tutma kuralı**: görev belgesi "problu olan = `uzman_seti_145_goruntu.csv`'deki"; PROMPT "yaş/cinsiyeti olan, yoksa `image_a`". High'da uzman seti listesi esas; low/normal'de yaş/cinsiyet kuralı. Uygulamada ikisi tutarlı olmalı, değilse raporlanır.
6. **Uzman anonim ID**: protokol `U1-001` + `ANAHTAR_uzman_N.csv` (uzman başına); görev/envanter `G001–G165` + tek `ANAHTAR_arastirmaci.csv`. Gerçek dosya (`ANAHTAR_arastirmaci.csv`) esas.
7. **Örneklem metni**: analiz planı §3.1 "150 görüntü ≈ %90 güç"; gerçek n = 145. Makale metninde sayı güncellenmeli (rapora not).
8. **Prob kalibrasyon aracı** (PROMPT §4.4, OpenCV tıklama aracı): görev belgesi Aşama 3'te yok. Kapsam dışı sayılıyor; Aşama 3 planında kullanıcıya sorulacak.
9. **Kurulum §4**: Aşama 6 için "maskeler + results.csv commit edilir" diyor; görev belgesi üretilmiş maskeleri `.gitignore`'a koyuyor. OOF maskeleri için karar Aşama 5'te (öneri: maskeler PNG olarak `outputs/` altında, git dışı; sadece `results.csv` commit).

## Ortam durumu

- COCO: high 158/28/30 = 216, low 214/45/44 = 303, normal 558/119/119 = 796 — envanterle birebir; her bölüntüde `_annotations.coco.json` var.
- **`data/inputs/` boş** — 6 girdi dosyasının hiçbiri yok (Aşama 1 bloklu). v3'te yalnızca eski `ölçümler ai guncel.xlsx` ve 15 görüntülük `calibration-first/last.xlsx` var; bunlar kullanılmayacak.
- v3: `splits.json`, `labels_smileline.csv`, `results/*.csv` boş (0 bayt); `best.pt` yok; `requirements.txt` sürümsüz.
- Python 3.11.5 (anaconda) mevcut; git `origin/master` ile senkron.
