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

## Kararlar (kullanıcı, 12 Eylül 2026 — çelişkileri kapatır)

1. 66 ölçümsüz high **çıkar**; `dataset.keep_unmeasured_high_in_train` bayrağı (varsayılan `false`); `true` ise yalnızca train, hiçbir analize girmez.
2. Eğitim yalnızca RTX 5090 iş istasyonunda; Colab defteri yok; `requirements-train.txt`, `scripts/train_all.sh`, `README_TRAINING.md`; tahmin PNG'leri + `results.csv`/`args.yaml` commit edilir.
3. `IMG_8645` ve `27-IMG_8645` farklı anahtarlar; belirsizlik yok; yalnızca `IMG_7366` belirsiz (Aşama 1'de doğrulandı: tek `row_ambiguous`).
4. Tekil eşleşme hedefi 149 (Aşama 1: 119 exact + 30 dash_base_fallback = 149).
5. Prob kalibrasyon aracı kapsam dışı; uzmanlar px/mm'i formda verir.
6. `data/inputs/` ve `data/expert/` versiyonlanır.

## Aşama 1'de öğrenilenler

- iPhone kökleri gruplar arasında tekrarlar (8 kök: `IMG_3068`, `IMG_3110`, `IMG_3129`, `IMG_3647`, `IMG_3660`, `IMG_3677`, `IMG_7350`, `IMG_8648`) ve `normal` içinde 4 ad yalnızca uzantıyla ayrılır (`IMG_1556_JPEG`/`IMG_1556_jpg` …; piksel düzeyinde farklı fotoğraflar). Kimlik = `uid = grup/kök`; `patient_id` = tutulan görüntünün `uid`'si.
- Excel yaş sütununda 4 hücre not/legend metnidir (`Not 1`, `Not 2`, `K: Kadın`, `E: Erkek`); yaş 630 satırda dolu (spec'teki 634 bu metinleri sayıyordu).
- 8 ad tiresiz yaş önekli (`60IMG_4347`); önek kuralı bunu da kapsar.
- Çapraz-grup çiftler: `low/IMG_1207` ↔ `normal/IMG_1178` (ikisinde de yaş/cinsiyet yok → `image_a` = low tutuldu), `low/IMG_9696` ↔ `normal/52-IMG_9695` (önekten yaş → normal tutuldu). Sonuç low 300 / normal 785 (beklenti ≈299/≈786; fark bu iki karardan).
- 2698×1799 ±2 px çerçevesinde 830 görüntü, dışında 485 (oracle duyarlılık analizinde ayrı tutulacak).

## Aşama 3'te öğrenilenler

- Seçilen yöntem `C_p25` (dev MAE 0.557; C_p10 0.578 → 0.02 mm toleransın hemen dışında), global ölçek 16.84 px/mm (yalnız dev). Holdout MAE 0.52 mm, ICC(2,1) 0.858. C yöntemi 28/145 görüntüde zenith bulamayıp A'ya düşer; raporda açık.
- Gözlemci içi (calibration.xlsx): ICC(2,1) 0.995 diş / 0.998 görüntü, SD 0.17 mm — audit ile birebir.
- Boşluk medyanı 8 px (IQR 5–12); beklenti 9 px ile uyumlu. Dudak-ankrajlı tahminciler dev'de daha kötü.
- Çerçeve dışı 22/145 görüntü: MAE 0.78 vs 0.46 mm; ölçek göstergeleri zıt yönde → ikili ölçek önerilmedi, duyarlılık satırı olarak raporlandı.
- `label_inconsistent` iki satır (IMG_9152, IMG_9206) COCO'da yok; dışlama etkisiz.

## Aşama 4'te öğrenilenler

- Uzman analizi sentetik formlarla uçtan uca çalışıyor; gerçek formlar `data/expert/Uzman_{1,2,3}_form.xlsx` olarak gelince aynı komut bayraksız çalıştırılır; `consensus.csv` (image,class) konursa `consensus_pending` vakalar çözülür.
- Model sınıfı = `per_image_results.csv` seçilen yöntem mm + kural motoru; global ölçek birincil, uzman ölçeği ikincil. Puanlama: katı / esnek / lenient2.
- Diş bazlı karma modelde hasta varyansı sınıra (≈ 0) düşerse MixedLM GA'ları tanımsız olur; kod küme-dayanıklı OLS'ye düşer ve bunu raporlar (sentetik veride bu oldu; gerçek veride hasta etkisi beklenir).
- Kappa yaygınlığa duyarlı: 145 görüntünün ~%78'i E1 → Fleiss/Cohen düşük çıkabilir; gözlenen uyum ve PABAK yanında raporlanıyor.

## Aşama 5'te öğrenilenler

- Ön-belirleme değişikliği (12 Eylül, gerçek form gelmeden): uzman analizinde birincil küme = 145 görüntü OOF tahminleri, ikincil = sabit test alt kümesi (n = 29 için κ GA'sı kullanılamayacak kadar geniş). Aşama 6'da da aynı.
- YOLO veri seti symlink'lerle `data/yolo_dataset/` altında; dosya adları `grup__kök` ve boşluk/parantez `_` yapılır (`yolo_index.csv` geri eşler). Listeler satır bazlı okunur.
- Fold eğitiminde valid = ana valid − held-out fold (best.pt seçimi held-out'a sızmaz); fold modeli yalnızca kendi fold'unu tahmin eder; test tahmini yalnız final modelden.
- `train_all.sh` `runs/<ad>/DONE` işaretleriyle yeniden başlatılabilir; 9 eğitim + toplama + değerlendirme.

## Aşama 7 (yerel kısım)

- `scripts/build_report.py` her çalıştırmada mevcut verilerle figür/tabloları üretir; iş istasyonu (Aşama 5/6) ve gerçek uzman formları (Aşama 4) gelmeden bekleyen ögeler "pending" yer tutucu olarak yazılır, `report_status.md` neyin gerektiğini listeler. `REVIZYON_OZETI.md` hakem maddesi ↔ çıktı eşlemesi.
- Post-hoc kalibrasyon (16 Eylül 2026): `measurement.offset_px = -13` (alt dişeti kenarı; Aşama 3 dev listesinde kestirildi, holdout ve test setinde doğrulandı; `06_prediction/offset_correction.md`, `offset_checks.md`). Piksel cinsinden tanımlı, her görüntünün geçerli px/mm'iyle mm'ye çevrilir; 0'ın altı 0'a kırpılır ve `clipped` bayrağı konur. **Düzeltmesiz = birincil, düzeltilmiş = ikincil; her tabloda ikisi de, birincil önce.**
- Sıra: iş istasyonunda `train_all.sh` → `git pull` → Aşama 6 (`scripts/run_prediction_eval.py`: yöntem ve ölçek config'ten sabit, yeniden uydurma yok; fallback oranı > %30 ise A_p25'e karşı yeniden değerlendirme — satır her durumda tabloda) → gerçek formlar gelince `run_expert_analysis.py --model-table outputs/06_prediction/per_image_results.csv` → `build_report.py`.

## Ortam durumu

- COCO: high 158/28/30 = 216, low 214/45/44 = 303, normal 558/119/119 = 796 — envanterle birebir; her bölüntüde `_annotations.coco.json` var.
- `data/inputs/` altında 6 girdi dosyası mevcut (12 Eylül). v3'teki eski `ölçümler ai guncel.xlsx` ve 15 görüntülük `calibration-first/last.xlsx` kullanılmaz.
- v3: `splits.json`, `labels_smileline.csv`, `results/*.csv` boş (0 bayt); `best.pt` yok; `requirements.txt` sürümsüz.
- Python 3.11.5 (anaconda) mevcut; git `origin/master` ile senkron.
