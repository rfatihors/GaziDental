# Claude Code'u başlatmadan önce — kurulum

## 1. Yerel depoyu güncelle

```bash
cd ~/<depo-yolu>/GaziDental
git pull origin master
ls gummy_smile_v4          # silinecek.md görünmeli
```

Depo Mac'te tam klon değilse (sparse ise), v3'ün tamamı lazım:

```bash
git sparse-checkout set gummy_smile_v3 gummy_smile_v4
```

`gummy_smile_v3/data/coco_dataset/` altında `high`, `low`, `normal` klasörlerinin hepsinin olduğunu kontrol et; v4 görüntülere buradan erişecek.

## 2. Belgeleri ve girdileri yerleştir

```bash
cd gummy_smile_v4
mkdir -p docs data/inputs data/expert
```

`docs/` içine (bu sohbetten indir):
- `Veri_envanteri.md`
- `GummySmile_v3_Teknik_Audit_Raporu_v2.md`
- `PROMPT_olcum_modulu_yeniden_yazim_v2.md`
- `Istatistik_analiz_plani.md`
- `Uzman_degerlendirme_protokolu.md`
- `Guc_analizi_hakem_cevabi_ve_metin.md`

`data/inputs/` içine:
- `Hasta_ID-_Ölçümler.xlsx` (hocadan gelen)
- `calibration.xlsx` (hocadan gelen)
- `ayni_hasta_aday_ciftler.csv`
- `olcumsuz_yuksek_gulme_hatti_66.csv`
- `uzman_seti_145_goruntu.csv`
- `ANAHTAR_arastirmaci.csv`

Görev dosyasını kök dizine koy: `CLAUDE_CODE_gummy_smile_v4_gorev.md`

Kontrol:

```bash
ls docs            # 6 dosya
ls data/inputs     # 6 dosya
```

## 3. Claude Code'u başlat

```bash
cd ~/<depo-yolu>/GaziDental/gummy_smile_v4
claude
```

İlk mesaj:

```
CLAUDE_CODE_gummy_smile_v4_gorev.md dosyasını oku ve Aşama 0'dan başla.
```

## 4. Beklenen akış

| Aşama | Nerede | Süre (tahmini) | Senin yapacağın |
|---|---|---|---|
| 0 Okuma | yerel | 15 dk | — |
| 1 Veri katmanı | yerel | 1–2 saat | Sayıları kontrol et: 692 satır, 149 eşleşme, 145 high |
| 2 Ölçüm + kural motoru | yerel | 2–3 saat | — |
| 3 Oracle | yerel | 1–2 saat | MAE ≈ 0.6 mm civarı çıkmalı; çok farklıysa dur |
| 4 Uzman analizi (sentetik) | yerel | 2 saat | — |
| 5 Eğitim hattı | yazılır | 1–2 saat | İş istasyonunda (RTX 5090): `git pull`, `README_TRAINING.md`'ye göre kurulum, `scripts/train_all.sh` — 9 eğitim, gece |
| 6 Tahmin doğruluğu | yerel | 1 saat | İş istasyonunda `git push` (maskeler + results.csv commit edilir), Mac'te `git pull` |
| 7 Rapor | yerel | 1–2 saat | — |

Uzman formları gelince: `data/expert/Uzman_{1,2,3}_form.xlsx` altına koy, `scripts/run_expert_analysis.py` çalıştır. Aşama 4 sentetik veriyle hazırlandığı için ek kod gerekmez.

## 5. Oturum kesilirse

Claude Code'u aynı klasörde yeniden başlat ve:

```
CLAUDE_CODE_gummy_smile_v4_gorev.md ve docs/OKUMA_NOTU.md dosyalarını oku, git log'a bak, kaldığın aşamadan devam et.
```

## Notlar

- Eğitim iş istasyonunda (RTX 5090, 32 GB). 9 eğitim × ~1–1.5 saat; `scripts/train_all.sh` gece çalışacak şekilde yazılıyor. 5090 Blackwell mimarisi olduğu için PyTorch'un CUDA 12.8 derlemesi şart (`torch>=2.7`, cu128); eski bir torch kuruluysa `CUDA error: no kernel image` benzeri hata alırsın, `README_TRAINING.md`'deki kontrol komutuyla başta doğrula.
- İş akışı: Claude Code Mac'te kodu yazar ve push eder → iş istasyonunda `git pull` + `train_all.sh` → maskeler ve results.csv commit/push → Mac'te `git pull` ile Aşama 6.
- Diğer sohbetteki içeriğe erişemiyorum; orada bu belgelerde olmayan bir karar varsa `docs/` altına kısa bir not olarak ekle, Claude Code onu da okusun.
