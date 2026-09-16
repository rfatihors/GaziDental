# Rebuttal — Türkçe özet

- Taslak: `RESPONSE_TO_REVIEWERS.md` (15 madde, 4 hakem). Tüm sayılar `outputs/` dosyalarından okundu; her cevabın altında kaynak dosya yorumu var.
- **Hazır (12)**: R1-1 (System diagram missing), R1-2 (Segmentation outputs not shown), R2-5 (Image counts (1,315 → 2,235 / 3,403)), R2-6 (Data leakage between splits), R2-10 (Boundary accuracy; lip vs gingiva performance gap), R2-power (G*Power calculation inappropriate), R2-fig6 (Figure 6 inconsistency (measurement vs reference)), R3-1 (Demographics / sex distribution), R4-1 (Sample size and external validity), R4-2 (Calibration (pixel to mm)), R4-5 (Data leakage (CLAIM 2024)), R4-6 (Intra-observer reliability of the reference).
- **Uzman verisi bekleyen (1)**: R4-3 (Circular validation of the E/T classification). Formlar analiz edilince `run_expert_analysis.py` → `manuscript_numbers.md`; ayrıca kalibrasyon cevabındaki uzman ölçeği cümlesi (R4-2 içinde [PENDING] işaretli).
- **Klinik ekipten metin bekleyen (2)**: R4-4 (Overlap with the earlier (J Dent 2026) cohort), R4-7 (Terminology, rare classes and limitations). Örtüşme paragrafı klinik ekibin 15 Eylül metninden taslak olarak kondu; sınırlılıklar için teknik maddeler listelendi.
- Ölçüm doğruluğu her yerde düzeltmesiz birincil, maske düzeyi düzeltme ikincil.
- Açıkça kabul edilen hatalar: eski ölçüm modülünün geometri hatası (eski Figure 6; v3 ve v1 kolonları geçersiz, figür yeniden üretildi), aynı hastanın iki fotoğrafının bölüntüler arasında bulunması (hasta düzeyi yeniden bölünme, yeniden eğitim), gözlemci içi dosyasında 15 vs 20 görüntü (tam dosyayla yeniden hesaplandı).
- Yapamadıklarımız açıkça yazıldı: E4 sınıfı veri setinde yok; dış geçerlilik yok (tek merkez/cihaz); pigmentasyon kaydı yok; demografi kısmi.
- Eksik: hakemlerin orijinal metni (`docs/Hakem_revizyonları.docx`) depoda yok; alıntı yerlerinde yer tutucu var. Belge gelince `python scripts/build_rebuttal.py --docx <dosya>` paragrafları listeler, alıntılar `docs/hakem_maddeleri.yaml`'a yapıştırılır ve betik yeniden çalıştırılır.
- Terminoloji: 'high smile line'; 'gummy smile / excessive gingival display' kullanılmadı.
