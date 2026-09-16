# Aşama 7 — Türkçe özet

- `scripts/build_report.py` mevcut verilerle 14 öge üretti, 1 öge bekliyor (`report_status.md`: ne gerektiği yazıyor). Veri gelince aynı komut yeniden çalıştırılır.
- Hazır: blok diyagramı, GT overlay, GT-maske ölçüm figürü (Figure 6 yerine), veri seti sayıları, demografi, mm doğruluğu (GT satırları), gözlemci içi.
- Bekleyen: segmentasyon örnekleri, öğrenme eğrisi, sınır hatası, test metrikleri (iş istasyonu); tahmin maskesi doğruluğu (Aşama 6); uzman uyumu (gerçek formlar).
- `REVIZYON_OZETI.md`: hakem maddesi ↔ çıktı eşlemesi (✅ / ⏳).
- Hakem cevabı taslağı: `RESPONSE_TO_REVIEWERS.md`, durum tablosu `REBUTTAL_DURUM.md`, Türkçe özet `REBUTTAL_OZET.md` (üretim: `scripts/build_rebuttal.py`).
