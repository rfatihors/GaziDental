import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import ttest_rel

# Dosya yolları
file_first = "dental/gazi_gum_project/intra_observer/calibration-first.xlsx"
file_last  = "dental/gazi_gum_project/intra_observer/calibration-last.xlsx"

# Geniş formatta dosyaları oku
df1 = pd.read_excel(file_first)
df2 = pd.read_excel(file_last)

# Kolon adlarını düzelt (ilk sütun isimlerini kontrol et)
df1 = df1.rename(columns={df1.columns[0]: 'patient_id'})
df2 = df2.rename(columns={df2.columns[0]: 'patient_id'})

# Geniş formatı uzun formata çevir (patient_id, tooth_region, measurement_value)
df1_long = df1.melt(id_vars=['patient_id'], var_name='tooth_region', value_name='Test')
df2_long = df2.melt(id_vars=['patient_id'], var_name='tooth_region', value_name='Retest')

# tooth_region'ı string'e çevir (eğer değilse)
df1_long['tooth_region'] = df1_long['tooth_region'].astype(str)
df2_long['tooth_region'] = df2_long['tooth_region'].astype(str)

# Merge (her hasta ve diş için iki ölçüm eşleşecek)
df = pd.merge(
    df1_long,
    df2_long,
    on=['patient_id', 'tooth_region'],
    how='inner'
)

# ICC ve paired t-test için uzun formata çevir
long_df = pd.DataFrame({
    'patient_id': list(df['patient_id']) * 2,
    'tooth_region': list(df['tooth_region']) * 2,
    'measurement_time': ['Test'] * len(df) + ['Retest'] * len(df),
    'rater': ['Observer_1'] * len(df) * 2,
    'measurement_value': pd.concat([df['Test'], df['Retest']], ignore_index=True)
})

print("Birleştirilmiş veri setinden ilk 10 satır:")
print(long_df.head(10))
print("-" * 50)

# --- SINIF-İÇİ KORELASYON (ICC) ve Paired t-test ---

print("\n--- Adım 2: Sınıf-İçi Korelasyon (ICC) ve Paired t-test Analizi Başlatılıyor ---")

icc_results_list = []
tooth_regions = sorted(df['tooth_region'].unique(), key=lambda x: int(x))

for region in tooth_regions:
    print(f"\n{region} bölgesi için analiz yapılıyor...")
    region_long_df = long_df[long_df['tooth_region'] == region]
    icc = pg.intraclass_corr(data=region_long_df, targets='patient_id', raters='measurement_time', ratings='measurement_value')
    icc.set_index('Type', inplace=True)
    icc3_value = icc.loc['ICC3', 'ICC']
    ci95 = icc.loc['ICC3', 'CI95%']
    # Paired t-test
    d = df[df['tooth_region'] == region]
    t_stat, p_value = ttest_rel(d['Test'], d['Retest'])
    is_consistent = "Evet" if p_value > 0.05 else "Hayır"
    print(f"{region} bölgesi için ICC3 (Mutlak Uyum) değeri: {icc3_value:.4f}")
    print(f"{region} bölgesi için Paired t-test p-değeri: {p_value:.4f} (Tutarlı mı? {is_consistent})")
    icc_results_list.append({
        'Diş Bölgesi': region,
        'ICC Değeri': icc3_value,
        '%95 Güven Aralığı': ci95,
        'Paired t-test p-değeri': f"{p_value:.4f}",
        'Ölçümler Tutarlı mı?': is_consistent
    })

print("\n" + "="*80)
print("         INTRA-OBSERVER TUTARLILIK VE SİSTEMATİK FARK ANALİZİ SONUÇLARI")
print("="*80)
results_df = pd.DataFrame(icc_results_list)
print(results_df.to_string(index=False))

print("\n--- Yorumlama ---")
print("ICC Değerleri, iki farklı zamanda yapılan ölçümlerin ne kadar tutarlı olduğunu gösterir.")
print(" - 0.90 ve üzeri: Mükemmel Güvenilirlik")
print(" - 0.75 – 0.90: İyi Güvenilirlik")
print(" - 0.50 – 0.75: Orta Düzeyde Güvenilirlik")
print(" - 0.50'den az:  Zayıf Güvenilirlik")
print("\nPaired t-test p-değeri > 0.05 ise: Ölçümler arasında sistematik bir fark yoktur, ölçümler tekrarlanabilir ve tutarlıdır.")
print("Paired t-test p-değeri <= 0.05 ise: Ölçümler arasında sistematik fark olabilir.")
print("="*80)
