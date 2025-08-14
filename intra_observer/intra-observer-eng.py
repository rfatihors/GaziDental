import argparse


def main(file_first: str, file_last: str) -> None:
    """Run intra-observer consistency analysis."""

    import pandas as pd
    import numpy as np
    import pingouin as pg
    from scipy.stats import ttest_rel

    # Read files in wide format
    df1 = pd.read_excel(file_first)
    df2 = pd.read_excel(file_last)

    # Fix column names (check the first column's name)
    df1 = df1.rename(columns={df1.columns[0]: 'patient_id'})
    df2 = df2.rename(columns={df2.columns[0]: 'patient_id'})

    # Convert wide format to long format (patient_id, tooth_region, measurement_value)
    df1_long = df1.melt(id_vars=['patient_id'], var_name='tooth_region', value_name='Test')
    df2_long = df2.melt(id_vars=['patient_id'], var_name='tooth_region', value_name='Retest')

    # Convert tooth_region to string (if not already)
    df1_long['tooth_region'] = df1_long['tooth_region'].astype(str)
    df2_long['tooth_region'] = df2_long['tooth_region'].astype(str)

    # Merge (to match two measurements for each patient and tooth)
    df = pd.merge(
        df1_long,
        df2_long,
        on=['patient_id', 'tooth_region'],
        how='inner'
    )

    # Convert to long format for ICC and paired t-test
    long_df = pd.DataFrame({
        'patient_id': list(df['patient_id']) * 2,
        'tooth_region': list(df['tooth_region']) * 2,
        'measurement_time': ['Test'] * len(df) + ['Retest'] * len(df),
        'rater': ['Observer_1'] * len(df) * 2,
        'measurement_value': pd.concat([df['Test'], df['Retest']], ignore_index=True)
    })

    print("First 10 rows from the merged dataset:")
    print(long_df.head(10))
    print("-" * 50)

    # --- INTRA-CLASS CORRELATION (ICC) and Paired t-test ---

    print("\n--- Step 2: Starting Intra-Class Correlation (ICC) and Paired t-test Analysis ---")

    icc_results_list = []
    tooth_regions = sorted(df['tooth_region'].unique(), key=lambda x: int(x))

    for region in tooth_regions:
        print(f"\nAnalyzing for tooth region: {region}...")
        region_long_df = long_df[long_df['tooth_region'] == region]
        icc = pg.intraclass_corr(data=region_long_df, targets='patient_id', raters='measurement_time', ratings='measurement_value')
        icc.set_index('Type', inplace=True)
        icc3_value = icc.loc['ICC3', 'ICC']
        ci95 = icc.loc['ICC3', 'CI95%']
        # Paired t-test
        d = df[df['tooth_region'] == region]
        t_stat, p_value = ttest_rel(d['Test'], d['Retest'])
        is_consistent = "Yes" if p_value > 0.05 else "No"
        print(f"ICC3 (Absolute Agreement) value for {region} region: {icc3_value:.4f}")
        print(f"Paired t-test p-value for {region} region: {p_value:.4f} (Consistent? {is_consistent})")
        icc_results_list.append({
            'Tooth Region': region,
            'ICC Value': icc3_value,
            '95% Confidence Interval': ci95,
            'Paired t-test p-value': f"{p_value:.4f}",
            'Measurements Consistent?': is_consistent
        })

    print("\n" + "="*80)
    print("     INTRA-OBSERVER CONSISTENCY AND SYSTEMATIC DIFFERENCE ANALYSIS RESULTS")
    print("="*80)
    results_df = pd.DataFrame(icc_results_list)
    print(results_df.to_string(index=False))

    print("\n--- Interpretation ---")
    print("ICC Values indicate how consistent measurements taken at two different times are:")
    print(" - 0.90 and above: Excellent Reliability")
    print(" - 0.75 – 0.90: Good Reliability")
    print(" - 0.50 – 0.75: Moderate Reliability")
    print(" - Less than 0.50: Poor Reliability")
    print("\nIf Paired t-test p-value > 0.05: There is no systematic difference between measurements, measurements are repeatable and consistent.")
    print("If Paired t-test p-value <= 0.05: There might be a systematic difference between measurements.")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intra-observer consistency analysis")
    parser.add_argument("file_first", help="Path to the first calibration Excel file")
    parser.add_argument("file_last", help="Path to the last calibration Excel file")
    args = parser.parse_args()
    main(args.file_first, args.file_last)
