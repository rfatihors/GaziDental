from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import ttest_rel


def _prepare_long_format(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.rename(columns={df.columns[0]: "patient_id"})
    return df.melt(id_vars=["patient_id"], var_name="tooth_region", value_name=value_name)


def run_intra_observer(file_first: Path, file_last: Path, output_csv: Path) -> Path:
    df1 = pd.read_excel(file_first)
    df2 = pd.read_excel(file_last)

    df1_long = _prepare_long_format(df1, "Test")
    df2_long = _prepare_long_format(df2, "Retest")

    df1_long["tooth_region"] = df1_long["tooth_region"].astype(str)
    df2_long["tooth_region"] = df2_long["tooth_region"].astype(str)

    merged = pd.merge(df1_long, df2_long, on=["patient_id", "tooth_region"], how="inner")

    long_df = pd.DataFrame(
        {
            "patient_id": list(merged["patient_id"]) * 2,
            "tooth_region": list(merged["tooth_region"]) * 2,
            "measurement_time": ["Test"] * len(merged) + ["Retest"] * len(merged),
            "rater": ["Observer_1"] * len(merged) * 2,
            "measurement_value": pd.concat([merged["Test"], merged["Retest"]], ignore_index=True),
        }
    )

    results: List[dict] = []
    tooth_regions = sorted(merged["tooth_region"].unique(), key=lambda x: int(x))
    for region in tooth_regions:
        region_long = long_df[long_df["tooth_region"] == region]
        icc_table = pg.intraclass_corr(
            data=region_long,
            targets="patient_id",
            raters="measurement_time",
            ratings="measurement_value",
        )
        icc_table.set_index("Type", inplace=True)
        icc3_value = icc_table.loc["ICC3", "ICC"]
        ci95 = icc_table.loc["ICC3", "CI95%"]

        region_df = merged[merged["tooth_region"] == region]
        t_stat, p_value = ttest_rel(region_df["Test"], region_df["Retest"])
        consistent = "Evet" if p_value > 0.05 else "Hayır"

        results.append(
            {
                "Diş Bölgesi": region,
                "ICC Değeri": float(icc3_value),
                "%95 Güven Aralığı": ci95,
                "Paired t-test p-değeri": float(p_value),
                "Ölçümler Tutarlı mı?": consistent,
            }
        )

    results_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    return output_csv
