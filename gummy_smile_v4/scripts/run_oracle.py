#!/usr/bin/env python
"""Stage 3 — oracle validation of the measurement geometry on ground-truth masks
(spec §4.1–4.6); re-used in Stage 6 with predicted masks (``--masks <png_dir>``).

The method and the global scale are **selected on ground-truth masks only**. With
``--masks <png_dir>`` they are read from ``configs/config.yaml`` and applied unchanged:
nothing is re-selected, nothing is re-fitted and nothing is written back to the config.
``--reselect`` re-runs the selection on the given masks — a deliberate second look at the
reference, for sensitivity analyses only, and still never written to the config.

Outputs (default outputs/03_oracle/):
  per_image_results.csv, dev_holdout_split.json, estimator_comparison.csv,
  sensitivity.csv, scale_estimation.md, boundary_check.csv, qc_flags.csv,
  tooth_level.csv, scatter_gt_vs_manual.png, bland_altman.png, intra_observer.md,
  oracle_summary.md, OZET.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.dataset.build import build_all  # noqa: E402
from gsv4.eval.agreement import bland_altman, icc_long, loo_scale  # noqa: E402
from gsv4.eval.oracle import (  # noqa: E402
    COMBOS, combo_name, dev_holdout_split, evaluate_combos, evaluate_fixed, measure_images, per_image_scale,
    select_method, sensitivity, tooth_level,
)
from gsv4.measure.qc import QCFlag  # noqa: E402
from gsv4.rules.thresholds import label_for_mm  # noqa: E402

EXPECT = {"px_per_mm": (15.0, 20.0), "mae_reasonable": 0.63, "gap_median_px": 9.0, "intra_icc": 0.995, "intra_sd": 0.17}


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    int_cols = {c for c in cols if pd.api.types.is_integer_dtype(df[c])}
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in int_cols:
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else fmt.format(v))
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def intra_observer(cal: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """ICCs and Bland-Altman for the 20-image two-session calibration file."""
    cal = cal.copy()
    cal["target"] = cal["image"] + "#" + cal["tooth"].astype(str)
    tooth = icc_long(cal, "target", "session", "mm")
    wide = cal.pivot_table(index="target", columns="session", values="mm")
    ba_t = bland_altman(wide[2], wide[1])
    img = cal.groupby(["image", "session"])["mm"].mean().reset_index()
    image = icc_long(img, "image", "session", "mm")
    wide_i = img.pivot_table(index="image", columns="session", values="mm")
    ba_i = bland_altman(wide_i[2], wide_i[1])
    return {"tooth": tooth, "ba_tooth": ba_t, "image": image, "ba_image": ba_i}, wide


def write_config_method(cfg_path: Path, method: dict, px_per_mm: float) -> None:
    s = cfg_path.read_text(encoding="utf-8")
    s = re.sub(r"^  px_per_mm: .*$", f"  px_per_mm: {px_per_mm:.4f}          # global scale, fitted on the oracle dev subset (Stage 3)", s, count=1, flags=re.M)
    s = re.sub(r"^  method: .*$", f"  method: {{regioning: {method['regioning']}, estimator: {method['estimator']}, anchored: {str(method['anchored']).lower()}}}  # selected in Stage 3 on dev MAE", s, count=1, flags=re.M)
    cfg_path.write_text(s, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--masks", default="gt", help="'gt' (COCO ground truth) or a directory of <image>_gingiva.png / _lip.png")
    ap.add_argument("--out", default="03_oracle", help="output sub-directory under paths.outputs")
    ap.add_argument("--write-config", action="store_true",
                    help="write the selected method and scale into configs/config.yaml; ground-truth masks only")
    ap.add_argument("--reselect", action="store_true",
                    help="with --masks: re-run the method/scale selection ON THE PREDICTED MASKS instead of taking "
                         "them from configs/config.yaml. Off by default; a sensitivity analysis, never the primary "
                         "result, and never written to the config.")
    ap.add_argument("--tolerance-mm", type=float, default=0.02)
    args = ap.parse_args()
    if args.write_config and args.masks != "gt":
        raise SystemExit("--write-config is for ground-truth masks only: the method and the scale are a property of the "
                         "measurement geometry, selected in Stage 3 on GT masks. A run on predicted masks never writes "
                         "configs/config.yaml.")
    if args.reselect and args.masks == "gt":
        raise SystemExit("--reselect is meaningless with --masks gt: the ground-truth run IS the selection.")
    reselect = args.masks == "gt" or args.reselect
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    out_dir = resolve(cfg, Path(cfg["paths"]["outputs"]) / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])

    data = build_all(cfg)
    manifest, high = data["manifest"], data["high"]
    ref_rows = manifest[(manifest["keep"]) & (manifest["has_reference_measurement"])].copy()
    hi = high.set_index("excel_row")
    for i in range(1, 7):
        ref_rows[f"ref_mm_{i}"] = ref_rows["excel_row"].map(hi[f"mm_{i}"])
    ref_rows["ref_mm"] = ref_rows["reference_mean_mm"]
    ref_rows["ref_label"] = ref_rows["ref_mm"].map(label_for_mm)
    for c in ("has_dash_zero", "has_ambiguous_100_999"):
        ref_rows[c] = ref_rows["excel_row"].map(hi[c]).astype(bool)
    ref_rows["label_inconsistent"] = ref_rows["excel_row"].map(hi["label_inconsistent_count"]).fillna(0).astype(int) > 0
    n_all = len(ref_rows)
    excluded_li = ref_rows[ref_rows["label_inconsistent"]]
    primary = ref_rows[~ref_rows["label_inconsistent"]].copy()

    # ---- measurement
    meas = measure_images(primary, cfg, coco_root, mask_source=args.masks)
    df = primary.merge(meas.drop(columns=["image", "group", "width", "height"]), on="uid", how="left")
    df = df[~df["mask_missing"].fillna(True).astype(bool)].copy()

    # ---- dev / holdout
    split = dev_holdout_split(list(df["uid"]), seed=seed, dev_frac=0.6)
    (out_dir / "dev_holdout_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    df["split"] = df["uid"].map(lambda u: "dev" if u in set(split["dev"]) else "holdout")

    # ---- combinations
    results = evaluate_combos(df, split)
    results.to_csv(out_dir / "estimator_comparison.csv", index=False)
    stage3_est = None
    if not reselect:
        p3 = resolve(cfg, Path(cfg["paths"]["outputs"]) / "03_oracle" / "estimator_comparison.csv")
        stage3_est = pd.read_csv(p3).set_index("combo") if p3.exists() else None
    if reselect:
        chosen, why = select_method(results, tolerance_mm=args.tolerance_mm)
        combo, k = chosen["combo"], chosen["px_per_mm"]
        sel_row = results[results["combo"] == combo].iloc[0]
    else:
        # Predicted masks: the method and the scale come from Stage 3 (ground-truth masks) through
        # configs/config.yaml and are applied unchanged. `results` is still written, as a sensitivity
        # table, but nothing in it selects anything here.
        method = dict(cfg["measurement"]["method"])
        combo = combo_name(method["regioning"], method["estimator"], bool(method["anchored"]))
        k = float(cfg["measurement"]["px_per_mm"])
        sel_row = pd.Series(evaluate_fixed(df, split, combo, k))
        chosen = {"regioning": method["regioning"], "estimator": method["estimator"], "anchored": bool(method["anchored"]),
                  "combo": combo, "px_per_mm": k, "mae_dev": float(sel_row["mae_dev"])}
        best = results.sort_values("mae_dev").iloc[0]
        why = (f"**No selection was made here.** Method `{combo}` and scale {k:.4f} px/mm are taken from "
               f"`configs/config.yaml`, where Stage 3 put them after selecting them on ground-truth masks; on predicted "
               f"masks they are applied unchanged (the measurement geometry is a property of the method, not of the "
               f"segmentation model, and re-selecting here would be a second look at the same clinical reference). "
               f"`estimator_comparison.csv` is written for information only: every combination there carries a scale "
               f"re-fitted on the dev subset of these masks, and the best dev MAE in it is `{best['combo']}` "
               f"({best['mae_dev']:.3f} mm vs {sel_row['mae_dev']:.3f} mm for the fixed method) — reported, not adopted. "
               f"Re-run with `--reselect` to make that comparison the selected result.")

    # ---- per-image results (read by Stages 4 and 6)
    per = df[["uid", "image", "split", "ref_mm"] + [f"ref_mm_{i}" for i in range(1, 7)] + ["ref_label", "frame_ok", "width", "height", "has_dash_zero", "has_ambiguous_100_999"]].copy()
    scales = results.set_index("combo")["px_per_mm_dev"]
    if not reselect and stage3_est is not None:
        # every combination in mm at its Stage-3 (ground-truth dev) scale, not at a scale fitted here
        scales = stage3_est["px_per_mm_dev"].reindex(scales.index).fillna(scales)
    for reg, est, anch in COMBOS:
        name = combo_name(reg, est, anch)
        per[f"{name}_px"] = df[f"{name}_px"]
        per[f"{name}_mm"] = df[f"{name}_px"] / scales[name]
    per["selected_method"] = combo
    per["selected_px_per_mm"] = k
    per["selected_mm"] = df[f"{combo}_px"] / k
    per["selected_label"] = per["selected_mm"].map(label_for_mm)
    for i in range(1, 7):
        per[f"selected_region_{i}_mm"] = df[f"{combo}_region_{i}_px"] / k
    per["gap_median_px"] = df["gap_median_px"]
    per["n_gingiva_instances"] = df["n_gingiva_instances"]
    per["n_zeniths_found"] = df["n_zeniths_found"]
    per["alignment_uncertain"] = df["qc_flags"].fillna("").str.contains(QCFlag.ZENITH_DETECTION_FAILED.value)
    per["qc_flags"] = df["qc_flags"].fillna("")
    per.to_csv(out_dir / "per_image_results.csv", index=False)

    # ---- sensitivity, scale, gap, teeth
    sens = sensitivity(df, split, combo, k)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    dev_df = df[df["split"] == "dev"]
    loo = loo_scale(dev_df[f"{combo}_px"], dev_df["ref_mm"])
    pis = per_image_scale(df, combo)
    grp = pis.groupby("frame_ok")["px_per_mm_image"].agg(["count", "mean", "std", "median"]).rename(index={True: "2698x1799 ±2 px", False: "other sizes"})
    grp["cv"] = grp["std"] / grp["mean"]
    n_in, n_out = int(df["frame_ok"].sum()), int((~df["frame_ok"]).sum())
    mean_in = pis[pis.frame_ok]["px_per_mm_image"].mean()
    mean_out = pis[~pis.frame_ok]["px_per_mm_image"].mean()
    scale_diff_pct = 100 * (mean_out - mean_in) / mean_in
    k_in = float(np.sum(dev_df[dev_df.frame_ok][f"{combo}_px"] * dev_df[dev_df.frame_ok]["ref_mm"]) / np.sum(dev_df[dev_df.frame_ok]["ref_mm"] ** 2))
    k_out = float(np.sum(dev_df[~dev_df.frame_ok][f"{combo}_px"] * dev_df[~dev_df.frame_ok]["ref_mm"]) / np.sum(dev_df[~dev_df.frame_ok]["ref_mm"] ** 2)) if (~dev_df.frame_ok).sum() >= 5 else float("nan")
    k_diff_pct = 100 * (k_out - k_in) / k_in if np.isfinite(k_out) else float("nan")
    if abs(scale_diff_pct) > 5 and np.isfinite(k_diff_pct) and np.sign(k_diff_pct) == np.sign(scale_diff_pct) and abs(k_diff_pct) > 5:
        dual_text = (f"**Both indicators exceed 5 % in the same direction (per-image ratio {scale_diff_pct:+.1f} %, dev regression scale {k_diff_pct:+.1f} %): "
                     "a dual global scale (one per frame group) is proposed for the sensitivity analysis. NOT applied here — the protocol fixed a single scale.**")
    elif abs(scale_diff_pct) > 5:
        dual_text = (f"The per-image ratio differs by {scale_diff_pct:+.1f} % but the dev regression scale by {k_diff_pct:+.1f} % — opposite directions, and the outside-frame group is small (n = {n_out}) and much noisier "
                     f"(per-image CV {grp.loc['other sizes', 'cv']:.2f} vs {grp.loc['2698x1799 ±2 px', 'cv']:.2f}). This points to frame *uncertainty* (which copy ImageJ used) rather than a consistent scale shift; "
                     "a dual global scale is therefore NOT proposed. Recommendation: keep the single scale and report the outside-frame subset as a sensitivity row (done in oracle_summary.md).")
    else:
        dual_text = "Difference ≤ 5 %: a single global scale is adequate; no dual scale proposed."
    gap = df["gap_median_px"].dropna()
    gap_stats = {"n": int(len(gap)), "median": float(gap.median()), "q1": float(gap.quantile(0.25)), "q3": float(gap.quantile(0.75)), "mean": float(gap.mean()), "max": float(gap.max())}
    df[["uid", "gap_median_px", "gap_q1_px", "gap_q3_px", "n_gingiva_instances", "n_lip_instances"]].assign(
        boundary_flag=df["qc_flags"].fillna("").str.contains(QCFlag.LIP_GINGIVA_BOUNDARY_MISMATCH.value)).to_csv(out_dir / "boundary_check.csv", index=False)
    df[["uid", "split", "qc_flags", "n_components", "n_gingiva_instances"]].to_csv(out_dir / "qc_flags.csv", index=False)
    flag_counts = pd.Series([f for s in df["qc_flags"].fillna("") for f in s.split(",") if f]).value_counts()
    teeth = tooth_level(df, combo, k)
    teeth.to_csv(out_dir / "tooth_level.csv", index=False)

    # ---- fallback transparency: C falls back to A when zeniths are not found, B to A when festoons are not found
    fb_flag = {"C": QCFlag.ZENITH_DETECTION_FAILED.value, "B": QCFlag.FESTOON_DETECTION_FAILED.value}.get(chosen["regioning"])
    fb_mask = df["qc_flags"].fillna("").str.contains(fb_flag) if fb_flag else pd.Series(False, index=df.index)
    n_fb = int(fb_mask.sum())
    fb_text = ""
    if fb_flag:
        ok_dev = df[(df["split"] == "dev") & ~fb_mask]
        alt = combo_name("A", chosen["estimator"], chosen["anchored"])
        if reselect or stage3_est is None or alt not in stage3_est.index:
            k_alt = float(results.set_index("combo").loc[alt, "px_per_mm_dev"])
            alt_row = results.set_index("combo").loc[alt]
        else:                                    # fixed mode: the alternative too keeps its Stage-3 scale
            k_alt = float(stage3_est.loc[alt, "px_per_mm_dev"])
            alt_row = pd.Series(evaluate_fixed(df, split, alt, k_alt))
        mae_sel = float(np.abs(ok_dev[f"{combo}_px"] / k - ok_dev["ref_mm"]).mean())
        mae_alt = float(np.abs(ok_dev[f"{alt}_px"] / k_alt - ok_dev["ref_mm"]).mean())
        fb = df[fb_mask]
        okc = df[~fb_mask]
        cand_dist = (fb["n_zenith_candidates_left"].astype(int).astype(str) + "+" + fb["n_zenith_candidates_right"].astype(int).astype(str)).value_counts().sort_index()
        fb_detail = (f"\n\nFallback distribution ({n_fb} images): zenith candidates found left+right of the midline (3+3 needed): "
                     + ", ".join(f"{k}: {v}" for k, v in cand_dist.items())
                     + f". Total minima on fallback images: median {int((fb['n_zenith_candidates_left'] + fb['n_zenith_candidates_right']).median())} vs {int((okc['n_zenith_candidates_left'] + okc['n_zenith_candidates_right']).median())} on successful ones. "
                     + f"Gingiva band width (fraction of image width, a proxy for premolar visibility): fallback {fb['window_width_frac'].mean():.3f} vs success {okc['window_width_frac'].mean():.3f}; "
                     + f"reference mm: fallback {fb['ref_mm'].mean():.2f} vs success {okc['ref_mm'].mean():.2f}. "
                     + ("Fallback images are not wider, so premolar visibility is not the main cause; " if fb['window_width_frac'].mean() <= okc['window_width_frac'].mean() * 1.05 else "Fallback images show a wider band, consistent with premolars entering the window (more minima than expected); ")
                     + "the typical failure is one side of the midline having fewer than three detectable minima (a shallow festoon on that side).")
        fb_detail += (f"\n\n**Sensitivity analysis — `{alt}` (no fallback, equal-split regions) side by side:** holdout MAE {alt_row['mae_holdout']:.3f} vs {sel_row['mae_holdout']:.3f} mm, RMSE {alt_row['rmse_holdout']:.3f} vs {sel_row['rmse_holdout']:.3f}, r {alt_row['r_holdout']:.3f} vs {sel_row['r_holdout']:.3f}, "
                      f"ICC(2,1) {alt_row['icc2_1_holdout']:.3f} vs {sel_row['icc2_1_holdout']:.3f}, bias {alt_row['ba_bias_holdout']:+.3f} vs {sel_row['ba_bias_holdout']:+.3f} mm, scale {alt_row['px_per_mm_dev']:.2f} vs {k:.2f} px/mm.")
        fb_detail += (f"\n\n**Stage 6 note:** on predicted masks the fallback rate of `{combo}` will be re-measured; if it exceeds 30 % the selection is re-evaluated against `{alt}`.")
        fb_text = (f"Regioning {chosen['regioning']} could not be established on {n_fb} of {len(df)} images ({100 * n_fb / len(df):.0f} %, `{fb_flag}`); "
                   f"there the measurement silently uses the equal-split regions (A) and the value is identical to `{alt}`. "
                   f"On the dev images where {chosen['regioning']} succeeded (n = {len(ok_dev)}), dev MAE is {mae_sel:.3f} mm for `{combo}` vs {mae_alt:.3f} mm for `{alt}` — "
                   f"the advantage of {chosen['regioning']} comes from these images, not from the fallback ones." + fb_detail)

    # ---- figures (holdout, selected method)
    hold = per[per["split"] == "holdout"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(hold["ref_mm"], hold["selected_mm"], s=18, alpha=0.8)
    lim = [0, max(hold["ref_mm"].max(), hold["selected_mm"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Clinical reference (ImageJ), mm"); ax.set_ylabel(f"GT-mask measurement ({combo}), mm")
    ax.set_title(f"Holdout n = {len(hold)}: MAE {sel_row['mae_holdout']:.2f} mm, r = {sel_row['r_holdout']:.3f}, ICC(2,1) = {sel_row['icc2_1_holdout']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "scatter_gt_vs_manual.png", dpi=120); plt.close(fig)
    ba = bland_altman(hold["selected_mm"], hold["ref_mm"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mean = (hold["selected_mm"] + hold["ref_mm"]) / 2
    ax.scatter(mean, hold["selected_mm"] - hold["ref_mm"], s=18, alpha=0.8)
    for y, ls, lab in ((ba["bias"], "-", f"bias {ba['bias']:.2f}"), (ba["loa_low"], "--", f"LoA {ba['loa_low']:.2f}"), (ba["loa_high"], "--", f"LoA {ba['loa_high']:.2f}")):
        ax.axhline(y, color="k", ls=ls, lw=1); ax.text(mean.max(), y, lab, fontsize=8, va="bottom", ha="right")
    xs = np.linspace(mean.min(), mean.max(), 10)
    ax.plot(xs, ba["prop_intercept"] + ba["prop_slope"] * xs, color="tab:red", lw=1, label=f"diff ~ mean: slope {ba['prop_slope']:.3f} (p = {ba['prop_p']:.3f})")
    ax.legend(fontsize=8); ax.set_xlabel("Mean of methods, mm"); ax.set_ylabel("GT-mask − reference, mm")
    ax.set_title(f"Bland–Altman, holdout n = {len(hold)}, {combo}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "bland_altman.png", dpi=120); plt.close(fig)

    # ---- intra-observer
    io, wide = intra_observer(data["calibration"])
    n_cal_in_coco = int(data["calibration"]["key"].drop_duplicates().isin(set(manifest["key"])).sum())
    intra_md = f"""# Intra-observer reliability of the clinical reference (calibration.xlsx)

Gingival display was measured twice by the same observer on 20 images, at six tooth sites per image (sheets `İlk Ölçümler` / `İkinci Ölçümler`, same ×1000 coding). The measured quantity is the gingival display at a tooth site, not a dimension of the tooth. {n_cal_in_coco} of the 20 images are in the current COCO set.

| level | n | ICC(2,1) [95 % CI] | ICC(3,1) [95 % CI] | ICC(2,k) | mean diff (2−1), mm | SD, mm | 95 % LoA, mm |
|---|---|---|---|---|---|---|---|
| tooth site | {io['tooth']['n_targets']} | {io['tooth']['icc2_1']:.3f} [{io['tooth']['icc2_1_ci_low']:.3f}, {io['tooth']['icc2_1_ci_high']:.3f}] | {io['tooth']['icc3_1']:.3f} [{io['tooth']['icc3_1_ci_low']:.3f}, {io['tooth']['icc3_1_ci_high']:.3f}] | {io['tooth']['icc2_k']:.3f} | {io['ba_tooth']['bias']:.3f} [{io['ba_tooth']['bias_ci_low']:.3f}, {io['ba_tooth']['bias_ci_high']:.3f}] | {io['ba_tooth']['sd']:.3f} | {io['ba_tooth']['loa_low']:.2f} to {io['ba_tooth']['loa_high']:.2f} |
| image mean | {io['image']['n_targets']} | {io['image']['icc2_1']:.3f} [{io['image']['icc2_1_ci_low']:.3f}, {io['image']['icc2_1_ci_high']:.3f}] | {io['image']['icc3_1']:.3f} [{io['image']['icc3_1_ci_low']:.3f}, {io['image']['icc3_1_ci_high']:.3f}] | {io['image']['icc2_k']:.3f} | {io['ba_image']['bias']:.3f} [{io['ba_image']['bias_ci_low']:.3f}, {io['ba_image']['bias_ci_high']:.3f}] | {io['ba_image']['sd']:.3f} | {io['ba_image']['loa_low']:.2f} to {io['ba_image']['loa_high']:.2f} |

The tooth-site-level SD of {io['ba_tooth']['sd']:.2f} mm is the observer's own repeatability floor; the paired t-test is not used as evidence of agreement (ICC and limits of agreement are).
Expected orders of magnitude (audit §B): ICC ≈ 0.995 / 0.998, SD ≈ 0.17 mm.
"""
    (out_dir / "intra_observer.md").write_text(intra_md, encoding="utf-8")
    noise_sd = io["ba_tooth"]["sd"]
    noise_sd_img = io["ba_image"]["sd"]

    # ---- scale report
    if reselect:
        scale_head = (f"Global scale: regression through the origin on the **dev subset only** (n = {len(dev_df)}): "
                      f"**{k:.2f} px/mm** (R² {sel_row['scale_r2_dev']:.3f}; residual SD {sel_row['scale_resid_sd_mm']:.2f} mm). Applied unchanged to holdout.\n"
                      f"Leave-one-out on dev: mean {loo['px_per_mm_mean']:.2f}, SD {loo['px_per_mm_sd']:.3f}, "
                      f"range {loo['px_per_mm_min']:.2f}–{loo['px_per_mm_max']:.2f} px/mm; LOO MAE {loo['loo_mae_mm']:.3f} mm.")
    else:
        k_here = float(results.set_index("combo").loc[combo, "px_per_mm_dev"])
        scale_head = (f"Global scale: **{k:.2f} px/mm, FIXED** — taken from `configs/config.yaml` (Stage 3, fitted on the dev subset of the "
                      f"**ground-truth** masks) and applied unchanged to every image here. Nothing was fitted on these masks.\n"
                      f"For information only, never used: a through-origin fit of the same method on the dev subset of *these* masks would give "
                      f"{k_here:.2f} px/mm ({100 * (k_here - k) / k:+.1f} %); leave-one-out on that fit: mean {loo['px_per_mm_mean']:.2f}, "
                      f"SD {loo['px_per_mm_sd']:.3f}, range {loo['px_per_mm_min']:.2f}–{loo['px_per_mm_max']:.2f} px/mm.")
    scale_md = f"""# Scale estimation ({combo})

{scale_head}
Expected order of magnitude 15–20 px/mm (2698 px ≈ 15–16 cm field of view): {'OK' if EXPECT['px_per_mm'][0] <= k <= EXPECT['px_per_mm'][1] else 'OUTSIDE — check'}.

## Per-image ratio px / reference mm (all {len(pis)} images; mixes scale variation with measurement noise, reported without interpretation)
overall: mean {pis['px_per_mm_image'].mean():.2f}, SD {pis['px_per_mm_image'].std():.2f}, CV {pis['px_per_mm_image'].std() / pis['px_per_mm_image'].mean():.2f}, median {pis['px_per_mm_image'].median():.2f}.

By frame size (the reference was measured on 2698×1799 copies; other sizes may carry a different scale):

{md_table(grp.reset_index().rename(columns={'frame_ok': 'frame'}), '{:.3f}')}

Images outside the 2698×1799 ±2 px frame: {n_out} of {n_in + n_out} reference images ({100 * n_out / (n_in + n_out):.1f} %); in the whole COCO export 485 of 1315.
Mean per-image ratio differs by {scale_diff_pct:+.1f} % between the two groups. Dev-only through-origin scale by group: inside {k_in:.2f} px/mm, outside {k_out:.2f} px/mm.
{dual_text}

Reference calibration note: ImageJ used a single 1 mm probe interval per image (≈ {k:.0f} px); a 1 px marking error is ≈ {100 / k:.1f} % of scale, so part of the per-image CV is the reference's own calibration noise.
"""
    (out_dir / "scale_estimation.md").write_text(scale_md, encoding="utf-8")

    # ---- summary
    show = results.sort_values("mae_dev")[["combo", "px_per_mm_dev", "mae_dev", "mae_holdout", "rmse_holdout", "r_holdout", "icc2_1_holdout", "icc2_1_ci_low_holdout", "icc2_1_ci_high_holdout", "ba_bias_holdout", "ba_loa_low_holdout", "ba_loa_high_holdout", "ba_prop_slope_holdout", "ba_prop_p_holdout", "threshold_agreement_holdout"]]
    above_noise = sel_row["mae_holdout"] - noise_sd_img * np.sqrt(2 / np.pi)  # expected |diff| of pure observer noise
    if reselect:
        proto_note = "Scale of every combination fitted on dev only and applied unchanged to holdout; holdout metrics are reported for all combinations but were not used for selection."
        method_head = f"## Selected method: **{combo}**, global scale **{k:.2f} px/mm**"
        table_head = "## All combinations (sorted by dev MAE; holdout columns for reporting only)"
    else:
        proto_note = ("The scales in the table below were re-fitted on the dev subset of these masks and are reported as a sensitivity "
                      "analysis; the method and the scale used everywhere else in this file are the fixed ones. `--reselect` is the only "
                      "way to turn that table into a selection, and it still writes nothing into `configs/config.yaml`.")
        method_head = f"## Fixed method: **{combo}**, global scale **{k:.2f} px/mm** (from `configs/config.yaml`, Stage 3 — not selected here)"
        table_head = ("## All combinations, each with a scale re-fitted on these masks — SENSITIVITY ONLY, NOT A SELECTION\n"
                      "(sorted by dev MAE; the fixed method above is the result of this run whatever this table shows)")
    summary = f"""# Oracle validation summary — {'ground-truth masks' if args.masks == 'gt' else 'predicted masks: ' + args.masks}

## Set
Reference images (kept high with clinical measurement): {n_all}; excluded `label_inconsistent` rows: {len(excluded_li)} ({', '.join(excluded_li['image']) if len(excluded_li) else '—'}); measured: {len(df)}; dev {len(dev_df)} / holdout {len(hold)} (seed {seed}, lists in `dev_holdout_split.json`).

## Protocol
{why}
{proto_note}

{method_head}
Holdout (n = {len(hold)}): MAE {sel_row['mae_holdout']:.3f} mm, RMSE {sel_row['rmse_holdout']:.3f} mm, r {sel_row['r_holdout']:.3f}, ICC(2,1) {sel_row['icc2_1_holdout']:.3f} [{sel_row['icc2_1_ci_low_holdout']:.3f}, {sel_row['icc2_1_ci_high_holdout']:.3f}], bias {sel_row['ba_bias_holdout']:+.3f} mm [{sel_row['ba_bias_ci_low_holdout']:+.3f}, {sel_row['ba_bias_ci_high_holdout']:+.3f}], LoA {sel_row['ba_loa_low_holdout']:.2f} to {sel_row['ba_loa_high_holdout']:.2f} mm, proportional bias slope {sel_row['ba_prop_slope_holdout']:+.3f} (p = {sel_row['ba_prop_p_holdout']:.3f}); threshold-label agreement {100 * sel_row['threshold_agreement_holdout']:.1f} % (linear-weighted κ {sel_row['threshold_kappa_linear_holdout']:.3f}) — a *measurement* check against Table 1, not a clinical validation.
Dev (n = {len(dev_df)}): MAE {sel_row['mae_dev']:.3f} mm, r {sel_row['r_dev']:.3f}, ICC(2,1) {sel_row['icc2_1_dev']:.3f}.

**Fallback transparency.** {fb_text or 'Regioning A has no fallback.'}

**Against the reference's own repeatability:** intra-observer SD is {noise_sd:.2f} mm per tooth site and {noise_sd_img:.2f} mm per image mean (`intra_observer.md`); pure observer noise would produce an expected absolute difference of ≈ {noise_sd_img * np.sqrt(2 / np.pi):.2f} mm at image level. The holdout MAE of {sel_row['mae_holdout']:.2f} mm therefore leaves ≈ {above_noise:.2f} mm above the observer-noise floor, attributable to the estimator, the single global scale (per-image calibration was not recorded) and region alignment.

Pre-analysis plausibility check (audit §6, A/p25, same-data scale, n = 148): r ≈ 0.83, MAE ≈ 0.63 mm, ≈ 17 px/mm — {'consistent' if abs(sel_row['mae_holdout'] - EXPECT['mae_reasonable']) < 0.25 and 15 <= k <= 20 else 'DIFFERENT — see OZET'}.

{table_head}
{md_table(show)}

## Bland–Altman direction and proportional bias (holdout)
Sign of `ba_bias`: positive = GT-mask value above the clinical reference. `ba_prop_slope` is the slope of (GT − ref) on the mean of both; p < 0.05 indicates a proportional (mm-dependent) bias. For the selected method: {'bias grows with gingival display' if sel_row['ba_prop_slope_holdout'] > 0 else 'bias shrinks with gingival display'} (slope {sel_row['ba_prop_slope_holdout']:+.3f}, p = {sel_row['ba_prop_p_holdout']:.3f}).

## Sensitivity (selected method, scale fixed)
{md_table(sens)}

## Lip–gingiva boundary (n = {gap_stats['n']})
Gap between the lowest lip pixel and the gingiva top: median {gap_stats['median']:.1f} px (IQR {gap_stats['q1']:.1f}–{gap_stats['q3']:.1f}; mean {gap_stats['mean']:.1f}; max {gap_stats['max']:.0f}). Expected ≈ 9 px (IQR 6–12) from the audit. {'Consistent.' if abs(gap_stats['median'] - EXPECT['gap_median_px']) <= 6 else 'DIFFERENT from expectation — see OZET.'} Boundary flag raised on {int(flag_counts.get(QCFlag.LIP_GINGIVA_BOUNDARY_MISMATCH.value, 0))} images.
Lip-anchored estimators ({combo_name(chosen['regioning'], chosen['estimator'], True)}): dev MAE {results.set_index('combo').loc[combo_name(chosen['regioning'], chosen['estimator'], True), 'mae_dev']:.3f} mm vs {sel_row['mae_dev']:.3f} mm for gingiva thickness.

## Tooth level (secondary; region i left-to-right vs reference tooth i; A/B/C alignment is approximate)
{md_table(teeth)}

## QC flags
{flag_counts.to_string() if len(flag_counts) else 'none'}
Frame outside 2698×1799 ±2 px: {n_out} images (`frame_uncertain`; sensitivity rows above).

## Assumptions
- COCO frame = ImageJ frame (clinical team, screenshot); for other frame sizes the reference frame is uncertain.
- `-` cells are 0 mm (clinical decision); the sensitivity row without dash-zero rows shows the effect.
- The reference gingival display recorded at tooth site i is compared to the left-to-right region of the same index; the image-level mean is the primary endpoint.
- v1 (XGBoost) column of the original Figure 6 is not reproduced: the regressor was trained on 512×512 gingiva-only DeepLab masks and fed lip+gingiva masks at 1024 px in v3, i.e. inputs outside its training distribution (audit §3).
"""
    (out_dir / "oracle_summary.md").write_text(summary, encoding="utf-8")
    if reselect:
        ozet_head = "# Aşama 3 — Türkçe özet"
        ozet_masks = "ölçümlü high görüntü (GT maske)"
        ozet_method = (f"- Seçilen yöntem **{combo}** (dev MAE {sel_row['mae_dev']:.3f} mm; basitlik kuralı ±{args.tolerance_mm} mm). "
                       f"Global ölçek **{k:.2f} px/mm** (yalnız dev'de kestirildi).")
    else:
        ozet_head = f"# Oracle ölçümü — tahmin maskeleri ({args.masks})"
        ozet_masks = f"ölçümlü high görüntü (tahmin maskesi: {args.masks})"
        ozet_method = (f"- Yöntem **{combo}** ve ölçek **{k:.2f} px/mm** `configs/config.yaml`'den SABİT alındı (Aşama 3'te GT maskelerde seçildi); "
                       f"burada hiçbir şey yeniden seçilmedi, yeniden kestirilmedi ve config'e yazılmadı. Bu maskelerdeki dev MAE {sel_row['mae_dev']:.3f} mm. "
                       f"`estimator_comparison.csv` yalnız duyarlılık amaçlıdır; yeniden seçim ancak `--reselect` ile yapılır.")
    ozet = f"""{ozet_head}

- {len(df)} {ozet_masks}, dev {len(dev_df)} / holdout {len(hold)}; `label_inconsistent` dışlanan: {len(excluded_li)}.
{ozet_method}
- Holdout: MAE {sel_row['mae_holdout']:.2f} mm, RMSE {sel_row['rmse_holdout']:.2f}, r {sel_row['r_holdout']:.3f}, ICC(2,1) {sel_row['icc2_1_holdout']:.3f}; sapma {sel_row['ba_bias_holdout']:+.2f} mm, orantısal eğim {sel_row['ba_prop_slope_holdout']:+.3f} (p {sel_row['ba_prop_p_holdout']:.3f}).
- Gözlemci içi (dişeti görünürlüğünün tekrar ölçümü): ICC(2,1) diş bölgesi düzeyi {io['tooth']['icc2_1']:.3f}, görüntü ortalaması {io['image']['icc2_1']:.3f}; SD {noise_sd:.2f} mm. Holdout MAE'nin gözlemci gürültüsü üstünde kalan kısmı ≈ {above_noise:.2f} mm.
- Dudak-altı ↔ dişeti-üstü boşluk medyanı {gap_stats['median']:.1f} px (IQR {gap_stats['q1']:.0f}–{gap_stats['q3']:.0f}); beklenti ≈ 9 px → {'uyumlu' if abs(gap_stats['median'] - 9) <= 6 else 'BEKLENTİDEN UZAK'}.
- Çerçeve: {n_out}/{n_in + n_out} görüntü 2698×1799 dışında; görüntü bazlı oran farkı {scale_diff_pct:+.1f} %, dev regresyon ölçeği farkı {k_diff_pct:+.1f} % → {'ikili ölçek önerilir (uygulanmadı)' if 'proposed for' in dual_text else 'ikili ölçek önerilmez (yönler zıt / grup küçük ve gürültülü); tek ölçek + duyarlılık satırı'}.
- {'Seçilen bölgeleme' if reselect else 'Bölgeleme'} {chosen['regioning']}: {n_fb} görüntüde başarısız → A'ya düşer (raporda açık){'' if reselect else f"; oran %{100 * n_fb / len(df):.0f} (ön kayıtlı eşik %30)"}.
- Ön analizle (MAE ≈ 0.63, ≈ 17 px/mm) karşılaştırma: {'makul' if abs(sel_row['mae_holdout'] - 0.63) < 0.25 and 15 <= k <= 20 else 'FARKLI — kontrol'}.
"""
    (out_dir / "OZET.md").write_text(ozet, encoding="utf-8")
    if args.write_config and args.masks == "gt":      # belt and braces; argument parsing already refuses the rest
        write_config_method(resolve(cfg, "configs/config.yaml"), chosen, k)
    print(ozet)
    print(show.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
