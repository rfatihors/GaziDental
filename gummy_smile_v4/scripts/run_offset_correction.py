#!/usr/bin/env python
"""Stage 6 addendum — post-hoc offset calibration of the pipeline measurement.

    python scripts/run_offset_correction.py [--max-shift 24] [--tolerance-mm 0.03]

Fits three corrections on the Stage-3 **dev** subset (60 %, seed from config, list in
outputs/03_oracle/dev_holdout_split.json) and reports them on **holdout** (40 %); nothing
is fitted on holdout. Writes outputs/06_prediction/offset_correction.{md,csv},
pixel_shift_curve.csv and figures/offset_correction.png. Recommends a correction by the
pre-stated rule but does not write anything into configs/config.yaml.
"""
from __future__ import annotations

import argparse
import json
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
from gsv4.eval.agreement import bland_altman  # noqa: E402
from gsv4.eval.offset import (  # noqa: E402
    SELECTION_TOLERANCE_MM, apply_constant, apply_regression, best_pixel_shift, fit_constant, fit_regression, select_correction,
    shift_lower_edge_up,
)
from gsv4.eval.oracle import combo_name  # noqa: E402
from gsv4.eval.prediction import agreement_metrics, label_confusion, md_table  # noqa: E402
from gsv4.measure.calibration import corrected_mm, offset_mm_at, offset_px_from_config  # noqa: E402
from gsv4.masks.extract import from_png  # noqa: E402
from gsv4.measure.gingival_display import measure_gingival_display  # noqa: E402
from gsv4.report.figures import PLOT_DPI  # noqa: E402
from gsv4.rules.thresholds import label_for_mm  # noqa: E402

COLS = ["subset", "correction", "n", "mae", "rmse", "r", "icc2_1", "icc2_1_ci_low", "icc2_1_ci_high", "ba_bias", "ba_bias_ci_low", "ba_bias_ci_high",
        "ba_loa_low", "ba_loa_high", "ba_prop_slope", "ba_prop_p", "threshold_agreement", "threshold_kappa_linear", "threshold_kappa_linear_ci_low",
        "threshold_kappa_linear_ci_high", "within_0_5_mm", "within_1_mm"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-shift", type=int, default=24, help="largest lower-edge shift tried, px")
    ap.add_argument("--tolerance-mm", type=float, default=SELECTION_TOLERANCE_MM)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    out_dir = outputs / "06_prediction"
    oof_dir = resolve(cfg, cfg["paths"]["predictions"]) / "oof"
    mcfg = cfg["measurement"]
    method = dict(mcfg["method"])
    combo = combo_name(method["regioning"], method["estimator"], bool(method["anchored"]))
    k = float(mcfg["px_per_mm"])
    seed = int(cfg["seed"])

    per = pd.read_csv(out_dir / "per_image_results.csv")
    split = json.loads((outputs / "03_oracle" / "dev_holdout_split.json").read_text())
    assert set(per.loc[per["split"] == "dev", "uid"]) == set(split["dev"]), "dev list differs from Stage 3"
    failed = per[per["empty_prediction"].astype(bool)]
    ok = per[~per["empty_prediction"].astype(bool)].copy()
    dev = ok[ok["split"] == "dev"].copy()
    hold = ok[ok["split"] == "holdout"].copy()
    b = pd.read_csv(out_dir / "boundary_error_oof.csv").set_index("uid")
    bias_px_dev = float(b.loc[dev["uid"], "gingiva_bottom_edge_bias_px"].mean())
    top_px_dev = float(b.loc[dev["uid"], "gingiva_top_edge_bias_px"].mean())

    # ---- (a) constant, (c) regression — fitted on dev
    fit_a = fit_constant(dev["selected_mm"], dev["ref_mm"])
    fit_c = fit_regression(dev["selected_mm"], dev["ref_mm"])
    for d in (dev, hold):
        d["constant"] = apply_constant(d["selected_mm"], fit_a)
        d["regression"] = apply_regression(d["selected_mm"], fit_c)
        d["none"] = d["selected_mm"]

    # ---- (b) pixel shift of the lower edge: re-measure every image for d = 0..max_shift
    print(f"[offset] re-measuring {len(ok)} OOF masks for lower-edge shifts 0..{args.max_shift} px …")
    px = {}
    for r in ok.itertuples(index=False):
        m = from_png(oof_dir / f"{r.image}_gingiva.png", oof_dir / f"{r.image}_lip.png", expected_shape=(int(r.height), int(r.width)))
        vals = []
        for d in range(args.max_shift + 1):
            g = shift_lower_edge_up(m.gingiva, d)
            res = measure_gingival_display(g, m.lip, px_per_mm=None, cfg=mcfg, method=method)
            vals.append(res.image_values.get((method["regioning"], method["estimator"], bool(method["anchored"])), np.nan))
        px[r.uid] = vals
    shifts = pd.DataFrame(px, index=range(args.max_shift + 1)).T / k        # uid × d, mm
    shifts.columns = [f"d{d}" for d in shifts.columns]
    if abs(shifts["d0"].reindex(ok["uid"]).to_numpy() - ok["selected_mm"].to_numpy()).max() > 1e-6:
        raise SystemExit("d = 0 re-measurement differs from per_image_results.csv — inconsistent inputs")
    curve_rows = []
    for d in range(args.max_shift + 1):
        col = f"d{d}"
        cd = shifts.loc[dev["uid"], col].to_numpy() - dev["ref_mm"].to_numpy()
        ch = shifts.loc[hold["uid"], col].to_numpy() - hold["ref_mm"].to_numpy()
        curve_rows.append({"d": d, "d_mm": d / k, "mae_dev": float(np.nanmean(np.abs(cd))), "bias_dev": float(np.nanmean(cd)),
                           "mae_holdout": float(np.nanmean(np.abs(ch))), "bias_holdout": float(np.nanmean(ch)),
                           "n_vanished": int(np.isnan(shifts[col]).sum() + (shifts[col] <= 0).sum())})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(out_dir / "pixel_shift_curve.csv", index=False)
    d_star = best_pixel_shift(curve)
    dev["pixel"] = shifts.loc[dev["uid"], f"d{d_star}"].to_numpy()
    hold["pixel"] = shifts.loc[hold["uid"], f"d{d_star}"].to_numpy()
    d_geom = int(round(bias_px_dev))

    # ---- comparison table
    rows = []
    for sub_name, d in (("dev", dev), ("holdout", hold)):
        for corr in ("none", "constant", "pixel", "regression"):
            m = agreement_metrics(d[corr], d["ref_mm"], n_boot=args.n_boot, seed=seed)
            rows.append({"subset": sub_name, "correction": corr, **m})
    table = pd.DataFrame(rows)[COLS]
    table.to_csv(out_dir / "offset_correction.csv", index=False)
    sel = select_correction(table, args.tolerance_mm)
    t = table.set_index(["subset", "correction"])
    h_none, h_a, h_b, h_c = (t.loc[("holdout", c)] for c in ("none", "constant", "pixel", "regression"))
    ct_none = label_confusion(hold["ref_mm"].map(label_for_mm), hold["none"].map(label_for_mm))
    ct_sel = label_confusion(hold["ref_mm"].map(label_for_mm), hold[sel["chosen"]].map(label_for_mm))

    # ---- figure: dev curve of the pixel shift + holdout Bland–Altman before/after the chosen correction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(curve["d_mm"], curve["mae_dev"], color="#0072B2", lw=2, label="dev MAE (used for the choice of d)")
    ax.plot(curve["d_mm"], curve["mae_holdout"], color="#D55E00", lw=1.2, ls="--", label="holdout MAE (reported only)")
    y0, y1 = ax.get_ylim()
    ax.axvline(d_star / k, color="0.3", lw=0.8, ls=":"); ax.text(d_star / k + 0.02, y0 + 0.30 * (y1 - y0), f"d* = {d_star} px\n(dev minimum)", fontsize=7.5, va="center", ha="left", color="0.2")
    ax.axvline(bias_px_dev / k, color="0.6", lw=0.8, ls=":"); ax.text(bias_px_dev / k - 0.02, y0 + 0.30 * (y1 - y0), f"edge bias\n{bias_px_dev:.1f} px", fontsize=7, va="center", ha="right", color="0.4")
    ax.set_xlabel(f"lower gingiva edge shifted up, mm (px / {k:.2f})"); ax.set_ylabel("MAE vs clinical reference, mm"); ax.legend(fontsize=8, loc="upper center")
    ax.set_title("(b) pixel-level correction: MAE against the shift", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    for corr, col, lab in (("none", "0.55", "uncorrected"), (sel["chosen"], "#0072B2", f"{sel['chosen']} correction")):
        diff = hold[corr] - hold["ref_mm"]; mean = (hold[corr] + hold["ref_mm"]) / 2
        ax.scatter(mean, diff, s=14, alpha=0.7, color=col, lw=0, label=f"{lab}: bias {diff.mean():+.2f}, LoA ±{1.96 * diff.std(ddof=1):.2f} mm")
        ax.axhline(diff.mean(), color=col, lw=1)
    ax.axhline(0, color="0.5", lw=0.8, ls="--"); ax.set_xlabel("mean of pipeline and reference, mm"); ax.set_ylabel("pipeline − reference, mm")
    ax.set_title(f"holdout (n = {len(hold)}), fitted on dev (n = {len(dev)})", fontsize=9); ax.legend(fontsize=8, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); (out_dir / "figures").mkdir(exist_ok=True); fig.savefig(out_dir / "figures" / "offset_correction.png", dpi=PLOT_DPI); plt.close(fig)

    # ---- report
    def row(name, r):
        return (f"| {name} | {int(r['n'])} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['r']:.3f} | {r['icc2_1']:.3f} [{r['icc2_1_ci_low']:.3f}, {r['icc2_1_ci_high']:.3f}] | "
                f"{r['ba_bias']:+.3f} [{r['ba_bias_ci_low']:+.3f}, {r['ba_bias_ci_high']:+.3f}] | {r['ba_loa_low']:.2f} to {r['ba_loa_high']:.2f} | {r['ba_prop_slope']:+.3f} ({r['ba_prop_p']:.3f}) | "
                f"{100 * r['threshold_agreement']:.1f} % | {r['threshold_kappa_linear']:.3f} [{r['threshold_kappa_linear_ci_low']:.3f}, {r['threshold_kappa_linear_ci_high']:.3f}] | {100 * r['within_1_mm']:.0f} % |")
    header = ("| subset / correction | n | MAE, mm | RMSE, mm | r | ICC(2,1) [95 % CI] | bias, mm [95 % CI] | 95 % LoA, mm | prop. slope (p) | label agreement | κ linear [95 % CI] | within 1 mm |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    names = {"none": "uncorrected", "constant": "(a) constant mm offset", "pixel": f"(b) lower edge −{d_star} px (mask level)", "regression": "(c) linear recalibration"}
    for sub in ("dev", "holdout"):
        for corr in ("none", "constant", "pixel", "regression"):
            lines.append(row(f"{sub}: {names[corr]}", t.loc[(sub, corr)]))
    md = f"""# Post-hoc offset calibration of the pipeline measurement (Stage 6 addendum)

**Status: post-hoc calibration, estimated on the Stage-3 dev subset, reported on holdout.** It was not part of the pre-registered protocol (Stage 3 fixed the method and scale on ground-truth masks; Stage 6 applied them unchanged to predicted masks). If adopted, the manuscript must present it as a calibration step derived after inspecting the out-of-fold error, with the fit set and the held-out evaluation stated. Nothing here was written into `configs/config.yaml`; the recommendation below is for the clinical team's decision.

## Why
On the OOF masks the lower gingiva edge is drawn systematically below the annotation (dev mean {bias_px_dev:+.1f} px = {bias_px_dev / k:+.2f} mm; upper edge {top_px_dev:+.1f} px = {top_px_dev / k:+.2f} mm), the pipeline − reference bias is +0.68 mm and the tooth-level mixed model gives +0.68 [+0.57, +0.80] mm with a proportional-bias slope near zero. A near-constant offset is the signature of a calibration error, which is correctable.

## Data and protocol
* Images: the {len(per)} reference images with OOF masks; **{len(failed)} segmentation failure(s)** ({', '.join(failed['image']) or '—'}: no gingiva predicted, no mm value) excluded from the fit and the evaluation and reported as a separate category.
* Split: Stage 3 dev / holdout lists (`outputs/03_oracle/dev_holdout_split.json`, seed {seed}): dev n = {len(dev)}, holdout n = {len(hold)}. **All three corrections are fitted on dev only; holdout numbers are reported once, nothing is re-fitted there.** Method {combo}, scale {k:.2f} px/mm unchanged.
* (a) constant: subtract the mean dev error, **{fit_a['offset_mm']:.3f} mm**.
* (b) pixel: shift the lower gingiva edge up by d px at mask level (column-wise removal of the lowest d pixels of the bottom run) and re-measure with the unchanged method; d chosen as the integer minimising dev MAE on a 0–{args.max_shift} px grid → **d* = {d_star} px ({d_star / k:.2f} mm)**. The dev boundary bias alone would suggest {d_geom} px; the curve is in `pixel_shift_curve.csv` (holdout column for information only).
* (c) regression: reference ≈ a + b · pipeline on dev → **a = {fit_c['a']:+.3f} mm, b = {fit_c['b']:.3f}**; corrected value = a + b · pipeline.

## Comparison (bootstrap CIs of κ: {args.n_boot} resamples)
{chr(10).join(lines)}

## Selection rule and recommendation
Rule stated in advance: lowest holdout MAE, but the simplest correction (constant mm) is preferred when it is within {args.tolerance_mm:.2f} mm of the best. Holdout MAE: constant {h_a['mae']:.3f}, pixel {h_b['mae']:.3f}, regression {h_c['mae']:.3f} (uncorrected {h_none['mae']:.3f}). Best = **{sel['best_holdout']}**; constant − best = {t.loc[('holdout', 'constant'), 'mae'] - sel['mae_best']:+.3f} mm → **recommended: {sel['chosen']}** ({names[sel['chosen']]}). Not applied to the config; decision pending.

Effect of the recommended correction on holdout: MAE {h_none['mae']:.3f} → {t.loc[('holdout', sel['chosen']), 'mae']:.3f} mm, bias {h_none['ba_bias']:+.3f} → {t.loc[('holdout', sel['chosen']), 'ba_bias']:+.3f} mm, ICC(2,1) {h_none['icc2_1']:.3f} → {t.loc[('holdout', sel['chosen']), 'icc2_1']:.3f}, LoA width {h_none['ba_loa_high'] - h_none['ba_loa_low']:.2f} → {t.loc[('holdout', sel['chosen']), 'ba_loa_high'] - t.loc[('holdout', sel['chosen']), 'ba_loa_low']:.2f} mm, label agreement {100 * h_none['threshold_agreement']:.1f} % → {100 * t.loc[('holdout', sel['chosen']), 'threshold_agreement']:.1f} %, κ {h_none['threshold_kappa_linear']:.3f} → {t.loc[('holdout', sel['chosen']), 'threshold_kappa_linear']:.3f}. A constant offset cannot change r or the spread of the differences; it moves the bias and, through the thresholds, the labels.

### Holdout label confusion, uncorrected
{md_table(ct_none.reset_index())}

### Holdout label confusion, {names[sel['chosen']]}
{md_table(ct_sel.reset_index())}

## How to present it in the manuscript
"The segmentation model places the lower gingival margin on average {bias_px_dev / k:.2f} mm below the annotated margin. A constant offset of {fit_a['offset_mm']:.2f} mm was therefore estimated post hoc on the development subset (n = {len(dev)}) of the out-of-fold predictions and applied unchanged to the held-out subset (n = {len(hold)}), where it reduced the mean absolute error from {h_none['mae']:.2f} to {h_a['mae']:.2f} mm and the bias from {h_none['ba_bias']:+.2f} to {h_a['ba_bias']:+.2f} mm. Uncorrected results are reported as the primary analysis." (Adjust to the chosen correction if it is not the constant one.)

## Caveats
* The offset was fitted on dev images whose reference measurements also determined the global scale in Stage 3; holdout is the only clean estimate of the corrected accuracy.
* If the correction is adopted, the same offset must be applied to the test-set (final model) predictions and to the expert-agreement analysis *before* those are looked at again.
* The segmentation failure category (n = {len(failed)}) is unaffected by any calibration; it needs a manual-review path.
"""
    # ---- adopted correction (config.yaml: measurement.offset_px), value level, and the verification
    off_px = offset_px_from_config(cfg)
    off_mm_global = offset_mm_at(k, off_px)
    per_t = pd.read_csv(out_dir / "per_image_results_test.csv")
    per_t = per_t[~per_t["empty_prediction"].astype(bool)]
    adopted = {}
    for name, d in (("dev", dev), ("holdout", hold), ("test high, final model", per_t)):
        c = corrected_mm(d[f"{combo}_px"], k, off_px)
        m = agreement_metrics(c["mm"], d["ref_mm"], n_boot=args.n_boot, seed=seed)
        m["n_clipped"] = c["n_clipped"]
        adopted[name] = m
    unc_test = agreement_metrics(per_t["selected_mm"], per_t["ref_mm"], n_boot=args.n_boot, seed=seed)
    _oof = per.set_index("uid").loc[per_t["uid"], "selected_mm"].to_numpy(float)
    paired = bland_altman(per_t["selected_mm"].to_numpy(float), _oof)   # final − fold models, same 29 images
    d_adopt = int(round(-off_px))
    mask_row = curve.set_index("d").loc[d_adopt] if d_adopt in set(curve["d"]) else None
    const_px = fit_a["offset_mm"] * k   # the constant-mm row expressed in pixels (magnitude)
    ver = (f"At the single global scale the adopted offset equals {off_mm_global:+.3f} mm. Holdout MAE with the value-level offset: {adopted['holdout']['mae']:.3f} mm"
           + (f"; mask-level shift of the same {d_adopt} px (row (b) of the table, re-measured): {mask_row['mae_holdout']:.3f} mm" if mask_row is not None else "")
           + f"; constant mm offset (row (a), {fit_a['offset_mm']:.3f} mm = {const_px:.1f} px): {h_a['mae']:.3f} mm. "
           + ("Value-level and mask-level results agree to within 0.01 mm, so the pixel definition reproduces the pixel row of the comparison table. " if mask_row is not None and abs(adopted['holdout']['mae'] - mask_row['mae_holdout']) < 0.01 else "")
           + (f"Note that {off_px:+.0f} px is the dev optimum of the mask-level shift (d*), not the pixel equivalent of the constant-mm row ({-const_px:.1f} px); the two differ by {abs(off_mm_global) - fit_a['offset_mm']:+.3f} mm in the corrected value and by {adopted['holdout']['mae'] - h_a['mae']:+.3f} mm in holdout MAE." if abs(abs(off_px) - const_px) > 0.5 else "The pixel value is the rounded equivalent of the constant-mm offset; results are identical to the constant-mm row."))
    adopted_rows = "\n".join(f"| {name} | {int(m['n'])} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['icc2_1']:.3f} [{m['icc2_1_ci_low']:.3f}, {m['icc2_1_ci_high']:.3f}] | {m['ba_bias']:+.3f} [{m['ba_bias_ci_low']:+.3f}, {m['ba_bias_ci_high']:+.3f}] | {m['ba_loa_low']:.2f} to {m['ba_loa_high']:.2f} | {100 * m['threshold_agreement']:.1f} % | {m['threshold_kappa_linear']:.3f} [{m['threshold_kappa_linear_ci_low']:.3f}, {m['threshold_kappa_linear_ci_high']:.3f}] | {int(m['n_clipped'])} |"
                             for name, m in adopted.items())
    # numbers from the other Stage-6 files for the manuscript paragraph (fallbacks when absent)
    def _read(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else None
    bset, stab, tdf, altdf = _read("boundary_by_set.csv"), _read("offset_stability.csv"), _read("offset_test_high.csv"), _read("offset_alternatives.csv")
    a6 = bset.set_index("set").loc["(a) OOF, 145 reference high"] if bset is not None else None
    top_txt = f"{a6['gingiva_top_edge_mae_mm_mean']:.2f} mm (bias {a6['gingiva_top_edge_bias_mm_mean']:+.2f} mm)" if a6 is not None else "n/a"
    bot_txt = f"{a6['gingiva_bottom_edge_mae_mm_mean']:.2f} mm (bias {a6['gingiva_bottom_edge_bias_mm_mean']:+.2f} mm)" if a6 is not None else "n/a"
    if stab is not None:
        f_ = stab[stab["grouping"] == "fold"]
        fold_txt = f"{f_['offset_mm'].min():.2f}–{f_['offset_mm'].max():.2f} mm across the five folds"
    else:
        fold_txt = "n/a"
    from scipy import stats as _st
    _ok = pd.concat([dev, hold])
    _g = [(p["selected_mm"] - p["ref_mm"]).to_numpy(float) for _, p in _ok.groupby("fold")]
    fold_p = float(_st.f_oneway(*_g).pvalue) if len(_g) > 1 else float("nan")
    fin_txt = ""
    if tdf is not None:
        fr = tdf[(tdf["masks"] == "final model") & (tdf["candidate"].str.endswith("uncorrected"))].iloc[0]
        fo = tdf[(tdf["masks"] == "fold models (OOF)") & (tdf["candidate"].str.endswith("uncorrected"))].iloc[0]
        fin_txt = f"the final model on the 29 test-set images showed the same bias ({fr['ba_bias']:+.3f} mm [{fr['ba_bias_ci_low']:+.2f}, {fr['ba_bias_ci_high']:+.2f}]) as the fold models on the same images ({fo['ba_bias']:+.3f} mm)"
    alt_txt = ""
    if altdf is not None:
        ha = altdf[(altdf["subset"] == "holdout") & (altdf["correction"] == "none")].sort_values("mae")
        alt_txt = f"lower percentiles of the column profile (p10, p5, minimum) and lip-anchored variants did not remove the bias (best alternative {ha.iloc[0]['candidate']}: holdout MAE {ha.iloc[0]['mae']:.2f} mm, residual bias {ha.iloc[0]['ba_bias']:+.2f} mm)"
    ho_u, ho_c, te_u, te_c = h_none, adopted["holdout"], unc_test, adopted["test high, final model"]
    paragraph = f"""## Adopted correction (config.yaml `measurement.offset_px = {off_px:+.0f}`) — value level, verification
The correction is defined in pixels because the physical finding is a fixed number of pixels at the lower edge; it is converted to mm with the scale valid for each image (global scale here; the experts' per-image scale in Stage 4). {ver}

| subset | n | MAE, mm | RMSE, mm | ICC(2,1) [CI] | bias, mm [CI] | 95 % LoA | label agreement | κ linear [CI] | clipped |
|---|---|---|---|---|---|---|---|---|---|
{adopted_rows}

## Manuscript paragraph (Methods / Results, post-hoc calibration)
The segmentation error was concentrated at the lower gingival margin. Against the annotated masks of the 145 reference images (out-of-fold predictions), the upper, lip-side edge of the gingiva was accurate (mean absolute error {top_txt}) whereas the lower edge was placed systematically too low (mean absolute error {bot_txt}), i.e. the model consistently included a thin strip of the festooned margin. The resulting over-measurement was constant rather than proportional: the mean error was {fold_txt} (one-way ANOVA p = {fold_p:.2f}), did not differ between image frame groups, and {fin_txt or 'was reproduced by the final model on the independent test images'} (paired difference final − fold models {paired['bias']:+.2f} mm [{paired['bias_ci_low']:+.2f}, {paired['bias_ci_high']:+.2f}]). A constant pixel offset of {off_px:+.0f} px ({off_mm_global:+.2f} mm at the global scale of {k:.2f} px/mm) was therefore estimated post hoc on the development subset of the out-of-fold predictions (n = {len(dev)}) and applied unchanged to the held-out subset (n = {len(hold)}) and to the final model's test-set images (n = {int(te_u['n'])}). Alternative estimators were examined first: {alt_txt or 'lower percentiles and lip-anchored variants did not remove the bias'}; the bias is a shift of the edge, not a spread within the profile, so a percentile cannot absorb it. On the held-out subset the correction reduced the mean absolute error from {ho_u['mae']:.2f} to {ho_c['mae']:.2f} mm and the bias from {ho_u['ba_bias']:+.2f} to {ho_c['ba_bias']:+.2f} mm (ICC(2,1) {ho_u['icc2_1']:.2f} → {ho_c['icc2_1']:.2f}; agreement with the Table 1 class {100 * ho_u['threshold_agreement']:.0f} % → {100 * ho_c['threshold_agreement']:.0f} %, linear-weighted κ {ho_u['threshold_kappa_linear']:.2f} → {ho_c['threshold_kappa_linear']:.2f}); on the independent test images from {te_u['mae']:.2f} to {te_c['mae']:.2f} mm (bias {te_u['ba_bias']:+.2f} → {te_c['ba_bias']:+.2f} mm). Because the offset was derived after inspecting the out-of-fold error, it is a post-hoc calibration: uncorrected results are reported as the primary analysis throughout, corrected results as a secondary analysis, and the offset is stated explicitly so that it can be re-estimated for any retrained model.
"""
    md += "\n" + paragraph
    (out_dir / "offset_correction.md").write_text(md, encoding="utf-8")
    print(md.split("## Selection rule")[1][:900])
    print(paragraph[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
