#!/usr/bin/env python
"""Seed spread of the reported measurement (outputs/09_final_rfdetr/PLAN.md, Amendment 5).

    python scripts/run_seed_spread.py --seeds 42 43 44

The five-fold out-of-fold pipeline is repeated with seeds 43 and 44 and measured with the SAME
fixed method and scale as everything else (``configs/config.yaml``; nothing is selected or fitted
here). This script reads each seed's out-of-fold masks, measures the 145 reference images, and
reports what Amendment 5 pre-registered:

  * per seed: MAE, RMSE, ICC(2,1), Bland-Altman bias and limits, class agreement, linear-weighted kappa;
  * over the seeds: mean +- SD of each of those;
  * between seeds: the paired difference in MAE with its 95 % bootstrap interval, computed by the
    same function the architecture comparison used, so the two spreads are directly comparable;
  * that spread against the 0.293 mm architecture lead, read from the comparison's own file.

**The final model stays seed 42.** Amendment 5 fixes that in advance: nothing here selects a model,
a configuration, a method or a scale, whichever seed turns out to have the lowest error.

Outputs (outputs/10_seed_spread/): per_seed.csv, summary.csv, paired_seeds.csv,
per_image_by_seed.csv, RESULTS.md, OZET.md.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.architecture import paired_difference  # noqa: E402
from gsv4.eval.oracle import combo_name, measure_images  # noqa: E402
from gsv4.eval.prediction import agreement_metrics, check_oof, md_table  # noqa: E402
from gsv4.rules.thresholds import label_for_mm  # noqa: E402

# what the spread is compared against (PLAN.md Amendment 5, A5.4)
ARCH_FILE = "08_architecture/paired_comparisons.csv"
REPORTED = ["mae", "rmse", "icc2_1", "ba_bias", "ba_loa_low", "ba_loa_high",
            "threshold_agreement", "threshold_kappa_linear"]


def check_seed_directory(oof: pd.DataFrame, seed: int, where: str) -> None:
    """Refuse a prediction table that cannot be reported, before it is measured.

    Two mistakes would be invisible in the numbers and are therefore fatal here: a directory that
    accumulated more than one seed (each seed must have its own, ``--masks-out``), and the
    label-space signature of `outputs/08_architecture/PROTOCOL.md` Amendment 2 — instances dropped
    for having no role, or no lip mask anywhere.
    """
    seeds_in = sorted({int(x) for x in oof["seed"].dropna()}) if "seed" in oof.columns else []
    if seeds_in != [seed]:
        raise SystemExit(f"{where} holds seeds {seeds_in}, expected [{seed}] — each seed needs its own directory")
    if "n_ignored" in oof.columns and int(oof["n_ignored"].fillna(0).sum()):
        raise SystemExit(f"seed {seed}: {int(oof['n_ignored'].sum())} instance(s) dropped for having no role — "
                         "that is the signature of a label-space fault (PROTOCOL.md Amendment 2); do not report this run")
    if "n_lip" in oof.columns and len(oof) and int((oof["n_lip"] == 0).sum()) == len(oof):
        raise SystemExit(f"seed {seed}: no image has a lip mask — label-space fault, do not report this run")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="10_seed_spread")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--masks-pattern", default="oof_rfdetr_s{seed}",
                    help="out-of-fold mask directory per seed, under outputs/05_predictions/")
    ap.add_argument("--reference-seed", type=int, default=42,
                    help="the seed of the reported final model; its masks are the ones Stage 6 already used")
    ap.add_argument("--reference-masks", default="oof_rfdetr",
                    help="mask directory of --reference-seed (it was written before the pattern existed)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    out_dir = outputs / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = resolve(cfg, cfg["paths"]["predictions"])
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    mcfg = cfg["measurement"]
    combo = combo_name(mcfg["method"]["regioning"], mcfg["method"]["estimator"], bool(mcfg["method"]["anchored"]))
    k = float(mcfg["px_per_mm"])
    off = float(mcfg.get("bottom_edge_offset_px", 0) or 0)
    if off:
        raise SystemExit(f"measurement.bottom_edge_offset_px is {off}: Amendment 3 dropped the post-hoc correction, "
                         "and the seed spread is reported on the uncorrected pipeline only")

    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    ref = manifest[manifest["keep"] & manifest["has_reference_measurement"]].copy()
    rows = ref[["uid", "image", "patient_id", "group", "split", "cv_fold", "width", "height", "frame_ok"]].reset_index(drop=True)
    stage3 = pd.read_csv(outputs / "03_oracle" / "per_image_results.csv").set_index("uid")
    if stage3["selected_method"].iloc[0] != combo or abs(float(stage3["selected_px_per_mm"].iloc[0]) - k) / k > 1e-3:
        raise SystemExit(f"config method/scale ({combo}, {k}) differ from Stage 3 — the seed spread must use the fixed ones")

    per_seed, long_rows = [], []
    for seed in args.seeds:
        name = args.reference_masks if seed == args.reference_seed else args.masks_pattern.format(seed=seed)
        mdir = pred_dir / name
        if not (mdir / "oof_predictions.csv").exists():
            raise SystemExit(f"seed {seed}: {mdir / 'oof_predictions.csv'} does not exist — run the folds for this seed first "
                             "(scripts/train_seed_spread.sh on the workstation)")
        oof = pd.read_csv(mdir / "oof_predictions.csv")
        check_seed_directory(oof, seed, str(mdir))
        chk = check_oof(oof, ref[["image", "uid", "cv_fold"]], mdir)
        if not chk["ok"]:
            raise SystemExit(f"seed {seed}: out-of-fold table is inconsistent — {chk['problems']}")
        print(f"[spread] seed {seed}: measuring {len(rows)} images from {mdir} with {combo} at {k:.4f} px/mm …")
        meas = measure_images(rows, cfg, coco_root, mask_source=str(mdir))
        d = rows.merge(meas.drop(columns=["image", "group", "width", "height"]), on="uid", how="left")
        d["selected_mm"] = d[f"{combo}_px"] / k
        d["ref_mm"] = d["uid"].map(stage3["ref_mm"])
        d["ref_label"] = d["uid"].map(stage3["ref_label"])
        d["seed"] = seed
        d["masks_dir"] = str(mdir.relative_to(ROOT)) if mdir.is_relative_to(ROOT) else str(mdir)
        long_rows.append(d[["uid", "image", "seed", "masks_dir", "selected_mm", "ref_mm", "ref_label"]])
        m = agreement_metrics(d["selected_mm"], d["ref_mm"], n_boot=args.n_boot, seed=int(cfg["seed"]))
        m.update({"seed": seed, "masks": str(mdir.name),
                  "n_empty_prediction": int(d["selected_mm"].isna().sum()),
                  "is_reported_model": seed == args.reference_seed})
        per_seed.append(m)

    ps = pd.DataFrame(per_seed)
    ps = ps[["seed", "masks", "is_reported_model", "n", "n_empty_prediction"] + [c for c in REPORTED if c in ps.columns]]
    ps.to_csv(out_dir / "per_seed.csv", index=False)
    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(out_dir / "per_image_by_seed.csv", index=False)

    summary = pd.DataFrame([{"metric": c, "mean": float(ps[c].mean()), "sd": float(ps[c].std(ddof=1)),
                             "min": float(ps[c].min()), "max": float(ps[c].max()), "range": float(ps[c].max() - ps[c].min()),
                             "n_seeds": int(len(ps))} for c in REPORTED if c in ps.columns])
    summary.to_csv(out_dir / "summary.csv", index=False)

    # paired differences, by the function the architecture comparison used
    pairs = []
    for a, b in combinations(args.seeds, 2):
        da = long[long["seed"] == a][["uid", "selected_mm", "ref_mm"]]
        db = long[long["seed"] == b][["uid", "selected_mm", "ref_mm"]]
        r = paired_difference(da, db, n_boot=args.n_boot, seed=int(cfg["seed"]))
        r.update({"seed_a": a, "seed_b": b, "abs_diff_mae_mm": abs(r.get("diff_mae_mm", float("nan")))})
        pairs.append(r)
    pr = pd.DataFrame(pairs)
    pr.to_csv(out_dir / "paired_seeds.csv", index=False)

    # the comparison Amendment 5 fixed in advance
    arch_path = outputs / ARCH_FILE
    arch_lead = arch_txt = None
    if arch_path.exists():
        ap_ = pd.read_csv(arch_path)
        w = ap_.sort_values("diff_mae_mm", ascending=False).iloc[0]
        arch_lead = float(w["diff_mae_mm"])
        arch_txt = f"{w['model_b']} over {w['model_a']}, {arch_lead:.3f} mm [{w['ci_low']:.3f}, {w['ci_high']:.3f}]"
    widest = pr.sort_values("abs_diff_mae_mm", ascending=False).iloc[0] if len(pr) else None
    verdict = "no architecture comparison file to compare against"
    if arch_lead is not None and widest is not None:
        ratio = float(widest["abs_diff_mae_mm"]) / arch_lead if arch_lead else float("nan")
        same_order = ratio >= 0.5 or bool(widest["excludes_zero"])
        verdict = (
            f"The widest between-seed paired difference is {widest['abs_diff_mae_mm']:.3f} mm "
            f"[{widest['ci_low']:.3f}, {widest['ci_high']:.3f}] (seeds {int(widest['seed_a'])} and {int(widest['seed_b'])}), "
            f"{ratio:.2f} times the architecture lead of {arch_lead:.3f} mm, and its interval "
            + ("excludes" if widest["excludes_zero"] else "contains") + " zero. "
            + ("**This is of the same order as the architecture difference.** Amendment 5 A5.4 therefore applies: it is "
               "written into the Limitations as a limitation of the architecture comparison — a difference of that size "
               "cannot be cleanly separated from run-to-run variation in the five-fold pipeline — and the architecture "
               "result carries that caveat wherever it is quoted. The architecture comparison is not re-run and its "
               "conclusion is not reversed on this evidence."
               if same_order else
               "**This is small relative to the architecture difference.** The architecture conclusion stands as written, "
               "and the seed spread is reported as the evidence that it does."))

    md = [f"# Seed spread of the reported measurement — {', '.join(str(s) for s in args.seeds)}", "",
          "Pre-registration: `outputs/09_final_rfdetr/PLAN.md`, Amendment 5, written and committed before these runs.", "",
          f"Five-fold out-of-fold pipeline repeated per seed, everything identical except the seed. Measurement method "
          f"**{combo}** at **{k:.2f} px/mm**, fixed in `configs/config.yaml` and not re-selected here; no post-hoc "
          f"calibration. Same {int(ps['n'].max())} reference images for every seed.", "",
          f"**The reported final model remains seed {args.reference_seed}.** Amendment 5 fixed that before these runs: "
          "no model, configuration, method or scale is chosen on this analysis.", "",
          "## Per seed", "", md_table(ps), "",
          "## Over the seeds (mean ± SD)", "", md_table(summary), "",
          "## Between seeds (paired, bootstrap over the shared images)", "",
          "Computed by the same `paired_difference` the architecture comparison used, so the two spreads are measured "
          "the same way. Positive means seed A has the larger error.", "", md_table(pr), "",
          "## Against the architecture difference", "",
          (f"Architecture comparison (`{ARCH_FILE}`): {arch_txt}." if arch_txt else "No architecture comparison file found."), "",
          verdict, ""]
    (out_dir / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")

    oz = [f"# Tohum yayılımı — Türkçe özet", "",
          f"- Tohumlar: {', '.join(str(s) for s in args.seeds)}; her biri 5 fold, aynı bölünme, aynı yöntem ({combo}, {k:.2f} px/mm), aynı {int(ps['n'].max())} görüntü.",
          f"- Nihai model **seed {args.reference_seed}** olarak kalır (PLAN.md Ek 5, koşudan önce sabitlendi).",
          "- MAE tohumlar arası: " + ", ".join(f"s{int(r.seed)} {r.mae:.3f}" for r in ps.itertuples()) +
          f" → ortalama {summary.loc[summary['metric'] == 'mae', 'mean'].iloc[0]:.3f} ± {summary.loc[summary['metric'] == 'mae', 'sd'].iloc[0]:.3f} mm.",
          "- ICC(2,1) tohumlar arası: " + ", ".join(f"s{int(r.seed)} {r.icc2_1:.3f}" for r in ps.itertuples()) + ".",
          (f"- En geniş eşleştirilmiş tohum farkı: {widest['abs_diff_mae_mm']:.3f} mm [{widest['ci_low']:.3f}, {widest['ci_high']:.3f}]"
           + (f"; mimari farkı {arch_lead:.3f} mm." if arch_lead else ".")
           if widest is not None else "- Tek tohum koşuldu: tohumlar arası fark hesaplanamaz."),
          "- Ayrıntı ve karar: `RESULTS.md`.", ""]
    (out_dir / "OZET.md").write_text("\n".join(oz), encoding="utf-8")
    print("\n".join(md[-6:]))
    print(f"[spread] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
