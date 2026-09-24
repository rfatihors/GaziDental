#!/usr/bin/env python
"""Re-measure a stored per-image table from the masks on disk and report every difference.

    python scripts/verify_measurement_reproducible.py                       # the reported Stage 6
    python scripts/verify_measurement_reproducible.py --stage6 06_prediction --oof-masks outputs/05_predictions/oof

A committed result table must be reproducible from the masks committed beside it. This script is
the check: it measures every reference image again with the method and scale fixed in
`configs/config.yaml`, compares the result with the stored table column by column, and exits
non-zero if anything moved. It is how the zenith tie of `outputs/09_final_rfdetr/PLAN.md`
Amendment 6 was found and how its fix is verified.

Reports, per column family and overall: how many images differ, the largest difference, and the
images themselves. `--tol` is the tolerance in pixels (default 1e-9: a stored table should be
reproducible to the bit, not approximately).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.eval.oracle import COMBOS, combo_name, measure_images  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage6", default="09_final_rfdetr", help="directory holding per_image_results.csv")
    ap.add_argument("--oof-masks", default="outputs/05_predictions/oof_rfdetr",
                    help="mask directory to re-measure, or the literal 'gt' to check a ground-truth run (Stage 3)")
    ap.add_argument("--tol", type=float, default=1e-9, help="pixel tolerance; the default demands bit-level reproduction")
    ap.add_argument("--out", default=None, help="write the per-image comparison here (CSV)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    stored_path = outputs / args.stage6 / "per_image_results.csv"
    gt = args.oof_masks == "gt"
    masks = Path(args.oof_masks)
    if not gt and not masks.is_absolute():
        masks = resolve(cfg, masks)
    stored = pd.read_csv(stored_path).set_index("uid")
    mcfg = cfg["measurement"]
    combo = combo_name(mcfg["method"]["regioning"], mcfg["method"]["estimator"], bool(mcfg["method"]["anchored"]))
    k = float(mcfg["px_per_mm"])

    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    ref = manifest[manifest["keep"] & manifest["has_reference_measurement"]].copy()
    cols_needed = ["uid", "image", "patient_id", "group", "split", "cv_fold", "width", "height", "frame_ok", "orig_split", "file_name"]
    rows = ref[[c for c in cols_needed if c in ref.columns]].reset_index(drop=True)
    print(f"[verify] {stored_path} against {masks}: {len(rows)} images, method {combo} at {k:.4f} px/mm")
    now = measure_images(rows, cfg, resolve(cfg, cfg["paths"]["coco_root"]),
                         mask_source="gt" if gt else str(masks)).set_index("uid")

    cols = [f"{combo_name(r, e, a)}_px" for r, e, a in COMBOS]
    cols = [c for c in cols if c in stored.columns and c in now.columns]
    shared = [u for u in stored.index if u in now.index]
    diffs = []
    for u in shared:
        for c in cols:
            s, n = float(stored.loc[u, c]), float(now.loc[u, c])
            if np.isnan(s) and np.isnan(n):
                continue
            d = abs(s - n) if not (np.isnan(s) or np.isnan(n)) else float("inf")
            if d > args.tol:
                diffs.append({"uid": u, "column": c, "stored_px": s, "recomputed_px": n, "abs_diff_px": d,
                              "abs_diff_mm": d / k})
    dd = pd.DataFrame(diffs)
    n_img = int(dd["uid"].nunique()) if len(dd) else 0
    print(f"[verify] columns compared: {len(cols)} per image; images differing: {n_img} of {len(shared)}")
    if len(dd):
        worst = dd.sort_values("abs_diff_px", ascending=False)
        print(f"[verify] largest difference: {worst.iloc[0]['abs_diff_px']:.6f} px "
              f"({worst.iloc[0]['abs_diff_mm']:.6f} mm) in {worst.iloc[0]['column']} of {worst.iloc[0]['uid']}")
        prim = dd[dd["column"] == f"{combo}_px"]
        if len(prim):
            print(f"[verify] the reported method ({combo}): {len(prim)} image(s), largest "
                  f"{prim['abs_diff_px'].max():.6f} px = {prim['abs_diff_mm'].max():.6f} mm")
        else:
            print(f"[verify] the reported method ({combo}) is unchanged on every image")
        print(worst.head(20).to_string(index=False))
        # what it does to the headline number
        sm_s = stored.loc[shared, f"{combo}_px"].astype(float) / k
        sm_n = now.loc[shared, f"{combo}_px"].astype(float) / k
        r = stored.loc[shared, "ref_mm"].astype(float)
        ok = np.isfinite(sm_s) & np.isfinite(sm_n) & np.isfinite(r)
        print(f"[verify] MAE from the stored table {np.abs(sm_s[ok] - r[ok]).mean():.6f} mm, "
              f"recomputed {np.abs(sm_n[ok] - r[ok]).mean():.6f} mm")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        dd.to_csv(out, index=False)
        print(f"[verify] -> {out}")
    if len(dd):
        print("[verify] NOT REPRODUCIBLE: the stored table does not follow from the masks beside it")
        return 1
    print("[verify] reproducible: every stored value follows from the masks on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
