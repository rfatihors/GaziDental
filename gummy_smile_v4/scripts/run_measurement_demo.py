#!/usr/bin/env python
"""Stage 2 check — measure ground-truth masks of a few images with the new module and
compare with the legacy v3 function on the same masks (spec §6.1).

Outputs (outputs/02_measure/):
  gt_overlay_examples.png   gingiva red, lip blue; title: group, merged gingiva instances, values
  v3_vs_v4.csv / .md        v3_value_px (v3 code on the merged lip+gingiva mask, as its pipeline
                            did), v3_gingiva_only_px (v3 code on the gingiva mask alone),
                            v4_value_px, v4_method, QC flags
  OZET.md
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # GaziDental/ -> import gummy_smile_v3 read-only

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.io.coco import load_annotations  # noqa: E402
from gsv4.masks.extract import from_coco  # noqa: E402
from gsv4.measure.gingival_display import measure_gingival_display  # noqa: E402
from gsv4.measure.qc import flags_to_str  # noqa: E402


def pick_images(manifest: pd.DataFrame, n_high: int = 3, n_normal: int = 2) -> pd.DataFrame:
    high = manifest[(manifest.group == "high") & (manifest.keep)].sort_values("uid").head(n_high)
    normal = manifest[(manifest.group == "normal") & (manifest.keep) & (manifest.n_gingiva >= 3)].sort_values("uid").head(n_normal)
    return pd.concat([high, normal])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    out_dir = resolve(cfg, Path(cfg["paths"]["outputs"]) / "02_measure")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(cfg, Path(cfg["paths"]["manifest_dir"]) / "dataset_manifest.csv"))
    sel = pick_images(manifest)

    from gummy_smile_v3.measurement.measure_gum_visibility import measure_gum_visibility  # legacy, read-only

    rows = []
    fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.7 * len(sel)))
    ann_cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        for ax, (_, r) in zip(axes, sel.iterrows()):
            key = (r.group, r.orig_split)
            if key not in ann_cache:
                ann_cache[key] = load_annotations(coco_root, r.group, r.orig_split, cfg["coco"])
            ann = ann_cache[key]
            img_path = coco_root / r.path
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            masks = from_coco(ann["annotations"], int(r.coco_image_id) if "coco_image_id" in r else next(im["id"] for im in ann["images"] if im["file_name"] == r.file_name), (h, w), cfg["coco"]["category_ids"])
            masks.check_shape((h, w))
            res = measure_gingival_display(masks.gingiva, masks.lip, px_per_mm=None, cfg=cfg["measurement"], keep_profile=True)

            # legacy v3 on the merged mask (its real input) and on gingiva alone
            merged = (masks.gingiva | (masks.lip if masks.lip is not None else False)).astype(np.uint8) * 255
            p_merged = Path(tmp) / f"{r.image}_merged.png"
            p_g = Path(tmp) / f"{r.image}_g.png"
            cv2.imwrite(str(p_merged), merged)
            cv2.imwrite(str(p_g), masks.gingiva.astype(np.uint8) * 255)
            v3_merged = measure_gum_visibility(img_path, p_merged, regions=6, px_per_mm=None)
            v3_g = measure_gum_visibility(img_path, p_g, regions=6, px_per_mm=None)

            rows.append({
                "uid": r.uid, "group": r.group, "width": w, "height": h,
                "gingiva_instances_merged": masks.n_gingiva_instances, "lip_instances": masks.n_lip_instances,
                "v3_value_px": round(v3_merged.gum_visibility_px, 2),
                "v3_gingiva_only_px": round(v3_g.gum_visibility_px, 2),
                "v4_value_px": round(res.gingival_display_px, 2),
                "v4_method": res.to_row()["method"],
                "v4_A_median_px": round(res.image_values[("A", "median", False)], 2),
                "v4_C_median_px": round(res.image_values[("C", "median", False)], 2),
                "v4_A_p25_lipanchored_px": round(res.image_values[("A", "p25", True)], 2),
                "gap_median_px": res.gap_median, "reference_mean_mm": r.reference_mean_mm,
                "qc_flags": flags_to_str(res.flags),
            })

            # overlay
            scale = 900 / w
            small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (900, int(h * scale)))
            over = small.astype(float)
            g_s = cv2.resize(masks.gingiva.astype(np.uint8), (900, int(h * scale)), interpolation=cv2.INTER_NEAREST) > 0
            over[g_s] = over[g_s] * 0.45 + np.array([255, 0, 0]) * 0.55
            if masks.lip is not None:
                l_s = cv2.resize(masks.lip.astype(np.uint8), (900, int(h * scale)), interpolation=cv2.INTER_NEAREST) > 0
                over[l_s] = over[l_s] * 0.55 + np.array([0, 60, 255]) * 0.45
            # crop to the mouth area
            ys = np.nonzero(g_s.any(axis=1) | (l_s.any(axis=1) if masks.lip is not None else False))[0]
            y0, y1 = max(0, ys.min() - 40), min(over.shape[0], ys.max() + 40)
            ax.imshow(over[y0:y1].astype(np.uint8))
            for zx in res.zeniths_c:
                ax.axvline(zx * scale, color="yellow", lw=0.8, alpha=0.8)
            ax.set_title(f"{r.group} | {r.image} | gingiva instances merged: {masks.n_gingiva_instances} | "
                         f"v4 {res.to_row()['method']} = {res.gingival_display_px:.1f} px | v3 (merged mask) = {v3_merged.gum_visibility_px:.1f} px"
                         + (f" | ref {r.reference_mean_mm:.2f} mm" if pd.notna(r.reference_mean_mm) else ""), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("GT masks: gingiva red, lip blue, yellow = method-C zeniths", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "gt_overlay_examples.png", dpi=100)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "v3_vs_v4.csv", index=False)
    md = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    (out_dir / "v3_vs_v4.md").write_text("# v3 vs v4 on the same ground-truth masks\n\n"
        "v3 = legacy `measure_gum_visibility` (largest contour, top-edge deviation) on the merged lip+gingiva mask exactly as its pipeline ran; "
        "v4 = gingiva-mask vertical thickness (method A, p25). Units: px at original resolution.\n\n" + md + "\n", encoding="utf-8")
    ozet = f"""# Aşama 2 — Türkçe özet

- 5 GT görüntü (3 high, 2 normal) ölçüldü; overlay `gt_overlay_examples.png`.
- v3 (birleşik maske, en büyük kontur = dudak) ve v4 (dişeti kalınlığı) değerleri `v3_vs_v4.md`'de; v3 değerleri dudak üst kenarının dalgalanmasıdır, dişeti görünürlüğüyle ilgisizdir.
- Normal grupta dişeti {', '.join(str(x) for x in df[df.group=='normal'].gingiva_instances_merged)} parçadan birleştirildi; v4 zenith bölgeleri sıfır okur.
"""
    (out_dir / "OZET.md").write_text(ozet, encoding="utf-8")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
