"""Manuscript figures (Stage 7). Every function returns a status dict
``{"name", "path", "status": "done"|"pending", "source"|"needs"}``; when the input data is
not available yet a placeholder PNG is written so the build is deterministic and the same
command fills it in later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PLOT_DPI = 300    # manuscript figures (plots)
PANEL_DPI = 150   # photo panels (kept under 2 MB)

MERMAID = """```mermaid
flowchart LR
    A[Smile photograph] --> B[YOLOv11x-seg<br/>imgsz 640, retina_masks]
    B --> C[Class-separated masks<br/>gingiva / lip, union of instances,<br/>original resolution]
    C --> D[Column-wise thickness profile t(x)<br/>longest vertical run, empty columns = 0]
    D --> E[Tooth regioning<br/>midline-anchored zeniths (C), fallback A]
    E --> F[Region values p25 -> image mean<br/>px / px_per_mm = mm]
    F --> G[Rule engine<br/>E1 <4, E2 3-6, E3 4-8, E4 >8 mm<br/>overlaps -> combined label]
    G --> H[Report: class, candidates,<br/>treatment alternatives, QC flags]
    C -. lip mask: window, midline,<br/>boundary check .-> D
```"""


def placeholder(path: Path, title: str, needs: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.3, f"PENDING — needs: {needs}", ha="center", va="center", fontsize=10, color="tab:red")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


def block_diagram(out_md: Path, out_png: Path) -> Dict[str, Any]:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("# System block diagram (Reviewer 1)\n\n" + MERMAID + "\n", encoding="utf-8")
    boxes = [
        ("Smile\nphotograph", "#e8eef7"), ("YOLOv11x-seg\nimgsz 640\nretina_masks", "#dbe9d8"),
        ("Class-separated\nmasks (gingiva, lip)\nunion of instances,\noriginal resolution", "#dbe9d8"),
        ("Thickness profile\nt(x): longest vertical\nrun per column,\nempty columns = 0", "#fbe9d0"),
        ("Tooth regioning\nmidline-anchored\nzeniths (C),\nfallback A", "#fbe9d0"),
        ("Region p25 →\nimage mean\npx / px_per_mm\n= mm", "#fbe9d0"),
        ("Rule engine\nE1 <4 · E2 3–6\nE3 4–8 · E4 >8 mm\noverlaps → E1-E2 / E2-E3", "#f4d9d9"),
        ("Report\nclass, candidates,\ntreatment alternatives,\nQC flags", "#e8eef7"),
    ]
    fig, ax = plt.subplots(figsize=(16, 3.4))
    ax.set_xlim(0, len(boxes) * 2.0); ax.set_ylim(0, 3); ax.axis("off")
    for i, (txt, col) in enumerate(boxes):
        x = i * 2.0 + 0.15
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.7, 1.8, boxstyle="round,pad=0.03", fc=col, ec="0.3", lw=1))
        ax.text(x + 0.85, 1.6, txt, ha="center", va="center", fontsize=7.6)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 2.13, 1.6), xytext=(x + 1.72, 1.6), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.25", mutation_scale=16))
    ax.annotate("lip mask: x-window, midline, boundary check, lip-anchored estimator", xy=(6.15, 0.75), xytext=(5.0, 0.2),
                fontsize=7, color="0.35", arrowprops=dict(arrowstyle="-|>", color="0.5", ls="--"))
    ax.set_title("Gingival display measurement and decision-support pipeline (v4)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_DPI); plt.close(fig)
    return {"name": "pipeline_block_diagram", "path": str(out_png), "status": "done", "source": "gsv4 code"}


def scatter_and_bland_altman(per_image: pd.DataFrame, out_png: Path, title: str, value_col: str = "selected_mm", ref_col: str = "ref_mm",
                             subset_col: Optional[str] = "split", subset_value: Optional[str] = "holdout") -> Dict[str, Any]:
    d = per_image.copy()
    label = ""
    if subset_col and subset_col in d.columns and subset_value:
        d = d[d[subset_col] == subset_value]
        label = f" ({subset_value})"
    d = d.dropna(subset=[value_col, ref_col])
    x, y = d[ref_col].to_numpy(float), d[value_col].to_numpy(float)
    diff, mean = y - x, (x + y) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(x, y, s=16, alpha=0.8); lim = [0, max(x.max(), y.max()) * 1.05]; ax.plot(lim, lim, "k--", lw=1, label="identity (y = x)"); ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Clinical reference measurement (ImageJ), mm"); ax.set_ylabel("Pipeline measurement, mm")
    r = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"n = {len(d)}{label}: MAE {np.abs(diff).mean():.2f} mm, r = {r:.3f}", fontsize=9)
    ax = axes[1]
    ax.scatter(mean, diff, s=16, alpha=0.8)
    for v, ls, lab in ((bias, "-", f"bias {bias:+.2f} mm"), (bias - 1.96 * sd, "--", f"95 % LoA {bias - 1.96 * sd:.2f} mm"), (bias + 1.96 * sd, "--", f"95 % LoA {bias + 1.96 * sd:.2f} mm")):
        ax.axhline(v, color="k", ls=ls, lw=1); ax.text(mean.max(), v, lab, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("Mean of the two methods, mm"); ax.set_ylabel("Difference (pipeline − reference), mm"); ax.set_title("Bland–Altman plot", fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=PLOT_DPI); plt.close(fig)
    return {"name": out_png.stem, "path": str(out_png), "status": "done", "n": int(len(d))}


def segmentation_examples(cfg, manifest: pd.DataFrame, mask_dirs: List[Path], out_png: Path, n: int = 6) -> Dict[str, Any]:
    """GT (left) and predicted (right) class masks side by side for n images; pending
    when no predicted masks exist yet."""
    import cv2

    from gsv4.config import resolve
    from gsv4.io.coco import load_annotations
    from gsv4.masks.extract import from_coco, from_png

    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    kept = manifest[manifest["keep"]]
    pred_dir = next((d for d in mask_dirs if d.exists() and any(d.glob("*_gingiva.png"))), None)
    if pred_dir is None:
        placeholder(out_png, "Segmentation examples (GT vs prediction)", "predicted masks in outputs/05_predictions/{oof,test} (workstation run)")
        return {"name": "segmentation_examples", "path": str(out_png), "status": "pending", "needs": "outputs/05_predictions/oof or test masks"}
    avail = {p.name.replace("_gingiva.png", "") for p in pred_dir.glob("*_gingiva.png")}
    sel = kept[kept["image"].isin(avail)]
    sel = pd.concat([sel[sel.group == "high"].head(max(1, n - 2)), sel[sel.group != "high"].head(2)]).head(n)
    if sel.empty:
        placeholder(out_png, "Segmentation examples", "predicted masks matching kept images")
        return {"name": "segmentation_examples", "path": str(out_png), "status": "pending", "needs": "masks for kept images"}
    cache: Dict[Any, Any] = {}
    fig, axes = plt.subplots(len(sel), 2, figsize=(12, 2.4 * len(sel)))
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, sel.iterrows()):
        key = (r["group"], r["orig_split"])
        if key not in cache:
            cache[key] = load_annotations(coco_root, r["group"], r["orig_split"], cfg["coco"])
        ann = cache[key]
        im = next(i for i in ann["images"] if i["file_name"] == r["file_name"])
        shape = (int(im["height"]), int(im["width"]))
        gt = from_coco(ann["annotations"], int(im["id"]), shape, cfg["coco"]["category_ids"])
        pm = from_png(pred_dir / f"{r['image']}_gingiva.png", pred_dir / f"{r['image']}_lip.png", expected_shape=shape)
        img = cv2.cvtColor(cv2.imread(str(coco_root / r["path"])), cv2.COLOR_BGR2RGB)
        for ax, masks, name in zip(row, (gt, pm), ("ground truth", f"prediction ({pred_dir.name})")):
            over = img.astype(float)
            over[masks.gingiva] = over[masks.gingiva] * 0.45 + np.array([255, 0, 0]) * 0.55
            if masks.lip is not None:
                over[masks.lip] = over[masks.lip] * 0.55 + np.array([0, 60, 255]) * 0.45
            ys = np.nonzero(gt.gingiva.any(axis=1) | (gt.lip.any(axis=1) if gt.lip is not None else False))[0]
            y0, y1 = max(0, ys.min() - 60), min(shape[0], ys.max() + 60)
            ax.imshow(over[y0:y1].astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{r['group']} | {r['image']} | {name}", fontsize=8)
    fig.suptitle("Ground truth (left) vs predicted (right): gingiva red, lip blue", fontsize=10)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=PANEL_DPI); plt.close(fig)
    return {"name": "segmentation_examples", "path": str(out_png), "status": "done", "source": str(pred_dir)}


def copy_or_placeholder(src: Path, out_png: Path, title: str, needs: str) -> Dict[str, Any]:
    import shutil

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, out_png)
        return {"name": out_png.stem, "path": str(out_png), "status": "done", "source": str(src)}
    placeholder(out_png, title, needs)
    return {"name": out_png.stem, "path": str(out_png), "status": "pending", "needs": needs}


def boundary_error_figure(csv: Path, out_png: Path) -> Dict[str, Any]:
    if not csv.exists():
        placeholder(out_png, "Boundary error (upper / lower gingiva edge)", "outputs/05_predictions/boundary_error.csv (evaluate_test on the workstation)")
        return {"name": "boundary_error", "path": str(out_png), "status": "pending", "needs": str(csv)}
    b = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ("gingiva_top_edge_mae_px", "gingiva_bottom_edge_mae_px", "gingiva_boundary_iou"),
                              ("Upper edge (lip side) MAE, px", "Lower edge (gingival margin) MAE, px", "Boundary IoU")):
        for g, part in b.groupby("group"):
            ax.hist(part[col].dropna(), bins=20, alpha=0.6, label=f"{g} (n={len(part)})")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("Test set: gingiva boundary quality of the final model vs ground truth", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=PLOT_DPI); plt.close(fig)
    return {"name": "boundary_error", "path": str(out_png), "status": "done", "source": str(csv)}
