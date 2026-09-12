#!/usr/bin/env python
"""Learning curve: retrain on nested 25/50/75 % subsets of the training split (the 100 %
point is the final model), validate on the same validation set, per-class mask mAP.

    python -m gsv4.train.learning_curve --fraction 0.25 [--dry-run]   # one point (restartable)
    python -m gsv4.train.learning_curve --collect [--dry-run]         # -> outputs/05_predictions/learning_curve.csv + .png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gsv4.config import load_config, resolve
from gsv4.train.common import is_done, mark_done, per_class_metrics, run_dir, train_model


def point_name(fraction: float) -> str:
    return "final" if fraction >= 1.0 else f"lc{int(round(fraction * 100))}"


def collect(cfg, dry_run: bool) -> Path:
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    pred = resolve(cfg, cfg["paths"]["predictions"])
    pred.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in cfg["yolo"]["learning_curve_fractions"]:
        name = point_name(f)
        lst = ds / "lists" / ("main_train.txt" if name == "final" else f"{name}_train.txt")
        n_train = len(lst.read_text().splitlines()) if lst.exists() else None
        best = run_dir(cfg, name) / "weights" / "best.pt"
        rec = {"fraction": f, "run": name, "n_train_images": n_train, "best_pt": str(best), "done": is_done(cfg, name)}
        if dry_run or not best.exists():
            rows.append(rec)
            continue
        from ultralytics import YOLO

        m = YOLO(str(best)).val(data=str(ds / "data_main.yaml"), split="val", imgsz=int(cfg["yolo"]["imgsz"]), plots=False, verbose=False,
                                project=str(run_dir(cfg, "lc_collect")), name=name, exist_ok=True)
        pc = per_class_metrics(m, {int(k): v for k, v in m.names.items()})
        for cname, vals in pc.items():
            for k, v in vals.items():
                rec[f"{cname}_{k}"] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = pred / "learning_curve.csv"
    df.to_csv(out, index=False)
    g, l = cfg["class_names"]["gingiva"], cfg["class_names"]["lip"]
    if not dry_run and f"{g}_seg_map50" in df.columns:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for cname, style in ((g, "o-"), (l, "s--")):
            ax.plot(df["n_train_images"], df[f"{cname}_seg_map50"], style, label=f"{cname} mask mAP@50")
            ax.plot(df["n_train_images"], df[f"{cname}_seg_map50_95"], style, alpha=0.5, label=f"{cname} mask mAP@50–95")
        ax.set_xlabel("training images"); ax.set_ylabel("validation mAP"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("Learning curve (nested stratified subsets, same validation set)", fontsize=9)
        fig.tight_layout(); fig.savefig(pred / "learning_curve.png", dpi=120); plt.close(fig)
        gain_lo = df.loc[df.fraction == 0.5, f"{g}_seg_map50"].iloc[0] - df.loc[df.fraction == 0.25, f"{g}_seg_map50"].iloc[0]
        gain_hi = df.loc[df.fraction == 1.0, f"{g}_seg_map50"].iloc[0] - df.loc[df.fraction == 0.75, f"{g}_seg_map50"].iloc[0]
        verdict = "plateau" if abs(gain_hi) < 0.25 * abs(gain_lo) else "not plateaued (report as limitation)"
        (pred / "learning_curve.md").write_text(f"# Learning curve\n\nGingiva mask mAP@50 gain 25→50 %: {gain_lo:+.4f}; 75→100 %: {gain_hi:+.4f} → **{verdict}** (rule: 75→100 gain < 1/4 of the 25→50 gain).\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
        mark_done(cfg, "lc_collect")
    print(df.to_string(index=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = resolve(cfg, cfg["paths"]["yolo_dataset"])
    if args.fraction is not None:
        name = point_name(args.fraction)
        if name == "final":
            raise SystemExit("the 100 % point is the final model: use gsv4.train.train --name final")
        train_model(cfg, ds / f"data_{name}.yaml", name, dry_run=args.dry_run)
    if args.collect:
        collect(cfg, args.dry_run)
    if args.fraction is None and not args.collect:
        ap.error("give --fraction or --collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
