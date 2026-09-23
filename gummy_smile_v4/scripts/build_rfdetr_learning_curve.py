#!/usr/bin/env python
"""Learning curve of the RF-DETR final model, assembled from its own COCO evaluations (PLAN.md §3).

    python scripts/build_rfdetr_learning_curve.py --out 09_final_rfdetr

Each point is one training run's own evaluation, written by scripts/rfdetr_train_predict.py as
``outputs/08_architecture/rfdetr_metrics_<tag>[_<split>].json``:

  25/50/75 %  -> rfdetr_metrics_lc{25,50,75}_s<seed>.json          (evaluated on `val`)
  100 %       -> rfdetr_metrics_<model>_s<seed>_val.json           (the final model, which is judged
                                                                    on `test`, evaluated on `val`
                                                                    as well so the points compare)

All four points must come from the same evaluator and the same split; a mismatch stops the script
rather than producing a curve whose points are not comparable. Nothing is read from the YOLO
learning curve, which stays in the appendix as the previous final model's own evidence.

Outputs (default outputs/09_final_rfdetr/): learning_curve.csv, learning_curve.md, learning_curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.report.figures import PLOT_DPI  # noqa: E402

# The metric the plateau rule reads, in order of preference. RF-DETR's evaluate() returns whatever
# its pycocotools wrapper produces, so the name is resolved against the file rather than assumed.
METRIC_CANDIDATES = ("segm_mAP_50", "segm_map50", "segm_mAP50", "segm_mAP_50_95", "segm_mAP",
                     "segm_map", "map50_segm", "segm_AP50", "segm_AP")
FRACTIONS = {25: 0.25, 50: 0.50, 75: 0.75}


def load_point(path: Path, fraction: float, run: str) -> dict:
    if not path.exists():
        raise SystemExit(f"missing learning-curve point ({100 * fraction:.0f} %): {path}\n"
                         "Run that training (scripts/train_final_rfdetr.sh) or, for the 100 % point, "
                         "`rfdetr_train_predict.py --variant main --seed <seed> --predict-only --evaluate --evaluate-split val`.")
    m = json.loads(path.read_text())
    return {"fraction": fraction, "run": run, "split": m.get("split"), "evaluator": m.get("evaluator"),
            "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            **{k: v for k, v in m.items() if isinstance(v, (int, float)) and k != "seed"}}


def pick_metric(df: pd.DataFrame, wanted: str | None) -> str:
    if wanted:
        if wanted not in df.columns:
            raise SystemExit(f"metric {wanted!r} is not in the evaluation files; available: {sorted(c for c in df.columns if df[c].dtype != object)}")
        return wanted
    for c in METRIC_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit("none of the expected segmentation-mAP keys is in the evaluation files; pass --metric explicitly. "
                     f"Available numeric keys: {sorted(c for c in df.columns if df[c].dtype != object and c != 'fraction')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="09_final_rfdetr", help="output sub-directory under paths.outputs")
    ap.add_argument("--metrics-dir", default="outputs/08_architecture", help="where rfdetr_metrics_*.json live")
    ap.add_argument("--model", default="rfdetr-seg-large", help="model label of the 100 % point, as the metrics file names it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default=None, help="metric the plateau rule reads (default: the first segmentation mAP key present)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    outputs = resolve(cfg, cfg["paths"]["outputs"])
    out_dir = outputs / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    md = resolve(cfg, args.metrics_dir)
    lists = resolve(cfg, Path(cfg["paths"]["yolo_dataset"]) / "lists")

    rows = [load_point(md / f"rfdetr_metrics_lc{n}_s{args.seed}.json", f, f"lc{n}") for n, f in FRACTIONS.items()]
    rows.append(load_point(md / f"rfdetr_metrics_{args.model}_s{args.seed}_val.json", 1.0, "final (reused)"))
    df = pd.DataFrame(rows)

    splits, evals = set(df["split"].dropna()), set(df["evaluator"].dropna())
    if len(splits) != 1:
        raise SystemExit(f"the points were evaluated on different splits {sorted(splits)} — they are not comparable. "
                         "Evaluate the 100 % point on the same split as the subsets (--evaluate-split val).")
    if len(evals) != 1:
        raise SystemExit(f"the points come from different evaluators {sorted(evals)} — one curve, one evaluator (PLAN.md 4).")
    split, evaluator = splits.pop(), evals.pop()

    counts = {"lc25": "lc25_train.txt", "lc50": "lc50_train.txt", "lc75": "lc75_train.txt", "final (reused)": "main_train.txt"}
    df["n_train_images"] = [len([ln for ln in (lists / counts[r]).read_text().splitlines() if ln.strip()])
                            if (lists / counts[r]).exists() else pd.NA for r in df["run"]]
    metric = pick_metric(df, args.metric)
    df = df[["fraction", "run", "n_train_images", "split", "evaluator", "source", metric]
            + [c for c in df.columns if c not in ("fraction", "run", "n_train_images", "split", "evaluator", "source", metric)]]
    df.to_csv(out_dir / "learning_curve.csv", index=False)

    v = df.set_index("fraction")[metric]
    gain_early, gain_late = float(v[0.50] - v[0.25]), float(v[1.00] - v[0.75])
    plateau = gain_late < gain_early / 4          # the rule the YOLO curve used, unchanged
    verdict = "plateau" if plateau else "still rising"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["fraction"] * 100, df[metric], "o-", color="#0072B2", lw=1.8)
    for _, r in df.iterrows():
        ax.annotate(f"{r[metric]:.3f}", (r["fraction"] * 100, r[metric]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("training subset, % of the training partition"); ax.set_ylabel(f"{metric} ({split} split)")
    ax.set_title(f"RF-DETR-Seg Large @624 learning curve — {verdict}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out_dir / "learning_curve.png", dpi=PLOT_DPI); plt.close(fig)

    text = f"""# Learning curve — RF-DETR-Seg Large @624 (PLAN.md 3)

Model: **RF-DETR-Seg Large @624, seed {args.seed}**; evaluator: **{evaluator}**; split: **{split}**; metric: **{metric}**.
Each point is that training run's own COCO evaluation (`{args.metrics_dir}/rfdetr_metrics_*.json`); the 100 % point is the
final model of PLAN.md 1, reused rather than retrained and evaluated on the same split as the subsets. The YOLOv11x
learning curve is not mixed in here: it belongs to the previous final model and stays in the appendix.

{metric} gain 25→50 %: {gain_early:+.4f}; 75→100 %: {gain_late:+.4f} → **{verdict}** (rule: 75→100 gain < 1/4 of the 25→50 gain).

| fraction | run | n_train_images | {metric} |
|---|---|---|---|
""" + "\n".join(f"| {r['fraction']:.2f} | {r['run']} | {r['n_train_images']} | {r[metric]:.4f} |" for _, r in df.iterrows()) + "\n"
    (out_dir / "learning_curve.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
