"""Synthetic expert forms and model output for end-to-end tests of Stage 4.

The generator knows the *true* class and mm of every image, so agreement statistics
of the produced forms are controllable: ``p_agree`` is the probability that an expert
reproduces the true class (otherwise a neighbouring class), ``mm_sd`` the expert's
measurement noise, ``scale_cv`` the spread of the per-image scale entries.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openpyxl
import pandas as pd

from gsv4.rules.thresholds import label_for_mm

CLASSES = ["E1", "E2", "E3", "E4"]
HEADER = ["Sıra", "Görüntü ID", "Ölçek (pixel/mm)", 13, 12, 11, 21, 22, 23, "Etiyoloji sınıfı", "İkinci aday", "Güven", "Not"]


def synthetic_key(n_images: int = 145, n_repeats: int = 20, seed: int = 42) -> pd.DataFrame:
    """ANAHTAR-like table: G001.. with the last ``n_repeats`` rows repeating earlier images."""
    rng = np.random.default_rng(seed)
    images = [f"IMG_{1000 + i}_jpg" for i in range(n_images)]
    rep = rng.choice(images, n_repeats, replace=False)
    rows = [{"goruntu_id": f"G{i + 1:03d}", "sira": i + 1, "orijinal": im, "tekrar": 0} for i, im in enumerate(images)]
    rows += [{"goruntu_id": f"G{n_images + j + 1:03d}", "sira": n_images + j + 1, "orijinal": im, "tekrar": 1} for j, im in enumerate(rep)]
    return pd.DataFrame(rows)


def synthetic_truth(key: pd.DataFrame, seed: int = 42, px_per_mm: float = 16.84) -> pd.DataFrame:
    """Per-image truth: 6 tooth mm values (mean in 0.5–7.5 mm), class from Table 1 (first candidate)."""
    rng = np.random.default_rng(seed)
    images = key.loc[key["tekrar"] == 0, "orijinal"].tolist()
    rows = []
    for im in images:
        mean = float(np.clip(rng.gamma(3.0, 1.0), 0.3, 7.5))
        teeth = np.clip(mean + rng.normal(0, 0.6, 6), 0, None)
        teeth[rng.random(6) < 0.05] = 0.0  # a tooth without visible gingiva
        m = float(teeth.mean())
        label = label_for_mm(m)
        rows.append({"image": im, "true_mean_mm": m, **{f"true_mm_{i + 1}": float(t) for i, t in enumerate(teeth)},
                     "true_label": label, "true_class": label.split("-")[0] if label.startswith("E") else "E1",
                     "true_px_per_mm": px_per_mm * float(np.exp(rng.normal(0, 0.05)))})
    return pd.DataFrame(rows)


def _neighbour(c: str, rng: np.random.Generator) -> str:
    i = CLASSES.index(c)
    j = i + rng.choice([-1, 1]) if 0 < i < 3 else (1 if i == 0 else 2)
    return CLASSES[int(j)]


def write_synthetic_forms(
    out_dir: Path, key: pd.DataFrame, truth: pd.DataFrame, seed: int = 42,
    p_agree: float = 0.8, mm_sd: float = 0.35, scale_cv: float = 0.06, messy: bool = True,
    per_expert: Optional[List[Dict[str, float]]] = None,
    empty_rows_expert2: Optional[set] = None,
) -> List[Path]:
    """Write ``Uzman_{1,2,3}_form.xlsx`` and return their paths."""
    rng = np.random.default_rng(seed)
    t = truth.set_index("image")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    empty_rows = {17, 58, 131} if empty_rows_expert2 is None else set(empty_rows_expert2)
    for e in range(1, 4):
        par = (per_expert or [{}] * 3)[e - 1]
        pa, sd, scv = par.get("p_agree", p_agree), par.get("mm_sd", mm_sd), par.get("scale_cv", scale_cv)
        bias = par.get("class_bias", 0)  # +1 shifts towards higher classes
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Degerlendirme"
        ws.append(["Uzman değerlendirme formu"])
        ws.append([f"Uzman {e}"])
        ws.append([None])
        ws.append(HEADER)
        # expert-specific shuffle of the presentation order
        order = key.sample(frac=1.0, random_state=seed + e).reset_index(drop=True) if False else key
        for _, k in order.iterrows():
            im = k["orijinal"]
            tr = t.loc[im]
            scale = tr["true_px_per_mm"] * float(np.exp(rng.normal(0, scv)))
            mm = [max(0.0, tr[f"true_mm_{i + 1}"] + rng.normal(0, sd)) if tr[f"true_mm_{i + 1}"] > 0 else 0.0 for i in range(6)]
            mm = [round(v, 2) for v in mm]
            cls = tr["true_class"] if rng.random() < pa else _neighbour(tr["true_class"], rng)
            if bias and rng.random() < abs(bias) * 0.3:
                cls = CLASSES[min(3, max(0, CLASSES.index(cls) + int(np.sign(bias))))]
            second = _neighbour(cls, rng) if rng.random() < 0.3 else None
            conf = int(rng.integers(2, 6))
            row = [int(k["sira"]), k["goruntu_id"], round(scale, 2), *mm, cls, second, conf, ""]
            if messy:
                r = rng.random()
                if r < 0.03:
                    row[11] = None                      # missing confidence
                elif r < 0.05:
                    row[2] = None                       # missing scale
                elif r < 0.07:
                    row[3 + int(rng.integers(0, 6))] = None  # tooth left blank (not visible)
                elif r < 0.08:
                    row[9] = "e2"                        # lower-case class
                if e == 2 and int(k["sira"]) in empty_rows:
                    row = [int(k["sira"]), k["goruntu_id"], None, None, None, None, None, None, None, None, None, None, None]  # skipped rows
            ws.append(row)
        p = out_dir / f"Uzman_{e}_form.xlsx"
        wb.save(p)
        paths.append(p)
    return paths


def synthetic_model_output(truth: pd.DataFrame, seed: int = 42, mm_sd: float = 0.45, px_per_mm: float = 16.84, splits: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """A ``per_image_results.csv``-like table: selected px/mm from the truth plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in truth.iterrows():
        regions = [max(0.0, r[f"true_mm_{i + 1}"] + rng.normal(0, mm_sd)) for i in range(6)]
        mean = float(np.mean(regions))
        rows.append({
            "uid": f"high/{r['image']}", "image": r["image"], "split": "holdout" if rng.random() < 0.4 else "dev",
            "ref_mm": r["true_mean_mm"], **{f"ref_mm_{i + 1}": r[f"true_mm_{i + 1}"] for i in range(6)},
            "ref_label": r["true_label"], "frame_ok": rng.random() > 0.15, "width": 2698, "height": 1799,
            "selected_method": "C_p25", "selected_px_per_mm": px_per_mm, "C_p25_px": mean * px_per_mm,
            "selected_mm": mean, "selected_label": label_for_mm(mean),
            **{f"selected_region_{i + 1}_mm": regions[i] for i in range(6)},
            "n_zeniths_found": 6 if rng.random() > 0.2 else 0, "qc_flags": "",
        })
    df = pd.DataFrame(rows)
    df["alignment_uncertain"] = df["n_zeniths_found"] < 6
    df.loc[df["alignment_uncertain"], "qc_flags"] = "zenith_detection_failed"
    return df
