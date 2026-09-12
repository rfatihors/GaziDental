"""Stratified, patient-level train/valid/test split with a fixed test set, plus CV folds.

* After cleaning, image == patient except for ``train_only`` twins, which follow their
  kept twin and force it into ``train``.
* High images are stratified by their reference class label (E1 / E1-E2 / E2-E3 / E3),
  low and normal by group only.
* All keys in the result are ``uid`` (``group/stem``).
* ``cv_folds`` assigns every kept high image with a reference measurement to one of
  ``n`` folds (stratified by label) for out-of-fold prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _allocate(n: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder rounding of ``n`` into train/valid/test."""
    raw = {k: n * float(v) for k, v in ratios.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rest]:
        base[k] += 1
    return base


def _split_group(images: List[str], strata: List[str], ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    n = len(images)
    counts = _allocate(n, ratios)
    strat = pd.Series(strata, index=images)
    use_strat = strat.nunique() > 1 and strat.value_counts().min() >= 3
    rest, test = train_test_split(images, test_size=counts["test"], random_state=seed, stratify=strat.values if use_strat else None)
    strat_rest = strat.loc[rest]
    use_strat2 = use_strat and strat_rest.value_counts().min() >= 2
    train, valid = train_test_split(rest, test_size=counts["valid"], random_state=seed, stratify=strat_rest.values if use_strat2 else None)
    out = {i: "train" for i in train}
    out.update({i: "valid" for i in valid})
    out.update({i: "test" for i in test})
    return out


def make_splits(manifest: pd.DataFrame, dataset_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    m = manifest[manifest["keep"]].copy()
    if "uid" not in m.columns:
        m["uid"] = m["group"] + "/" + m["image"]
    m["reference_label"] = m["reference_label"].fillna("none")
    free = m[m["split_constraint"] != "train_only"]
    train_only = m[m["split_constraint"] == "train_only"]
    assign: Dict[str, str] = {}
    for group, ratios in dataset_cfg["split_ratios"].items():
        g = free[free["group"] == group].sort_values("uid")
        if g.empty:
            continue
        strata = list(g["reference_label"]) if group == "high" else ["all"] * len(g)
        assign.update(_split_group(list(g["uid"]), strata, ratios, seed))
    # train-only images: their patient (kept twin) must be in train
    forced: List[str] = []
    for _, r in train_only.iterrows():
        assign[r["uid"]] = "train"
        twins = free[(free["patient_id"] == r["patient_id"]) & (free["uid"] != r["uid"])]["uid"]
        for t in twins:
            if assign.get(t) != "train":
                assign[t] = "train"
                forced.append(t)
    # patient-level sanity: a patient must not straddle splits
    pid = m.set_index("uid")["patient_id"]
    per_patient = pd.Series(assign).groupby(pid.reindex(list(assign)).values).nunique()
    if (per_patient > 1).any():
        raise RuntimeError(f"patients straddle splits: {per_patient[per_patient > 1].index.tolist()[:5]}")

    # CV folds for measured high images
    measured = m[(m["group"] == "high") & (m["has_reference_measurement"]) & (m["split_constraint"] != "train_only")].sort_values("uid")
    n_folds = int(dataset_cfg.get("cv_folds", 5))
    folds: Dict[str, int] = {}
    if len(measured) >= n_folds:
        labels = measured["reference_label"].values
        if pd.Series(labels).value_counts().min() < n_folds:
            labels = np.array(["all"] * len(measured))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (_, idx) in enumerate(skf.split(np.zeros(len(measured)), labels)):
            for i in idx:
                folds[measured["uid"].iloc[i]] = f

    by_split = {s: sorted(i for i, v in assign.items() if v == s) for s in ("train", "valid", "test")}
    counts = {g: {s: int(((free["group"] == g) & (free["uid"].map(assign) == s)).sum()) for s in ("train", "valid", "test")} for g in dataset_cfg["split_ratios"]}
    return {
        "seed": seed,
        "ratios": dataset_cfg["split_ratios"],
        "stratification": {"high": "reference_label", "low": "group", "normal": "group"},
        "counts": counts,
        "images": dict(sorted(assign.items())),
        "by_split": by_split,
        "train_only": sorted(train_only["uid"]),
        "forced_to_train_by_twin": sorted(forced),
        "cv_folds": {"n_folds": n_folds, "stratification": "reference_label", "assignments": dict(sorted(folds.items()))},
    }
