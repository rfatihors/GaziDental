"""Categorical agreement: Cohen's kappa (unweighted / linear), Fleiss' kappa, PABAK,
observed agreement, per-class sensitivity/specificity with Wilson CIs, and bootstrap
CIs. Every function tolerates classes with zero cases (E4 is absent in this dataset).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

CLASSES = ["E1", "E2", "E3", "E4"]


def _index(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    lut = {c: i for i, c in enumerate(classes)}
    return np.asarray([lut[x] for x in labels], dtype=int)


def confusion(a: Sequence[str], b: Sequence[str], classes: Sequence[str] = CLASSES) -> np.ndarray:
    k = len(classes)
    m = np.zeros((k, k), dtype=float)
    for i, j in zip(_index(a, classes), _index(b, classes)):
        m[i, j] += 1
    return m


def cohen_kappa(a: Sequence[str], b: Sequence[str], classes: Sequence[str] = CLASSES, weights: Optional[str] = None) -> float:
    """Cohen's kappa; ``weights=None`` unweighted, ``'linear'`` or ``'quadratic'`` on the class order."""
    m = confusion(a, b, classes)
    n = m.sum()
    if n == 0:
        return math.nan
    k = len(classes)
    idx = np.arange(k)
    if weights is None:
        w = (idx[:, None] != idx[None, :]).astype(float)
    elif weights == "linear":
        w = np.abs(idx[:, None] - idx[None, :]) / (k - 1)
    elif weights == "quadratic":
        w = ((idx[:, None] - idx[None, :]) / (k - 1)) ** 2
    else:
        raise ValueError(weights)
    p_obs = m / n
    expected = np.outer(m.sum(axis=1), m.sum(axis=0)) / (n * n)
    disagreement_obs = float((w * p_obs).sum())
    disagreement_exp = float((w * expected).sum())
    if disagreement_exp == 0:
        return math.nan  # both raters use a single class: kappa undefined
    return 1 - disagreement_obs / disagreement_exp


def observed_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = list(a), list(b)
    return float(np.mean([x == y for x, y in zip(a, b)])) if a else math.nan


def pabak(a: Sequence[str], b: Sequence[str], classes: Sequence[str] = CLASSES) -> float:
    """Prevalence- and bias-adjusted kappa: k * p_o - 1 over k - 1 (Byrt 1993, generalised)."""
    k = len(classes)
    po = observed_agreement(a, b)
    return (k * po - 1) / (k - 1)


def fleiss_kappa(ratings: Sequence[Sequence[str]], classes: Sequence[str] = CLASSES) -> float:
    """Fleiss' kappa for n subjects each rated by the same number of raters.
    ``ratings[i]`` = the labels given to subject i."""
    counts = np.asarray([[list(r).count(c) for c in classes] for r in ratings], dtype=float)
    n, k = counts.shape
    if n == 0:
        return math.nan
    m = counts.sum(axis=1)
    if not np.all(m == m[0]) or m[0] < 2:
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of raters per subject")
    m = m[0]
    p_j = counts.sum(axis=0) / (n * m)
    p_i = (np.sum(counts * (counts - 1), axis=1)) / (m * (m - 1))
    p_bar = p_i.mean()
    p_e = float(np.sum(p_j**2))
    if p_e == 1:
        return math.nan
    return float((p_bar - p_e) / (1 - p_e))


def wilson_ci(x: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (math.nan, math.nan)
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_class_metrics(ref: Sequence[str], pred: Sequence[str], classes: Sequence[str] = CLASSES) -> List[Dict[str, float]]:
    """One-vs-rest sensitivity / specificity / PPV / agreement per class with counts and Wilson CIs."""
    ref, pred = list(ref), list(pred)
    out = []
    for c in classes:
        tp = sum(1 for r, p in zip(ref, pred) if r == c and p == c)
        fn = sum(1 for r, p in zip(ref, pred) if r == c and p != c)
        fp = sum(1 for r, p in zip(ref, pred) if r != c and p == c)
        tn = sum(1 for r, p in zip(ref, pred) if r != c and p != c)
        n_pos, n_neg, n_pred = tp + fn, fp + tn, tp + fp
        sens = tp / n_pos if n_pos else math.nan
        spec = tn / n_neg if n_neg else math.nan
        ppv = tp / n_pred if n_pred else math.nan
        out.append({
            "class": c, "n_reference": n_pos, "n_predicted": n_pred, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": sens, "sensitivity_ci_low": wilson_ci(tp, n_pos)[0], "sensitivity_ci_high": wilson_ci(tp, n_pos)[1],
            "specificity": spec, "specificity_ci_low": wilson_ci(tn, n_neg)[0], "specificity_ci_high": wilson_ci(tn, n_neg)[1],
            "ppv": ppv, "ppv_ci_low": wilson_ci(tp, n_pred)[0], "ppv_ci_high": wilson_ci(tp, n_pred)[1],
        })
    return out


def bootstrap_ci(stat, n: int, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap of ``stat(indices)`` over case resampling. ``stat`` receives an
    integer index array and returns a float (NaN results are dropped and counted)."""
    rng = np.random.default_rng(seed)
    vals = []
    n_nan = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat(idx)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            n_nan += 1
        else:
            vals.append(v)
    if not vals:
        return {"ci_low": math.nan, "ci_high": math.nan, "n_boot": n_boot, "n_undefined": n_nan}
    return {"ci_low": float(np.percentile(vals, 100 * alpha / 2)), "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_boot": n_boot, "n_undefined": n_nan}


def kappa_bundle(ref: Sequence[str], pred: Sequence[str], classes: Sequence[str] = CLASSES, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Linear-weighted (primary), unweighted, observed agreement and PABAK, each with a bootstrap CI."""
    ref = np.asarray(list(ref)); pred = np.asarray(list(pred))
    n = len(ref)
    out: Dict[str, float] = {"n": n}
    for name, fn in (
        ("kappa_linear", lambda r, p: cohen_kappa(r, p, classes, "linear")),
        ("kappa_unweighted", lambda r, p: cohen_kappa(r, p, classes, None)),
        ("observed_agreement", lambda r, p: observed_agreement(r, p)),
        ("pabak", lambda r, p: pabak(r, p, classes)),
    ):
        out[name] = fn(ref, pred) if n else math.nan
        ci = bootstrap_ci(lambda idx, fn=fn: fn(ref[idx], pred[idx]), n, n_boot, seed) if n else {"ci_low": math.nan, "ci_high": math.nan}
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = ci["ci_low"], ci["ci_high"]
    return out
