"""Expert-agreement analyses (docs/Istatistik_analiz_plani.md §2–3, protocol §5).

Inputs: three expert forms joined to the anonymisation key, and the model table
(``per_image_results.csv`` of Stage 3 or 6: selected-method mm, region mm, QC flags).

Scoring of the model's class against an expert class:
* ``strict``   — the model's first candidate only (``E1-E2`` counts as E1);
* ``lenient``  — agreement if the expert's primary class is in the model's candidate set;
* ``lenient2`` — as lenient, also accepting the expert's second candidate (reported only).
For kappa the model label under lenient scoring is set to the expert's class when it is
a candidate, otherwise to the strict label.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gsv4.eval.agreement import bland_altman, icc_long, icc_two_raters, mixed_model_diff
from gsv4.eval.kappa import CLASSES, bootstrap_ci, cohen_kappa, fleiss_kappa, kappa_bundle, per_class_metrics
from gsv4.io.forms import TEETH
from gsv4.rules.thresholds import label_for_mm, matching_classes

BOUNDARIES = [3.0, 4.0, 6.0, 8.0]


# ----------------------------------------------------------------------------- model side
def _label_columns(m: pd.DataFrame, scale: str) -> None:
    m[f"model_label_{scale}"] = m[f"model_mm_{scale}"].map(label_for_mm)
    m[f"model_candidates_{scale}"] = m[f"model_mm_{scale}"].map(lambda v: matching_classes(v))
    m[f"model_strict_{scale}"] = m[f"model_candidates_{scale}"].map(lambda c: c[0] if c else None)


def model_table(per_image: pd.DataFrame, expert_scale: Optional[pd.Series] = None, offset_px: float = 0.0) -> pd.DataFrame:
    """Model class/mm per image from the selected method under four "scales":
    ``global`` (primary; uncorrected, global px/mm), ``global_corrected`` (secondary; the
    pixel offset of config.yaml applied at the global scale), and when ``expert_scale``
    (image -> px/mm) is given ``expert`` / ``expert_corrected`` with the experts' mean
    per-image scale. The corrected columns are ``model_mm_<scale>`` with a
    ``model_clipped_<scale>`` flag; region columns ``model_region_<i>_mm_<scale>``."""
    from gsv4.measure.calibration import corrected_mm

    m = per_image.copy()
    method = m["selected_method"].iloc[0]
    px_col = f"{method}_px"
    k_global = m["selected_px_per_mm"].astype(float) if "selected_px_per_mm" in m.columns else np.nan
    m["model_mm_global"] = m["selected_mm"]
    m["model_offset_px"] = float(offset_px)
    if px_col in m.columns:
        c = corrected_mm(m[px_col], k_global, offset_px)
        m["model_mm_global_corrected"] = c["mm"]
        m["model_clipped_global_corrected"] = c["clipped"]
        for i in range(1, 7):
            rc = f"{method}_region_{i}_px"
            if rc in m.columns:
                m[f"model_region_{i}_mm_global"] = m[rc] / k_global
                m[f"model_region_{i}_mm_global_corrected"] = corrected_mm(m[rc], k_global, offset_px)["mm"]
    else:
        m["model_mm_global_corrected"] = np.nan
        m["model_clipped_global_corrected"] = False
    if expert_scale is not None and px_col in m.columns:
        m["expert_px_per_mm"] = m["image"].map(expert_scale)
        m["model_mm_expert"] = m[px_col] / m["expert_px_per_mm"]
        c = corrected_mm(m[px_col], m["expert_px_per_mm"], offset_px)
        m["model_mm_expert_corrected"] = c["mm"]
        m["model_clipped_expert_corrected"] = c["clipped"]
        for i in range(1, 7):
            rc = f"{method}_region_{i}_px"
            if rc in m.columns:
                m[f"model_region_{i}_mm_expert"] = m[rc] / m["expert_px_per_mm"]
                m[f"model_region_{i}_mm_expert_corrected"] = corrected_mm(m[rc], m["expert_px_per_mm"], offset_px)["mm"]
    else:
        m["expert_px_per_mm"] = np.nan
        m["model_mm_expert"] = np.nan
        m["model_mm_expert_corrected"] = np.nan
        m["model_clipped_expert_corrected"] = False
    for scale in ("global", "global_corrected", "expert", "expert_corrected"):
        _label_columns(m, scale)
    m["dist_to_threshold_mm"] = m["model_mm_global"].map(lambda v: min(abs(v - b) for b in BOUNDARIES) if pd.notna(v) else np.nan)
    if "alignment_uncertain" not in m.columns:
        m["alignment_uncertain"] = m.get("qc_flags", pd.Series("", index=m.index)).fillna("").str.contains("zenith_detection_failed")
    return m


# ----------------------------------------------------------------------------- expert side
def reference_standard(first: Dict[int, pd.DataFrame], consensus: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Majority of the three primary classes; all-different -> ``consensus_pending`` unless a
    blinded consensus table (image, class) is supplied."""
    frames = []
    for e, df in first.items():
        frames.append(df.set_index("image")[["class_primary", "class_secondary", "confidence"]].rename(
            columns={"class_primary": f"class_e{e}", "class_secondary": f"second_e{e}", "confidence": f"conf_e{e}"}))
    ref = pd.concat(frames, axis=1)
    cols = [c for c in ref.columns if c.startswith("class_e")]

    def majority(row):
        votes = [v for v in row[cols] if isinstance(v, str)]
        if len(votes) < 2:
            return None, "insufficient_votes"
        vc = pd.Series(votes).value_counts()
        if vc.iloc[0] >= 2:
            return vc.index[0], "unanimous" if vc.iloc[0] == len(votes) else "majority"
        return None, "consensus_pending"

    res = ref.apply(majority, axis=1, result_type="expand")
    ref["reference_class"], ref["reference_kind"] = res[0], res[1]
    ref["n_votes"] = ref[cols].notna().sum(axis=1)
    if consensus is not None and len(consensus):
        c = consensus.set_index("image")["class"]
        pend = ref["reference_kind"] == "consensus_pending"
        ref.loc[pend & ref.index.isin(c.index), "reference_class"] = ref.index[pend & ref.index.isin(c.index)].map(c)
        ref.loc[pend & ref.index.isin(c.index), "reference_kind"] = "consensus"
    return ref.reset_index()


def score_labels(model: pd.DataFrame, expert_class: pd.Series, expert_second: Optional[pd.Series], mode: str, scale: str = "global") -> pd.DataFrame:
    """Return DataFrame(image, ref, pred) for kappa under strict / lenient / lenient2 scoring."""
    m = model.set_index("image")
    rows = []
    for image, ref in expert_class.items():
        if image not in m.index or not isinstance(ref, str):
            continue
        cands = m.at[image, f"model_candidates_{scale}"]
        strict = m.at[image, f"model_strict_{scale}"]
        if strict is None or (isinstance(strict, float) and math.isnan(strict)) or not cands:
            continue  # model unclassified / no visible gingiva
        pred = strict
        if mode == "lenient" and ref in cands:
            pred = ref
        elif mode == "lenient2":
            second = expert_second.get(image) if expert_second is not None else None
            if ref in cands:
                pred = ref
            elif isinstance(second, str) and second in cands:
                pred = ref  # counted as agreement: expert's second candidate matches
        rows.append({"image": image, "ref": ref, "pred": pred, "strict": strict, "n_candidates": len(cands)})
    return pd.DataFrame(rows)


def class_agreement(model: pd.DataFrame, reference: pd.DataFrame, subsets: Dict[str, Sequence[str]], n_boot: int, seed: int,
                    scales: Sequence[str] = ("global",)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Model vs reference class under every scoring mode, scale and subset; plus per-class table."""
    ref = reference.set_index("image")
    rows, per_class = [], []
    for scale in scales:
        for subset, uids in subsets.items():
            imgs = [i for i in uids if i in ref.index and isinstance(ref.at[i, "reference_class"], str)]
            n_pending = int(sum(1 for i in uids if i in ref.index and ref.at[i, "reference_kind"] == "consensus_pending"))
            for mode in ("strict", "lenient", "lenient2"):
                sc = score_labels(model, ref.loc[imgs, "reference_class"], None, mode, scale)
                if sc.empty:
                    continue
                b = kappa_bundle(sc["ref"], sc["pred"], CLASSES, n_boot=n_boot, seed=seed)
                rows.append({"scale": scale, "subset": subset, "scoring": mode, "n_consensus_pending_excluded": n_pending,
                             "n_model_unclassified": len(imgs) - len(sc), **b})
                if mode == "strict":
                    for pc in per_class_metrics(sc["ref"], sc["pred"], CLASSES):
                        per_class.append({"scale": scale, "subset": subset, **pc})
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def strata_by_threshold(model: pd.DataFrame, reference: pd.DataFrame, uids: Sequence[str], near_mm: float = 0.5) -> pd.DataFrame:
    ref = reference.set_index("image")
    imgs = [i for i in uids if i in ref.index and isinstance(ref.at[i, "reference_class"], str)]
    sc = score_labels(model, ref.loc[imgs, "reference_class"], None, "strict").set_index("image")
    sc["dist"] = model.set_index("image").loc[sc.index, "dist_to_threshold_mm"]
    sc["stratum"] = np.where(sc["dist"] < near_mm, f"within {near_mm} mm of a threshold", f">= {near_mm} mm from thresholds")
    rows = []
    for s, part in sc.groupby("stratum"):
        rows.append({"stratum": s, "n": len(part), "observed_agreement": float((part["ref"] == part["pred"]).mean()),
                     "kappa_linear": cohen_kappa(part["ref"], part["pred"], CLASSES, "linear"),
                     "n_disagreements": int((part["ref"] != part["pred"]).sum())})
    return pd.DataFrame(rows)


def intra_expert(pairs: Dict[int, pd.DataFrame], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for e, p in pairs.items():
        ok = p.dropna(subset=["class_primary_first", "class_primary_repeat"])
        rec: Dict[str, Any] = {"expert": e, "n_repeats": len(p), "n_class_pairs": len(ok)}
        if len(ok) >= 2:
            k = cohen_kappa(ok["class_primary_first"], ok["class_primary_repeat"], CLASSES, "linear")
            rec["kappa_linear"] = k
            a, b = ok["class_primary_first"].to_numpy(), ok["class_primary_repeat"].to_numpy()
            ci = bootstrap_ci(lambda idx: cohen_kappa(a[idx], b[idx], CLASSES, "linear"), len(ok), n_boot, seed)
            rec["kappa_linear_ci_low"], rec["kappa_linear_ci_high"] = ci["ci_low"], ci["ci_high"]
            rec["kappa_unweighted"] = cohen_kappa(ok["class_primary_first"], ok["class_primary_repeat"], CLASSES)
            rec["observed_agreement"] = float((ok["class_primary_first"] == ok["class_primary_repeat"]).mean())
        okm = p.dropna(subset=["mean_mm_first", "mean_mm_repeat"])
        rec["n_mm_pairs"] = len(okm)
        if len(okm) >= 3:
            icc = icc_two_raters(okm["mean_mm_first"], okm["mean_mm_repeat"])
            rec.update({"icc2_1_image": icc["icc2_1"], "icc2_1_image_ci_low": icc["icc2_1_ci_low"], "icc2_1_image_ci_high": icc["icc2_1_ci_high"]})
            ba = bland_altman(okm["mean_mm_repeat"], okm["mean_mm_first"])
            rec.update({"mm_bias": ba["bias"], "mm_sd": ba["sd"], "mm_loa_low": ba["loa_low"], "mm_loa_high": ba["loa_high"]})
            long = []
            for t in TEETH:
                for _, r in p.iterrows():
                    long.append({"target": f"{r['image']}#{t}", "session": 1, "mm": r[f"mm_{t}_first"]})
                    long.append({"target": f"{r['image']}#{t}", "session": 2, "mm": r[f"mm_{t}_repeat"]})
            tl = pd.DataFrame(long).dropna()
            if tl["target"].nunique() >= 3:
                icc_t = icc_long(tl, "target", "session", "mm")
                rec.update({"icc2_1_tooth_naive": icc_t["icc2_1"], "n_tooth_pairs": icc_t["n_targets"]})
        oks = p.dropna(subset=["scale_px_per_mm_first", "scale_px_per_mm_repeat"])
        if len(oks) >= 3:
            rec["scale_icc2_1"] = icc_two_raters(oks["scale_px_per_mm_first"], oks["scale_px_per_mm_repeat"])["icc2_1"]
            rec["scale_cv_repeat"] = float(np.mean(np.abs(oks["scale_px_per_mm_first"] - oks["scale_px_per_mm_repeat"]) / ((oks["scale_px_per_mm_first"] + oks["scale_px_per_mm_repeat"]) / 2)))
        rows.append(rec)
    return pd.DataFrame(rows)


def inter_expert(first: Dict[int, pd.DataFrame], clinical_ref: Optional[pd.Series], uids: Sequence[str], n_boot: int, seed: int) -> Dict[str, Any]:
    """Fleiss kappa (class) and ICC(2,1)/(2,k) (image-mean mm) among experts, with and
    without the clinical reference observer; expert scale agreement."""
    cls = pd.DataFrame({e: df.set_index("image")["class_primary"] for e, df in first.items()}).loc[[u for u in uids if all(u in df["image"].values for df in first.values())]]
    complete = cls.dropna()
    ratings = complete.values.tolist()
    fk = fleiss_kappa(ratings, CLASSES) if len(ratings) else math.nan
    arr = np.asarray(ratings, dtype=object)
    ci = bootstrap_ci(lambda idx: fleiss_kappa(arr[idx].tolist(), CLASSES), len(ratings), n_boot, seed) if len(ratings) else {"ci_low": math.nan, "ci_high": math.nan}
    pair_k = {f"e{a}_e{b}": cohen_kappa(complete[a], complete[b], CLASSES, "linear") for a in first for b in first if a < b}
    out: Dict[str, Any] = {"n_class": len(complete), "fleiss_kappa": fk, "fleiss_kappa_ci_low": ci["ci_low"], "fleiss_kappa_ci_high": ci["ci_high"], "pairwise_kappa_linear": pair_k}
    # mm, image level
    mm = pd.DataFrame({f"expert_{e}": df.set_index("image")["mean_mm"] for e, df in first.items()}).loc[complete.index.union(cls.index)]
    long = mm.dropna().reset_index().melt(id_vars="image", var_name="rater", value_name="mm")
    if long["image"].nunique() >= 3:
        icc = icc_long(long, "image", "rater", "mm")
        out.update({"mm_n": icc["n_targets"], "mm_icc2_1": icc["icc2_1"], "mm_icc2_1_ci_low": icc["icc2_1_ci_low"], "mm_icc2_1_ci_high": icc["icc2_1_ci_high"],
                    "mm_icc2_k": icc["icc2_k"], "mm_icc2_k_ci_low": icc["icc2_k_ci_low"], "mm_icc2_k_ci_high": icc["icc2_k_ci_high"]})
    if clinical_ref is not None:
        mm4 = mm.copy()
        mm4["clinical_reference"] = mm4.index.map(clinical_ref)
        long4 = mm4.dropna().reset_index().melt(id_vars="image", var_name="rater", value_name="mm")
        if long4["image"].nunique() >= 3:
            icc4 = icc_long(long4, "image", "rater", "mm")
            out.update({"mm4_n": icc4["n_targets"], "mm4_icc2_1": icc4["icc2_1"], "mm4_icc2_1_ci_low": icc4["icc2_1_ci_low"], "mm4_icc2_1_ci_high": icc4["icc2_1_ci_high"],
                        "mm4_icc2_k": icc4["icc2_k"], "mm4_icc2_k_ci_low": icc4["icc2_k_ci_low"], "mm4_icc2_k_ci_high": icc4["icc2_k_ci_high"]})
    # scale agreement
    sc = pd.DataFrame({f"expert_{e}": df.set_index("image")["scale_px_per_mm"] for e, df in first.items()})
    longs = sc.dropna().reset_index().melt(id_vars="image", var_name="rater", value_name="scale")
    if longs["image"].nunique() >= 3:
        iccs = icc_long(longs, "image", "rater", "scale")
        cv = (sc.std(axis=1) / sc.mean(axis=1)).dropna()
        out.update({"scale_n": iccs["n_targets"], "scale_icc2_1": iccs["icc2_1"], "scale_icc2_1_ci_low": iccs["icc2_1_ci_low"], "scale_icc2_1_ci_high": iccs["icc2_1_ci_high"],
                    "scale_mean": float(sc.mean(axis=1).mean()), "scale_sd_between_images": float(sc.mean(axis=1).std()),
                    "scale_cv_within_image_median": float(cv.median()), "scale_cv_within_image_mean": float(cv.mean())})
    out["expert_mean_mm"] = mm.mean(axis=1)
    out["expert_scale_mean"] = sc.mean(axis=1)
    return out


def mm_agreement(model: pd.DataFrame, first: Dict[int, pd.DataFrame], expert_mean_mm: pd.Series, clinical_ref: Optional[pd.Series],
                 uids: Sequence[str], scale: str = "global") -> pd.DataFrame:
    """Image-level ICC(2,1) and Bland–Altman of the model against each expert, the expert mean
    and the clinical reference, for one model scale."""
    m = model.set_index("image")
    col = f"model_mm_{scale}"
    comps = {f"expert_{e}": df.set_index("image")["mean_mm"] for e, df in first.items()}
    comps["expert_mean"] = expert_mean_mm
    if clinical_ref is not None:
        comps["clinical_reference"] = clinical_ref
    rows = []
    for name, s in comps.items():
        d = pd.DataFrame({"model": m[col], "other": s}).loc[[u for u in uids if u in m.index]].dropna()
        if len(d) < 3:
            rows.append({"scale": scale, "comparator": name, "n": len(d)})
            continue
        icc = icc_two_raters(d["model"], d["other"])
        ba = bland_altman(d["model"], d["other"])
        rows.append({"scale": scale, "comparator": name, "n": len(d), "mae": float(np.abs(d["model"] - d["other"]).mean()),
                     "rmse": float(np.sqrt(np.mean((d["model"] - d["other"]) ** 2))),
                     "icc2_1": icc["icc2_1"], "icc2_1_ci_low": icc["icc2_1_ci_low"], "icc2_1_ci_high": icc["icc2_1_ci_high"],
                     "bias": ba["bias"], "bias_ci_low": ba["bias_ci_low"], "bias_ci_high": ba["bias_ci_high"],
                     "loa_low": ba["loa_low"], "loa_high": ba["loa_high"], "prop_slope": ba["prop_slope"], "prop_p": ba["prop_p"]})
    return pd.DataFrame(rows)


def tooth_long(model: pd.DataFrame, first: Dict[int, pd.DataFrame], uids: Sequence[str], scale: str = "global") -> pd.DataFrame:
    """Long table: image (patient), tooth, model region mm, expert mean mm per tooth, diff, alignment flag."""
    m = model.set_index("image")
    rows = []
    for u in uids:
        if u not in m.index:
            continue
        for i, t in enumerate(TEETH, start=1):
            mc = f"model_region_{i}_mm_{scale}" if f"model_region_{i}_mm_{scale}" in m.columns else (f"selected_region_{i}_mm" if scale == "global" else None)
            if mc is None:
                continue
            if mc not in m.columns:
                continue
            vals = [df.set_index("image").at[u, f"mm_{t}"] for df in first.values() if u in df["image"].values]
            vals = [v for v in vals if pd.notna(v)]
            if not vals or pd.isna(m.at[u, mc]):
                continue
            rows.append({"patient": u, "tooth": t, "tooth_index": i, "model_mm": float(m.at[u, mc]), "expert_mm": float(np.mean(vals)),
                         "n_experts": len(vals), "alignment_uncertain": bool(m.at[u, "alignment_uncertain"])})
    d = pd.DataFrame(rows)
    if len(d):
        d["diff"] = d["model_mm"] - d["expert_mm"]
    return d


def tooth_mixed_models(long: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    if long.empty or long["patient"].nunique() < 5:
        return out
    for rhs in ("1", "C(tooth)", "C(tooth) + alignment_uncertain"):
        try:
            r = mixed_model_diff(long, rhs, group="patient", diff="diff")
            out.append(r)
        except Exception as exc:  # noqa: BLE001 - report, never crash the pipeline
            out.append({"formula": f"diff ~ {rhs} + (1 | patient)", "error": str(exc)})
    return out
