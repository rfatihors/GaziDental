"""Dataset cleaning -> ``dataset_manifest.csv`` (spec §4.8 with the 12 Sep 2026 decisions).

Nothing is deleted or moved on disk. Rules, in order:

1. ``row_ambiguous`` images (two Excel rows carry the name; ``IMG_7366``) are dropped.
2. High-smile-line images without a reference measurement (the 66-image list) are dropped
   (default) or, with ``keep_unmeasured_high_in_train``, kept as ``train_only`` images that
   never enter valid/test nor any analysis.
3. Same-patient pairs: one image per patient. If a member is already dropped the pair is
   resolved. Otherwise, for pairs touching the high group the image in the expert set
   (= the probed, measured photograph) is kept; for other pairs the member with age/sex
   on the low/normal sheets is kept; ties fall back to ``image_a``.
4. ``patient_id`` = ``uid`` of the kept image; dropped twins inherit it. (Not the normalised
   key: four different photographs in ``normal`` share a key and differ only by extension.)
Images are identified by ``uid`` = ``group/stem`` because iPhone numbers repeat across groups.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DROP_ROW_AMBIGUOUS = "row_ambiguous"
DROP_NO_REFERENCE = "no_reference_measurement"
DROP_TWIN_HELD_OUT = "unmeasured_twin_of_held_out"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _has_demo(row: pd.Series) -> bool:
    age = row.get("age")
    sex = row.get("sex")
    return (age is not None and not (isinstance(age, float) and math.isnan(age))) or (sex is not None and not (isinstance(sex, float) and math.isnan(sex)))


def build_manifest(
    inventory: pd.DataFrame,
    high_matches: pd.DataFrame,
    demo_low: pd.DataFrame,
    demo_normal: pd.DataFrame,
    pairs: pd.DataFrame,
    unmeasured_high: pd.DataFrame,
    expert_set: pd.DataFrame,
    dataset_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (manifest, summary). See module docstring for the rules."""
    m = inventory.copy()
    if "uid" not in m.columns:
        m["uid"] = m["group"] + "/" + m["image"]
    m = m.set_index("uid", drop=False)
    stem_index: Dict[str, List[str]] = {}
    for uid, stem in zip(m["uid"], m["image"]):
        stem_index.setdefault(stem, []).append(uid)

    def to_uid(stem: str, group: Optional[str] = None) -> Optional[str]:
        """Resolve an input-CSV stem to a uid (group given, or unique across groups)."""
        if isinstance(group, str) and group:
            uid = f"{group}/{stem}"
            return uid if uid in m.index else None
        cands = stem_index.get(str(stem), [])
        if len(cands) > 1:
            raise ValueError(f"stem {stem} is ambiguous across groups; a group column is required")
        return cands[0] if cands else None
    m["orig_split"] = m["split"]
    m["keep"] = True
    m["drop_reason"] = ""
    m["split_constraint"] = ""
    m["twin_of"] = ""
    m["has_reference_measurement"] = False
    m["reference_mean_mm"] = math.nan
    m["reference_label"] = None
    m["match_kind"] = "not_applicable"
    m["excel_row"] = pd.array([None] * len(m), dtype="Int64")
    m["age"] = math.nan
    m["sex"] = None
    m["age_source"] = ""
    m["source_sheet"] = ""
    m["in_expert_set"] = False

    # --- reference measurements and demographics for the high group
    hm = high_matches.copy()
    if "uid" not in hm.columns:
        hm["uid"] = hm.get("group", "high") + "/" + hm["image"] if "group" in hm.columns else "high/" + hm["image"]
    hm = hm.set_index("uid")
    for image, r in hm.iterrows():
        if image not in m.index:
            continue
        m.at[image, "match_kind"] = r["match_kind"]
        if r["match_kind"] in ("exact", "dash_base_fallback"):
            m.at[image, "has_reference_measurement"] = not (isinstance(r["reference_mean_mm"], float) and math.isnan(r["reference_mean_mm"]))
            m.at[image, "reference_mean_mm"] = r["reference_mean_mm"]
            m.at[image, "reference_label"] = r.get("reference_label")
            m.at[image, "excel_row"] = int(r["excel_row"]) if r["excel_row"] is not None and not (isinstance(r["excel_row"], float) and math.isnan(r["excel_row"])) else None
            m.at[image, "source_sheet"] = "high"
            age = r.get("age")
            if age is not None and not (isinstance(age, float) and math.isnan(age)):
                m.at[image, "age"] = float(age)
                m.at[image, "age_source"] = r.get("age_source") or "sheet"
            sex = r.get("sex")
            if isinstance(sex, str):
                m.at[image, "sex"] = sex
    for demo, sheet in ((demo_low, "low"), (demo_normal, "normal")):
        if demo is None or demo.empty:
            continue
        d = demo.drop_duplicates("key").set_index("key")
        for image, key in m["key"].items():
            if key in d.index and m.at[image, "group"] == sheet:
                r = d.loc[key]
                m.at[image, "source_sheet"] = sheet
                if r.get("age") is not None and not (isinstance(r.get("age"), float) and math.isnan(r.get("age"))):
                    m.at[image, "age"] = float(r["age"])
                    m.at[image, "age_source"] = "sheet"
                if isinstance(r.get("sex"), str):
                    m.at[image, "sex"] = r["sex"]
    # age from the file-name prefix when the sheet gave nothing
    for image, r in m.iterrows():
        if (isinstance(r["age"], float) and math.isnan(r["age"])) and r["age_prefix"] is not None and not (isinstance(r["age_prefix"], float) and math.isnan(r["age_prefix"])):
            m.at[image, "age"] = float(r["age_prefix"])
            m.at[image, "age_source"] = "prefix"

    expert_images = {u for u in (to_uid(str(i), g) for i, g in zip(expert_set["image"], expert_set["group"] if "group" in expert_set.columns else ["high"] * len(expert_set))) if u}
    m["in_expert_set"] = m["uid"].isin(expert_images)

    # --- rule 1: row-ambiguous
    excluded = set(dataset_cfg.get("row_ambiguous_exclude", []))
    for image, r in m.iterrows():
        if r["match_kind"] == "row_ambiguous" or r["base"] in {e.lower().replace("_", "") for e in excluded}:
            m.at[image, "keep"] = False
            m.at[image, "drop_reason"] = DROP_ROW_AMBIGUOUS

    # --- pairs -> connected components
    uf = _UnionFind()
    pair_order: Dict[str, int] = {}
    for i, p in pairs.iterrows():
        a = to_uid(str(p["image_a"]), p.get("group_a"))
        b = to_uid(str(p["image_b"]), p.get("group_b"))
        if a is not None and b is not None:
            uf.union(a, b)
            pair_order.setdefault(a, i)
    comp: Dict[str, List[str]] = {}
    for image in m.index:
        if image in uf.parent:
            comp.setdefault(uf.find(image), []).append(image)

    # --- rule 2: unmeasured high
    keep_train_only = bool(dataset_cfg.get("keep_unmeasured_high_in_train", False))
    unmeasured = {u for u in (to_uid(str(i), g) for i, g in zip(unmeasured_high["image"], unmeasured_high["group"] if "group" in unmeasured_high.columns else ["high"] * len(unmeasured_high))) if u}
    for image in unmeasured:
        if not m.at[image, "keep"]:
            continue
        if keep_train_only:
            m.at[image, "split_constraint"] = "train_only"
        else:
            m.at[image, "keep"] = False
            m.at[image, "drop_reason"] = DROP_NO_REFERENCE

    # --- rule 3: one image per component
    pair_decisions = []
    for root, members in comp.items():
        alive = [i for i in members if m.at[i, "keep"] and m.at[i, "split_constraint"] != "train_only"]
        train_only_members = [i for i in members if m.at[i, "keep"] and m.at[i, "split_constraint"] == "train_only"]
        if not alive:
            # only train-only images alive: keep the first as the patient anchor
            kept = train_only_members[0] if train_only_members else None
        elif len(alive) == 1:
            kept = alive[0]
        else:
            touches_high = any(m.at[i, "group"] == "high" for i in alive)
            if touches_high:
                cands = [i for i in alive if m.at[i, "in_expert_set"]]
                rule = "expert_set"
            else:
                cands = [i for i in alive if _has_demo(m.loc[i])]
                rule = "demographics"
            if len(cands) != 1:
                cands = sorted(alive, key=lambda i: (pair_order.get(i, 10**9), i))[:1]
                rule += "->image_a"
            kept = cands[0]
            for i in alive:
                if i != kept:
                    m.at[i, "keep"] = False
                    m.at[i, "drop_reason"] = f"duplicate_of:{m.at[kept, 'image']}"
            pair_decisions.append({"kept": kept, "dropped": [i for i in alive if i != kept], "rule": rule})
        if kept is None:
            continue
        pid = m.at[kept, "uid"]
        for i in members:
            m.at[i, "patient_id"] = pid
            if i != kept:
                m.at[i, "twin_of"] = m.at[kept, "image"]
        # train-only twin of a held-out image cannot be decided here; recorded for the splitter
    m["patient_id"] = m.apply(lambda r: r["patient_id"] if isinstance(r.get("patient_id"), str) and r["patient_id"] else r["uid"], axis=1)

    # expert-set cross-check: kept high images with a reference must equal the expert set
    kept_high_ref = set(m[(m["keep"]) & (m["group"] == "high") & (m["has_reference_measurement"]) & (m["split_constraint"] != "train_only")]["uid"])
    mismatch = sorted((kept_high_ref ^ expert_images))

    summary = {
        "start": m.groupby("group").size().to_dict(),
        "kept": m[m["keep"]].groupby("group").size().to_dict(),
        "dropped": m[~m["keep"]].groupby(["group", "drop_reason"]).size().to_dict(),
        "train_only": m[m["split_constraint"] == "train_only"].groupby("group").size().to_dict(),
        "kept_with_reference": m[(m["keep"]) & (m["has_reference_measurement"])].groupby("group").size().to_dict(),
        "pair_components": len(comp),
        "pair_decisions": pair_decisions,
        "expert_set_mismatch": mismatch,
        "n_patients": int(m[m["keep"]]["patient_id"].nunique()),
    }
    cols = [
        "uid", "image", "group", "orig_split", "keep", "drop_reason", "patient_id", "twin_of", "split_constraint",
        "has_reference_measurement", "reference_mean_mm", "reference_label", "match_kind", "excel_row",
        "in_expert_set", "age", "sex", "age_source", "source_sheet", "width", "height", "frame_ok",
        "n_gingiva", "n_lip", "key", "file_name", "path",
    ]
    return m[cols].reset_index(drop=True), summary
