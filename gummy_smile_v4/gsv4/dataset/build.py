"""End-to-end data layer: inputs -> inventory -> matches -> manifest -> splits.

Used by ``scripts/build_manifest.py`` (Stage 1) and re-used by later stages that need
the reference table joined to the manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from gsv4.config import input_path, resolve
from gsv4.io.coco import coco_inventory
from gsv4.io.excel_parser import parse_summary, read_calibration, read_demo_sheet, read_high_sheet
from gsv4.io.naming import MatchResult, match_names
from gsv4.dataset.manifest import build_manifest
from gsv4.dataset.split import make_splits


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def match_group(inventory: pd.DataFrame, excel: pd.DataFrame, group: str) -> pd.DataFrame:
    """Tiered matching of one COCO group against the high sheet; one row per COCO image."""
    excel_keys: Dict[str, List[int]] = {}
    for i, k in enumerate(excel["key"]):
        excel_keys.setdefault(k, []).append(i)
    g = inventory[inventory["group"] == group]
    dup_keys = set(g.loc[g["key"].duplicated(keep=False), "key"])
    coco_keys = dict(zip(g.loc[~g["key"].isin(dup_keys), "key"], g.loc[~g["key"].isin(dup_keys), "image"]))
    res = match_names(coco_keys, excel_keys, all_coco_keys=set(inventory["key"]))
    for _, r in g[g["key"].isin(dup_keys)].iterrows():   # two different photos, same key (extension only)
        res[r["image"]] = MatchResult(r["image"], r["key"], "coco_key_collision")
    rows = []
    for image, r in res.items():
        rec: Dict[str, Any] = {
            "uid": f"{group}/{image}", "image": image, "group": group, "coco_key": r.coco_key, "match_kind": r.match_kind,
            "excel_key": r.excel_key, "excel_rows": ";".join(str(x) for x in r.excel_rows),
            "excel_row": None, "reference_mean_mm": float("nan"), "reference_label": None,
            "age": float("nan"), "sex": None, "age_source": None, "complete": None,
            "has_dash_zero": None, "has_ambiguous_100_999": None, "label_inconsistent_count": None,
        }
        if r.match_kind in ("exact", "dash_base_fallback"):
            e = excel.iloc[r.excel_rows[0]]
            rec.update({
                "excel_row": int(e["excel_row"]), "reference_mean_mm": float(e["mean_mm"]),
                "reference_label": e["reference_label"], "age": e["age"], "sex": e["sex"],
                "age_source": e["age_source"], "complete": bool(e["complete"]),
                "has_dash_zero": bool(e["has_dash_zero"]), "has_ambiguous_100_999": bool(e["has_ambiguous_100_999"]),
                "label_inconsistent_count": int(e["label_inconsistent_count"]),
            })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("image").reset_index(drop=True)


def build_all(cfg: Dict[str, Any]) -> Dict[str, Any]:
    coco_root = resolve(cfg, cfg["paths"]["coco_root"])
    inventory = coco_inventory(coco_root, cfg["coco"])

    xlsx = input_path(cfg, "measurements_xlsx")
    high = read_high_sheet(xlsx, cfg["excel"])
    low = read_demo_sheet(xlsx, cfg["excel"]["low_sheet"], cfg["excel"])
    normal = read_demo_sheet(xlsx, cfg["excel"]["normal_sheet"], cfg["excel"])
    calibration = read_calibration(input_path(cfg, "calibration_xlsx"), cfg["excel"])

    matches = {g: match_group(inventory, high, g) for g in cfg["coco"]["groups"]}

    pairs = read_csv(input_path(cfg, "same_patient_pairs_csv"))
    unmeasured = read_csv(input_path(cfg, "unmeasured_high_csv"))
    expert_set = read_csv(input_path(cfg, "expert_set_csv"))
    key = read_csv(input_path(cfg, "anonymisation_key_csv"))

    manifest, msummary = build_manifest(
        inventory, matches["high"], low, normal, pairs, unmeasured, expert_set, cfg["dataset"]
    )
    splits = make_splits(manifest, cfg["dataset"], int(cfg["seed"]))
    manifest["split"] = manifest["uid"].map(splits["images"]).fillna("")
    manifest["cv_fold"] = manifest["uid"].map(splits["cv_folds"]["assignments"]).astype("Int64")

    return {
        "inventory": inventory, "high": high, "low": low, "normal": normal, "calibration": calibration,
        "parse_summary": parse_summary(high), "matches": matches, "pairs": pairs, "unmeasured": unmeasured,
        "expert_set": expert_set, "key": key, "manifest": manifest, "manifest_summary": msummary,
        "splits": splits,
    }
