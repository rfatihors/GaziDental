#!/usr/bin/env python
"""Stage 1 — data layer: parse the clinical workbook, match names to the COCO export,
build the cleaned manifest and the fixed splits, and write the reports.

Outputs
  data/manifest/dataset_manifest.csv, splits.json, manifest_summary.md
  outputs/01_data/parse_report.md, manifest_summary.md, demographics.md, split_summary.md,
  matches_high.csv, unmatched_manual.csv, name_collisions_coco.csv, duplicates.csv,
  SAPMALAR.md (only when a number deviates from the expectations), OZET.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsv4.config import load_config, resolve  # noqa: E402
from gsv4.dataset.build import build_all  # noqa: E402

EXPECT = {
    "excel_rows": 692, "excel_complete_min": 690, "label_inconsistent_max": 5,
    "high_unique_matches": 149, "dash_base_fallback": 30,
    "kept": {"high": 145, "low": 299, "normal": 786},
    "high_split": {"train": 87, "valid": 29, "test": 29},
    "demographics": {"high_age": None, "low_age": 99, "normal_age": 181},
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = resolve(cfg, Path(cfg["paths"]["outputs"]) / "01_data")
    man_dir = resolve(cfg, cfg["paths"]["manifest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    r = build_all(cfg)
    inv, high, ps = r["inventory"], r["high"], r["parse_summary"]
    matches, manifest, ms, splits = r["matches"], r["manifest"], r["manifest_summary"], r["splits"]
    deviations = []

    # ---------------- parse report
    mh = matches["high"]
    kinds_high = mh["match_kind"].value_counts().to_dict()
    n_unique = int(mh["match_kind"].isin(["exact", "dash_base_fallback"]).sum())
    sec = {g: matches[g]["match_kind"].value_counts().to_dict() for g in ("low", "normal")}
    unmatched_excel_rows = set(range(len(high)))
    for g in matches.values():
        for rows in g.loc[g["match_kind"].isin(["exact", "dash_base_fallback", "row_ambiguous", "name_ambiguous"]), "excel_rows"]:
            unmatched_excel_rows -= {int(x) for x in str(rows).split(";") if x}
    unmatched = high.iloc[sorted(unmatched_excel_rows)][["excel_row", "raw_name", "key", "mean_mm", "complete"]]
    unmatched.to_csv(out_dir / "unmatched_manual.csv", index=False)
    mh.to_csv(out_dir / "matches_high.csv", index=False)
    pd.concat([matches["low"], matches["normal"]]).query("match_kind != 'unmatched'").to_csv(out_dir / "matches_low_normal_secondary.csv", index=False)
    coll = pd.concat(matches.values()).query("match_kind in ('name_ambiguous', 'coco_key_collision')")
    coll.to_csv(out_dir / "name_collisions_coco.csv", index=False)
    keyset = set(inv["key"])
    plain_dash = sorted(k for k in keyset if (k + "~") in keyset)
    stem_dups = inv.groupby("image")["group"].agg(list)
    stem_dups = stem_dups[stem_dups.map(len) > 1]
    dups = high[high["key"].duplicated(keep=False)].sort_values("key")[["excel_row", "raw_name", "key", "mean_mm", "age", "sex"]]
    dups.to_csv(out_dir / "duplicates.csv", index=False)
    sizes = inv.groupby(["width", "height"]).size().sort_values(ascending=False)
    lbl_incons = []
    for _, e in high[high["label_inconsistent_count"] > 0].iterrows():
        for i in range(1, 7):
            if e[f"label_inconsistent_{i}"]:
                lbl_incons.append({"excel_row": e["excel_row"], "name": e["raw_name"], "tooth_index": i, "raw": e[f"raw_{i}"], "mm": e[f"mm_{i}"], "kind": e[f"kind_{i}"], "excel_label": e[f"label_{i}"], "expected": e[f"expected_label_{i}"]})
    lbl_df = pd.DataFrame(lbl_incons)

    if ps["n_rows"] != EXPECT["excel_rows"]:
        deviations.append(f"Excel high sheet rows: {ps['n_rows']} (expected {EXPECT['excel_rows']})")
    if ps["n_complete"] < EXPECT["excel_complete_min"]:
        deviations.append(f"complete rows: {ps['n_complete']} (expected >= {EXPECT['excel_complete_min']})")
    if ps["n_label_inconsistent"] > EXPECT["label_inconsistent_max"]:
        deviations.append(f"label-inconsistent cells: {ps['n_label_inconsistent']} (expected <= {EXPECT['label_inconsistent_max']})")
    if n_unique != EXPECT["high_unique_matches"]:
        deviations.append(f"unique high matches: {n_unique} (expected {EXPECT['high_unique_matches']})")

    parse_md = f"""# Stage 1 — parse report

Source: `{cfg['inputs']['measurements_xlsx']}`, sheet `{cfg['excel']['high_sheet']}` (header row {cfg['excel']['high_header_row']}).

## Rows
| metric | value |
|---|---|
| rows with a name | {ps['n_rows']} |
| rows with all six teeth readable | {ps['n_complete']} |
| rows with six values > 0 | {ps['n_all_positive']} |
| rows with at least one `-` (0 mm) | {ps['n_with_dash_zero']} |
| duplicate normalised names (rows) | {ps['n_duplicate_keys']} — {', '.join(ps['duplicate_keys'])} |
| age-prefixed names | {ps['n_prefixed']} (prefix equals YAŞ column in {ps['n_prefix_matches_sheet']}) |
| age available (sheet / prefix) | {ps['n_age']} ({ps['n_age_from_sheet']} / {ps['n_age_from_prefix']}) |
| sex available | {ps['n_sex']} |

## Cell types (6 × {ps['n_rows']} = {6 * ps['n_rows']} cells)
| parse_kind | count |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in ps["cell_kinds"].items()) + f"""

## Distribution of the image-level mean (complete rows)
median {ps['mean_mm_median']:.2f} mm, 95th percentile {ps['mean_mm_p95']:.2f} mm, max {ps['mean_mm_max']:.2f} mm.

## Consistency with the clinicians' E labels
{ps['n_labelled_cells']} labelled cells; {ps['n_label_inconsistent']} disagree with Table 1 applied to the parsed value:

{md_table(lbl_df) if not lbl_df.empty else '(none)'}

These cells are flagged `label_inconsistent` and excluded from the primary oracle analysis.

## COCO export
{len(inv)} images: """ + ", ".join(f"{g} {n}" for g, n in inv.groupby('group').size().items()) + f"""; gingiva instances {int(inv['n_gingiva'].sum())}, lip instances {int(inv['n_lip'].sum())}.
Frame check (2698×1799 ± 2 px): {int(inv['frame_ok'].sum())} images inside, {int((~inv['frame_ok']).sum())} outside (`frame_uncertain`). {len(sizes)} distinct sizes; top: """ + ", ".join(f"{w}×{h} ({n})" for (w, h), n in sizes.head(5).items()) + f""".

## Name matching (high sheet ↔ COCO)
`high` group ({len(mh)} images): """ + ", ".join(f"{k} {v}" for k, v in sorted(kinds_high.items())) + f""" → **{n_unique} unique matches** (exact + dash_base_fallback).
Secondary (assumed name collisions, reported only): low {sec['low']}, normal {sec['normal']}.
Excel rows without any COCO match: {len(unmatched)} (`unmatched_manual.csv`) — photographs of the earlier study.
Name collisions: {len(coll)} images (`name_collisions_coco.csv`): """ + ", ".join(coll["image"]) + f""".
COCO-internal: {len(plain_dash)} bases exist both plain and dashed ({', '.join(plain_dash)}); {len(stem_dups)} stems repeat across groups ({', '.join(f'{k} [{"/".join(v)}]' for k, v in stem_dups.items())}) — different photographs (audit §5.1), hence images are identified by `group/stem`.

## Calibration workbook
{r['calibration']['image'].nunique()} images × 2 sessions, {len(r['calibration'])} cells; parse kinds: {r['calibration']['parse_kind'].value_counts().to_dict()}.
"""
    (out_dir / "parse_report.md").write_text(parse_md, encoding="utf-8")

    # ---------------- manifest
    manifest.to_csv(man_dir / "dataset_manifest.csv", index=False)
    kept = ms["kept"]
    for g, exp in EXPECT["kept"].items():
        got = int(kept.get(g, 0))
        tol = 0 if g == "high" else 3
        if abs(got - exp) > tol:
            deviations.append(f"kept {g}: {got} (expected ≈ {exp})")
    if ms["expert_set_mismatch"]:
        deviations.append(f"kept measured high ≠ expert set: {ms['expert_set_mismatch']}")
    drop_rows = pd.DataFrame([{"group": g, "reason": rsn, "n": n} for (g, rsn), n in ms["dropped"].items()])
    drop_rows["reason_class"] = drop_rows["reason"].str.split(":").str[0]
    drop_summary = drop_rows.groupby(["group", "reason_class"])["n"].sum().reset_index()
    man_md = f"""# Stage 1 — manifest summary

Mode: `keep_unmeasured_high_in_train = {cfg['dataset']['keep_unmeasured_high_in_train']}` (clinical decision: unmeasured high images are dropped).

| group | start | dropped | kept | kept with reference | train-only |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {g} | {ms['start'].get(g, 0)} | {ms['start'].get(g, 0) - kept.get(g, 0)} | {kept.get(g, 0)} | {ms['kept_with_reference'].get(g, 0)} | {ms['train_only'].get(g, 0)} |"
        for g in cfg["coco"]["groups"]
    ) + f"""
| **total** | {sum(ms['start'].values())} | {sum(ms['start'].values()) - sum(kept.values())} | {sum(kept.values())} | {sum(ms['kept_with_reference'].values())} | {sum(ms['train_only'].values())} |

Patients after cleaning: {ms['n_patients']} (image = patient).

## Drop reasons
{md_table(drop_summary)}

## Same-patient pairs
{len(r['pairs'])} pairs → {ms['pair_components']} connected components. Resolved by an earlier drop: {ms['pair_components'] - len(ms['pair_decisions'])}; decided by rule: {len(ms['pair_decisions'])}.

| kept | dropped | rule |
|---|---|---|
""" + "\n".join(f"| {d['kept']} | {', '.join(d['dropped'])} | {d['rule']} |" for d in ms["pair_decisions"]) + f"""

## Cross-check with the expert set (145)
Kept high images with a reference measurement vs `uzman_seti_145_goruntu.csv`: {'identical' if not ms['expert_set_mismatch'] else 'MISMATCH ' + str(ms['expert_set_mismatch'])}.
"""
    (out_dir / "manifest_summary.md").write_text(man_md, encoding="utf-8")
    (man_dir / "manifest_summary.md").write_text(man_md, encoding="utf-8")

    # ---------------- demographics
    k = manifest[manifest["keep"]]
    demo_rows = []
    for g in cfg["coco"]["groups"]:
        gg = k[k["group"] == g]
        demo_rows.append({
            "group": g, "n": len(gg), "age_n": int(gg["age"].notna().sum()), "age_mean": gg["age"].mean(), "age_sd": gg["age"].std(),
            "age_min": gg["age"].min(), "age_max": gg["age"].max(), "sex_n": int(gg["sex"].notna().sum()),
            "female": int((gg["sex"] == "F").sum()), "male": int((gg["sex"] == "M").sum()),
            "age_from_prefix": int((gg["age_source"] == "prefix").sum()),
        })
    demo = pd.DataFrame(demo_rows)
    tot = k
    demo_md = f"""# Stage 1 — demographic coverage (Reviewer 3)

Age/sex were recorded only for part of the cohort (recording started later). Reported on the cleaned dataset, "n with a record" basis.

{md_table(demo, '{:.1f}')}

All kept images: n = {len(tot)}, age recorded {int(tot['age'].notna().sum())} ({100*tot['age'].notna().mean():.1f} %), mean {tot['age'].mean():.1f} ± {tot['age'].std():.1f} y (range {tot['age'].min():.0f}–{tot['age'].max():.0f}); sex recorded {int(tot['sex'].notna().sum())} ({100*tot['sex'].notna().mean():.1f} %): {int((tot['sex']=='F').sum())} female, {int((tot['sex']=='M').sum())} male.

Reference set (145 high with clinical measurement): age {int(k[k.has_reference_measurement]['age'].notna().sum())}/145, sex {int(k[k.has_reference_measurement]['sex'].notna().sum())}/145.

Age source among kept images with age: sheet {int((tot['age_source']=='sheet').sum())}, file-name prefix {int((tot['age_source']=='prefix').sum())}.
"""
    (out_dir / "demographics.md").write_text(demo_md, encoding="utf-8")

    # ---------------- splits
    with (man_dir / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=1, ensure_ascii=False)
    hs = splits["counts"]["high"]
    if hs != EXPECT["high_split"]:
        deviations.append(f"high split {hs} (expected {EXPECT['high_split']})")
    fold_sizes = pd.Series(splits["cv_folds"]["assignments"]).value_counts().sort_index().to_dict()
    lab_by_split = k[k.group == "high"].groupby(["split", "reference_label"]).size().unstack(fill_value=0)
    lab_by_fold = k[k.cv_fold.notna()].groupby(["cv_fold", "reference_label"]).size().unstack(fill_value=0)
    split_md = f"""# Stage 1 — split summary

Seed {splits['seed']}; patient level (= image level after cleaning); stratified by smile-line group and, within `high`, by the reference class label; fixed test set.

| group | train | valid | test | total |
|---|---|---|---|---|
""" + "\n".join(f"| {g} | {c['train']} | {c['valid']} | {c['test']} | {sum(c.values())} |" for g, c in splits["counts"].items()) + f"""

Train-only images: {len(splits['train_only'])}; images forced to train by a train-only twin: {len(splits['forced_to_train_by_twin'])}.

## High group by reference label
{md_table(lab_by_split.reset_index())}

## 5-fold CV over the {len(splits['cv_folds']['assignments'])} measured high images
Fold sizes: {fold_sizes}

{md_table(lab_by_fold.reset_index())}

Original Roboflow split of the kept high test images (for the record): {k[(k.group=='high') & (k.split=='test')]['orig_split'].value_counts().to_dict()}.
"""
    (out_dir / "split_summary.md").write_text(split_md, encoding="utf-8")

    # ---------------- deviations + summary
    if deviations:
        (out_dir / "SAPMALAR.md").write_text("# Aşama 1 — beklentiden sapmalar\n\n" + "\n".join(f"- {d}" for d in deviations) + "\n", encoding="utf-8")
    elif (out_dir / "SAPMALAR.md").exists():
        (out_dir / "SAPMALAR.md").unlink()
    ozet = f"""# Aşama 1 — Türkçe özet

- Excel yüksek sayfası: {ps['n_rows']} satır, {ps['n_complete']} tam; hücre türleri parse_report.md'de; E etiketiyle uyumsuz {ps['n_label_inconsistent']} hücre.
- `high` grubunda {n_unique} tekil eşleşme (exact {kinds_high.get('exact', 0)}, dash_base_fallback {kinds_high.get('dash_base_fallback', 0)}); belirsiz {kinds_high.get('name_ambiguous', 0) + kinds_high.get('row_ambiguous', 0)}.
- Temizlik sonrası: high {kept.get('high', 0)}, low {kept.get('low', 0)}, normal {kept.get('normal', 0)}; toplam {sum(kept.values())}. Uzman seti ile karşılaştırma: {'birebir' if not ms['expert_set_mismatch'] else 'FARK VAR'}.
- Bölünme (seed {splits['seed']}): high {hs}; low {splits['counts']['low']}; normal {splits['counts']['normal']}; 5 kat: {fold_sizes}.
- Sapma: {'yok' if not deviations else '; '.join(deviations)}.
"""
    (out_dir / "OZET.md").write_text(ozet, encoding="utf-8")
    print(ozet)
    print("deviations:", deviations or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
