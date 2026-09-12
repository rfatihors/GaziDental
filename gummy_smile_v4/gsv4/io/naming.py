"""Image-name normalisation and tiered Excel <-> COCO matching (spec §4.2).

Three rules, all confirmed with the clinical team:

1. A trailing dot in the Excel name (``IMG_2544.``) is a *different* photograph from
   ``IMG_2544``. Roboflow turned that dot into a dash (``IMG_2544-_jpeg``). Both are
   normalised to a ``~`` marker that is kept, never merged away.
2. A leading number followed by a dash (``25-IMG_4552``) — or glued to the ``IMG_``
   token (``60IMG_4347``, eight such rows) — is the patient's age and is a different
   photograph from ``IMG_4552``. The prefix is kept in the key and also returned as
   ``age_prefix``. ``111 (1)`` is a plain name, not a prefix.
3. Lower-case with Turkish ``ı``/``İ`` mapped to ``i``; the Roboflow ``.rf.<hash>``
   suffix, extensions and ``_jpg``-style suffixes are removed; whitespace and
   underscores are dropped. Parenthesised counters (``IMG_7004 (2)``) are kept as
   ``(2)`` so they cannot collide with a genuine five-digit number.

Matching is tiered: exact key -> dash-base fallback (COCO ``x~`` to Excel ``x`` only
when no COCO image ``x`` exists anywhere) -> ambiguous.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

_RF_SUFFIX = re.compile(r"\.rf\.[A-Za-z0-9]+(\.[A-Za-z0-9]+)?$")
_EXT = re.compile(r"\.(jpe?g|png|bmp|tiff?)$", re.IGNORECASE)
_UNDERSCORE_EXT = re.compile(r"_(jpe?g|png|bmp|tiff?)$", re.IGNORECASE)
_AGE_PREFIX = re.compile(r"^(\d{1,3})(?:-(?=\S)|(?=[A-Za-z]{2,4}_))")  # 25-IMG_4552 or 60IMG_4347
_KEEP = re.compile(r"[^a-z0-9()]")


@dataclass(frozen=True)
class NormalizedName:
    raw: str
    base: str
    dot: bool
    age_prefix: Optional[int]
    key: str = field(default="")

    @property
    def prefix_stripped(self) -> bool:
        return self.age_prefix is not None


def coco_stem(file_name: str) -> str:
    """``IMG_7307_jpg.rf.7EZ0jr.jpg`` -> ``IMG_7307_jpg`` (the id used in every CSV)."""
    return _RF_SUFFIX.sub("", str(file_name).strip())


def _lower_tr(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower().replace("ı", "i")


def _finish(raw: str, core: str, dot: bool) -> NormalizedName:
    core = core.strip()
    age: Optional[int] = None
    m = _AGE_PREFIX.match(core)
    if m:
        age = int(m.group(1))
        core = core[m.end():]
    base = _KEEP.sub("", _lower_tr(core))
    key = (f"{age}-" if age is not None else "") + base + ("~" if dot else "")
    return NormalizedName(raw=raw, base=base, dot=dot, age_prefix=age, key=key)


def normalize_excel_name(raw: object) -> NormalizedName:
    """Normalise a name as written by the clinician (``IMG_2544.``, ``25-IMG_4552``, ``IMG_78701.jpg``)."""
    s = str(raw).strip()
    core = _EXT.sub("", s)          # a real extension is not a marker
    dot = core.endswith(".")
    core = core.rstrip(".")
    return _finish(s, core, dot)


def normalize_coco_name(raw: object) -> NormalizedName:
    """Normalise a COCO ``file_name`` or stem (``IMG_2544-_jpeg.rf.x.jpeg``, ``23-IMG_4080_JPG``)."""
    s = str(raw).strip()
    core = coco_stem(s)
    core = _EXT.sub("", core)
    core = _UNDERSCORE_EXT.sub("", core)
    dot = core.endswith("-")
    core = core.rstrip("-")
    return _finish(s, core, dot)


@dataclass
class MatchResult:
    image: str
    coco_key: str
    match_kind: str                 # exact | dash_base_fallback | name_ambiguous | row_ambiguous | unmatched
    excel_key: Optional[str] = None
    excel_rows: List[int] = field(default_factory=list)


def match_names(
    coco_keys: Dict[str, str],
    excel_keys: Dict[str, List[int]],
    all_coco_keys: Optional[Set[str]] = None,
) -> Dict[str, MatchResult]:
    """Match COCO images (key -> image stem) to Excel rows (key -> row indices).

    ``all_coco_keys`` is the set of keys over the *whole* COCO export (all groups) and
    decides whether a dash-base fallback is safe.
    """
    all_keys = set(all_coco_keys) if all_coco_keys is not None else set(coco_keys)
    out: Dict[str, MatchResult] = {}
    for key, image in coco_keys.items():
        rows = excel_keys.get(key)
        if rows:
            kind = "exact" if len(rows) == 1 else "row_ambiguous"
            out[image] = MatchResult(image, key, kind, key, list(rows))
            continue
        if key.endswith("~"):
            base = key[:-1]
            rows = excel_keys.get(base)
            if rows:
                if base in all_keys:
                    out[image] = MatchResult(image, key, "name_ambiguous", base, list(rows))
                else:
                    kind = "dash_base_fallback" if len(rows) == 1 else "row_ambiguous"
                    out[image] = MatchResult(image, key, kind, base, list(rows))
                continue
        out[image] = MatchResult(image, key, "unmatched")
    # rule (c): plain COCO image ``x`` whose dashed twin ``x~`` also exists in COCO while
    # Excel only has ``x`` -> which photograph the row belongs to is unknown
    for res in out.values():
        if (
            res.match_kind == "exact"
            and not res.coco_key.endswith("~")
            and (res.coco_key + "~") in all_keys
            and (res.coco_key + "~") not in excel_keys
        ):
            res.match_kind = "name_ambiguous"
    return out
