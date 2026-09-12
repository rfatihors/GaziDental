"""Name normalisation and tiered Excel <-> COCO matching (spec §4.2, tests 12)."""
from gsv4.io.naming import (
    coco_stem,
    match_names,
    normalize_coco_name,
    normalize_excel_name,
)


def test_coco_stem_strips_roboflow_hash_and_extension():
    assert coco_stem("IMG_7307_jpg.rf.7EZ0jrzH8e8LkUYfRKtu.jpg") == "IMG_7307_jpg"
    assert coco_stem("IMG_2544-_jpeg.rf.abc.jpeg") == "IMG_2544-_jpeg"
    assert coco_stem("IMG_2544-_jpeg") == "IMG_2544-_jpeg"


def test_trailing_dot_and_dash_are_the_same_marker_and_are_kept():
    excel = normalize_excel_name("IMG_2544.")
    coco = normalize_coco_name("IMG_2544-_jpeg.rf.abc.jpeg")
    assert excel.key == coco.key == "img2544~"
    assert excel.dot and coco.dot
    assert excel.base == "img2544"


def test_plain_name_does_not_match_dotted_name():
    plain = normalize_excel_name("ımg_2544")  # Turkish dotless i
    dotted = normalize_coco_name("IMG_2544-_jpeg.rf.abc.jpeg")
    assert plain.key == "img2544"
    assert plain.key != dotted.key


def test_extension_in_excel_is_not_a_dot_marker():
    assert normalize_excel_name("IMG_78701.jpg").key == "img78701"
    assert normalize_excel_name("IMG_78701.jpg").dot is False
    # double dot before extension: the first dot is the marker
    assert normalize_excel_name("IMG_6849..jpg").key == "img6849~"


def test_age_prefix_is_extracted_and_preserved_in_key():
    n = normalize_excel_name("25-IMG_4552")
    assert n.base == "img4552"
    assert n.age_prefix == 25
    assert n.prefix_stripped is True
    assert n.key == "25-img4552"
    assert normalize_excel_name("IMG_4552").key == "img4552"
    assert normalize_coco_name("23-IMG_4080_JPG.rf.x.jpg").key == "23-img4080"
    glued = normalize_excel_name("60IMG_4347")
    assert glued.age_prefix == 60 and glued.key == "60-img4347"
    assert normalize_excel_name("111 (1)").age_prefix is None


def test_parenthesised_counters_stay_distinct():
    assert normalize_coco_name("IMG_7004 (2)_jpg.rf.x.jpg").key == "img7004(2)"
    assert normalize_excel_name("111 (1)").key == "111(1)"
    assert normalize_coco_name("111 (2)-_jpg.rf.x.jpg").key == "111(2)~"


def test_coco_uppercase_extension_variants():
    assert normalize_coco_name("gummy42_JPG.rf.x.jpg").key == "gummy42"
    assert normalize_coco_name("IMG_1207_JPG.rf.x.jpg").key == "img1207"


def _excel(keys):
    """Map key -> list of Excel row indices."""
    out = {}
    for i, k in enumerate(keys):
        out.setdefault(k, []).append(i)
    return out


def test_match_exact():
    coco = {"img2544~": "IMG_2544-_jpeg"}
    res = match_names(coco, _excel(["img2544~"]), all_coco_keys=set(coco))
    assert res["IMG_2544-_jpeg"].match_kind == "exact"
    assert res["IMG_2544-_jpeg"].excel_rows == [0]


def test_match_dash_base_fallback_only_when_base_absent_in_coco():
    coco = {"img6545~": "IMG_6545-_jpg"}
    res = match_names(coco, _excel(["img6545"]), all_coco_keys=set(coco))
    assert res["IMG_6545-_jpg"].match_kind == "dash_base_fallback"
    assert res["IMG_6545-_jpg"].excel_key == "img6545"

    coco2 = {"img6545~": "IMG_6545-_jpg", "img6545": "IMG_6545_jpg"}
    res2 = match_names(coco2, _excel(["img6545"]), all_coco_keys=set(coco2))
    assert res2["IMG_6545-_jpg"].match_kind == "name_ambiguous"
    assert res2["IMG_6545_jpg"].match_kind == "name_ambiguous"


def test_base_absent_elsewhere_in_coco_counts_too():
    # base image lives in another group -> still ambiguous
    coco = {"img6545~": "IMG_6545-_jpg"}
    res = match_names(coco, _excel(["img6545"]), all_coco_keys={"img6545~", "img6545"})
    assert res["IMG_6545-_jpg"].match_kind == "name_ambiguous"


def test_match_row_ambiguous_when_excel_has_two_rows():
    coco = {"img7366": "IMG_7366_jpg"}
    res = match_names(coco, _excel(["img7366", "img7366"]), all_coco_keys=set(coco))
    assert res["IMG_7366_jpg"].match_kind == "row_ambiguous"
    assert res["IMG_7366_jpg"].excel_rows == [0, 1]


def test_prefixed_and_unprefixed_are_different_keys():
    coco = {"img8645": "IMG_8645_jpg"}
    res = match_names(coco, _excel(["img8645", "27-img8645"]), all_coco_keys=set(coco))
    assert res["IMG_8645_jpg"].match_kind == "exact"
    assert res["IMG_8645_jpg"].excel_rows == [0]


def test_prefix_is_never_stripped_for_matching():
    coco = {"img4552": "IMG_4552_jpg"}
    res = match_names(coco, _excel(["25-img4552"]), all_coco_keys=set(coco))
    assert res["IMG_4552_jpg"].match_kind == "unmatched"


def test_dotted_excel_never_matches_plain_coco():
    coco = {"img2544": "IMG_2544_jpg"}
    res = match_names(coco, _excel(["img2544~"]), all_coco_keys=set(coco))
    assert res["IMG_2544_jpg"].match_kind == "unmatched"
