"""Form reader on synthetic (messy) forms."""
import math

import pandas as pd
import pytest

from gsv4.io.forms import form_qc_summary, join_key, parse_class, read_form, split_repeats
from tests.synth import synthetic_key, synthetic_truth, write_synthetic_forms

CFG = {"form_sheet": "Degerlendirme", "form_header_row": 3}


def test_parse_class_variants():
    assert parse_class("E2") == ("E2", None, None)
    assert parse_class(" e3 ") == ("E3", None, None)
    assert parse_class("E1-E2") == ("E1", "E2", "combined_class_in_cell")
    assert parse_class(None)[2] == "missing_class"
    assert parse_class("gummy")[2] == "invalid_class"


def test_read_messy_forms_keeps_all_rows_and_flags(tmp_path):
    key = synthetic_key(30, 5, seed=1)
    truth = synthetic_truth(key, seed=1)
    paths = write_synthetic_forms(tmp_path, key, truth, seed=1, messy=True, empty_rows_expert2={3, 10, 20})
    for i, p in enumerate(paths, start=1):
        f = read_form(p, CFG)
        assert len(f) == 35, "no row may be dropped"
        assert set(f.columns) >= {"goruntu_id", "scale_px_per_mm", "mm_13", "mm_23", "class_primary", "class_secondary", "confidence", "form_flags", "row_empty"}
        j = join_key(f, key)
        assert j["key_missing"].sum() == 0
        first, pairs = split_repeats(j)
        assert len(first) == 30 and len(pairs) == 5
        assert {"class_primary_first", "class_primary_repeat", "mean_mm_first", "mean_mm_repeat"} <= set(pairs.columns)
        qc = form_qc_summary(f)
        if i == 2:
            assert qc["n_rows_empty"] == 3
            assert (f.loc[f["row_empty"], "form_flags"].str.contains("row_empty")).all()
            assert f.loc[f["row_empty"], "class_primary"].isna().all()


def test_reader_flags_blank_tooth_zero_and_missing_values(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Degerlendirme"
    for _ in range(3):
        ws.append([None])
    ws.append(["Sıra", "Görüntü ID", "Ölçek (pixel/mm)", 13, 12, 11, 21, 22, 23, "Etiyoloji sınıfı", "İkinci aday", "Güven", "Not"])
    ws.append([1, "G001", 17.2, 2.1, 2.3, None, 0, 1.9, 2.0, "E1", None, 4, "ok"])
    ws.append([2, "G002", None, 3.1, 3.3, 3.0, 3.2, 2.9, 3.0, "e2", "E1", None, None])
    ws.append([3, "G003", 0, 3.1, "x", 3.0, 3.2, 2.9, 3.0, "E9", "E1", 7, None])
    ws.append([4, "G004", 16.5, None, None, None, None, None, None, None, None, None, None])
    ws.append([None] * 13)
    p = tmp_path / "f.xlsx"; wb.save(p)
    f = read_form(p, CFG).set_index("goruntu_id")
    assert len(f) == 4
    assert math.isnan(f.loc["G001", "mm_11"]) and "mm_11_missing" in f.loc["G001", "form_flags"]
    assert f.loc["G001", "mm_21"] == 0.0 and f.loc["G001", "n_mm_filled"] == 5
    assert math.isnan(f.loc["G001", "mean_mm_complete"]) and not math.isnan(f.loc["G001", "mean_mm"])
    assert "scale_missing" in f.loc["G002", "form_flags"] and f.loc["G002", "class_primary"] == "E2"
    assert "confidence_missing" in f.loc["G002", "form_flags"]
    g3 = f.loc["G003"]
    assert "scale_zero" in g3["form_flags"] and "mm_12_unparsed" in g3["form_flags"] and "invalid_class" in g3["form_flags"] and "confidence_out_of_range" in g3["form_flags"]
    assert g3["class_primary"] is None or (isinstance(g3["class_primary"], float) and math.isnan(g3["class_primary"]))
    assert bool(f.loc["G004", "row_empty"]) is True and "row_empty" in f.loc["G004", "form_flags"]
