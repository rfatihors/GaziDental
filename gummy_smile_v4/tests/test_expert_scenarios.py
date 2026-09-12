"""End-to-end Stage 4 on synthetic forms with known agreement levels."""
import numpy as np
import pandas as pd
import pytest

from gsv4.eval.expert import class_agreement, inter_expert, intra_expert, mm_agreement, model_table, reference_standard, tooth_long, tooth_mixed_models
from gsv4.io.forms import join_key, read_form, split_repeats
from tests.synth import synthetic_key, synthetic_model_output, synthetic_truth, write_synthetic_forms

CFG = {"form_sheet": "Degerlendirme", "form_header_row": 3}


def _run(tmp_path, p_agree, mm_sd, scale_cv=0.06, seed=3, n=120):
    key = synthetic_key(n, 20, seed=seed)
    truth = synthetic_truth(key, seed=seed)
    per = synthetic_model_output(truth, seed=seed, mm_sd=0.3)
    write_synthetic_forms(tmp_path, key, truth, seed=seed, p_agree=p_agree, mm_sd=mm_sd, scale_cv=scale_cv, messy=False)
    first, pairs = {}, {}
    for e in (1, 2, 3):
        f = join_key(read_form(tmp_path / f"Uzman_{e}_form.xlsx", CFG), key)
        first[e], pairs[e] = split_repeats(f)
    images = list(truth["image"])
    inter = inter_expert(first, per.set_index("image")["ref_mm"], images, n_boot=200, seed=42)
    exp_mean, exp_scale = inter.pop("expert_mean_mm"), inter.pop("expert_scale_mean")
    model = model_table(per, exp_scale)
    ref = reference_standard(first)
    cls, pc = class_agreement(model, ref, {"all": images}, n_boot=200, seed=42, scales=["global", "expert"])
    return dict(inter=inter, model=model, ref=ref, cls=cls, pc=pc, first=first, pairs=pairs, per=per, exp_mean=exp_mean, images=images)


def test_high_agreement_scenario(tmp_path):
    r = _run(tmp_path, p_agree=0.95, mm_sd=0.15, scale_cv=0.02)
    strict = r["cls"][(r["cls"].scoring == "strict") & (r["cls"].scale == "global")].iloc[0]
    assert strict["kappa_linear"] > 0.6
    assert strict["kappa_linear_ci_low"] <= strict["kappa_linear"] <= strict["kappa_linear_ci_high"]
    lenient = r["cls"][(r["cls"].scoring == "lenient") & (r["cls"].scale == "global")].iloc[0]
    assert lenient["observed_agreement"] >= strict["observed_agreement"]
    assert r["inter"]["fleiss_kappa"] > 0.7
    assert r["inter"]["mm_icc2_1"] > 0.95
    assert r["inter"]["scale_icc2_1"] > 0.5
    assert (r["ref"]["reference_kind"] == "consensus_pending").sum() < 5
    e4 = r["pc"][(r["pc"]["class"] == "E4") & (r["pc"].scale == "global")].iloc[0]
    assert e4["n_reference"] == 0 and np.isnan(e4["sensitivity"])


def test_random_classes_give_near_zero_kappa(tmp_path):
    r = _run(tmp_path, p_agree=0.0, mm_sd=0.15, seed=5)
    # p_agree = 0 -> experts always pick a neighbouring class: systematic disagreement, kappa well below the high scenario
    strict = r["cls"][(r["cls"].scoring == "strict") & (r["cls"].scale == "global")].iloc[0]
    assert strict["kappa_linear"] < 0.3


def test_mm_noise_lowers_icc_monotonically(tmp_path):
    lo = _run(tmp_path / "a", p_agree=0.9, mm_sd=0.1)
    hi = _run(tmp_path / "b", p_agree=0.9, mm_sd=1.0)
    assert lo["inter"]["mm_icc2_1"] > hi["inter"]["mm_icc2_1"]
    mm_lo = mm_agreement(lo["model"], lo["first"], lo["exp_mean"], lo["per"].set_index("image")["ref_mm"], lo["images"]).set_index("comparator")
    assert mm_lo.at["expert_mean", "icc2_1"] > 0.8
    assert mm_lo.at["clinical_reference", "n"] == len(lo["images"])


def test_intra_expert_and_mixed_models(tmp_path):
    r = _run(tmp_path, p_agree=0.9, mm_sd=0.2)
    intra = intra_expert(r["pairs"], n_boot=100, seed=42)
    assert len(intra) == 3 and (intra["n_repeats"] == 20).all()
    assert (intra["icc2_1_image"] > 0.9).all()
    long = tooth_long(r["model"], r["first"], r["images"])
    assert set(long.columns) >= {"patient", "tooth", "diff", "alignment_uncertain"}
    mixed = tooth_mixed_models(long)
    assert len(mixed) == 3 and all("error" not in m for m in mixed)
    assert "alignment_uncertain[T.True]" in mixed[2]["fixed_effects"].index
    assert abs(mixed[0]["fixed_effects"].loc["Intercept", "estimate"]) < 0.3
