"""Propagate the inpainting error into the real-MS-lesion results."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import coauthor_inpaint as CI  # noqa: E402


def test_error_ratios_are_the_457_pseudo_lesions():
    e = CI.error_ratios()
    assert len(e) == 457
    assert abs(np.median(e) - (-0.0860)) < 1e-3 and abs(np.median(np.abs(e)) - 0.2359) < 1e-3


def test_true_contrast_is_the_measured_one_minus_the_inpainting_error():
    assert np.allclose(CI.factor(np.array([0.2, -0.5, 0.0])), [0.8, 1.5, 1.0])


def test_error_draws_are_reproducible_and_resampled_from_the_ratios():
    a, b = CI.draw_errors(7, n_draws=3, seed=0), CI.draw_errors(7, n_draws=3, seed=0)
    assert a.shape == (3, 7) and np.array_equal(a, b)
    assert set(np.round(a.ravel(), 12)) <= set(np.round(CI.error_ratios(), 12))


@pytest.mark.skipif(not os.path.exists(CI.OUT_NPZ), reason="GPU run not done yet")
def test_unperturbed_draw_reproduces_the_published_lesion_rows():
    z = np.load(CI.OUT_NPZ)
    rows = {CI.key(l): l for l in json.load(open(f"{ROOT}/out_public/d3reallesion_rows_s0.json"))["rows"]}
    keys = [tuple(k) for k in json.loads(str(z["keys"]))]
    assert len(keys) == len(rows) == 721
    assert np.allclose(z["tru"][0], [rows[k]["tru_t2"] for k in keys], atol=1e-5)
    for nm in json.loads(str(z["models"])):
        if nm.startswith("LoHiResGAN"):          # forward pass not deterministic on the GPU used (see test_coauthor_kscale.py)
            continue
        ref = np.array([rows[k]["out"][nm]["real"] for k in keys])
        assert np.nanmax(np.abs(z[f"out|{nm}"][0] - ref)) < 1e-3, nm


@pytest.mark.skipif(not os.path.exists(CI.OUT_NPZ), reason="GPU run not done yet")
def test_unperturbed_draw_reproduces_lohiresgan_pooled_values():
    s = CI.summarise()["unperturbed"]
    pub = json.load(open(f"{ROOT}/out_public/d3reallesion.json"))["transplant"]["LoHiResGAN s0"]
    assert abs(s["mean_R"]["LoHiResGAN s0"] - pub["R_real"]) < 0.005


@pytest.mark.skipif(not os.path.exists(CI.OUT_NPZ), reason="GPU run not done yet")
def test_summary_has_one_entry_per_perturbed_draw():
    s = CI.summarise()
    assert s["n_lesions"] == 534 and len(s["draws"]) == 20
    assert abs(s["unperturbed"]["rho"] - 0.9860139860139862) < 1e-9
    assert abs(s["unperturbed"]["erased_pick"] - 0.8220973782771536) < 1e-9
