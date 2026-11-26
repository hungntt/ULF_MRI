"""Erasure with D's lesion response scaled by k."""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import coauthor_kscale as K  # noqa: E402

ERASE = json.load(open(f"{ROOT}/out_public/d1_physcheck_erasure.json"))["splits"]


def test_k_is_the_matched_unit_k_of_each_size_bin():
    kt = K.k_table()
    for op in ("HFC", "HFE"):
        for r in (2, 3, 4, 6):
            assert kt["test"][op][r] == ERASE["test"][op]["k"]["fwhm"][str(r)][0]
            assert kt["ext_v5"][op][r] == ERASE["ext_v5"][op]["k"]["fwhm"][str(r)][0]
        assert kt["ext_on_hfe"][op] == kt["test"][op]
    assert kt["test"]["HFC"][3] == kt["test"]["HFC"][4]          # 6 and 8 mm share the 3-5 mm bin


def test_kscaled_input_is_the_lesioned_input_at_k1():
    x0 = torch.rand(4, 3, 8, 8) * 0.5
    xl = (x0 + 0.1 * torch.rand(4, 3, 8, 8)).clamp(0, 1)
    assert torch.equal(K.kscaled(x0, xl, torch.ones(4)), xl)


def test_kscaled_input_multiplies_the_lesion_response():
    x0 = torch.full((2, 3, 4, 4), 0.2)
    xl = torch.full((2, 3, 4, 4), 0.3)
    out = K.kscaled(x0, xl, torch.tensor([2.0, 0.5]))
    assert torch.allclose(out[0], torch.full((3, 4, 4), 0.4)) and torch.allclose(out[1], torch.full((3, 4, 4), 0.25))


def test_kscaled_input_stays_in_range():
    out = K.kscaled(torch.full((1, 3, 2, 2), 0.9), torch.full((1, 3, 2, 2), 1.0), torch.tensor([3.0]))
    assert float(out.max()) == 1.0


@pytest.mark.skipif(not os.path.exists(K.rows_path("test")), reason="GPU run not done yet")
def test_sites_are_the_stage_sites():
    z, s = np.load(K.rows_path("test")), np.load(f"{ROOT}/out_public/d3stage_rows_test_T2.npz")
    for k in ("meta_row", "meta_cy", "meta_cx", "meta_r", "meta_c"):
        assert np.array_equal(z[k], s[k])


@pytest.mark.skipif(not os.path.exists(K.rows_path("test")), reason="GPU run not done yet")
def test_k1_reproduces_the_stage_rows():
    """Exact for every deterministic model. LoHiResGAN's forward pass is not deterministic on this GPU (two identical
    calls differ by up to 0.023 in output), and wavelet diffusion samples, so those two are checked separately."""
    z, s = np.load(K.rows_path("test")), np.load(f"{ROOT}/out_public/d3stage_rows_test_T2.npz")
    dy = s["meta_dy"]
    worst = {}
    for key in z.files:
        if key.startswith(("dp1|", "dpa1|")) and not any(m in key for m in ("wavelet diffusion", "LoHiResGAN")):
            ref = key.replace("dp1|", "dp|").replace("dpa1|", "dpa|")
            worst[key] = float(np.max(np.abs(z[key] / dy - s[ref] / dy)))
    assert len(worst) == 2 * 2 * 31, len(worst)                     # 2 arms x 2 operators x 31 models
    assert max(worst.values()) < 1e-3, sorted(worst.items(), key=lambda kv: -kv[1])[:3]


@pytest.mark.skipif(not os.path.exists(K.rows_path("test")), reason="GPU run not done yet")
def test_k1_reproduces_lohiresgan_pooled_values():
    z, s = np.load(K.rows_path("test")), np.load(f"{ROOT}/out_public/d3stage_rows_test_T2.npz")
    dy = s["meta_dy"]
    keys = [k for k in z.files if k.startswith(("dp1|", "dpa1|")) and "LoHiResGAN" in k]
    assert len(keys) == 2 * 2 * 5
    for key in keys:
        a, b = z[key] / dy, s[key.replace("dp1|", "dp|").replace("dpa1|", "dpa|")] / dy
        assert abs(a.mean() - b.mean()) < 0.002 and abs((a < 0.25).mean() - (b < 0.25).mean()) < 0.01, key


@pytest.mark.skipif(not os.path.exists(K.rows_path("test")), reason="GPU run not done yet")
def test_summary_compares_k1_and_kscaled_for_every_configuration_and_arm():
    s = K.summarise(splits=("test",))["test"]["HFC"]
    for arm in ("syn", "acq"):
        v = s[arm]
        assert len(v["per_config"]) == 13 and -1 <= v["rho_meanR"] <= 1
        for c, p in v["per_config"].items():
            assert set(p) >= {"erased_k1", "erased_k", "drop_perseed", "drop_seedmean", "meanR_k1", "meanR_k"}
        assert {"dE_seedmean", "dE_perseed", "dR"} <= set(v["delta"])
