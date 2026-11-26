"""Which enhancer models each analysis used."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import registry_table as RT  # noqa: E402

FIVE = {"U-Net L1", "U-Net L1+SSIM", "U-Net absolute", "cGAN / pix2pix", "LoHiResGAN", "corpus-calibrated"}


def configs(pairs):
    return {c for c, _ in pairs}


def test_zoo_is_12_configurations_36_models_with_six_five_seed():
    z = RT.model_sets()["zoo"]
    assert len(z) == 36 and len(configs(z)) == 12
    seeds = {c: sum(1 for cc, _ in z if cc == c) for c in configs(z)}
    assert {c for c, n in seeds.items() if n == 5} == FIVE
    assert all(n in (1, 5) for n in seeds.values())


def test_h2lo_and_kscaled_use_the_same_11_seed0_configurations_without_diffusion():
    s = RT.model_sets()
    for k in ("h2lo", "kscale_seed0"):
        assert len(s[k]) == 11 and {sd for _, sd in s[k]} == {0}
        assert "wavelet diffusion" not in configs(s[k])
    assert configs(s["h2lo"]) == configs(s["kscale_seed0"])


def test_location_and_map_use_the_five_seed_configurations_at_seed0():
    s = RT.model_sets()
    assert configs(s["location"]) == FIVE and {sd for _, sd in s["location"]} == {0}
    assert configs(s["map_xarch"]) == FIVE


def test_real_ms_uses_all_12_configurations_at_seed0():
    r = RT.model_sets()["real_ms"]
    assert len(r) == 12 and len(configs(r)) == 12 and {sd for _, sd in r} == {0}


def test_every_configuration_has_a_family():
    for name, pairs in RT.model_sets().items():
        for c, _ in pairs:
            assert c in RT.FAMILY, (name, c)


def test_markdown_names_every_analysis():
    md = RT.markdown(RT.model_sets())
    for k in RT.ANALYSES:
        assert RT.ANALYSES[k]["label"] in md
