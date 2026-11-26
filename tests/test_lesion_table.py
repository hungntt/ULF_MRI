"""Unique lesion instances by size, site, contrast and polarity."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import lesion_table as LT  # noqa: E402


def test_tissue_class_maps_synthseg_labels():
    assert LT.tissue_class(2) == "WM" and LT.tissue_class(41) == "WM"
    assert LT.tissue_class(3) == "cortical GM" and LT.tissue_class(42) == "cortical GM"
    assert LT.tissue_class(4) == "CSF" and LT.tissue_class(24) == "CSF"
    assert LT.tissue_class(10) == "deep GM"
    assert LT.tissue_class(0) == "background"


def test_stage_test_cohort_has_96_bright_lesions_per_subject():
    t = LT.stage_table("test")
    assert t["n"] == 480 and t["n_subjects"] == 5
    assert set(t["per_subject"].values()) == {96}
    assert set(t["by_size_mm"]) == {4, 6, 8, 12}
    assert t["by_polarity"] == {"bright": 480}


def test_stage_test_tissue_counts_cover_every_lesion():
    t = LT.stage_table("test", tissue=True)
    assert sum(t["by_tissue"].values()) == 480


def test_detect_test_set_has_1605_lesions_by_size():
    t = LT.detect_table()
    assert t["n"] == 1605
    assert t["by_size_mm"] == {4: 402, 6: 390, 8: 389, 12: 424}


def test_real_ms_measurable_lesions_match_the_paper():
    t = LT.realms_table()
    assert t["n_kept"] == 721
    assert t["n_measurable"] == 534


def test_atlas_counts_match_the_paper():
    t = LT.atlas_table()
    assert t["n_placed"] == 697
    assert t["n_measurable"] == 684


def test_six_and_eight_mm_share_one_k_bin():
    kb = LT.k_bins()
    assert kb["fwhm"][6] == kb["fwhm"][8]
    assert kb["fwhm"][4] != kb["fwhm"][6]


def test_physcheck_counts_have_subjects_and_instances_per_bin():
    p = LT.physcheck_table()
    assert set(p) == {"HFC|D_HFC", "HFE|D_HFE", "V5|D_HFC", "V5|D_HFE"}
    for comp in p.values():
        for b in comp.values():
            assert b["n"] >= 0 and 0 <= b["n_subj"] <= 19


def test_physcheck_counts_are_the_instances_that_gave_k():
    import json
    A = json.load(open(f"{ROOT}/out_public/d1_physcheck.json"))["analyses"]
    p = LT.physcheck_table()
    for comp in ("HFC|D_HFC", "V5|D_HFE"):
        for b, v in p[comp].items():
            a = A[f"T2|{comp}|sulcal_csf|3d|{b}|reg|c0.05"]
            assert (v["n"], v["n_subj"]) == (a["n"], a["n_subj"]), (comp, b)
