"""Handcrafted baseline for the retention map."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import coauthor_baseline as HB  # noqa: E402


def test_distance_to_csf_is_zero_in_csf_and_grows_away_from_it():
    seg = np.full((3, 9, 9), 2, np.int16)
    seg[:, :, 0] = 24
    d = HB.csf_distance(seg)
    assert d[1, 4, 0] == 0 and d[1, 4, 3] == 3 and d[1, 4, 8] == 8


def test_tissue_one_hot_has_one_class_per_site():
    oh = HB.tissue_one_hot(["WM", "cortical GM", "CSF", "other"])
    assert oh.shape == (4, len(HB.TISSUES)) and np.array_equal(oh.sum(1), np.ones(4))


def test_footprint_features_of_a_flat_image():
    y = np.full((1, 32, 32), 0.3, np.float32)
    f = HB.footprint_features(y, rows=[0], cy=[16], cx=[16], r=[3])
    assert np.allclose(f["core_mean"], 0.3) and np.allclose(f["grad"], 0) and np.allclose(f["std"], 0)


def test_labels_are_the_cached_map_labels():
    tr, te = HB.labels("train"), HB.labels("test")
    assert len(tr) == 5754 and len(te) == 1605


@pytest.mark.skipif(not os.path.exists(HB.OUT_JSON), reason="baseline not fitted yet")
def test_results_reproduce_radius_alone_and_report_both_models():
    r = json.load(open(HB.OUT_JSON))
    assert abs(r["auroc"]["radius"] - 0.7271520916544041) < 1e-9
    assert {"logistic", "boosting"} <= set(r["auroc"])
    assert set(r["within_radius"]["boosting"]) == {"4", "6", "8", "12"}
