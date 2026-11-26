"""Dark against bright lesions at matched local intensity."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import coauthor_darkbright as DB  # noqa: E402

OOD = json.load(open(f"{ROOT}/out_public/d3ood_T2.json"))


def test_tertiles_split_values_into_three_equal_groups():
    lab, edges = DB.tertiles(np.arange(9.0))
    assert list(lab) == [0, 0, 0, 1, 1, 1, 2, 2, 2] and len(edges) == 2


def test_paired_counts_tabulate_discordant_pairs():
    t = DB.paired_counts(np.array([True, True, False, False, True]), np.array([True, False, False, True, False]))
    assert t == dict(both=1, dark_only=2, bright_only=1, neither=1)


@pytest.mark.skipif(not os.path.exists(DB.OUT_JSON), reason="GPU run not done yet")
def test_rerun_reproduces_the_published_families():
    rows = json.load(open(DB.OUT_JSON))["rows"]
    for fam, key in (("bright", "iso   (training family)"), ("dark", "hypo  (negative)")):
        kept = [x for x in rows[fam] if x["kept"]]
        assert len(kept) == OOD[key]["n"] == 120
        er = np.mean([x["retention"] < 0.25 for x in kept])
        assert abs(er - OOD[key]["erased"]) <= 1 / 120 + 1e-12


@pytest.mark.skipif(not os.path.exists(DB.OUT_JSON), reason="GPU run not done yet")
def test_bright_and_dark_lesions_share_site_size_and_magnitude():
    rows = json.load(open(DB.OUT_JSON))["rows"]
    assert len(rows["bright"]) == len(rows["dark"])
    for b, d in zip(rows["bright"], rows["dark"]):
        assert (b["sid"], b["row"], b["cy"], b["cx"], b["r"]) == (d["sid"], d["row"], d["cy"], d["cx"], d["r"])
        assert b["c"] == -d["c"] > 0
        assert b["y_ring"] == d["y_ring"]
