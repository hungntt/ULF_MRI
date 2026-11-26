"""What one record of the 71% figure is."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import trace_detect_denominator as TD  # noqa: E402

PUB = json.load(open(f"{ROOT}/out_public/d3detect_T2.json"))


def test_cached_labels_are_the_published_records_in_order():
    lab = TD.cached_test_labels()
    assert len(lab) == len(PUB) == 1605
    for (row, cy, cx, r, c, ret), x in zip(lab, PUB):
        assert (row, r, c) == (x["row"], x["r"], x["c"]) and ret == x["retention"]


def test_every_record_belongs_to_one_test_subject():
    per = TD.per_subject(TD.cached_test_labels())
    assert len(per) == 5 and sum(v["n"] for v in per.values()) == 1605
    assert np.isclose(sum(v["n"] * v["erased"] for v in per.values()) / 1605,
                      np.mean([x["retention"] < 0.25 for x in PUB]))


@pytest.mark.skipif(not os.path.exists(TD.OUT_JSON), reason="GPU regeneration not run yet")
def test_regeneration_reproduces_the_published_records():
    g = json.load(open(TD.OUT_JSON))
    kept = [s for s in g["sites"] if s["keep"]]
    assert len(kept) == 1605
    lab = TD.cached_test_labels()
    assert all((s["row"], s["cy"], s["cx"], s["r"]) == tuple(int(v) for v in l[:4]) for s, l in zip(kept, lab))
    assert max(abs(s["retention"] - x["retention"]) for s, x in zip(kept, PUB)) < 1e-4


@pytest.mark.skipif(not os.path.exists(TD.OUT_JSON), reason="GPU regeneration not run yet")
def test_keep_flag_depends_only_on_the_inserted_contrast():
    g = json.load(open(TD.OUT_JSON))
    for s in g["sites"]:
        assert s["keep"] == (abs(s["tru"]) > 0.25 * abs(s["c"]))
