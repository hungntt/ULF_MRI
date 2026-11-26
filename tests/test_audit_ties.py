"""Ties at the erasure threshold."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import audit_ties as A  # noqa: E402


def test_tie_counts_separates_exact_and_near_ties():
    c = A.tie_counts(np.array([0.25, 0.25 + 1e-12, 0.1, np.nan]), 0.25)
    assert c["n"] == 3
    assert c["exact"] == 1
    assert c["near"] == 2
    assert np.isclose(c["erased_lt"], 1 / 3) and np.isclose(c["erased_le"], 2 / 3)


def test_json_walk_collects_only_retention_keys():
    vals = A.json_retentions({"rows": [{"retention": 0.25, "x": 0.25}, {"R": 0.1, "R_acq": 0.3}]})
    assert sorted(vals) == [0.1, 0.25, 0.3]


def test_npz_ratios_divide_responses_by_inserted_contrast():
    z = {"meta_dy": np.array([1.0, 2.0]), "dp|HFC|m": np.array([0.25, 1.0]), "sid_idx": np.array([0, 1])}
    out = dict(A.npz_retentions(z))
    assert list(out) == ["dp|HFC|m"]
    assert np.allclose(out["dp|HFC|m"], [0.25, 0.5])


def test_detect_file_has_no_ties_at_quarter():
    rec = A.audit_file(f"{ROOT}/out_public/d3detect_T2.json")
    assert rec["n"] == 1605
    assert rec["thr"]["0.25"]["exact"] == 0
