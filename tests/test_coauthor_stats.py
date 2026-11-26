"""Seed-averaged retention, selection, metric association,
subject bootstrap and the acquired arm, from the matched d3stage lesions."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp import coauthor_stats as CS  # noqa: E402

STAGE = json.load(open(f"{ROOT}/out_public/d3stage_T2.json"))["splits"]


def toy():
    """Config A with two seeds and config B with one, three lesions in two subjects."""
    R = np.array([[0.20, 0.10, 0.60],
                  [0.30, 0.10, 0.40],
                  [0.05, 0.50, 0.30]])
    meta = dict(models=["A s0", "A s1", "B"], config=["A", "A", "B"], seed=[0, 1, 0],
                sid=np.array(["s1", "s1", "s2"]), subjects=["s1", "s2"])
    return R, meta


# ---- seed averaging ---------------------------------------------------------------------------------
def test_load_keeps_every_model_and_every_matched_test_lesion():
    R, meta = CS.load("test", "HFC", "syn")
    assert R.shape == (37, 480)
    assert meta["models"][0] == "no enhancer" and len(set(meta["config"])) == 13
    assert sorted(set(meta["sid"])) == sorted(meta["subjects"]) and len(meta["subjects"]) == 5


def test_per_subject_mean_reproduces_the_stage_summary():
    for op in ("HFC", "HFE"):
        for arm, key in (("syn", "R"), ("acq", "R_acq")):
            R, meta = CS.load("test", op, arm)
            ref = STAGE["test"]["per_model"][op]
            for i, nm in enumerate(meta["models"]):
                for s in meta["subjects"]:
                    got = R[i, meta["sid"] == s].mean()
                    assert abs(got - ref[nm]["per_subject"][s][key]) < 1e-6, (op, arm, nm, s)


def test_seed_mean_averages_seeds_within_configuration():
    R, meta = toy()
    Rbar, cfgs = CS.seed_mean(R, meta)
    assert cfgs == ["A", "B"]
    assert np.allclose(Rbar, [[0.25, 0.10, 0.50], [0.05, 0.50, 0.30]])


def test_the_two_erasure_estimands_follow_their_definitions():
    R, meta = toy()
    Rbar, cfgs = CS.seed_mean(R, meta)
    # A: seed-mean [0.25, 0.10, 0.50] -> erased only lesion 2 (0.25 is not < 0.25) -> 1/3
    #    per seed: s0 erases lesions 1, 2 (0.20, 0.10) -> 2/3; s1 erases lesion 2 -> 1/3; mean 1/2
    assert np.allclose(CS.erased_seedmean(Rbar, 0.25), [1 / 3, 1 / 3])
    assert np.allclose(CS.erased_perseed(R, meta, cfgs, 0.25), [1 / 2, 1 / 3])


def test_mean_retention_is_the_same_under_both_estimands():
    R, meta = CS.load("test", "HFC", "syn")
    Rbar, cfgs = CS.seed_mean(R, meta)
    s = CS.summary(R, meta)
    for i, c in enumerate(cfgs):
        per_seed = [R[j].mean() for j, cc in enumerate(meta["config"]) if cc == c]
        assert np.isclose(s[c]["mean_R"], Rbar[i].mean()) and np.isclose(s[c]["mean_R"], np.mean(per_seed))


def test_summary_ranks_enhancers_and_reports_seed_spread():
    s = CS.summary(*CS.load("test", "HFC", "syn"))
    ranks = sorted(v["rank"] for c, v in s.items() if c != "no enhancer")
    assert ranks == list(range(1, 13)) and s["no enhancer"]["rank"] is None
    assert s["cGAN / pix2pix"]["n_seeds"] == 5 and s["cGAN / pix2pix"]["seed_sd"] > 0
    assert s["wavelet diffusion"]["n_seeds"] == 1 and s["wavelet diffusion"]["seed_sd"] is None
    for thr in ("0.1", "0.25", "0.5"):
        assert 0 <= s["U-Net L1"]["erased_seedmean"][thr] <= 1


def test_task6_covers_every_cohort_operator_and_available_arm():
    t6 = CS.task6()
    assert set(t6) == {"test", "ext_on_hfe", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"}
    for sp, v in t6.items():
        assert set(v) == {"HFC", "HFE"}
        arms = {"syn", "acq"} if sp in ("test", "ext_on_hfe", "ext_v5") else {"syn"}
        assert set(v["HFC"]) == arms, sp


def test_e6_verdicts_follow_the_numbers():
    E = CS.check_e6(CS.task6())
    assert set(E) == {"E6a", "E6b", "E6c"}
    for e in E.values():
        assert isinstance(e["pass"], bool) and e["value"]


# ---- selection ---------------------------------------------------------------------------------
def test_delta_is_leader_minus_pick_for_retention_and_pick_minus_leader_for_erasure():
    R, meta = toy()
    d = CS.delta(R, meta, pick="B", leader="A")
    assert np.isclose(d["dR"], 0.0)                                   # both average 0.2833
    assert np.isclose(d["dE_seedmean"], 0.0)                          # 1/3 vs 1/3
    assert np.isclose(d["dE_perseed"], 1 / 3 - 1 / 2)                 # B 1/3, A 1/2
    assert np.isclose(d["per_subject"]["s2"]["dR"], 0.50 - 0.30)      # lesion 3 only


def test_crossfit_chooses_on_the_other_subjects_only():
    # A is best on s1 and s2 but worst on s3; Dice prefers B on every subject.
    R = np.array([[0.9, 0.9, 0.0],
                  [0.5, 0.5, 0.5],
                  [0.1, 0.1, 0.95]])
    meta = dict(models=["A", "B", "C"], config=["A", "B", "C"], seed=[0, 0, 0],
                sid=np.array(["s1", "s2", "s3"]), subjects=["s1", "s2", "s3"])
    dice = {"A": {"s1": 0.5, "s2": 0.5, "s3": 0.5}, "B": {"s1": 0.9, "s2": 0.9, "s3": 0.9},
            "C": {"s1": 0.1, "s2": 0.1, "s3": 0.1}}
    cf = CS.crossfit(R, meta, dice)
    f3 = cf["folds"]["s3"]
    assert f3["leader"] == "A" and f3["pick"] == "B"                  # chosen without s3
    assert np.isclose(f3["dR"], 0.0 - 0.5)                            # scored on s3: A 0.0, B 0.5
    assert cf["folds"]["s1"]["leader"] == "C"                         # on s2, s3: A 0.45, B 0.5, C 0.525


def test_asis_picks_are_the_published_selections():
    assert CS.asis_picks("test") == ("corpus-calibrated", "cGAN / pix2pix")
    assert CS.asis_picks("ext_on_hfe") == ("corpus, uncalibrated", "cGAN / pix2pix")


def test_dice_is_averaged_over_seeds_per_subject():
    d = CS.dice_by_config("test")
    assert len(d) == 12 and all(len(v) == 5 for v in d.values())
    z = json.load(open(f"{ROOT}/out_public/d3zoo_synthseg_T2.json"))
    runs = [nm for nm in z["runs"] if nm.startswith("cGAN / pix2pix")]
    s0 = z["sids"][0]
    assert np.isclose(d["cGAN / pix2pix"][s0], np.mean([z["runs"][nm]["dice"][0] for nm in runs]))


def test_task7_has_asis_and_crossfit_for_every_paired_cohort_operator_and_arm():
    t7 = CS.task7()
    assert set(t7) == {"test", "ext_on_hfe", "ext_v5"}
    for sp in t7:
        for op in ("HFC", "HFE"):
            for arm in ("syn", "acq"):
                v = t7[sp][op][arm]
                assert set(v) == {"asis", "crossfit"}
                assert v["crossfit"]["n_folds"] == len(v["asis"]["per_subject"])


def test_e7_verdicts_follow_the_numbers():
    E = CS.check_e7(CS.task7())
    assert set(E) == {"E7a", "E7b"}
    for e in E.values():
        assert isinstance(e["pass"], bool) and e["value"]


# ---- metric association and leave-family-out ---------------------------------------------------------------------------------
def test_metric_by_model_reads_each_runs_per_subject_values():
    m = CS.metric_by_model("test", "ssim")
    z = json.load(open(f"{ROOT}/out_public/d3zoo_T2.json"))
    v = z["U-Net L1 s3"]
    assert m["U-Net L1 s3"] == dict(zip(v["sids"], v["ssim"]))
    assert len(m) == 36


def test_hierarchical_bootstrap_of_a_perfect_ranking_stays_perfect():
    cfg = ["A", "A", "B", "C", "C"]
    subjects = ["s1", "s2"]
    metric = {"A": {"s1": 0.1, "s2": 0.1}, "B": {"s1": 0.2, "s2": 0.2}, "C": {"s1": 0.3, "s2": 0.3}}
    met = {nm: metric[c] for nm, c in zip(["A0", "A1", "B", "C0", "C1"], cfg)}
    ret = {nm: {s: 10 * metric[c][s] for s in subjects} for nm, c in zip(["A0", "A1", "B", "C0", "C1"], cfg)}
    out = CS.rho_boot(met, ret, dict(zip(["A0", "A1", "B", "C0", "C1"], cfg)), subjects, B=500)
    assert np.isclose(out["rho"], 1.0) and out["lo"] >= 0.99


def test_leave_family_out_drops_the_corpus_family_and_names_the_new_pick():
    lfo = CS.leave_family_out("test", "HFC", "syn")
    assert len(lfo["configs"]) == 9 and not any(c.startswith("corpus") for c in lfo["configs"])
    assert not lfo["pick"].startswith("corpus") and lfo["leader"] == "cGAN / pix2pix"


def test_task8_and_e8_cover_the_paired_cohorts():
    t8 = CS.task8(B=200)
    assert set(t8) == {"test", "ext_on_hfe", "ext_v5"}
    assert set(t8["test"]["HFC"]["syn"]["rho"]) == {"ssim", "psnr", "synthseg"}
    E = CS.check_e8(t8)
    assert set(E) == {"E8a", "E8b"} and all(isinstance(e["pass"], bool) for e in E.values())


# ---- subject bootstrap ---------------------------------------------------------------------------------
def test_resample_weights_count_how_often_each_subject_was_drawn():
    _, meta = toy()
    w = CS.lesion_weights(meta, ["s1", "s1"])
    assert list(w) == [2, 2, 0]


def test_number_of_distinct_subject_resamples():
    assert CS.distinct_resamples(5) == 126
    assert CS.distinct_resamples(11) == 352716


def test_subject_bootstrap_of_identical_subjects_has_no_spread():
    R = np.array([[0.1, 0.3, 0.1, 0.3]])
    meta = dict(models=["A"], config=["A"], seed=[0], sid=np.array(["s1", "s1", "s2", "s2"]), subjects=["s1", "s2"])
    b = CS.boot_subjects(lambda w: float(np.average(R[0] < 0.25, weights=w)), meta, B=200)
    assert b["point"] == 0.5 and b["lo"] == b["hi"] == 0.5
    assert b["per_subject"] == {"s1": 0.5, "s2": 0.5} and b["exploratory"] is True


def test_map_auroc_point_is_the_published_four_seed_mean():
    m = CS.map_auroc_boot(B=50)
    assert abs(m["point"] - 0.8266530151055587) < 1e-9
    assert len(m["per_subject"]) == 5


def test_task9_and_e9():
    t9 = CS.task9(B=200)
    assert {"test", "ext_v5", "map_auroc"} <= set(t9)
    E = CS.check_e9(t9, CS.task7())
    assert set(E) == {"E9a", "E9b"} and all(isinstance(e["pass"], bool) for e in E.values())
