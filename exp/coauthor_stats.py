"""From the matched d3stage lesions.

Every model of the zoo (12 configurations, 36 models) and the no-enhancer row saw the same lesions at the same
sites (out_public/d3stage_rows_<split>_T2.npz, exp/d3stage.py, dense plan, seed 11). Retention per lesion is
R = dp / dy (synthetic arm) or dpa / dy (acquired-input arm), as in d3stage. Lesions with too small a core or
ring, or a zero inserted contrast, are dropped exactly as d3stage.summary does.

Two erasure estimands, reported side by side:
  erased_seedmean   fraction of lesions whose seed-averaged retention is below the threshold
  erased_perseed    mean over seeds of each trained model's erased fraction (what one deployed model does)
For single-run configurations they coincide. Mean retention and the ranking are the same under both.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/coauthor_stats.py
"""
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
NOENH = "no enhancer"
THRS = (0.10, 0.25, 0.50)
THR = 0.25


def _tkey(t):
    return f"{t:g}"


def load(split, op="HFC", arm="syn", source="stage"):
    """-> (R[model, lesion], meta). source 'stage' = d3stage rows; 'kscale' = exp/coauthor_kscale.py rows."""
    z = np.load(f"{OUT}/d3stage_rows_{split}_T2.npz")
    m = json.load(open(f"{OUT}/d3stage_rows_{split}_T2.meta.json"))
    dy = z["meta_dy"]
    keep = z["meta_good"].astype(bool) & (np.abs(dy) > 0)
    src = z if source == "stage" else np.load(f"{OUT}/coauthor_kscale_rows_{split}_T2.npz")
    key = {"syn": "dp", "acq": "dpa"}[arm]
    models = list(m["models"])
    R = np.stack([src[f"{key}|{op}|{nm}"][keep] / dy[keep] for nm in models]).astype(np.float64)
    subj = np.array(m["subjects"])
    meta = dict(models=models, config=[m["models"][nm][0] for nm in models], seed=[m["models"][nm][1] for nm in models],
                sid=subj[z["sid_idx"][keep]], subjects=list(subj), r=z["meta_r"][keep], c=z["meta_c"][keep],
                split=split, op=op, arm=arm, source=source)
    return R, meta


def configs(meta):
    out = []
    for c in meta["config"]:
        if c not in out:
            out.append(c)
    return out


def seed_mean(R, meta):
    """-> (Rbar[config, lesion], configs in first-seen order)."""
    cfgs = configs(meta)
    cfg = np.array(meta["config"])
    return np.stack([R[cfg == c].mean(0) for c in cfgs]), cfgs


def erased_seedmean(Rbar, thr):
    return (Rbar < thr).mean(1)


def erased_perseed(R, meta, cfgs, thr):
    cfg = np.array(meta["config"])
    return np.array([(R[cfg == c] < thr).mean(1).mean() for c in cfgs])


def summary(R, meta, thrs=THRS):
    """Per configuration: mean R, rank among enhancers (1 = most retention), both erasure estimands, seed spread."""
    Rbar, cfgs = seed_mean(R, meta)
    cfg = np.array(meta["config"])
    mean = Rbar.mean(1)
    order = [c for _, c in sorted(((-mean[i], c) for i, c in enumerate(cfgs) if c != NOENH))]
    out = {}
    for i, c in enumerate(cfgs):
        per_seed = R[cfg == c].mean(1)
        out[c] = dict(mean_R=float(mean[i]), rank=None if c == NOENH else order.index(c) + 1,
                      n_seeds=int(len(per_seed)), per_seed_mean_R=per_seed.tolist(),
                      seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                      erased_seedmean={_tkey(t): float((Rbar[i] < t).mean()) for t in thrs},
                      erased_perseed={_tkey(t): float((R[cfg == c] < t).mean(1).mean()) for t in thrs},
                      erased_perseed_sd={_tkey(t): float(np.std((R[cfg == c] < t).mean(1), ddof=1))
                                         if len(per_seed) > 1 else None for t in thrs},
                      per_subject_mean_R={s: float(Rbar[i, meta["sid"] == s].mean()) for s in meta["subjects"]})
    return out


# ---- seed averaging ---------------------------------------------------------------------------------------
STAGE_SPLITS = ("test", "ext_on_hfe", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip")
PAIRED = ("test", "ext_on_hfe", "ext_v5")
FIVE = ("U-Net L1", "U-Net L1+SSIM", "U-Net absolute", "cGAN / pix2pix", "LoHiResGAN", "corpus-calibrated")
ADVERSARIAL = ("cGAN / pix2pix", "LoHiResGAN")


def task6():
    """Seed-averaged summary for every cohort and operator; the acquired arm where a 64 mT scan exists."""
    return {sp: {op: {arm: summary(*load(sp, op, arm)) for arm in (("syn", "acq") if sp in PAIRED else ("syn",))}
                 for op in ("HFC", "HFE")} for sp in STAGE_SPLITS}


def check_e6(t6):
    s = t6["test"]["HFC"]["syn"]
    rank = {c: v["rank"] for c, v in s.items() if v["rank"]}
    first, last = min(rank, key=rank.get), max(rank, key=rank.get)
    gap = {c: abs(s[c]["erased_seedmean"]["0.25"] - s[c]["erased_perseed"]["0.25"]) for c in FIVE}
    sd = {c: s[c]["seed_sd"] for c in FIVE if c not in ADVERSARIAL}
    return dict(
        E6a=dict(value=f"first {first}, last {last}",
                 **{"pass": first == "cGAN / pix2pix" and last == "U-Net L1+SSIM"}),
        E6b=dict(value=f"largest |seed-mean - per-seed| erasure at 0.25: {max(gap.values()) * 100:.1f} pp "
                       f"({max(gap, key=gap.get)})", **{"pass": max(gap.values()) <= 0.05}),
        E6c=dict(value=f"largest seed s.d. of mean R, non-adversarial five-seed: {max(sd.values()):.3f} "
                       f"({max(sd, key=sd.get)})", **{"pass": max(sd.values()) <= 0.02}))


# ---- selection ---------------------------------------------------------------------------------------
ZOO_SFX = {"test": "", "ext_on_hfe": "_ext_on_hfe", "ext_v5": "_ext_v5"}


def subset(R, meta, mask):
    m = dict(meta, **{k: meta[k][mask] for k in ("sid", "r", "c") if k in meta})
    m["subjects"] = [s for s in meta["subjects"] if s in set(m["sid"])]
    return R[:, mask], m


def delta(R, meta, pick, leader, thr=THR):
    """Retention gap R(leader) - R(pick) and erasure gaps E(pick) - E(leader), on matched lesions."""
    def one(R_, meta_):
        Rbar, cfgs = seed_mean(R_, meta_)
        i, j = cfgs.index(pick), cfgs.index(leader)
        Es, Ep = erased_seedmean(Rbar, thr), erased_perseed(R_, meta_, cfgs, thr)
        return dict(dR=float(Rbar[j].mean() - Rbar[i].mean()), dE_seedmean=float(Es[i] - Es[j]),
                    dE_perseed=float(Ep[i] - Ep[j]), R_pick=float(Rbar[i].mean()), R_leader=float(Rbar[j].mean()),
                    E_pick_seedmean=float(Es[i]), E_leader_seedmean=float(Es[j]),
                    E_pick_perseed=float(Ep[i]), E_leader_perseed=float(Ep[j]))
    out = dict(pick=pick, leader=leader, thr=thr, **one(R, meta))
    out["per_subject"] = {s: one(*subset(R, meta, meta["sid"] == s)) for s in meta["subjects"]}
    return out


def asis_picks(split):
    """(SynthSeg-Dice pick, retention leader) as published (out_public/d3zoo_stats<sfx>_T2.json)."""
    st = json.load(open(f"{OUT}/d3zoo_stats{ZOO_SFX[split]}_T2.json"))["selection"]
    return st["synthseg"]["pick"], st["ret"]["pick"]


def dice_by_config(split):
    """SynthSeg Dice on the acquired 64 mT input, per configuration (seeds averaged) and subject."""
    z = json.load(open(f"{OUT}/d3zoo{ZOO_SFX[split]}_T2.json"))
    s = json.load(open(f"{OUT}/d3zoo_synthseg{ZOO_SFX[split]}_T2.json"))
    out = {}
    for nm, v in s["runs"].items():
        if nm in z:                                          # skips the unenhanced 64 mT input
            out.setdefault(z[nm]["enhancer"], []).append(v["dice"])
    return {c: dict(zip(s["sids"], np.mean(v, 0).tolist())) for c, v in out.items()}


def crossfit(R, meta, dice, thr=THR):
    """Leave one subject out: choose the retention leader and the Dice pick on the others, score on it."""
    Rbar, cfgs = seed_mean(R, meta)
    cand = [c for c in cfgs if c != NOENH and c in dice]
    folds = {}
    for s in meta["subjects"]:
        tr = meta["sid"] != s
        others = [t for t in meta["subjects"] if t != s]
        leader = max(cand, key=lambda c: Rbar[cfgs.index(c), tr].mean())
        pick = max(cand, key=lambda c: np.mean([dice[c][t] for t in others]))
        d = delta(*subset(R, meta, ~tr), pick=pick, leader=leader, thr=thr)
        d.pop("per_subject")
        folds[s] = d
    v = list(folds.values())
    return dict(folds=folds, leaders=[f["leader"] for f in v], picks=[f["pick"] for f in v],
                dR=float(np.mean([f["dR"] for f in v])),
                dE_seedmean=float(np.mean([f["dE_seedmean"] for f in v])),
                dE_perseed=float(np.mean([f["dE_perseed"] for f in v])),
                n_positive_seedmean=int(sum(f["dE_seedmean"] > 0 for f in v)),
                n_positive_perseed=int(sum(f["dE_perseed"] > 0 for f in v)), n_folds=len(v))



def task7(thr=THR):
    out = {}
    for sp in PAIRED:
        pick, leader = asis_picks(sp)
        dice = dice_by_config(sp)
        out[sp] = {op: {arm: dict(asis=delta(*load(sp, op, arm), pick=pick, leader=leader, thr=thr),
                                  crossfit=crossfit(*load(sp, op, arm), dice=dice, thr=thr))
                        for arm in ("syn", "acq")} for op in ("HFC", "HFE")}
    return out


def check_e7(t7):
    a, cf = t7["test"]["HFC"]["syn"]["asis"], t7["test"]["HFC"]["syn"]["crossfit"]
    e7a = a["dE_seedmean"] >= 0.10 and a["dE_perseed"] >= 0.10 and a["dR"] >= 0.05
    e7b = (cf["leaders"].count("cGAN / pix2pix") == cf["n_folds"] and cf["picks"].count("corpus-calibrated") >= 4
           and cf["dE_seedmean"] >= 0.10 and cf["dE_perseed"] >= 0.10
           and cf["n_positive_seedmean"] == cf["n_folds"] and cf["n_positive_perseed"] == cf["n_folds"])
    return dict(
        E7a={"pass": bool(e7a), "value": f"as-is dE {100 * a['dE_seedmean']:.1f} pp (seed-mean), "
                                         f"{100 * a['dE_perseed']:.1f} pp (per-seed), dR {a['dR']:.3f}"},
        E7b={"pass": bool(e7b), "value": f"cross-fitted: pix2pix leads in {cf['leaders'].count('cGAN / pix2pix')}/"
                                         f"{cf['n_folds']} folds, Dice pick corpus-calibrated in "
                                         f"{cf['picks'].count('corpus-calibrated')}/{cf['n_folds']} "
                                         f"(others: {sorted(set(cf['picks']) - {'corpus-calibrated'})}), dE "
                                         f"{100 * cf['dE_seedmean']:.1f} / {100 * cf['dE_perseed']:.1f} pp, positive in "
                                         f"{cf['n_positive_seedmean']} / {cf['n_positive_perseed']} subjects"})


# ---- metric association and leave-family-out ---------------------------------------------------------------------------------------
CORPUS = ("corpus-calibrated", "corpus, uncalibrated", "corpus, physics operator")
PUBLISHED_RHO = {"ssim": 0.34, "psnr": 0.23, "synthseg": 0.25}      # paper line 406, d3zoo_stats_T2.json


def metric_by_model(split, key):
    """Lesion-free image metric per trained model and subject: 'ssim', 'psnr' (d3zoo) or 'synthseg' (Dice)."""
    z = json.load(open(f"{OUT}/d3zoo{ZOO_SFX[split]}_T2.json"))
    if key == "synthseg":
        s = json.load(open(f"{OUT}/d3zoo_synthseg{ZOO_SFX[split]}_T2.json"))
        return {nm: dict(zip(s["sids"], v["dice"])) for nm, v in s["runs"].items() if nm in z}
    return {nm: dict(zip(v["sids"], v[key])) for nm, v in z.items()}


def ret_by_model(R, meta):
    return {nm: {s: float(R[i, meta["sid"] == s].mean()) for s in meta["subjects"]}
            for i, nm in enumerate(meta["models"]) if nm != NOENH}


def _spearman(x, y):
    from scipy.stats import spearmanr
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y)[0])


def rho_boot(met, ret, cfg_of, subjects, B=10000, seed=0):
    """Spearman over configurations (seeds averaged) with a bootstrap resampling configurations, seeds and
    subjects (as d3zoo_stats), and a subject-only bootstrap."""
    models = [nm for nm in met if nm in ret]
    cfgs = sorted({cfg_of[nm] for nm in models})
    idx = {c: [i for i, nm in enumerate(models) if cfg_of[nm] == c] for c in cfgs}
    M = np.array([[met[nm][s] for s in subjects] for nm in models])
    Q = np.array([[ret[nm][s] for s in subjects] for nm in models])
    w0 = np.full(len(subjects), 1 / len(subjects))
    point = lambda w, pick: (np.array([(M[ii] @ w).mean() for ii in pick]), np.array([(Q[ii] @ w).mean() for ii in pick]))
    rho = _spearman(*point(w0, [idx[c] for c in cfgs]))
    rng = np.random.default_rng(seed)
    full, subj = [], []
    for _ in range(B):
        w = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects)) / len(subjects)
        subj.append(_spearman(*point(w, [idx[c] for c in cfgs])))
        cc = rng.integers(len(cfgs), size=len(cfgs))
        pick = [list(rng.choice(idx[cfgs[k]], len(idx[cfgs[k]]))) for k in cc]
        full.append(_spearman(*point(w, pick)))
    full, subj = np.array(full), np.array(subj)
    q = lambda v, a: float(np.nanpercentile(v, a))
    return dict(rho=rho, lo=q(full, 2.5), hi=q(full, 97.5), lo_subj=q(subj, 2.5), hi_subj=q(subj, 97.5),
                n=len(cfgs), B=B, nan_replicates=int(np.isnan(full).sum()))


def leave_family_out(split, op, arm, thr=THR, family=CORPUS):
    """Selection gap with the corpus-pretrained family (which holds the SynthSeg pick) removed. Descriptive."""
    R, meta = load(split, op, arm)
    Rbar, cfgs = seed_mean(R, meta)
    keep = [c for c in cfgs if c != NOENH and c not in family]
    dice = dice_by_config(split)
    pick = max(keep, key=lambda c: np.mean(list(dice[c].values())))
    leader = max(keep, key=lambda c: Rbar[cfgs.index(c)].mean())
    return dict(configs=keep, **delta(R, meta, pick=pick, leader=leader, thr=thr))


def task8(B=10000):
    out = {}
    for sp in PAIRED:
        cfg_of = {nm: v["enhancer"] for nm, v in json.load(open(f"{OUT}/d3zoo{ZOO_SFX[sp]}_T2.json")).items()}
        mets = {k: metric_by_model(sp, k) for k in PUBLISHED_RHO}
        out[sp] = {}
        for op in ("HFC", "HFE"):
            out[sp][op] = {}
            for arm in ("syn", "acq"):
                R, meta = load(sp, op, arm)
                ret = ret_by_model(R, meta)
                out[sp][op][arm] = dict(rho={k: rho_boot(m, ret, cfg_of, meta["subjects"], B=B) for k, m in mets.items()},
                                        leave_family_out=leave_family_out(sp, op, arm))
    return out


def check_e8(t8):
    r = t8["test"]["HFC"]["syn"]["rho"]
    lf = t8["test"]["HFC"]["syn"]["leave_family_out"]
    near = {k: abs(r[k]["rho"] - v) <= 0.2 for k, v in PUBLISHED_RHO.items()}
    zero = {k: r[k]["lo"] < 0 < r[k]["hi"] for k in PUBLISHED_RHO}
    return dict(
        E8a={"pass": all(near.values()) and all(zero.values()),
             "value": ", ".join(f"{k} {r[k]['rho']:+.2f} [{r[k]['lo']:+.2f}, {r[k]['hi']:+.2f}] (published "
                                f"{PUBLISHED_RHO[k]:+.2f})" for k in PUBLISHED_RHO)},
        E8b={"pass": lf["dE_seedmean"] >= 0.05 and lf["dE_perseed"] >= 0.05,
             "value": f"without the corpus family the SynthSeg pick is {lf['pick']}; it erases "
                      f"{100 * lf['dE_seedmean']:.1f} / {100 * lf['dE_perseed']:.1f} pp more than {lf['leader']}"})


# ---- subject bootstrap ---------------------------------------------------------------------------------------
def distinct_resamples(n):
    """Distinct multisets of n subjects drawn with replacement from n: C(2n - 1, n)."""
    from math import comb
    return comb(2 * n - 1, n)


def lesion_weights(meta, chosen):
    from collections import Counter
    k = Counter(chosen)
    return np.array([k.get(s, 0) for s in meta["sid"]], float)


def boot_subjects(stat, meta, B=10000, seed=0):
    """Subject-cluster bootstrap of stat(weights per lesion). Exploratory: with 5 subjects there are only
    126 distinct resamples, so the per-subject values and the sign count are reported beside the interval."""
    subj = list(meta["subjects"])
    rng = np.random.default_rng(seed)
    vals = np.array([stat(lesion_weights(meta, rng.choice(subj, len(subj)))) for _ in range(B)])
    per = {s: stat(lesion_weights(meta, [s])) for s in subj}
    return dict(point=stat(np.ones(len(meta["sid"]))), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), per_subject=per, n_positive=int(sum(v > 0 for v in per.values())),
                n_subjects=len(subj), distinct_resamples=distinct_resamples(len(subj)), B=B, exploratory=True)


def headline_boot(split, op, arm, thr=THR, B=10000):
    """Erasure of the SynthSeg pick, the retention leader and no enhancer, and the gaps, with subject intervals."""
    R, meta = load(split, op, arm)
    Rbar, cfgs = seed_mean(R, meta)
    cfg = np.array(meta["config"])
    pick, leader = asis_picks(split)
    ip, il, i0 = cfgs.index(pick), cfgs.index(leader), cfgs.index(NOENH)
    es = lambda i: (lambda w: float(np.average(Rbar[i] < thr, weights=w)))
    ep = lambda c: (lambda w: float(np.mean([np.average(r < thr, weights=w) for r in R[cfg == c]])))
    mr = lambda i: (lambda w: float(np.average(Rbar[i], weights=w)))
    stats = {f"erased_seedmean|{pick}": es(ip), f"erased_seedmean|{leader}": es(il), f"erased|{NOENH}": es(i0),
             f"erased_perseed|{pick}": ep(pick), f"erased_perseed|{leader}": ep(leader),
             "dE_seedmean": lambda w: es(ip)(w) - es(il)(w), "dE_perseed": lambda w: ep(pick)(w) - ep(leader)(w),
             "dR": lambda w: mr(il)(w) - mr(ip)(w)}
    return dict(pick=pick, leader=leader, **{k: boot_subjects(f, meta, B=B) for k, f in stats.items()})


def map_auroc_boot(B=10000, thr=THR):
    """Retention-map AUROC for erasure (mean of four training seeds), subject-cluster bootstrap."""
    from sklearn.metrics import roc_auc_score
    from exp.lesion_table import row_sid
    seeds = [json.load(open(f"{OUT}/d3detect_T2{t}.json")) for t in ("", "_s1", "_s2", "_s3")]
    sid, _ = row_sid("test")
    meta = dict(sid=np.array([sid[x["row"]] for x in seeds[0]]), subjects=sorted({sid[x["row"]] for x in seeds[0]}))
    y = np.array([x["retention"] < thr for x in seeds[0]])
    score = [np.array([-x["detect"] for x in rows]) for rows in seeds]
    for rows in seeds[1:]:
        assert [x["row"] for x in rows] == [x["row"] for x in seeds[0]]
    stat = lambda w: float(np.mean([roc_auc_score(y[w > 0], s[w > 0], sample_weight=w[w > 0]) for s in score]))
    return boot_subjects(stat, meta, B=B)


def task9(B=10000):
    out = {sp: {op: {arm: headline_boot(sp, op, arm, B=B) for arm in ("syn", "acq")} for op in ("HFC", "HFE")}
           for sp in PAIRED}
    out["map_auroc"] = map_auroc_boot(B=B)
    return out


def check_e9(t9, t7):
    acq = {sp: t7[sp]["HFC"]["acq"]["asis"] for sp in PAIRED}
    e9a = all(v["dE_seedmean"] >= 0.08 and v["dE_perseed"] >= 0.08 for v in acq.values())
    b = t9["test"]["HFC"]["syn"]
    e9b = b["dE_seedmean"]["lo"] > 0 and b["dE_perseed"]["lo"] > 0
    return dict(
        E9a={"pass": bool(e9a), "value": "acquired-arm dE (seed-mean / per-seed): " + ", ".join(
            f"{sp} {100 * v['dE_seedmean']:.1f} / {100 * v['dE_perseed']:.1f} pp" for sp, v in acq.items())},
        E9b={"pass": bool(e9b), "value": f"test dE 95% subject interval [{100 * b['dE_seedmean']['lo']:.1f}, "
                                         f"{100 * b['dE_seedmean']['hi']:.1f}] (seed-mean), [{100 * b['dE_perseed']['lo']:.1f}, "
                                         f"{100 * b['dE_perseed']['hi']:.1f}] (per-seed) pp; exploratory, 126 distinct resamples"})

# ---- driver ---------------------------------------------------------------------------------------
def main():
    res = dict(meta=dict(threshold=THR, thresholds=list(THRS),
                         lesions="d3stage dense plan (seed 11), matched across all models",
                         estimands=dict(erased_seedmean="erasure of the seed-averaged retention",
                                        erased_perseed="mean over seeds of each model's erased fraction")))
    res["task6"] = task6()
    res["task7"] = task7()
    res["task8"] = task8()
    res["task9"] = task9()
    res["expectations"] = {**check_e6(res["task6"]), **check_e7(res["task7"]), **check_e8(res["task8"]),
                           **check_e9(res["task9"], res["task7"])}
    json.dump(res, open(f"{OUT}/coauthor_stats_T2.json", "w"), indent=1)
    for k, v in res["expectations"].items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['value']}")


if __name__ == "__main__":
    main()
