"""Where in the pipeline does a lesion die: in the scanner term D, or in the enhancer G?

Retention R = (output lesion contrast) / (inserted 3T lesion contrast) is a product of two
stages, R_D (3T -> 64 mT, D alone) and R_G|D (64 mT -> output, G given D). d3dfactor.py showed
this for 11 seed-0 enhancers under two operators. This script extends it to the full zoo and
adds what is needed to read it: every seed of every enhancer, six cohorts, a
no-enhancer reference row, the acquired-input arm, subject-level statistics, and a variance
decomposition that says how much of the spread in R comes from the operator, the enhancer, the
seed, the subject or the lesion.

Method. Identical bright lesions (diameter 2r = 4, 6, 8, 12 mm for the code's 1/e^2 radius
r = 2, 3, 4, 6 voxels; contrast 0.12 or 0.25; core b > 0.6, ring 0.05 < b < 0.25) enter the
3T slice at identical sites for every (operator D in {HFC, HFE}) x (model G). Per lesion,
dy, dd, dp are the core-minus-ring contrasts of the inserted lesion, of D(y_l) - D(y) and of
G(D(y_l)) - G(D(y)). Per subject: R = mean(dp/dy), R_D = mean(dd)/mean(dy), R_G|D = mean(dp)/mean(dd)
(the d3dfactor conventions, so R is not exactly R_D * R_G|D), the erased fraction (R < 0.25)
and, where an acquired 64 mT scan exists, R_acq: the lesion enters as x_real + [D(y_l) - D(y)]
near the lesion (d3detreal.lesion_input). The no-enhancer row is G = identity on the centre
slice, so its R_G|D is 1 by construction and its R must equal R_D. Spill-over is the energy of
dp outside the lesion footprint (b > 0.05, dilated 2 px, inside the brain) over the energy
in the core. Intervals are subject bootstraps. The ranking Spearman between the two operators
is computed on enhancer means (seeds averaged) with a subject bootstrap.

Two sampling modes: `pooled` reproduces d3dfactor's sample (96 random rows, 4 sites, seed 11)
for the sanity check against out_public/d3dfactor_T2.json; `dense` (default) draws rows per
subject so that subject-level means are stable. The zoo here has 12 enhancers / 36 models; the
three Mean-Teacher enhancers of the 15-configuration registry need the unpaired 64 mT pool,
which was withdrawn, so they are absent.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3stage.py run [--mode dense|pooled]
    python exp/d3stage.py summary      # JSON + figure from the saved lesion rows
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
sys.path.insert(0, ".")
from ulfkit import data as D, train as T
from exp.d3detect import site_pool, blobs
from exp.d3detreal import lesion_input
from exp.d3dfactor import load_op, RADII, CS
from exp import d3zoo as Z

OUT = os.environ.get("ULF_OUT", "./out")
FIG = os.environ.get("ULF_FIG", "./fig")
OPS = {"HFC": f"{OUT}/d1_forward_T2.pt", "HFE": f"{OUT}_hfe/d1_forward_T2.pt"}
SPLITS = ["test", "ext_on_hfe", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"]
NOENH = "no enhancer"
THR = 0.25


class Ident(nn.Module):
    """G = identity on the centre slice: the output is the 64 mT image itself."""
    def forward(self, x):
        return x[:, 1:2]


def registry_all():
    """-> [(run name, enhancer, seed, ckpt, kind)] for every model whose checkpoint exists."""
    reg = [r for r in Z.registry() if r[4] == "diff" or os.path.exists(f"{OUT}/{r[3]}.pt")]
    return reg


def make_plan(coh, mode, seed=11):
    """Lesion sites, fixed once per cohort so that every model and operator sees the same ones.
    -> [(rows, [(slice-in-batch, cy, cx, r, c)])], batches of 8 rows."""
    rng = np.random.default_rng(seed)
    if mode == "pooled":                      # d3dfactor.run, draw for draw
        rows = np.sort(rng.choice(len(coh.hf), min(96, len(coh.hf)), replace=False))
    else:
        k = int(min(24, max(8, 280 // len(coh.subjects))))
        rr = np.random.default_rng(seed + 1)
        pick = [rr.choice(coh.subject_rows(s), min(k, len(coh.subject_rows(s))), replace=False)
                for s in coh.subjects]
        rows = np.sort(np.concatenate(pick))
    plan = []
    for i in range(0, len(rows), 8):
        rb = rows[i:i + 8]
        y = torch.from_numpy(coh.hf[rb].astype(np.float32)).cuda()
        ok = site_pool(y)
        jobs = []
        for s in range(len(y)):
            cand = torch.nonzero(ok[s])
            if len(cand) < 8: continue
            for _ in range(4):
                cy, cx = cand[rng.integers(len(cand))].tolist()
                jobs.append((s, cy, cx, int(rng.choice(RADII)), float(rng.choice(CS))))
        if jobs:
            plan.append((rb, jobs))
    return plan


def subject_index(coh):
    sid = np.empty(len(coh.hf), int)
    for k, s in enumerate(coh.man):
        sid[s["slice_start"]:s["slice_start"] + s["slice_count"]] = k
    return sid


def con(d, core, ring):
    """Mean core minus mean ring of d (n,1,H,W), per lesion."""
    cf, rf = core.float(), ring.float()
    return ((d * cf).sum((1, 2, 3)) / cf.sum((1, 2, 3)) - (d * rf).sum((1, 2, 3)) / rf.sum((1, 2, 3)))


@torch.no_grad()
def evaluate(coh, plan, ops, nets, with_acq):
    """-> meta (per lesion, shared), dd[op], and per (op, model): dp, dpa, spill (np arrays)."""
    rec = {(o, nm): dict(dp=[], dpa=[], spill=[]) for o in ops for nm, _ in nets}
    dd_rec = {o: [] for o in ops}
    meta = dict(row=[], r=[], c=[], cy=[], cx=[], dy=[], good=[])
    for rb, jobs in plan:
        y = torch.from_numpy(coh.hf[rb].astype(np.float32)).cuda()
        xr = torch.from_numpy(coh.ulf[rb].astype(np.float32)).cuda() if with_acq else None
        si = torch.tensor([j[0] for j in jobs], device="cuda")
        cyt = torch.tensor([j[1] for j in jobs], device="cuda")
        cxt = torch.tensor([j[2] for j in jobs], device="cuda")
        rt = torch.tensor([j[3] for j in jobs], device="cuda")
        b = blobs(cyt, cxt, rt) * (y[si] > 0).float()
        c = torch.tensor([j[4] for j in jobs], device="cuda")[:, None, None, None]
        y0 = y[si]; yl = (y0 + c * b).clamp(0, 1)
        core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
        good = (core.sum((1, 2, 3)) >= 4) & (ring.sum((1, 2, 3)) >= 8)
        dy = con(yl - y0, core, ring)
        fp = F_.max_pool2d((b > 0.05).float(), 5, 1, 2) > 0
        outside = (y0 > 0) & ~fp
        for k, j in enumerate(jobs):
            meta["row"].append(int(rb[j[0]])); meta["r"].append(j[3]); meta["c"].append(j[4])
            meta["cy"].append(j[1]); meta["cx"].append(j[2])
        meta["dy"] += dy.cpu().tolist(); meta["good"] += good.cpu().tolist()
        for o, f in ops.items():
            x0, xl = f(y0), f(yl)
            dd_rec[o] += con((xl - x0)[:, 1:2], core, ring).cpu().tolist()
            if with_acq:
                xa0 = xr[si]
                xal = lesion_input(xa0, y0, yl, lambda v, g: f(v), None, cyt, cxt, rt)
            for nm, net in nets:
                p0, pl, a0, al = [], [], [], []
                for k0 in range(0, len(jobs), 12):
                    sl = slice(k0, k0 + 12)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p0.append(net(x0[sl]).float().clamp(0, 1)); pl.append(net(xl[sl]).float().clamp(0, 1))
                        if with_acq:
                            a0.append(net(xa0[sl]).float().clamp(0, 1)); al.append(net(xal[sl]).float().clamp(0, 1))
                dp = torch.cat(pl) - torch.cat(p0)
                R = rec[(o, nm)]
                R["dp"] += con(dp, core, ring).cpu().tolist()
                R["spill"] += (((dp ** 2) * outside).sum((1, 2, 3)) / (dp ** 2 * core).sum((1, 2, 3)).clamp(min=1e-12)).cpu().tolist()
                if with_acq:
                    R["dpa"] += con(torch.cat(al) - torch.cat(a0), core, ring).cpu().tolist()
    meta = {k: np.array(v) for k, v in meta.items()}
    dd = {o: np.array(v, np.float32) for o, v in dd_rec.items()}
    rec = {k: {kk: np.array(vv, np.float32) for kk, vv in v.items()} for k, v in rec.items()}
    return meta, dd, rec


def run(splits, mode, tag, chunk=6):
    T.seed_all(0)
    assert all(os.path.exists(p) for p in OPS.values()), OPS
    ops = {o: load_op(p) for o, p in OPS.items()}
    reg = registry_all()
    if mode == "pooled":                       # sanity run: the 11 seed-0 enhancers of d3dfactor
        reg = [r for r in reg if r[2] == 0 and r[4] != "diff"]
    store = {}
    for split in splits:
        if os.path.exists(f"{OUT}/d3stage_rows{tag}_{split}_T2.npz"):
            print(f"  {split}: rows exist, skipped"); continue
        coh = D.PairedCohort(split, "T2")
        with_acq = coh.ulf is not None
        plan = make_plan(coh, mode)
        sidx = subject_index(coh)
        print(f"\n== {split} ({mode}): {len(coh.subjects)} subjects, "
              f"{sum(len(j) for _, j in plan)} lesion sites, acquired arm {with_acq}", flush=True)
        allrec, ddall, meta = {}, None, None
        todo = [("ident", None)] + [(r[0], r) for r in reg]
        for c0 in range(0, len(todo), chunk):
            part = todo[c0:c0 + chunk]
            nets = [(NOENH, Ident().cuda()) if r is None else (r[0], Z.load(r[4], r[3])) for _, r in part]
            m, dd, rec = evaluate(coh, plan, ops, nets, with_acq)
            if meta is None: meta, ddall = m, dd
            allrec.update(rec)
            print(f"  chunk {c0 // chunk + 1}/{(len(todo) + chunk - 1) // chunk} done", flush=True)
            del nets; torch.cuda.empty_cache()
        store[split] = dict(meta=meta, dd=ddall, rec=allrec, sid=sidx[meta["row"]],
                            subjects=coh.subjects, with_acq=with_acq,
                            models={**{NOENH: (NOENH, 0, "identity")}, **{r[0]: (r[1], r[2], r[3]) for r in reg}})
        save_rows(store[split], split, tag)
        del plan, store[split]; torch.cuda.empty_cache()


def save_rows(s, split, tag):
    """One npz per cohort: per-lesion arrays (the 'site rows'), float32."""
    z = {f"meta_{k}": v for k, v in s["meta"].items()}
    z["sid_idx"] = s["sid"]
    for o, v in s["dd"].items():
        z[f"dd|{o}"] = v
    for (o, nm), R in s["rec"].items():
        for k, v in R.items():
            if len(v): z[f"{k}|{o}|{nm}"] = v
    np.savez_compressed(f"{OUT}/d3stage_rows{tag}_{split}_T2.npz", **z)
    json.dump(dict(subjects=s["subjects"], with_acq=s["with_acq"], models=s["models"]),
              open(f"{OUT}/d3stage_rows{tag}_{split}_T2.meta.json", "w"))


def load_rows(split, tag=""):
    z = np.load(f"{OUT}/d3stage_rows{tag}_{split}_T2.npz")
    m = json.load(open(f"{OUT}/d3stage_rows{tag}_{split}_T2.meta.json"))
    return z, m


# ---- summary -------------------------------------------------------------------------------

def stage_stats(dy, dd, dp, dpa=None):
    """The d3dfactor conventions on one group of lesions."""
    out = dict(n=int(len(dy)), R=float(np.mean(dp / dy)), R_D=float(dd.mean() / dy.mean()),
               R_GD=float(dp.mean() / dd.mean()), erased=float(np.mean(dp / dy < THR)))
    if dpa is not None:
        out.update(R_acq=float(np.mean(dpa / dy)), erased_acq=float(np.mean(dpa / dy < THR)))
    return out


def ci(v, B=2000, seed=0):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_subject_mean(mat, B=2000, seed=0):
    """mat (subjects,) -> mean and 95% subject-bootstrap CI."""
    rng = np.random.default_rng(seed); n = len(mat)
    reps = [mat[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(mat.mean()), ci(reps)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def variance_decomp(A, subj, r, c, models, enh):
    """Balanced ANOVA sums of squares of site-level R. A[op, model, lesion]."""
    A = A.astype(np.float64)
    gm = A.mean(); tot = ((A - gm) ** 2).sum(); nO, nM, nL = A.shape
    out = {}
    out["operator"] = nM * nL * ((A.mean((1, 2)) - gm) ** 2).sum()
    mm = A.mean((0, 2))                                   # model means
    en = np.array(enh); ue = sorted(set(enh))
    ss_model = nO * nL * ((mm - gm) ** 2).sum()
    ss_enh = 0.0
    for e in ue:
        ix = np.where(en == e)[0]
        ss_enh += len(ix) * nO * nL * (mm[ix].mean() - gm) ** 2
    out["enhancer"] = ss_enh; out["seed (within enhancer)"] = ss_model - ss_enh
    lm = A.mean((0, 1))                                   # lesion means
    us = np.unique(subj)
    sm = np.array([lm[subj == s].mean() for s in us]); sub_ct = np.array([(subj == s).sum() for s in us])
    out["subject"] = nO * nM * float((sub_ct * (sm - gm) ** 2).sum())
    lc = lm - np.array([sm[list(us).index(s)] for s in subj])   # within-subject lesion effect
    rr = np.array([lc[r == v].mean() for v in sorted(set(r))]); vals = sorted(set(r))
    lc2 = lc - np.array([rr[vals.index(v)] for v in r])
    ss_size = nO * nM * sum((r == v).sum() * lc[r == v].mean() ** 2 for v in vals)
    cc = sorted(set(c))
    ss_con = nO * nM * sum((c == v).sum() * lc2[c == v].mean() ** 2 for v in cc)
    ss_les_all = nO * nM * ((lm - gm) ** 2).sum()
    out["lesion size"] = ss_size; out["lesion contrast"] = ss_con
    out["other lesion-level (site)"] = ss_les_all - out["subject"] - ss_size - ss_con
    # operator x enhancer interaction (on enhancer means)
    cell = np.array([[A[o][en == e].mean() for e in ue] for o in range(nO)])
    wts = np.array([(en == e).sum() for e in ue], float)
    ie = 0.0
    for o in range(nO):
        for k, e in enumerate(ue):
            mo = A[o].mean(); me = A[:, en == e].mean()
            ie += nL * wts[k] * (cell[o, k] - mo - me + gm) ** 2
    out["operator x enhancer"] = ie
    out["other interactions and residual"] = tot - sum(out.values())
    return {k: float(v / tot) for k, v in out.items()}, float(tot)


def summary(tag="", out_json=None):
    reg = {}
    P = {}
    for split in SPLITS:
        if not os.path.exists(f"{OUT}/d3stage_rows{tag}_{split}_T2.npz"):
            continue
        z, m = load_rows(split, tag)
        P[split] = (z, m)
    res = dict(meta=dict(
        lesion_convention="r_vox is the 1/e^2 radius in 1 mm voxels; diam_mm = 2*r_vox",
        r_vox=list(RADII), diam_mm=[2 * r for r in RADII], contrasts=list(CS), erased_threshold=THR,
        operators={"HFC": OPS["HFC"], "HFE": OPS["HFE"]},
        zoo_note="12 enhancers / 36 models; the 3 Mean-Teacher enhancers of the 15-configuration "
                 "registry are absent (they need the withdrawn unpaired 64 mT pool).",
        lesion_aware_T2_reference="none exists (d2_bothtoT1_aug* are T1 models); not trained in this round",
        negative_controls="base_unet_mismatched_T2.pt is absent from out_public; not trained in this round"))
    res["splits"] = {}
    for split, (z, m) in P.items():
        sids = np.array(m["subjects"]); ns = len(sids)
        sidx = z["sid_idx"]; good = z["meta_good"].astype(bool)
        dy = z["meta_dy"]; r = z["meta_r"]; c = z["meta_c"]
        keep = good & (np.abs(dy) > 0)
        models = m["models"]
        sp = dict(n_lesions=int(keep.sum()), n_subjects=ns, subjects=sids.tolist(), with_acq=m["with_acq"],
                  per_model={}, per_enhancer={})
        for o in OPS:
            dd = z[f"dd|{o}"]
            sp["per_model"][o] = {}
            for nm, (enh, seed, ck) in models.items():
                dp = z[f"dp|{o}|{nm}"]
                dpa = z.get(f"dpa|{o}|{nm}") if f"dpa|{o}|{nm}" in z.files else None
                spill = z[f"spill|{o}|{nm}"]
                pooled = stage_stats(dy[keep], dd[keep], dp[keep], None if dpa is None else dpa[keep])
                per_sub = {}
                for k, s in enumerate(sids):
                    q = keep & (sidx == k)
                    if q.sum() >= 5:
                        per_sub[s] = stage_stats(dy[q], dd[q], dp[q], None if dpa is None else dpa[q])
                rd = dp[keep] / dy[keep]
                ent = dict(enhancer=enh, seed=seed, ckpt=ck, pooled=pooled, per_subject=per_sub,
                           frac_R_gt_1=float(np.mean(rd > 1)), frac_R_gt_1p5=float(np.mean(rd > 1.5)),
                           spill_median=float(np.median(spill[keep])), spill_mean=float(np.mean(spill[keep])))
                sp["per_model"][o][nm] = ent
        # enhancer-level (seeds averaged), subject-level values, with subject-bootstrap CIs
        enhs = sorted({v[0] for v in models.values()}, key=lambda e: (e != NOENH, e))
        keys = ["R", "R_D", "R_GD", "erased", "R_acq", "erased_acq"]
        mats = {}
        for o in OPS:
            for e in enhs:
                nms = [nm for nm, v in models.items() if v[0] == e]
                for key in keys:
                    if key in ("R_acq", "erased_acq") and not m["with_acq"]: continue
                    mat = np.array([[sp["per_model"][o][nm]["per_subject"].get(s, {}).get(key, np.nan) for s in sids]
                                    for nm in nms])
                    mats[(o, e, key)] = np.nanmean(mat, 0)       # (subjects,) seed-averaged
        sp["subject_ids_used"] = sids.tolist()
        sp["per_enhancer"] = {o: {} for o in OPS}
        for o in OPS:
            for e in enhs:
                d = dict(n_models=sum(v[0] == e for v in models.values()))
                for key in keys:
                    if (o, e, key) in mats:
                        v = mats[(o, e, key)]; v = v[~np.isnan(v)]
                        mu, lo = boot_subject_mean(v)
                        d[key] = dict(mean=mu, ci=lo)
                nms = [nm for nm, v in models.items() if v[0] == e]
                d["frac_R_gt_1"] = float(np.mean([sp["per_model"][o][nm]["frac_R_gt_1"] for nm in nms]))
                d["frac_R_gt_1p5"] = float(np.mean([sp["per_model"][o][nm]["frac_R_gt_1p5"] for nm in nms]))
                d["spill_median"] = float(np.mean([sp["per_model"][o][nm]["spill_median"] for nm in nms]))
                sp["per_enhancer"][o][e] = d
        # ranking agreement between the operators, subject bootstrap over enhancer means
        zoo = [e for e in enhs if e != NOENH]
        for key in ("R", "R_GD"):
            a = np.array([mats[("HFC", e, key)] for e in zoo]); b = np.array([mats[("HFE", e, key)] for e in zoo])
            ok = ~(np.isnan(a).any(0) | np.isnan(b).any(0)); a, b = a[:, ok], b[:, ok]
            rng = np.random.default_rng(0); reps = []
            for _ in range(10000):
                ix = rng.integers(0, a.shape[1], a.shape[1])
                reps.append(spearman(a[:, ix].mean(1), b[:, ix].mean(1)))
            reps = np.array(reps)
            sp[f"rank_rho_{key}_HFC_vs_HFE"] = dict(
                rho=spearman(a.mean(1), b.mean(1)), ci=[float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))],
                n_enhancers=len(zoo), n_subjects=int(a.shape[1]))
        # variance decomposition on site-level R (crossed operator x model x lesion)
        ml = [nm for nm in models if nm != NOENH]
        A = np.stack([np.stack([z[f"dp|{o}|{nm}"][keep] / dy[keep] for nm in ml]) for o in OPS])
        vd, tot = variance_decomp(A, sidx[keep], r[keep], c[keep], ml, [models[nm][0] for nm in ml])
        sp["variance_decomposition"] = dict(fractions=vd, total_ss=tot, n_models=len(ml), n_lesions=int(keep.sum()),
                                            note="balanced ANOVA SS of site-level R; sequential for subject, size, contrast")
        # reference checks
        idn = sp["per_model"]
        sp["identity_check"] = {o: dict(R_noenh=idn[o][NOENH]["pooled"]["R"], R_D=idn[o][NOENH]["pooled"]["R_D"],
                                        R_GD=idn[o][NOENH]["pooled"]["R_GD"],
                                        abs_diff=abs(idn[o][NOENH]["pooled"]["R"] - idn[o][NOENH]["pooled"]["R_D"]))
                                for o in OPS}
        res["splits"][split] = sp
    json.dump(res, open(out_json or f"{OUT}/d3stage_T2.json", "w"), indent=1)
    return res


def sanity(tag="_pooled"):
    """Pooled-mode rows against out_public/d3dfactor_T2.json (same sites, same 11 seed-0 enhancers).
    Pass: R_D to 1e-4 (the operator is deterministic), R and R_G|D to 0.005, erased to 0.015
    (a few lesions near the 0.25 threshold flip with bf16 batch composition, mainly LoHiResGAN)."""
    ref = json.load(open(f"{OUT}/d3dfactor_T2.json"))
    worst = dict(R=0.0, R_D=0.0, R_GD=0.0, erased=0.0); cells = 0; idn_rows = {}
    for split, rs in ref.items():
        z, m = load_rows(split, tag)
        dy = z["meta_dy"]; keep = z["meta_good"].astype(bool)
        for o in OPS:
            dd = z[f"dd|{o}"]
            for nm, (enh, seed, ck) in m["models"].items():
                if seed != 0 or enh not in rs["enhancers"]: continue
                mine = stage_stats(dy[keep], dd[keep], z[f"dp|{o}|{nm}"][keep])
                old = rs["per"][f"{o}|{enh}"]
                assert mine["n"] == old["n"], (split, o, enh, mine["n"], old["n"])
                for k in worst: worst[k] = max(worst[k], abs(mine[k] - old[k]))
                cells += 1
            idn = stage_stats(dy[keep], dd[keep], z[f"dp|{o}|{NOENH}"][keep])
            idn_rows[f"{split}|{o}"] = dict(R=idn["R"], R_D=idn["R_D"], R_GD=idn["R_GD"],
                                            R_D_d3dfactor=rs["per"][f"{o}|{rs['enhancers'][0]}"]["R_D"])
            print(f"  {split:11s} {o}: identity R {idn['R']:.4f}  R_G|D {idn['R_GD']:.4f}  R_D {idn['R_D']:.4f}  "
                  f"(d3dfactor R_D {idn_rows[f'{split}|{o}']['R_D_d3dfactor']:.4f})")
    ok = worst["R_D"] < 1e-4 and worst["R"] < 0.005 and worst["R_GD"] < 0.005 and worst["erased"] < 0.015
    print(f"  {cells} cells; max |mine - d3dfactor|: {worst}  -> {'PASS' if ok else 'FAIL'}")
    json.dump(dict(max_abs_diff=worst, n_cells=cells, passed=bool(ok), identity=idn_rows),
              open(f"{OUT}/d3stage_sanity_T2.json", "w"), indent=1)
    return ok


def figure(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = [s for s in ("test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip") if s in res["splits"]]
    fig, axes = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 4.6), sharey=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        pe = res["splits"][split]["per_enhancer"]
        enhs = [e for e in pe["HFC"] if e != NOENH]
        enhs.sort(key=lambda e: pe["HFC"][e]["R"]["mean"])
        yy = np.arange(len(enhs))
        for o, mk, col, off in (("HFC", "o", "#0072B2", -0.15), ("HFE", "s", "#D55E00", 0.15)):
            ax.barh(yy + off, [pe[o][NOENH]["R_D"]["mean"]] * len(enhs), height=0.25, color=col, alpha=0.18,
                    label=f"R_D {o}")
            ax.scatter([pe[o][e]["R_GD"]["mean"] for e in enhs], yy + off, marker="x", color=col, s=22,
                       label=f"R_G|D {o}")
            ax.scatter([pe[o][e]["R"]["mean"] for e in enhs], yy + off, marker=mk, color=col, s=26,
                       label=f"R {o}")
        ax.axvline(0.25, color="grey", lw=0.6, ls=":")
        ax.set_yticks(yy); ax.set_yticklabels(enhs if ax is np.atleast_1d(axes)[0] else [], fontsize=7)
        ax.set_title(split + f" (n={res['splits'][split]['n_subjects']})", fontsize=9)
        ax.set_xlabel("retention", fontsize=8); ax.set_xlim(0, 0.9)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].legend(fontsize=5.5, loc="lower right", ncol=1)
    fig.suptitle("Stage-wise retention: shaded bar R_D (no enhancer), x R_G|D, markers R", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "sanity"])
    ap.add_argument("--mode", default="dense", choices=["dense", "pooled"])
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    a = ap.parse_args()
    if a.cmd == "run":
        tag = "" if a.mode == "dense" else "_pooled"
        run(a.splits, a.mode, tag)
    elif a.cmd == "sanity":
        sanity()
    elif a.cmd == "summary":
        res = summary()
        figure(res, f"{FIG}/stage_retention.pdf")
        print("wrote", f"{OUT}/d3stage_T2.json")
