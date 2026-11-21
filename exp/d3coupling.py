"""Is polarity inversion the healthy-tissue T1/T2 coupling, applied to a lesion that breaks it?

Healthy tissue couples T1 and T2 inversely (CSF and oedema: T2-bright, T1-dark). A network that
synthesises 3T T1 with access to 64 mT T2 can learn that coupling instead of reading the T1
input. d3lesiontype.py tested four sign quadrants; here the T1 lesion contrast is swept
continuously, c1 = f * |c2| for f in -1..+1, with T2-bright (c2 > 0) and T2-dark (c2 < 0)
lesions, through routes A (64 mT T1), B (64 mT T2) and C (both), in every cohort.

Per lesion it stores raw contrasts (core minus ring): inserted 3T T1 and T2, and each route's
synthesised-T1 response. Three tests of the mechanism:
  1. the wrong-sign rate as a function of f, per route and T2 sign;
  2. the response of each route fitted as  resp = a * c1 + b * c2  (subject-bootstrap CIs):
     route A should read T1 (a > 0, b ~ 0), route B can only follow the coupling (a ~ 0,
     b < 0), route C mixes them; C then inverts a lesion when b*c2 outweighs a*c1 with the
     opposite sign, i.e. on the side f > -b/a of the predicted boundary;
  3. the coupling itself, measured on the training subjects' 3T images: mean 3T T1 per 3T T2
     band over the whole brain. T1 falls monotonically as T2 rises from white matter to CSF, but
     turns back below the white-matter level, so only T2-BRIGHT foci have a learnable T1 sign.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3coupling.py [splits...]
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, ".")
from ulfkit import data as D, train as T
from exp.d3detect import operator, site_pool, blobs
from exp.d3qa import enhancer
from exp.d3pipe import d2_model, pack

OUT = os.environ.get("ULF_OUT", "./out")
FRACS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
RADII, CS = (2, 3, 4, 6), (0.12, 0.25)


@torch.no_grad()
def sweep(split, nets, f1, f2, n_slices=8, n_sites=4, seed=5, bs=8):
    """-> rows: dict(sid, r, f, s2, c1, c2, A, B, C) with raw core-minus-ring contrasts."""
    mc = D.MultiContrastCohort(split)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    eA, eB, eC = nets
    rows = []
    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        sel = pos[np.linspace(len(pos) // 5, 4 * len(pos) // 5, n_slices).astype(int)]
        _, hf = mc.batch(sel)
        y1 = torch.from_numpy(hf[:, 0:1].astype(np.float32)).cuda()
        y2 = torch.from_numpy(hf[:, 1:2].astype(np.float32)).cuda()
        ok = site_pool(y1) & site_pool(y2)          # mid-intensity in both: no clipping
        x1_0, x2_0 = f1(y1, gen), f2(y2, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p0 = {"A": eA(x1_0).float().clamp(0, 1), "B": eB(pack(x2=x2_0)).float().clamp(0, 1), "C": eC(pack(x1_0, x2_0)).float().clamp(0, 1)}
        jobs = []                                   # the same sites and sizes for every (f, s2)
        for s in range(len(sel)):
            cand = torch.nonzero(ok[s])
            if len(cand) < 8: continue
            for _ in range(n_sites):
                cy, cx = cand[rng.integers(len(cand))].tolist()
                jobs.append((s, cy, cx, int(rng.choice(RADII)), float(rng.choice(CS))))
        for s2 in (+1.0, -1.0):
            for f in FRACS:
                for b0 in range(0, len(jobs), bs):
                    ch = jobs[b0:b0 + bs]
                    si = torch.tensor([t[0] for t in ch], device="cuda")
                    b = blobs(torch.tensor([t[1] for t in ch], device="cuda"),
                              torch.tensor([t[2] for t in ch], device="cuda"),
                              torch.tensor([t[3] for t in ch], device="cuda")) * (y1[si] > 0).float()
                    mag = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                    y1l = (y1[si] + f * mag * b).clamp(0, 1)
                    y2l = (y2[si] + s2 * mag * b).clamp(0, 1)
                    x1l, x2l = f1(y1l, gen), f2(y2l, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pl = {"A": eA(x1l).float().clamp(0, 1), "B": eB(pack(x2=x2l)).float().clamp(0, 1), "C": eC(pack(x1l, x2l)).float().clamp(0, 1)}
                    core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                    con = lambda d, k: float(d[k][core[k]].mean() - d[k][ring[k]].mean())
                    d1, d2 = y1l - y1[si], y2l - y2[si]
                    dp = {k: pl[k] - p0[k][si] for k in pl}
                    for k in range(len(ch)):
                        if core[k].sum() < 4 or ring[k].sum() < 8: continue
                        rows.append(dict(sid=sid, r=ch[k][3], f=f, s2=s2, c1=con(d1, k), c2=con(d2, k),
                                         **{r: con(dp[r], k) for r in dp}))
        print(f"  {split} {sid}: {len(rows)} rows", flush=True)
    return rows


BANDS = ((0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.01))


def coupling_bands(n_rows=200, seed=0):
    """The coupling a network can learn: mean 3T T1 per 3T T2 intensity band over the training
    subjects' brains (both contrasts > 0), with each band's share of brain voxels. (A slope over
    the mid-intensity lesion sites alone is the wrong window: the inverse coupling comes from the
    tissue classes, white matter to CSF.)"""
    mc = D.MultiContrastCohort("train")
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(mc.sid), min(n_rows, len(mc.sid)), replace=False)
    _, hf = mc.batch(np.sort(pos))
    a, b = hf[:, 0].astype(np.float64), hf[:, 1].astype(np.float64)
    m = (a > 0) & (b > 0); a, b = a[m], b[m]
    out = [dict(t2=[lo, hi], t1=float(a[(b >= lo) & (b < hi)].mean()), frac=float(((b >= lo) & (b < hi)).mean()))
           for lo, hi in BANDS]
    return out, float(np.corrcoef(a, b)[0, 1])


def fit(rows, route, n_boot=1000, seed=0):
    """resp = a*c1 + b*c2 (no intercept), with a subject-bootstrap 95% CI."""
    X = np.array([[x["c1"], x["c2"]] for x in rows]); y = np.array([x[route] for x in rows])
    sid = np.array([x["sid"] for x in rows]); us = sorted(set(sid)); ix = {u: np.where(sid == u)[0] for u in us}
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        i = np.concatenate([ix[u] for u in rng.choice(us, len(us))])
        reps.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(reps, 2.5, 0), np.percentile(reps, 97.5, 0)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(a=float(coef[0]), b=float(coef[1]), a_ci=[float(lo[0]), float(hi[0])],
                b_ci=[float(lo[1]), float(hi[1])], r2=float(r2))


def summarise(rows):
    out = {"fit": {r: fit(rows, r) for r in "ABC"}, "wrong": {}}
    for s2 in (+1.0, -1.0):
        for f in FRACS:
            if f == 0: continue
            R = [x for x in rows if x["s2"] == s2 and x["f"] == f and abs(x["c1"]) > 1e-4]
            out["wrong"][f"{s2:+.0f}|{f:+.2f}"] = {r: float(np.mean([np.sign(x[r]) != np.sign(x["c1"]) for x in R])) for r in "ABC"}
    a, b = out["fit"]["C"]["a"], out["fit"]["C"]["b"]
    out["boundary_f"] = float(-b / a) if a else None    # C inverts T2-bright lesions with f beyond this
    return out


def main(splits):
    T.seed_all(5)
    f1, f2 = operator("T1", "learned"), operator("T2", "learned")
    eA = enhancer(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
    nets = (eA, d2_model("T2toT1"), d2_model("bothtoT1"))
    bands, rho = coupling_bands()
    print(f"healthy-tissue coupling on the training subjects (r = {rho:+.2f}): mean 3T T1 by 3T T2 band")
    for q in bands:
        print(f"  T2 {q['t2'][0]:.1f}-{q['t2'][1]:.1f}: T1 {q['t1']:.3f}  ({q['frac']*100:4.1f}% of brain)")
    res = {"coupling": dict(bands=bands, r=rho), "splits": {}}
    for split in splits:
        rows = sweep(split, nets, f1, f2)
        s = summarise(rows)
        res["splits"][split] = s
        json.dump(rows, open(f"{OUT}/d3coupling_rows_{split}.json", "w"))
        print(f"\n== {split}: {len(rows)} lesion responses")
        for r in "ABC":
            q = s["fit"][r]
            print(f"  route {r}: resp = {q['a']:+.3f} [{q['a_ci'][0]:+.3f},{q['a_ci'][1]:+.3f}] * c1 "
                  f"{q['b']:+.3f} [{q['b_ci'][0]:+.3f},{q['b_ci'][1]:+.3f}] * c2   (R^2 {q['r2']:.2f})")
        print(f"  route C predicted inversion boundary for T2-bright lesions: f > {s['boundary_f']:+.2f}")
        print(f"  % wrong sign by f = c1/|c2| (A / B / C):")
        for s2, lab in ((+1.0, "T2-bright"), (-1.0, "T2-dark")):
            cells = []
            for f in FRACS:
                if f == 0: continue
                w = s["wrong"][f"{s2:+.0f}|{f:+.2f}"]
                cells.append(f"{f:+.2f}: {w['A']*100:3.0f}/{w['B']*100:3.0f}/{w['C']*100:3.0f}")
            print(f"    {lab:9s} " + "  ".join(cells))
        json.dump(res, open(f"{OUT}/d3coupling.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"])
