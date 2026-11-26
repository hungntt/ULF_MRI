"""Erasure under k-scaled D, all 36 models, both arms.

Review B (exp/d1_physcheck.py erase) scaled D's lesion response by the k of each lesion's size bin for the 11
seed-0 configurations, synthetic arm only. Here the d3stage lesions (exp/d3stage.make_plan, dense, seed 11, so
the result matches exp/coauthor_stats.py) pass through every model of the zoo plus no enhancer, with

    synthetic arm   x_k  = clamp(D(y) + k [D(y_l) - D(y)], 0, 1)
    acquired arm    xa_k = x + k [D(y_l) - D(y)] m        (m: radius max(4r, 24) px, as d3detreal.lesion_input)

k = T_real / T_D of the lesion's thickness bin (primary FWHM rule, per-tile alignment, sulcal CSF, point
estimate), from out_public/d1_physcheck_erasure.json, matched unit: operator HFC takes HFC|D_HFC and operator
HFE takes HFE|D_HFE on OpenNeuro (test and ext_on_hfe), V5|D_* on Zenodo. On the test split the k = 1 inputs
are also run, as a reproduction check against out_public/d3stage_rows_test_T2.npz.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/coauthor_kscale.py run [split ...]
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
SPLITS = ("test", "ext_on_hfe", "ext_v5")
K_SPLIT = {"test": "test", "ext_on_hfe": "test", "ext_v5": "ext_v5"}
CHECK_SPLIT = "test"


def rows_path(split):
    return f"{OUT}/coauthor_kscale_rows_{split}_T2.npz"


def k_table():
    """-> {split: {operator: {r: k}}} (r in voxels; diameter 2r mm)."""
    E = json.load(open(f"{OUT}/d1_physcheck_erasure.json"))["splits"]
    return {sp: {op: {int(r): float(v[0]) for r, v in E[K_SPLIT[sp]][op]["k"]["fwhm"].items()}
                 for op in ("HFC", "HFE")} for sp in SPLITS}


def kscaled(x0, xl, k):
    """Lesioned input with the operator's lesion response multiplied by k (one k per lesion)."""
    return (x0 + k.view(-1, 1, 1, 1).to(x0) * (xl - x0)).clamp(0, 1)


@torch.no_grad()
def evaluate(coh, plan, ops, nets, with_acq, kmap, with_k1):
    from exp.d3detect import blobs
    from exp.d3stage import con
    keys = ["dp", "dpa"] + (["dp1", "dpa1"] if with_k1 else [])
    rec = {(o, nm): {k: [] for k in keys} for o in ops for nm, _ in nets}
    kk = {o: [] for o in ops}
    meta = dict(row=[], r=[], c=[], cy=[], cx=[])
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
        near = (blobs(cyt, cxt, torch.clamp(rt.float() * 4, min=24)) > 0.05).float()
        for j in jobs:
            meta["row"].append(int(rb[j[0]])); meta["r"].append(j[3]); meta["c"].append(j[4])
            meta["cy"].append(j[1]); meta["cx"].append(j[2])
        for o, f in ops.items():
            k = torch.tensor([kmap[o][j[3]] for j in jobs], device="cuda")
            kk[o] += k.cpu().tolist()
            x0, xl = f(y0), f(yl)
            ins = {"dp": kscaled(x0, xl, k)}
            if with_k1:
                ins["dp1"] = xl
            if with_acq:
                xa0 = xr[si]
                ins["dpa"] = xa0 + k.view(-1, 1, 1, 1) * (xl - x0) * near
                if with_k1:
                    ins["dpa1"] = xa0 + (xl - x0) * near
            for nm, net in nets:
                def pred(x):
                    out = []
                    for k0 in range(0, len(jobs), 12):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out.append(net(x[k0:k0 + 12]).float().clamp(0, 1))
                    return torch.cat(out)
                base = {"syn": pred(x0), "acq": pred(xa0) if with_acq else None}
                for key, xin in ins.items():
                    p0 = base["acq" if key.startswith("dpa") else "syn"]
                    rec[(o, nm)][key] += con(pred(xin) - p0, core, ring).cpu().tolist()
    meta = {k: np.array(v) for k, v in meta.items()}
    rec = {k: {kk_: np.array(vv, np.float32) for kk_, vv in v.items()} for k, v in rec.items()}
    return meta, {o: np.array(v, np.float32) for o, v in kk.items()}, rec


def run(splits, chunk=6):
    from ulfkit import data as D, train as T
    from exp import d3stage as S, d3zoo as Z
    from exp.d3dfactor import load_op
    T.seed_all(0)
    ops = {o: load_op(p) for o, p in S.OPS.items()}
    reg = S.registry_all()
    kt = k_table()
    for split in splits:
        if os.path.exists(rows_path(split)):
            print(f"  {split}: rows exist, skipped"); continue
        coh = D.PairedCohort(split, "T2")
        with_acq = coh.ulf is not None
        plan = S.make_plan(coh, "dense")
        ref = np.load(f"{OUT}/d3stage_rows_{split}_T2.npz")
        todo = [None] + reg
        out, meta, kk = {}, None, None
        for c0 in range(0, len(todo), chunk):
            part = todo[c0:c0 + chunk]
            nets = [(S.NOENH, S.Ident().cuda()) if r is None else (r[0], Z.load(r[4], r[3])) for r in part]
            m, kk, rec = evaluate(coh, plan, ops, nets, with_acq, kt[split], with_k1=split == CHECK_SPLIT)
            if meta is None:
                meta = m
                for key in ("row", "cy", "cx", "r", "c"):
                    assert np.array_equal(meta[key], ref[f"meta_{key}"]), f"{split}: sites differ from d3stage ({key})"
            for (o, nm), R in rec.items():
                for key, v in R.items():
                    out[f"{key}|{o}|{nm}"] = v
            print(f"  {split}: chunk {c0 // chunk + 1}/{(len(todo) + chunk - 1) // chunk} done", flush=True)
            del nets; torch.cuda.empty_cache()
        z = {f"meta_{k}": v for k, v in meta.items()}
        z.update({f"k|{o}": v for o, v in kk.items()}, **out)
        z["meta_dy"], z["meta_good"], z["sid_idx"] = ref["meta_dy"], ref["meta_good"], ref["sid_idx"]
        np.savez_compressed(rows_path(split), **z)
        json.dump(dict(k=kt[split], k_rule="FWHM 1.18 r, per-tile alignment, sulcal CSF, point estimate",
                       with_acq=with_acq, with_k1=split == CHECK_SPLIT, n=int(len(meta["row"]))),
                  open(rows_path(split).replace(".npz", ".meta.json"), "w"), indent=1)
        print(f"  {split}: saved {rows_path(split)}", flush=True)


def summarise(splits=SPLITS, thr=0.25):
    """k = 1 (d3stage rows) against k-scaled rows: erasure per configuration under both estimands, the ranking
    Spearman on mean R, and the SynthSeg-pick gap, for both operators and both arms."""
    from scipy.stats import spearmanr
    from exp import coauthor_stats as CS
    t = str(thr)
    out = {}
    for sp in splits:
        if not os.path.exists(rows_path(sp)):
            continue
        pick, leader = CS.asis_picks(sp)
        out[sp] = {}
        for op in ("HFC", "HFE"):
            out[sp][op] = {}
            for arm in ("syn", "acq"):
                s1 = CS.summary(*CS.load(sp, op, arm, "stage"))
                Rk, mk = CS.load(sp, op, arm, "kscale")
                sk = CS.summary(Rk, mk)
                enh = [c for c in s1 if c != CS.NOENH]
                per = {c: dict(meanR_k1=s1[c]["mean_R"], meanR_k=sk[c]["mean_R"],
                               erased_k1=s1[c]["erased_perseed"][t], erased_k=sk[c]["erased_perseed"][t],
                               drop_perseed=s1[c]["erased_perseed"][t] - sk[c]["erased_perseed"][t],
                               drop_seedmean=s1[c]["erased_seedmean"][t] - sk[c]["erased_seedmean"][t]) for c in s1}
                d = CS.delta(Rk, mk, pick=pick, leader=leader, thr=thr)
                d.pop("per_subject")
                out[sp][op][arm] = dict(per_config=per, delta=d,
                                        rho_meanR=float(spearmanr([s1[c]["mean_R"] for c in enh], [sk[c]["mean_R"] for c in enh]).statistic),
                                        rho_erased=float(spearmanr([s1[c]["erased_perseed"][t] for c in enh],
                                                                   [sk[c]["erased_perseed"][t] for c in enh]).statistic))
    return out


def check_e10(s):
    v = s["test"]["HFC"]
    drops = {c: p["drop_perseed"] for c, p in v["syn"]["per_config"].items() if c != "no enhancer"}
    ok = (all(0.03 <= d <= 0.15 for d in drops.values()) and v["syn"]["rho_meanR"] >= 0.9
          and all(v[a]["delta"]["dE_seedmean"] >= 0.08 and v[a]["delta"]["dE_perseed"] >= 0.08 for a in ("syn", "acq")))
    return {"pass": bool(ok),
            "value": f"erasure drop per configuration {100 * min(drops.values()):.1f}-{100 * max(drops.values()):.1f} pp "
                     f"(per-seed); rho(mean R, k=1 vs k-scaled) {v['syn']['rho_meanR']:.2f}; dE k-scaled synthetic "
                     f"{100 * v['syn']['delta']['dE_seedmean']:.1f} / {100 * v['syn']['delta']['dE_perseed']:.1f} pp, acquired "
                     f"{100 * v['acq']['delta']['dE_seedmean']:.1f} / {100 * v['acq']['delta']['dE_perseed']:.1f} pp"}


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(sys.argv[2:] or SPLITS)
    else:
        res = summarise()
        res["E10a"] = check_e10(res)
        json.dump(res, open(f"{OUT}/coauthor_kscale_T2.json", "w"), indent=1)
        print(res["E10a"])
