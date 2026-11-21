"""Specificity, lesion-conditioned: does inserting a lesion make the enhancer draw focal
structure somewhere else?

A usable specificity measure is needed. The blob-count false-lesion rate of
d3spec.py is dominated by anatomy, and the spill ratio of d3stage.py divides by the core
response, so it grows when the lesion is erased. This script measures the off-lesion response
on its own scale, against the contrast that was inserted.

Method. Same lesions and sites as d3stage.py (make_plan, dense mode, seed 11), both operators
D in {HFC, HFE}, every model of the zoo and the no-enhancer row (G = identity on D's output).
Synthetic arm only, full fp32 (fp32_everywhere). Per lesion, dp = G(D(y_l)) - G(D(y)) is
scanned with a zero-sum matched filter of the lesion's own size: mean over the core (b > 0.6)
minus mean over the ring (0.05 < b < 0.25), the same contrast that defines R, so the filter
at the lesion centre returns the lesion's own dp (checked: meta centre_chk). Its largest
same-sign (pos) and opposite-sign (neg) values are taken over centres far from the lesion
(distance > 2 (1.22 r + 2) px, so the template cannot overlap the footprint) whose template
lies fully inside the brain, and divided by the inserted core-minus-ring contrast dy.

    off-lesion false-structure rate = fraction of lesions with pos / dy >= 0.25

i.e. inserting the lesion made the output show, somewhere else, a structure of the same size
and sign with at least a quarter of the inserted contrast (0.25 is the erasure threshold).
A diffuse change does not count, because the filter is zero-sum. box_far (largest |3x3 mean|
of dp in the same region) is kept to show the diffuse part. The core contrast dp is recomputed
and checked against the d3stage rows (bf16 there, fp32 here: differences of a few 1e-3).

Expectations recorded before any scoring (2026-11-19), for a first version that used the
3x3-box peak instead of the matched filter:
  E1  no enhancer (D alone): rate <= 1% in every cohort and operator.
  E2  every enhancer: rate <= 5% on the test split under D_HFC.
  E3  across enhancers (seeds averaged), Spearman(rate, mean R) > 0 on the test split, i.e. the
      enhancers that keep more lesion contrast also draw more of it elsewhere.
Provenance: a smoke test of the box version (96 test lesions) gave 27% for D alone because D
itself spreads the lesion response 9-14 px, so the measure became the matched filter. A second
smoke test (128 test lesions; no enhancer, U-Net L1+SSIM, pix2pix, LoHiResGAN, diffusion) was
seen before the full run: 1-2%, 0-1%, 22-27%, 24-33%, 28%. The expectations were not changed.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3offlesion.py run
    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3offlesion.py summary
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, ".")
from ulfkit import data as D, train as T
from exp.d3detect import blobs
from exp.d3dfactor import load_op, RADII, CS
from exp import d3zoo as Z
from exp.d3stage import (OPS, SPLITS, NOENH, Ident, registry_all, make_plan, subject_index, con,
                         load_rows, boot_subject_mean, spearman)

OUT = os.environ.get("ULF_OUT", "./out")
THR = 0.25
FAR_K = 2.0


def templates():
    """Zero-sum matched filters, one per radius: mean over the core (b > 0.6) minus mean over
    the ring (0.05 < b < 0.25) of a lesion of that radius, i.e. the contrast that defines R."""
    out = {}
    for r in RADII:
        h = int(np.ceil(1.22 * r)) + 1
        yy, xx = torch.meshgrid(torch.arange(-h, h + 1), torch.arange(-h, h + 1), indexing="ij")
        b = torch.exp(-2.0 * (yy ** 2 + xx ** 2).float() / r ** 2)
        core, ring = (b > 0.6).float(), ((b > 0.05) & (b < 0.25)).float()
        out[r] = ((core / core.sum() - ring / ring.sum())[None, None].cuda(), 2 * h + 1)
    return out


@torch.no_grad()
def evaluate(coh, plan, ops, nets):
    """Per lesion and (operator, model): the core contrast dp (as in d3stage), the largest
    same-sign (pos) and opposite-sign (neg) matched-filter response of dp at centres far from
    the lesion, the distance of the pos peak, and the far 3x3-box peak (diffuse change)."""
    tpl = templates()
    rec = {(o, nm): dict(dp=[], pos=[], neg=[], pos_dist=[], box_far=[]) for o in ops for nm, _ in nets}
    meta = dict(row=[], r=[], c=[], dy=[], good=[], centre_chk=[])
    for rb, jobs in plan:
        y = torch.from_numpy(coh.hf[rb].astype(np.float32)).cuda()
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
        H, W = y0.shape[-2:]
        gy = torch.arange(H, device="cuda").view(1, 1, H, 1).float()
        gx = torch.arange(W, device="cuda").view(1, 1, 1, W).float()
        dist = ((gy - cyt.view(-1, 1, 1, 1)) ** 2 + (gx - cxt.view(-1, 1, 1, 1)) ** 2).sqrt()
        rr = rt.view(-1, 1, 1, 1).float()
        far = dist > FAR_K * (1.22 * rr + 2)          # template cannot overlap the lesion footprint
        brain = (y0 > 0).float()
        valid = torch.zeros_like(far)
        for r, (k, K) in tpl.items():                 # template fully inside the brain
            inside = -F_.max_pool2d(-brain, K, 1, K // 2) > 0.5
            valid = torch.where(rr == r, inside & far, valid)
        box_ok = (F_.avg_pool2d(brain, 3, 1, 1) > 0.999) & far
        for j in jobs:
            meta["row"].append(int(rb[j[0]])); meta["r"].append(j[3]); meta["c"].append(j[4])
        meta["dy"] += dy.cpu().tolist(); meta["good"] += good.cpu().tolist()
        ninf = torch.tensor(-1e9, device="cuda")
        first = True
        for o, f in ops.items():
            x0, xl = f(y0), f(yl)
            for nm, net in nets:
                p0, pl = [], []
                for k0 in range(0, len(jobs), 12):
                    sl = slice(k0, k0 + 12)
                    p0.append(net(x0[sl]).float().clamp(0, 1)); pl.append(net(xl[sl]).float().clamp(0, 1))
                dp = torch.cat(pl) - torch.cat(p0)
                mf = torch.zeros_like(dp)
                for r, (k, K) in tpl.items():
                    mf = torch.where(rr == r, F_.conv2d(dp, k, padding=K // 2), mf)
                if first:                             # matched filter at the centre == con(dp)
                    at = mf[torch.arange(len(jobs)), 0, cyt, cxt]
                    meta["centre_chk"] += (at - con(dp, core, ring)).abs().cpu().tolist()
                    first = False
                pv = torch.where(valid, mf, ninf).flatten(1)
                nv = torch.where(valid, -mf, ninf).flatten(1)
                pmax, parg = pv.max(1)
                box = F_.avg_pool2d(dp, 3, 1, 1)
                R = rec[(o, nm)]
                R["dp"] += con(dp, core, ring).cpu().tolist()
                R["pos"] += pmax.clamp(min=0).cpu().tolist()
                R["neg"] += nv.max(1).values.clamp(min=0).cpu().tolist()
                R["pos_dist"] += dist.flatten(1).gather(1, parg[:, None])[:, 0].cpu().tolist()
                R["box_far"] += torch.where(box_ok, box.abs(), ninf).flatten(1).max(1).values.clamp(min=0).cpu().tolist()
    meta = {k: np.array(v) for k, v in meta.items()}
    rec = {k: {kk: np.array(vv, np.float32) for kk, vv in v.items()} for k, v in rec.items()}
    return meta, rec


def fp32_everywhere():
    """dp is a difference of two outputs and the far peak is a maximum over thousands of
    pixels, so it must not pick up rounding: no bf16 autocast (also inside d3dfactor.load_op and
    the diffusion sampler, hence the patch) and no TF32."""
    import contextlib
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.autocast = lambda *a, **k: contextlib.nullcontext()


def run(splits, chunk=6):
    fp32_everywhere()
    T.seed_all(0)
    ops = {o: load_op(p) for o, p in OPS.items()}
    reg = registry_all()
    for split in splits:
        path = f"{OUT}/d3offlesion_rows_{split}_T2.npz"
        if os.path.exists(path):
            print(f"  {split}: rows exist, skipped"); continue
        coh = D.PairedCohort(split, "T2")
        plan = make_plan(coh, "dense")
        sidx = subject_index(coh)
        print(f"\n== {split}: {len(coh.subjects)} subjects, {sum(len(j) for _, j in plan)} lesion sites", flush=True)
        allrec, meta = {}, None
        todo = [None] + reg
        for c0 in range(0, len(todo), chunk):
            part = todo[c0:c0 + chunk]
            nets = [(NOENH, Ident().cuda()) if r is None else (r[0], Z.load(r[4], r[3])) for r in part]
            m, rec = evaluate(coh, plan, ops, nets)
            if meta is None: meta = m
            allrec.update(rec)
            print(f"  chunk {c0 // chunk + 1}/{(len(todo) + chunk - 1) // chunk} done", flush=True)
            del nets; torch.cuda.empty_cache()
        z = {f"meta_{k}": v for k, v in meta.items()}
        z["sid_idx"] = sidx[meta["row"]]
        for (o, nm), R in allrec.items():
            for k, v in R.items():
                z[f"{k}|{o}|{nm}"] = v
        models = {NOENH: (NOENH, 0, "identity"), **{r[0]: (r[1], r[2], r[3]) for r in reg}}
        np.savez_compressed(path + ".tmp.npz", **z)
        os.replace(path + ".tmp.npz", path)
        json.dump(dict(subjects=coh.subjects, models=models),
                  open(f"{OUT}/d3offlesion_rows_{split}_T2.meta.json", "w"))
        del plan; torch.cuda.empty_cache()


def summary():
    res = dict(meta=dict(threshold=THR, far_factor=FAR_K, smoothing="3x3 box",
                         region="brain interior outside footprint (b > 0.05 dilated 2 px)",
                         far_region="also beyond FAR_K * (1.22 r + 2) px from the lesion centre",
                         lesions="d3stage.make_plan dense, seed 11, synthetic arm"),
               splits={})
    for split in SPLITS:
        p = f"{OUT}/d3offlesion_rows_{split}_T2.npz"
        if not os.path.exists(p): continue
        z = np.load(p); m = json.load(open(p.replace(".npz", ".meta.json")))
        sids = m["subjects"]; sidx = z["sid_idx"]; dy = z["meta_dy"]
        keep = z["meta_good"].astype(bool) & (np.abs(dy) > 0)
        # sanity against the d3stage rows (same lesions, same models)
        chk = None
        if os.path.exists(f"{OUT}/d3stage_rows_{split}_T2.npz"):
            zs, _ = load_rows(split)
            diffs = [float(np.max(np.abs(zs[f"dp|{o}|{nm}"] - z[f"dp|{o}|{nm}"])))
                     for o in OPS for nm in m["models"] if f"dp|{o}|{nm}" in zs.files]
            chk = dict(max_abs_ddp=max(diffs), n_models=len(diffs),
                       same_dy=bool(np.allclose(zs["meta_dy"], dy, atol=1e-6)))
        sp = dict(n_lesions=int(keep.sum()), n_subjects=len(sids), sanity_vs_d3stage=chk,
                  centre_check_max=float(z["meta_centre_chk"][keep].max()), per_enhancer={})
        enhs = sorted({v[0] for v in m["models"].values()}, key=lambda e: (e != NOENH, e))
        for o in OPS:
            sp["per_enhancer"][o] = {}
            for e in enhs:
                nms = [nm for nm, v in m["models"].items() if v[0] == e]
                def per_sub(fn):
                    return np.array([np.mean([fn(nm, keep & (sidx == k)) for nm in nms])
                                     for k in range(len(sids)) if (keep & (sidx == k)).sum() >= 5])
                rate = lambda key: (lambda nm, q: np.mean(z[f"{key}|{o}|{nm}"][q] / dy[q] >= THR))
                ent = {}
                rate5 = lambda nm, q: np.mean(z[f"pos|{o}|{nm}"][q] / dy[q] >= 0.5)
                for name, fn in [("rate", rate("pos")), ("rate_05", rate5), ("rate_neg", rate("neg")),
                                 ("R", lambda nm, q: np.mean(z[f"dp|{o}|{nm}"][q] / dy[q]))]:
                    mean, iv = boot_subject_mean(per_sub(fn))
                    ent[name] = dict(mean=mean, ci=iv)
                allpos = np.concatenate([z[f"pos|{o}|{nm}"][keep] / dy[keep] for nm in nms])
                ent["pos_over_dy_median"] = float(np.median(allpos))
                ent["pos_over_dy_p95"] = float(np.percentile(allpos, 95))
                dist = np.concatenate([z[f"pos_dist|{o}|{nm}"][keep] for nm in nms])
                hit = np.concatenate([z[f"pos|{o}|{nm}"][keep] / dy[keep] >= THR for nm in nms])
                ent["median_dist_px_of_hits"] = float(np.median(dist[hit])) if hit.any() else None
                ent["box_far_over_dy_median"] = float(np.median(np.concatenate(
                    [z[f"box_far|{o}|{nm}"][keep] / dy[keep] for nm in nms])))
                ent["n_models"] = len(nms)
                sp["per_enhancer"][o][e] = ent
            es = [e for e in enhs if e != NOENH]
            sp.setdefault("rho_rate_vs_R", {})[o] = dict(
                all=spearman([sp["per_enhancer"][o][e]["rate"]["mean"] for e in es],
                             [sp["per_enhancer"][o][e]["R"]["mean"] for e in es]),
                neg=spearman([sp["per_enhancer"][o][e]["rate_neg"]["mean"] for e in es],
                             [sp["per_enhancer"][o][e]["R"]["mean"] for e in es]))
        res["splits"][split] = sp
    json.dump(res, open(f"{OUT}/d3offlesion_T2.json", "w"), indent=1)
    for split, sp in res["splits"].items():
        print(f"\n== {split}  n={sp['n_lesions']}  sanity={sp['sanity_vs_d3stage']}")
        for o in OPS:
            print(f"  {o}  rho(rate, R) = {sp['rho_rate_vs_R'][o]['all']:+.2f} (neg {sp['rho_rate_vs_R'][o]['neg']:+.2f})")
            for e, v in sorted(sp["per_enhancer"][o].items(), key=lambda kv: -kv[1]["R"]["mean"]):
                print(f"    {e:28s} R {v['R']['mean']:.3f}  rate {100 * v['rate']['mean']:5.1f}% "
                      f"[{100 * v['rate']['ci'][0]:.1f}, {100 * v['rate']['ci'][1]:.1f}]  >=0.5 {100 * v['rate_05']['mean']:5.1f}%  "
                      f"neg {100 * v['rate_neg']['mean']:5.1f}%  med {v['pos_over_dy_median']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "summary"])
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    a = ap.parse_args()
    if a.stage == "run":
        run(a.splits)
    else:
        summary()
