"""Inpainting error propagated into the real-MS results.

The MS lesions were transplanted as a fractional contrast m = (y - y0) / y0 against an inpainted counterpart y0.
On pseudo-lesions (each lesion's shape moved to healthy white matter of the same patient and inpainted the same
way, exp/d3reallesion.py inplace) the inpainting adds a core-minus-ring contrast e * C to a lesion of contrast C,
with e = tru_pseudo / tru_real (457 pseudo-lesions; median -0.09, median |e| 0.24). A measured lesion contrast is
then C_meas = C_true + e C_meas, so the true lesion is the transplanted one scaled by (1 - e).

`run` replays the transplant of exp/d3reallesion.run (synthetic arm, central slice, one lesion per image) for the
12 seed-0 enhancers and no enhancer, with each lesion's m scaled by (1 - e_j), e_j resampled with replacement
from the 457 ratios, independently per lesion, for 20 draws, plus the unperturbed draw 0 as a reproduction check.
`summarise` recomputes the paper's two numbers per draw on the 534 lesions of the published analysis: the
Spearman rho between the real-lesion and matched-Gaussian enhancer rankings (Gaussians unperturbed; they carry
no inpainting error), and the erased fraction of the SynthSeg pick (corpus-calibrated s0).

    ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_SCRATCH=datasets/scratch_public \
        python exp/coauthor_inpaint.py run         # GPU
    ... python exp/coauthor_inpaint.py
"""
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
OUT_NPZ = f"{OUT}/coauthor_inpaint_T2.npz"
OUT_JSON = f"{OUT}/coauthor_inpaint_T2.json"
N_DRAWS = 20
PICK = "corpus-calibrated s0"


def error_ratios():
    from exp.d3reallesion import KEEP_MIN
    rows = json.load(open(f"{OUT}/d3reallesion_inplace_rows.json"))["rows"]
    return np.array([r["tru"] / r["tru_real"] for r in rows
                     if r["kind"] == "pseudo" and abs(r.get("tru_real", 0)) >= KEEP_MIN])


def factor(e):
    return 1.0 - np.asarray(e, float)


def draw_errors(n, n_draws=N_DRAWS, seed=0):
    return np.random.default_rng(seed).choice(error_ratios(), size=(n_draws, n), replace=True)


def key(l):
    return (l["patient"], l["target"], int(l["lesion"]))


def run(n_draws=N_DRAWS, seed=0):
    import torch
    from ulfkit import data as D, train as T
    from exp import d3zoo as Z
    from exp.d3detect import operator
    from exp import d3reallesion as RLm
    T.seed_all(0)
    f2 = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    cu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()
    d1c = lambda y: torch.cat([f2(y[i:i + 16], gen)[:, :1] for i in range(0, len(y), 16)]).cpu()
    coh = {}
    prep, keys = [], []
    for (_, split, _), pj, od in RLm.load_pairs():
        if split not in coh:
            coh[split] = (D.PairedCohort(split, "T2"), D.PairedCohort(split, "T1"))
        P = RLm.prepare_pair(pj, od, *coh[split])
        recs = [r for r in pj["lesions"] if r["lesion"] in pj["kept"]]
        G = [RLm.lesion_geometry(P, r) for r in recs]
        kc = [g["kc"] for g in G]
        with torch.no_grad():
            x2_base = d1c(cu(P["y2"])[:, None])            # every slab row, batched as d3reallesion.run
        prep.append(dict(y2=P["y2"][kc], m2=P["m2"][kc], s=np.stack([g["s2"] for g in G]).astype(np.float32),
                         G=G, kc=kc, x2_base=x2_base))
        keys += [(pj["patient"], pj["target"], int(r["lesion"])) for r in recs]
    n = len(keys)
    E = np.concatenate([np.zeros((1, n)), draw_errors(n, n_draws, seed)])
    print(f"  {len(prep)} placements, {n} lesions, {n_draws} draws", flush=True)
    names = [RLm.NOENH] + [r[0] for r in RLm.enhancers()]
    nets = {RLm.NOENH: RLm.Ident()}
    nets.update({r[0]: Z.load(r[4], r[3]) for r in RLm.enhancers()})
    g3 = lambda x: x.cuda().repeat(1, 3, 1, 1)
    run_ = lambda net, x: torch.cat([RLm.run_net(net, g3(x[i:i + 16])) for i in range(0, len(x), 16)]).cpu().numpy()[:, 0]
    tru = np.zeros((n_draws + 1, n))
    out = {nm: np.zeros((n_draws + 1, n)) for nm in names}
    part = OUT_NPZ.replace(".npz", ".partial.npz")
    start = 0
    if os.path.exists(part):                         # resume after an interruption (same seed, keys and errors)
        zp = np.load(part)
        assert json.loads(str(zp["keys"])) == [list(k) for k in keys] and np.array_equal(zp["errors"], E)
        start = int(zp["done"])
        tru[:start] = zp["tru"][:start]
        for nm in names:
            out[nm][:start] = zp[f"out|{nm}"][:start]
        print(f"  resuming at draw {start}", flush=True)
    with torch.no_grad():
        # lesion-free outputs over every slab row, then the lesions' slices (bf16 outputs depend on batch
        # composition, so this matches d3reallesion.run exactly)
        p0 = {nm: [run_(net, q["x2_base"])[q["kc"]] for q in prep] for nm, net in nets.items()}
        for d in range(start, n_draws + 1):
            j0 = 0
            for qi, q in enumerate(prep):
                k = len(q["G"])
                fac = cu(factor(E[d, j0:j0 + k]).astype(np.float32))[:, None, None, None]
                y2 = cu(q["y2"])[:, None]
                yl = (y2 * (1 + fac * cu(q["m2"])[:, None] * cu(q["s"])[:, None])).clamp(0, 1)
                dy = (yl - y2).cpu().numpy()[:, 0]
                xl = d1c(yl)
                for j, g in enumerate(q["G"]):
                    tru[d, j0 + j] = RLm.contrast(dy[j], g["core2"], g["ring2"])
                for nm, net in nets.items():
                    pr = run_(net, xl) - p0[nm][qi]
                    for j, g in enumerate(q["G"]):
                        out[nm][d, j0 + j] = RLm.contrast(pr[j], g["core2"], g["ring2"])
                j0 += k
            print(f"  draw {d}/{n_draws} done", flush=True)
            np.savez_compressed(part, keys=json.dumps(keys), errors=E, done=d + 1, tru=tru,
                                **{f"out|{nm}": v for nm, v in out.items()})
    np.savez_compressed(OUT_NPZ, keys=json.dumps(keys), models=json.dumps(names), errors=E, tru=tru,
                        **{f"out|{nm}": v for nm, v in out.items()})


def summarise():
    from scipy.stats import spearmanr
    from exp import d3reallesion as RLm
    z = np.load(OUT_NPZ)
    keys = [tuple(k) for k in json.loads(str(z["keys"]))]
    names = [nm for nm in json.loads(str(z["models"])) if nm != RLm.NOENH]
    rows = {key(l): l for l in json.load(open(f"{OUT}/d3reallesion_rows_s0.json"))["rows"]}
    fin = lambda l: all(np.isfinite(l["out"][nm][k]) for nm in [RLm.NOENH] + names for k in ("real", "gauss", "all2d", "all3d"))
    ok = [i for i, k in enumerate(keys) if abs(rows[k]["tru_t2"]) >= RLm.KEEP_MIN
          and abs(rows[k]["tru_g_t2"]) >= RLm.KEEP_MIN and fin(rows[k])]
    Rg = {nm: np.array([RLm.ratio(rows[keys[i]]["out"][nm]["gauss"], rows[keys[i]]["tru_g_t2"]) for i in ok]) for nm in names}
    mg = np.array([np.nanmean(Rg[nm]) for nm in names])

    def one(d):
        Rr = {nm: np.array([RLm.ratio(z[f"out|{nm}"][d, i], z["tru"][d, i]) for i in ok]) for nm in names}
        mr = np.array([np.nanmean(Rr[nm]) for nm in names])
        er = {nm: float(np.nanmean(Rr[nm] < RLm.ERASED)) for nm in names}
        best = [names[i] for i in np.argsort(-mr)[:2]]
        return dict(rho=float(spearmanr(mr, mg).statistic), erased_pick=er[PICK], best_two=best,
                    erased_best_two=[er[b] for b in best], mean_R={nm: float(v) for nm, v in zip(names, mr)},
                    sign_flipped=int(np.sum(np.sign(z["tru"][d, ok]) != np.sign(z["tru"][0, ok]))))
    draws = [one(d) for d in range(1, len(z["tru"]))]
    rho = np.array([v["rho"] for v in draws]); ep = np.array([v["erased_pick"] for v in draws])
    res = dict(n_lesions=len(ok), n_draws=len(draws), unperturbed=one(0), draws=draws,
               rho_median=float(np.median(rho)), rho_range=[float(rho.min()), float(rho.max())],
               erased_pick_median=float(np.median(ep)), erased_pick_range=[float(ep.min()), float(ep.max())],
               error_model="m scaled by (1 - e), e resampled from the 457 pseudo-lesion ratios tru_pseudo / tru_real")
    res["E11a"] = {"pass": bool(np.median(rho) >= 0.95 and abs(np.median(ep) - 0.82) <= 0.05),
                   "value": f"rho median {np.median(rho):.3f} (range {rho.min():.3f}-{rho.max():.3f}); SynthSeg-pick erasure "
                            f"median {100 * np.median(ep):.1f}% (range {100 * ep.min():.1f}-{100 * ep.max():.1f}%) against 82%"}
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    res = summarise()
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "draws"}, indent=1))
