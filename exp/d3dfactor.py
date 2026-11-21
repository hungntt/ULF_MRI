"""Does retention factorise into a scanner term and an enhancer term?

Retention R = (output lesion contrast) / (inserted 3T lesion contrast) runs through two maps,
the learned degradation D (3T -> 64 mT) and the enhancer G. Write it as

    R = R_D * R_G|D,    R_D   = 64 mT lesion contrast / 3T lesion contrast   (D alone)
                        R_G|D = output lesion contrast / 64 mT lesion contrast (G given D)

with each term a ratio of mean core-minus-ring contrasts over identical lesions. A diagnostic on
the old and new operators showed R_D moving from 0.63 to 0.25 with the training cohort of D,
while R_G|D stayed at 0.56-0.62. This script asks the public-data version of that question:
two operators trained on the SAME 15 OpenNeuro subjects and the same 3T, one per 64 mT Swoop
unit (HFC: out_public, HFE: out_public_hfe), applied to every seed-0 enhancer of the zoo.

Reported per cohort: R, R_D and R_G|D for each (D, G); and the Spearman correlation of the
enhancer ranking on R under the two operators. If the ranking survives a change of D, enhancer
comparisons do not depend on how strongly D attenuates lesions.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3dfactor.py [splits...]
"""
import json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from ulfkit import data as D, models as Mo, train as T
from exp.d3detect import site_pool, blobs
from exp import d3zoo as Z

OUT = os.environ.get("ULF_OUT", "./out")
OPS = {"HFC": f"{OUT}/d1_forward_T2.pt", "HFE": f"{OUT}_hfe/d1_forward_T2.pt"}
RADII, CS = (2, 3, 4, 6), (0.12, 0.25)


def load_op(path):
    net = Mo.UNet(1, 1, base=32, residual_from=(0, 1)).cuda()
    net.load_state_dict(torch.load(path)); net.eval()
    def f(y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = net(y).float().clamp(0, 1)
        return d.repeat(1, 3, 1, 1) * (y > 0)
    return f


@torch.no_grad()
def run(split, ops, nets, n_rows=96, sites=4, seed=11):
    """-> {(op, enhancer): dict(r=[...], dy=[...], dd=[...], dp=[...])}: per-lesion contrasts."""
    coh = D.PairedCohort(split, "T2")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(coh.hf), min(n_rows, len(coh.hf)), replace=False))
    rec = {(o, nm): dict(r=[], dy=[], dd=[], dp=[]) for o in ops for nm, _ in nets}
    for i in range(0, len(rows), 8):
        y = torch.from_numpy(coh.hf[rows[i:i + 8]].astype(np.float32)).cuda()
        ok = site_pool(y)
        jobs = []
        for s in range(len(y)):
            cand = torch.nonzero(ok[s])
            if len(cand) < 8: continue
            for _ in range(sites):
                cy, cx = cand[rng.integers(len(cand))].tolist()
                jobs.append((s, cy, cx, int(rng.choice(RADII)), float(rng.choice(CS))))
        if not jobs: continue
        si = torch.tensor([j[0] for j in jobs], device="cuda")
        b = blobs(torch.tensor([j[1] for j in jobs], device="cuda"), torch.tensor([j[2] for j in jobs], device="cuda"),
                  torch.tensor([j[3] for j in jobs], device="cuda")) * (y[si] > 0).float()
        c = torch.tensor([j[4] for j in jobs], device="cuda")[:, None, None, None]
        y0 = y[si]; yl = (y0 + c * b).clamp(0, 1)
        core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
        good = [k for k in range(len(jobs)) if core[k].sum() >= 4 and ring[k].sum() >= 8]
        con = lambda d, k: float(d[k][core[k]].mean() - d[k][ring[k]].mean())
        for o, f in ops.items():
            x0, xl = f(y0), f(yl)
            dd = (xl - x0)[:, 1:2]
            for nm, net in nets:
                p0, pl = [], []
                for k0 in range(0, len(jobs), 12):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p0.append(net(x0[k0:k0 + 12]).float().clamp(0, 1))
                        pl.append(net(xl[k0:k0 + 12]).float().clamp(0, 1))
                dp = torch.cat(pl) - torch.cat(p0)
                R = rec[(o, nm)]
                for k in good:
                    R["r"].append(jobs[k][3]); R["dy"].append(con(yl - y0, k))
                    R["dd"].append(con(dd, k)); R["dp"].append(con(dp, k))
    return rec


def summarise(rec):
    out = {}
    for (o, nm), R in rec.items():
        dy, dd, dp = (np.array(R[k]) for k in ("dy", "dd", "dp"))
        out[f"{o}|{nm}"] = dict(n=len(dy), R=float(np.mean(dp / dy)), R_D=float(dd.mean() / dy.mean()),
                                R_GD=float(dp.mean() / dd.mean()), erased=float(np.mean(dp / dy < 0.25)))
    return out


def main(splits):
    T.seed_all(0)
    assert all(os.path.exists(p) for p in OPS.values()), OPS
    ops = {o: load_op(p) for o, p in OPS.items()}
    reg = [r for r in Z.registry() if r[2] == 0 and r[4] != "diff" and os.path.exists(f"{OUT}/{r[3]}.pt")]
    res = {}
    for split in splits:
        nets = [(r[1], Z.load(r[4], r[3])) for r in reg]
        s = summarise(run(split, ops, nets))
        names = [nm for nm, _ in nets]
        a = [s[f"HFC|{n}"]["R"] for n in names]; b = [s[f"HFE|{n}"]["R"] for n in names]
        rho = float(spearmanr(a, b).statistic)
        res[split] = dict(per=s, rank_rho_HFC_vs_HFE=rho, enhancers=names)
        print(f"\n== {split}: {len(names)} enhancers, {s[f'HFC|{names[0]}']['n']} lesions each")
        print(f"  {'enhancer':28s} {'R (HFC)':>8s} {'R (HFE)':>8s} {'R_D HFC':>8s} {'R_D HFE':>8s} {'R_G|D HFC':>10s} {'R_G|D HFE':>10s}")
        for n in names:
            u, v = s[f"HFC|{n}"], s[f"HFE|{n}"]
            print(f"  {n:28s} {u['R']:8.3f} {v['R']:8.3f} {u['R_D']:8.3f} {v['R_D']:8.3f} {u['R_GD']:10.3f} {v['R_GD']:10.3f}")
        print(f"  enhancer ranking on R, HFC vs HFE operator: Spearman {rho:+.3f}", flush=True)
        json.dump(res, open(f"{OUT}/d3dfactor_T2.json", "w"), indent=1)
        del nets; torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"])
