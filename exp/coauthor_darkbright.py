"""Dark against bright lesions at matched local intensity.

Paper line 339: for the absolute-output U-Net, 97% of dark lesions were erased against 64% of bright ones (120
each, out_public/d3ood_T2.json). exp/d3ood.run_family draws both families with the same seed, rows, sites, radii
and contrast index, so every dark lesion has a bright twin at the same site with the same size and magnitude.
`run` replays the two families (without the retention map, which draws no random numbers) and keeps per-lesion
rows with the lesion-free 3T intensity in the core and ring, the fraction of the core clipped at 0 or 1, and the
SynthSeg tissue class at the centre. `analyse` compares the twins within strata of local background intensity
(tertiles of the ring mean), tissue class and size.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/coauthor_darkbright.py run    # GPU, ~1 min
    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/coauthor_darkbright.py
"""
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
OUT_JSON = f"{OUT}/coauthor_darkbright_T2.json"
THR = 0.25
FAMS = {"bright": (0.12, 0.25), "dark": (-0.12, -0.25)}       # d3ood FAMILIES 'iso' and 'hypo'


def tertiles(v):
    edges = np.quantile(v, [1 / 3, 2 / 3])
    return np.digitize(v, edges), edges


def paired_counts(dark_erased, bright_erased):
    d, b = np.asarray(dark_erased, bool), np.asarray(bright_erased, bool)
    return dict(both=int((d & b).sum()), dark_only=int((d & ~b).sum()), bright_only=int((~d & b).sum()),
                neither=int((~d & ~b).sum()))


def run_family(cs, test, enh, fwd, n_sites=4, bs=16, seed=3):
    """exp/d3ood.run_family for an isotropic family, keeping every lesion and its local intensities."""
    import torch
    from exp.d3detect import site_pool
    from exp.d3ood import profile
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    recs = []
    with torch.no_grad():
        for sid in test.subjects:
            idx = test.subject_rows(sid)
            sel = idx[np.linspace(len(idx) // 5, 4 * len(idx) // 5, 6).astype(int)]
            for i0 in range(0, len(sel), 4):
                j = sel[i0:i0 + 4]
                y0 = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
                ok = site_pool(y0, r_max=8)
                x0 = fwd(y0, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p0 = enh(x0).float().clamp(0, 1)
                jobs = []
                for s in range(len(j)):
                    cand = torch.nonzero(ok[s])
                    if len(cand) < 8:
                        continue
                    for _ in range(n_sites):
                        cy, cx = cand[rng.integers(len(cand))].tolist()
                        jobs.append((s, cy, cx, int(rng.choice((2, 3, 4, 6))), float(rng.choice(cs))))
                for b0 in range(0, len(jobs), bs):
                    ch = jobs[b0:b0 + bs]
                    si = torch.tensor([t[0] for t in ch], device="cuda")
                    cy = torch.tensor([t[1] for t in ch], device="cuda")
                    cx = torch.tensor([t[2] for t in ch], device="cuda")
                    rr = torch.tensor([t[3] for t in ch], device="cuda")
                    cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                    b = profile("iso", cy, cx, rr, rng=rng) * (y0[si] > 0).float()
                    raw = y0[si] + cc * b
                    yl = raw.clamp(0, 1)
                    dyl = yl - y0[si]
                    core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                    xl = fwd(yl, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pl = enh(xl).float().clamp(0, 1)
                    dp = pl - p0[si]
                    for k, t in enumerate(ch):
                        good = bool(core[k].sum() >= 4 and ring[k].sum() >= 8)
                        tru = float(dyl[k][core[k]].mean() - dyl[k][ring[k]].mean()) if good else 0.0
                        rec = float(dp[k][core[k]].mean() - dp[k][ring[k]].mean()) if good else 0.0
                        ret = float(np.clip(rec / np.sign(tru) / max(abs(tru), 1e-6), -0.5, 1.5)) if tru else 0.0
                        recs.append(dict(sid=sid, row=int(j[t[0]]), cy=t[1], cx=t[2], r=t[3], c=t[4], tru=tru,
                                         retention=ret, kept=good and abs(tru) >= 0.25 * abs(t[4]),
                                         y_core=float(y0[t[0]][core[k]].mean()), y_ring=float(y0[t[0]][ring[k]].mean()),
                                         clip_frac=float(((raw[k] < 0) | (raw[k] > 1))[core[k]].float().mean())))
    return recs


def run(contrast="T2"):
    import torch
    from ulfkit import data as D, train as T
    from exp.d3qa import enhancer
    from exp.d3detect import operator
    T.seed_all(3)                                                     # as d3ood.main
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    fwd = operator(contrast, "learned")
    rows = {fam: run_family(cs, test, enh, fwd) for fam, cs in FAMS.items()}
    json.dump(dict(meta=dict(task="dark against bright lesions", replay="exp/d3ood.run_family, seed 3, iso and hypo",
                             model="base_unet_noresid_T2.pt"), rows=rows), open(OUT_JSON, "w"), indent=1)


def _stratum(d, b):
    return dict(n=int(len(d)), erased_dark=float(d.mean()), erased_bright=float(b.mean()),
                diff=float(d.mean() - b.mean()), **paired_counts(d, b))


def analyse():
    from scipy.stats import binomtest
    from exp.lesion_table import tissue_at
    rows = json.load(open(OUT_JSON))["rows"]
    B, Dk = rows["bright"], rows["dark"]
    both = [i for i in range(len(B)) if B[i]["kept"] and Dk[i]["kept"]]
    eb = np.array([B[i]["retention"] < THR for i in both]); ed = np.array([Dk[i]["retention"] < THR for i in both])
    ring = np.array([B[i]["y_ring"] for i in both])
    lab, edges = tertiles(ring)
    tis = np.array(tissue_at("test", [B[i]["row"] for i in both], [B[i]["cy"] for i in both], [B[i]["cx"] for i in both]))
    size = np.array([2 * B[i]["r"] for i in both])
    pc = paired_counts(ed, eb)
    strata = {f"T{k + 1}": _stratum(ed[lab == k], eb[lab == k]) for k in range(3)}
    for k in range(3):
        strata[f"T{k + 1}"]["ring_range"] = [float(ring[lab == k].min()), float(ring[lab == k].max())]
    by_tissue = {t: _stratum(ed[tis == t], eb[tis == t]) for t in sorted(set(tis))}
    by_size = {int(s): _stratum(ed[size == s], eb[size == s]) for s in sorted(set(size))}
    n_ok = [v for v in strata.values() if v["n"] >= 20]
    pooled = float(np.sum([v["n"] * v["diff"] for v in n_ok]) / max(1, np.sum([v["n"] for v in n_ok])))
    disc = pc["dark_only"] + pc["bright_only"]
    res = dict(n_pairs=len(both), erased_dark=float(ed.mean()), erased_bright=float(eb.mean()), paired=pc,
               mcnemar_p=float(binomtest(pc["dark_only"], disc, 0.5).pvalue) if disc else 1.0,
               dark_core_clipped=int(sum(Dk[i]["clip_frac"] > 0 for i in both)),
               ring_tertile_edges=edges.tolist(), by_ring_tertile=strata, by_tissue=by_tissue, by_size=by_size,
               pooled_diff_over_tertiles=pooled,
               mean_R_dark=float(np.mean([Dk[i]["retention"] for i in both])),
               mean_R_bright=float(np.mean([B[i]["retention"] for i in both])))
    e = all(v["diff"] > 0 for v in n_ok) and pooled >= 0.15
    res["E13a"] = {"pass": bool(e), "value": f"dark minus bright erasure per ring-intensity tertile: " + ", ".join(
        f"{k} {100 * v['diff']:+.0f} pp (n {v['n']})" for k, v in strata.items()) + f"; pooled {100 * pooled:+.0f} pp"}
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    res = analyse()
    json.dump(dict(json.load(open(OUT_JSON)), analysis=res), open(OUT_JSON, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_tissue", "by_size")}, indent=1))
