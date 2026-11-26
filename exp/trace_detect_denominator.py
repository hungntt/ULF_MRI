"""The denominator of "71% of 1605 lesions".

One record of out_public/d3detect_T2.json is one inserted 2D Gaussian lesion at one site of one test slice,
passed through one model (the absolute-output U-Net, checkpoint base_unet_noresid_T2.pt). The records are
the labels cached by exp/d3detect_eval.py in datasets/npy_public/detect_test_T2.npy (row, cy, cx, r, c, R).
make_labels drops a site when the clamped inserted contrast is below 25% of the nominal one; that count is
not stored, so `regen` replays make_labels(n_sites=5, seed=1) with identical RNG use and batching and keeps
every site with its keep flag.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/trace_detect_denominator.py regen   # GPU
    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/trace_detect_denominator.py report
"""
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
DATA = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy_public")
OUT_JSON = f"{OUT}/coauthor_denominator_T2.json"
THR = 0.25


def cached_test_labels():
    """-> [(row, cy, cx, r, c, retention)] in the order of d3detect_T2.json."""
    a = np.load(f"{DATA}/detect_test_T2.npy", allow_pickle=True)
    return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4]), float(r[5])) for r in a]


def per_subject(lab):
    from exp.lesion_table import row_sid
    sid, _ = row_sid("test")
    out = {}
    for row, *_, ret in lab:
        out.setdefault(sid[row], []).append(ret < THR)
    return {s: dict(n=len(v), erased=float(np.mean(v))) for s, v in sorted(out.items())}


def regen(contrast="T2", n_sites=5, seed=1, bs=24):
    """Replay exp/d3detect.make_labels for the locked test set, keeping dropped sites."""
    import torch
    from ulfkit import data as D, train as T
    from exp.d3qa import enhancer
    from exp.d3detect import operator, site_pool, blobs, BRIGHT
    T.seed_all(1)                                                    # as d3detect_eval.__main__
    test = D.PairedCohort("test", contrast)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{contrast}.pt")); enh.eval()
    fwd = operator(contrast, "learned")
    radii, cs = (2, 3, 4, 6), BRIGHT
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    sites = []
    with torch.no_grad():
        for sid in test.subjects:
            idx = test.subject_rows(sid)
            for i0 in range(0, len(idx), 8):
                j = idx[i0:i0 + 8]
                y0 = torch.from_numpy(test.hf[j].astype(np.float32)).cuda()
                ok = site_pool(y0)
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
                        jobs.append((s, cy, cx, int(rng.choice(radii)), float(rng.choice(cs))))
                for b0 in range(0, len(jobs), bs):
                    ch = jobs[b0:b0 + bs]
                    si = torch.tensor([t[0] for t in ch], device="cuda")
                    cy = torch.tensor([t[1] for t in ch], device="cuda")
                    cx = torch.tensor([t[2] for t in ch], device="cuda")
                    rr = torch.tensor([t[3] for t in ch], device="cuda")
                    cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                    b = blobs(cy, cx, rr) * (y0[si] > 0).float()
                    yl = (y0[si] + cc * b).clamp(0, 1)
                    dy = yl - y0[si]
                    core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                    tru = torch.stack([(dy[k][core[k]].mean() - dy[k][ring[k]].mean()) for k in range(len(ch))])
                    xl = fwd(yl, gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pl = enh(xl).float().clamp(0, 1)
                    dp = pl - p0[si]
                    rec = torch.stack([(dp[k][core[k]].mean() - dp[k][ring[k]].mean()) for k in range(len(ch))])
                    keep = tru.abs() > 0.25 * cc[:, 0, 0, 0].abs()
                    den = torch.where(tru.abs() < 1e-6, torch.full_like(tru, 1e-6), tru)
                    ret = (rec / den).clamp(-0.5, 1.5)
                    for k, t in enumerate(ch):
                        sites.append(dict(sid=sid, row=int(j[t[0]]), cy=t[1], cx=t[2], r=t[3], c=t[4],
                                          tru=float(tru[k]), retention=float(ret[k]), keep=bool(keep[k])))
            print(f"  {sid}: {len(sites)} sites, {sum(s['keep'] for s in sites)} kept", flush=True)
    res = dict(meta=dict(task="denominator of the retention-map test set", replay="exp/d3detect.make_labels(test, n_sites=5, seed=1)",
                         model="base_unet_noresid_T2.pt (absolute-output U-Net, seed 0)", threshold=THR),
               sites=sites)
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    return res


def summary():
    g = json.load(open(OUT_JSON))
    S = g["sites"]
    kept = [s for s in S if s["keep"]]
    drop = [s for s in S if not s["keep"]]
    subj = sorted({s["sid"] for s in S})
    return dict(n_sites=len(S), n_kept=len(kept), n_dropped=len(drop),
                dropped_by_size={2 * r: sum(s["r"] == r for s in drop) for r in (2, 3, 4, 6)},
                slices=len({s["row"] for s in S}),
                erased=float(np.mean([s["retention"] < THR for s in kept])),
                per_subject={sid: dict(n=sum(s["sid"] == sid for s in kept), dropped=sum(s["sid"] == sid for s in drop),
                                       erased=float(np.mean([s["retention"] < THR for s in kept if s["sid"] == sid])))
                             for sid in subj})


if __name__ == "__main__":
    if sys.argv[1:] == ["regen"]:
        regen()
    print(json.dumps(summary(), indent=1))
