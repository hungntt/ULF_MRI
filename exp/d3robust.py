"""Do the retention findings survive changes to the measurement itself?

Every retention claim rests on choices made once: erasure means R < 0.25, lesions are Gaussian
profiles 4-12 mm across, core and ring are level sets of that profile, and the zoo leaves out
models trained to fail. One can ask whether any of these choices produced the result.
Four checks, each with its expected outcome written into out_public/d3robust_T2.json
("preregistration") before anything runs:

  1. threshold  erasure at R < 0.10, 0.25 and 0.50, and the ECDF of R, for the enhancers that
                SynthSeg Dice and SSIM select, the retention leader, and no enhancer. The picks are
                read from d3zoo_stats*_T2.json, not re-decided. Do the gap between the Dice pick and
                the leader, and the Dice pick's rank, hold at every threshold?
  2. large      positive control: diameters 16, 20 and 25 mm (r = 8, 10, 12.5 voxels, the code's
                1/e^2 radius), one lesion per slice, site_pool(r_max=16), contrasts 0.12 and 0.25,
                seed-0 zoo. Synthetic arm on every cohort; acquired-input arm
                (d3detreal.lesion_input) where a 64 mT scan exists. Erasure should be near zero.
  3. disc       sharp-edged disc (antialiased indicator of dist <= r) scored with distance-based
                regions: core dist <= 0.7r, ring r+2 <= dist <= r+2+w with the area of the core.
                The Gaussian is re-scored with the same regions, so the effect of the region
                definition and the effect of the profile are separated.
  4. controls   a U-Net trained on subject-mismatched pairs (baselines.MismatchSampler) and a U-Net
                trained on 3 subjects. They never enter a ranking statistic.

Sites for checks 1, 3 and 4 are drawn exactly as in d3arch.run (seed 11, 8 slices x 5 sites per
subject), so the Gaussian rows must reproduce out_public/d3zoo*_T2.json; that is the sanity check.
Every output stores the size both as r_vox (code radius) and diam_mm = 2r.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3robust.py prereg
    python exp/d3robust.py std [--models zoo|ctl] [--splits ...]   # checks 1, 3, 4 (site rows)
    python exp/d3robust.py large [--splits ...]                    # check 2
    python exp/d3robust.py diag_large                              # why check 2 failed (see docstring)
    python exp/d3robust.py train_lowdata --n_subj 3 --seed 0       # base_unet_n3_seed0_T2
    python exp/d3robust.py ssim_ctl                                # controls on acquired input
    python exp/d3robust.py summary                                 # JSON + fig_public/robust*.pdf
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from ulfkit import data as D, train as T
from exp.d3detect import operator, site_pool, blobs, BRIGHT
from exp.d3detreal import lesion_input
from exp import d3zoo as Z

OUT = os.environ.get("ULF_OUT", "./out")
FIG = os.environ.get("ULF_FIG", "./fig")
STORE = f"{OUT}/d3robust_T2.json"
SPLITS = ["test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"]
STD_RADII = (2, 3, 4, 6)
LARGE_RADII = (8.0, 10.0, 12.5)
THRS = (0.10, 0.25, 0.50)
CONDS = ("gauss_b", "gauss_d", "disc_d")
NOENH = "no enhancer"
SEED = 11                                   # d3arch.run's site seed
CTL = [  # (run name, control, seed, checkpoint, kind)
    ("mismatched pairs s0", "mismatched pairs", 0, "base_unet_mismatched_T2", "unet_res"),
    ("mismatched pairs s1", "mismatched pairs", 1, "base_unet_mismatched_seed1_T2", "unet_res"),
    ("mismatched pairs s2", "mismatched pairs", 2, "base_unet_mismatched_seed2_T2", "unet_res"),
    ("3 subjects s0", "3 subjects", 0, "base_unet_n3_seed0_T2", "unet_res"),
    ("3 subjects s1", "3 subjects", 1, "base_unet_n3_seed1_T2", "unet_res"),
    ("3 subjects s2", "3 subjects", 2, "base_unet_n3_seed2_T2", "unet_res"),
]


class Ident(nn.Module):
    """G = identity on the centre slice: the output is D's 64 mT rendering, so R = R_D."""
    def forward(self, x):
        return x[:, 1:2]


def models(which):
    """-> [(run name, enhancer, seed, ckpt, kind)] whose checkpoints exist."""
    if which == "ctl":
        reg = CTL
    else:
        reg = [r for r in Z.registry() if r[4] == "diff" or os.path.exists(f"{OUT}/{r[3]}.pt")]
        if which == "zoo0":
            reg = [r for r in reg if r[2] == 0]
        reg = reg + [(NOENH, NOENH, 0, None, "ident")]
    return [r for r in reg if r[4] in ("ident", "diff") or os.path.exists(f"{OUT}/{r[3]}.pt")]


def load(kind, ck):
    return Ident().eval() if kind == "ident" else Z.load(kind, ck)


def fwd_net(net, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(x).float().clamp(0, 1)


def dist_map(cy, cx, H=256, W=256):
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    return torch.sqrt(((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2).float())[:, None]


def disc(dist, r):
    """Antialiased indicator of dist <= r: linear pixel coverage over a one-voxel edge."""
    return (r + 0.5 - dist).clamp(0, 1)


def ring_width(r):
    """w such that the annulus r+2 <= dist <= r+2+w has the area of the core dist <= 0.7r."""
    return np.sqrt((r + 2) ** 2 + 0.49 * r ** 2) - (r + 2)


def con(d, core, ring):
    """Per-image mean(d | core) - mean(d | ring); d, core, ring: (B,1,H,W)."""
    cm, rm = core.float(), ring.float()
    return ((d * cm).sum((1, 2, 3)) / cm.sum((1, 2, 3)).clamp(min=1)
            - (d * rm).sum((1, 2, 3)) / rm.sum((1, 2, 3)).clamp(min=1))


# ------------------------------------------------------------------ check 1, 3, 4: standard radii
@torch.no_grad()
def run_std(split, nets, seed=SEED, n_sites=5, conds=CONDS):
    """d3arch.run's sites and lesions, scored under three (profile, region) conditions.
    -> dict of per-site arrays; rec_<cond> is (n_models, n_sites)."""
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    coh = D.PairedCohort(split, "T2")
    fwd = operator("T2", "learned")
    M = {k: [] for k in ("sid", "row", "cy", "cx", "r_vox", "c")}
    tru = {c: [] for c in conds}; nco = {c: [] for c in conds}; nri = {c: [] for c in conds}
    rec = {c: {nm: [] for nm, _ in nets} for c in conds}
    for sid in coh.subjects:
        idx = coh.subject_rows(sid)
        sel = idx[np.linspace(len(idx) // 5, 4 * len(idx) // 5, 8).astype(int)]
        for i0 in range(0, len(sel), 4):
            j = sel[i0:i0 + 4]
            y0 = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0)
            x0 = fwd(y0, gen)
            base = {nm: fwd_net(net, x0) for nm, net in nets}
            jobs = []
            for s in range(len(j)):                      # identical rng calls to d3arch.run
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                for _ in range(n_sites):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    jobs.append((s, cy, cx, int(rng.choice(STD_RADII)), float(rng.choice(BRIGHT))))
            for b0 in range(0, len(jobs), 12):
                ch = jobs[b0:b0 + 12]
                si = torch.tensor([t[0] for t in ch], device="cuda")
                cy = torch.tensor([t[1] for t in ch], device="cuda")
                cx = torch.tensor([t[2] for t in ch], device="cuda")
                rr = torch.tensor([t[3] for t in ch], device="cuda")
                cc = torch.tensor([t[4] for t in ch], device="cuda")[:, None, None, None]
                brain = (y0[si] > 0).float()
                b = blobs(cy, cx, rr) * brain
                dist = dist_map(cy, cx)
                rf = rr.float()[:, None, None, None]
                w = torch.sqrt((rf + 2) ** 2 + 0.49 * rf ** 2) - (rf + 2)
                regions = {"gauss_b": ("gauss", b > 0.6, (b > 0.05) & (b < 0.25)),
                           "gauss_d": ("gauss", (dist <= 0.7 * rf) & (brain > 0),
                                       (dist >= rf + 2) & (dist <= rf + 2 + w) & (brain > 0))}
                regions["disc_d"] = ("disc", regions["gauss_d"][1], regions["gauss_d"][2])
                yl = {"gauss": (y0[si] + cc * b).clamp(0, 1)}
                if "disc_d" in conds:
                    yl["disc"] = (y0[si] + cc * disc(dist, rf) * brain).clamp(0, 1)
                xl = {p: fwd(v, gen) for p, v in yl.items()}
                for c in conds:
                    p, core, ring = regions[c]
                    tru[c].append(con(yl[p] - y0[si], core, ring).cpu())
                    nco[c].append(core.sum((1, 2, 3)).cpu()); nri[c].append(ring.sum((1, 2, 3)).cpu())
                for nm, net in nets:
                    pl = {p: fwd_net(net, v) for p, v in xl.items()}
                    for c in conds:
                        p, core, ring = regions[c]
                        rec[c][nm].append(con(pl[p] - base[nm][si], core, ring).cpu())
                for t in ch:
                    M["sid"].append(sid); M["row"].append(int(j[t[0]])); M["cy"].append(t[1])
                    M["cx"].append(t[2]); M["r_vox"].append(t[3]); M["c"].append(t[4])
        print(f"  {split} {sid}: {len(M['sid'])} sites", flush=True)
    out = {k: np.array(v) for k, v in M.items()}
    out["diam_mm"] = 2.0 * out["r_vox"]
    for c in conds:
        out[f"tru_{c}"] = torch.cat(tru[c]).numpy()
        out[f"ncore_{c}"] = torch.cat(nco[c]).numpy(); out[f"nring_{c}"] = torch.cat(nri[c]).numpy()
        out[f"rec_{c}"] = np.stack([torch.cat(rec[c][nm]).numpy() for nm, _ in nets])
    return out


def rows_path(mode, split):
    return f"{OUT}/d3robust_rows_{mode}_{split}_T2.npz"


def run_chunked(fn, split, reg, chunk, **kw):
    """Run fn on chunks of models (sites depend only on the seed) and merge the rec arrays."""
    merged = None
    for c0 in range(0, len(reg), chunk):
        part = reg[c0:c0 + chunk]
        nets = [(r[0], load(r[4], r[3])) for r in part]
        print(f"  chunk {[r[0] for r in part]}", flush=True)
        o = fn(split, nets, **kw)
        if merged is None:
            merged = o
        else:
            for k, v in o.items():
                if k.startswith("rec"):
                    merged[k] = np.concatenate([merged[k], v])
                else:
                    assert np.array_equal(merged[k], v) if v.dtype.kind in "biuUO" else np.allclose(merged[k], v), k
        del nets; torch.cuda.empty_cache()
    merged["models"] = np.array([r[0] for r in reg]); merged["enhancer"] = np.array([r[1] for r in reg])
    merged["seed"] = np.array([r[2] for r in reg]); merged["ckpt"] = np.array([str(r[3]) for r in reg])
    return merged


# ------------------------------------------------------------------ check 2: large lesions
@torch.no_grad()
def run_large(split, nets, seed=SEED, n_slices=30, r_max=16):
    """One large lesion per slice; synthetic arm, and the acquired arm where a 64 mT scan exists."""
    T.seed_all(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    coh = D.PairedCohort(split, "T2")
    paired = coh.ulf is not None
    fwd = operator("T2", "learned")
    k_ring = float(np.sqrt(np.log(20) / 2))           # b > 0.05 reaches k_ring * radius
    M = {k: [] for k in ("sid", "row", "cy", "cx", "r_vox", "c")}
    tru, nco, nri, leak, clip, inbr = [], [], [], [], [], []
    rec = {nm: [] for nm, _ in nets}; acq = {nm: [] for nm, _ in nets}
    for sid in coh.subjects:
        idx = coh.subject_rows(sid)
        sel = np.unique(idx[np.linspace(len(idx) // 5, 4 * len(idx) // 5, n_slices).astype(int)])
        for i0 in range(0, len(sel), 12):
            j = sel[i0:i0 + 12]
            y0 = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
            ok = site_pool(y0, r_max=r_max)
            jobs = []
            for s in range(len(j)):
                cand = torch.nonzero(ok[s])
                if len(cand) < 8: continue
                cy, cx = cand[rng.integers(len(cand))].tolist()
                jobs.append((s, cy, cx, float(rng.choice(LARGE_RADII)), float(rng.choice(BRIGHT))))
            if not jobs: continue
            x0 = fwd(y0, gen)
            xr = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda() if paired else None
            si = torch.tensor([t[0] for t in jobs], device="cuda")
            cy = torch.tensor([t[1] for t in jobs], device="cuda")
            cx = torch.tensor([t[2] for t in jobs], device="cuda")
            rr = torch.tensor([t[3] for t in jobs], device="cuda")
            cc = torch.tensor([t[4] for t in jobs], device="cuda")[:, None, None, None]
            brain = (y0[si] > 0).float()
            b = blobs(cy, cx, rr) * brain
            yl = (y0[si] + cc * b).clamp(0, 1)
            core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
            tru.append(con(yl - y0[si], core, ring).cpu())
            nco.append(core.sum((1, 2, 3)).cpu()); nri.append(ring.sum((1, 2, 3)).cpu())
            xl = fwd(yl, gen)
            # acquired-arm neighbourhood (d3detreal.lesion_input): does it hold D's response,
            # and does it stay inside the 256 x 256 slab?
            near = (blobs(cy, cx, torch.clamp(rr * 4, min=24)) > 0.05).float()
            resp = (xl - x0[si])[:, 1:2] ** 2
            leak.append((resp * (1 - near)).sum((1, 2, 3)).cpu() / resp.sum((1, 2, 3)).clamp(min=1e-12).cpu())
            rn = torch.clamp(rr * 4, min=24) * k_ring
            clip.append(((cy - rn < 0) | (cy + rn > 255) | (cx - rn < 0) | (cx + rn > 255)).cpu())
            inbr.append(((near * brain).sum((1, 2, 3)) / near.sum((1, 2, 3))).cpu())
            xa = lesion_input(xr[si], y0[si], yl, fwd, gen, cy, cx, rr) if paired else None
            for nm, net in nets:
                dp = fwd_net(net, xl) - fwd_net(net, x0)[si]
                rec[nm].append(con(dp, core, ring).cpu())
                if paired:
                    da = fwd_net(net, xa) - fwd_net(net, xr)[si]
                    acq[nm].append(con(da, core, ring).cpu())
            for t in jobs:
                M["sid"].append(sid); M["row"].append(int(j[t[0]])); M["cy"].append(t[1])
                M["cx"].append(t[2]); M["r_vox"].append(t[3]); M["c"].append(t[4])
        print(f"  {split} {sid}: {len(M['sid'])} sites", flush=True)
    out = {k: np.array(v) for k, v in M.items()}
    out["diam_mm"] = 2.0 * out["r_vox"]
    out.update(tru_gauss_b=torch.cat(tru).numpy(), ncore_gauss_b=torch.cat(nco).numpy(),
               nring_gauss_b=torch.cat(nri).numpy(), leak=torch.cat(leak).numpy(),
               clip=torch.cat(clip).numpy(), near_in_brain=torch.cat(inbr).numpy(),
               rec_gauss_b=np.stack([torch.cat(rec[nm]).numpy() for nm, _ in nets]))
    if paired:
        out["rec_acq"] = np.stack([torch.cat(acq[nm]).numpy() for nm, _ in nets])
    return out


# ------------------------------------------------------------------ check 2: why large lesions fail
def _smooth(a, m, sigma):
    """Masked (normalised) Gaussian smoothing of (B,1,H,W) a inside mask m."""
    k = int(4 * sigma) | 1
    g = torch.exp(-(torch.arange(k, device=a.device) - k // 2).float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    def conv(x):
        x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
        return torch.nn.functional.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return conv(a * m) / conv(m).clamp(min=1e-6)


def _slope(u, v):
    """OLS slope of v on u, and corr^2."""
    u, v = u - u.mean(), v - v.mean()
    return float((u * v).sum() / (u * u).sum()), float((u * v).sum() ** 2 / ((u * u).sum() * (v * v).sum()))


@torch.no_grad()
def diag_large(splits=("test", "ext_v5"), sigma=4.0):
    """Probes run after the positive control failed (16-25 mm lesions not near-fully retained).
    1. core-only and ring-only gains, mean(dp | region) / mean(dy | region), on the large-run sites:
       if the ring subtraction were the cause, the core gain would be near 1.
    2. flat offsets of radius 40 vox (80 mm disc) and the whole brain at the same sites: the gain
       mean(dp)/mean(dy) in a 5-vox central disc is the pipeline's low-frequency intensity gain.
    3. intensity slopes on real data, after masked Gaussian smoothing (sigma 4 vox) in tissue of the
       site-pool range (3T 0.12-0.45): acquired 64 mT on 3T (the scanner's gain), D(3T) on 3T (D's),
       G(acquired) on acquired (each enhancer's), and corr^2 of 3T vs acquired, the ceiling that
       regression dilution puts on the gain G o D of any enhancer trained to predict 3T from 64 mT."""
    reg = models("zoo0"); nets = [(r[0], load(r[4], r[3])) for r in reg]
    fwd = operator("T2", "learned")
    res = {}
    for split in splits:
        rw = load_rows("large", split)
        coh = D.PairedCohort(split, "T2")
        ok, _ = retention(rw, "gauss_b")
        idx = np.where(ok)[0]
        g = {k: {nm: [] for nm, _ in nets} for k in ("core", "ring", "flat80", "brain")}
        for b0 in range(0, len(idx), 12):
            q = idx[b0:b0 + 12]
            y0 = torch.from_numpy(coh.hf[rw["row"][q]].astype(np.float32)).cuda()
            cy = torch.tensor(rw["cy"][q], device="cuda"); cx = torch.tensor(rw["cx"][q], device="cuda")
            rr = torch.tensor(rw["r_vox"][q], device="cuda").float()
            cc = torch.tensor(rw["c"][q], device="cuda").float()[:, None, None, None]
            brain = (y0 > 0).float()
            b = blobs(cy, cx, rr) * brain
            dist = dist_map(cy, cx)
            centre = (dist <= 5) & (brain > 0)
            ins = {"gauss": b, "flat80": disc(dist, torch.full_like(dist, 40.0)) * brain, "brain": brain}
            regs = {"core": ("gauss", b > 0.6), "ring": ("gauss", (b > 0.05) & (b < 0.25)),
                    "flat80": ("flat80", centre), "brain": ("brain", centre)}
            yl = {k: (y0 + cc * v).clamp(0, 1) for k, v in ins.items()}
            x0 = fwd(y0, None); xl = {k: fwd(v, None) for k, v in yl.items()}
            mean = lambda d, m: (d * m).sum((1, 2, 3)) / m.float().sum((1, 2, 3)).clamp(min=1)
            for nm, net in nets:
                p0 = fwd_net(net, x0); pl = {k: fwd_net(net, v) for k, v in xl.items()}
                for k, (p, m) in regs.items():
                    g[k][nm].append((mean(pl[p] - p0, m) / mean(yl[p] - y0, m)).cpu())
        out = dict(n_sites=int(len(idx)), gains={}, by_diam={})
        for k in g:
            out["gains"][k] = {nm: float(torch.cat(v).mean()) for nm, v in g[k].items()}
        for r in LARGE_RADII:
            m = rw["r_vox"][idx] == r
            out["by_diam"][f"{2 * r:g} mm"] = {k: {nm: float(torch.cat(v)[torch.from_numpy(m)].mean())
                                                   for nm, v in g[k].items()} for k in ("core", "ring")}
        # probe 3: slopes on real data
        sl = {"scanner": [], "D": [], "corr2": []}; slG = {nm: [] for nm, _ in nets}
        for sid in coh.subjects:
            rows = coh.subject_rows(sid)
            rows = rows[len(rows) // 5: 4 * len(rows) // 5]
            ys, xs, ds, ps = [], [], [], {nm: [] for nm, _ in nets}
            for i in range(0, len(rows), 16):
                j = rows[i:i + 16]
                y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
                x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
                m = (-torch.nn.functional.max_pool2d(-(y > 0).float(), 11, 1, 5) > 0.5).float()
                ys.append(_smooth(y, m, sigma)[m > 0]); xs.append(_smooth(x[:, 1:2], m, sigma)[m > 0])
                ds.append(_smooth(fwd(y, None)[:, 1:2], m, sigma)[m > 0])
                for nm, net in nets:
                    ps[nm].append(_smooth(fwd_net(net, x), m, sigma)[m > 0])
            ys, xs, ds = torch.cat(ys), torch.cat(xs), torch.cat(ds)
            t = (ys > 0.12) & (ys < 0.45)
            a, c2 = _slope(ys[t], xs[t]); sl["scanner"].append(a); sl["corr2"].append(c2)
            sl["D"].append(_slope(ys[t], ds[t])[0])
            for nm in slG:
                slG[nm].append(_slope(xs[t], torch.cat(ps[nm])[t])[0])
        out["slopes"] = dict(sigma_vox=sigma, scanner_64mT_on_3T=float(np.mean(sl["scanner"])),
                             D_on_3T=float(np.mean(sl["D"])), corr2_3T_vs_64mT=float(np.mean(sl["corr2"])),
                             G_on_64mT={nm: float(np.mean(v)) for nm, v in slG.items()},
                             predicted_GoD_gain={nm: float(np.mean(sl["D"]) * np.mean(v)) for nm, v in slG.items()})
        res[split] = out
        print(f"\n== {split}: {len(idx)} sites")
        print(f"  slopes: scanner {out['slopes']['scanner_64mT_on_3T']:.3f}  D {out['slopes']['D_on_3T']:.3f}"
              f"  corr^2 {out['slopes']['corr2_3T_vs_64mT']:.3f}")
        for nm, _ in nets:
            print(f"  {nm:26s} core {out['gains']['core'][nm]:.3f} ring {out['gains']['ring'][nm]:.3f}"
                  f"  80mm {out['gains']['flat80'][nm]:.3f} brain {out['gains']['brain'][nm]:.3f}"
                  f"  G slope {out['slopes']['G_on_64mT'][nm]:.3f} -> D*G {out['slopes']['predicted_GoD_gain'][nm]:.3f}",
                  flush=True)
    json.dump(res, open(f"{OUT}/d3robust_diag_large_T2.json", "w"), indent=1)


# ------------------------------------------------------------------ check 4: controls
def train_lowdata(n_subj, seed, steps):
    """baselines.py --method unet, unchanged, on the first n_subj training subjects (sorted, the
    nested-subset convention of d1.py). D, the retention protocol and selection are unchanged."""
    from exp import baselines as B
    tag = f"unet_n{n_subj}_seed{seed}"
    assert not os.path.exists(f"{OUT}/base_{tag}_T2.pt"), f"base_{tag}_T2.pt exists; not overwriting"
    orig = D.dev_split

    def nested(contrast):
        coh, tr, val = orig(contrast)
        sub = sorted(tr)[:n_subj]
        print(f"  low-data control: training subjects {sub} (of {len(tr)}), selection on {val}", flush=True)
        return coh, sub, val
    B.D.dev_split = nested
    sys.argv = ["baselines.py", "--contrast", "T2", "--method", "unet", "--steps", str(steps),
                "--seed", str(seed), "--tag", tag]
    B.main()


@torch.no_grad()
def ssim_ctl(splits=("test", "ext_v5")):
    """Brain-masked SSIM / PSNR on the ACQUIRED 64 mT input (d3zoo's in_fn), per subject."""
    res = {}
    for split in splits:
        coh = D.PairedCohort(split, "T2")
        in_fn = lambda rows: (coh.ulf[rows].astype(np.float32), coh.ulf[rows][:, 1:2].astype(np.float32),
                              coh.hf[rows].astype(np.float32))
        res[split] = {}
        for nm, ctl, seed, ck, kind in models("ctl"):
            net = load(kind, ck)
            q = T.evaluate(net, coh, coh.subjects, in_fn, bands=False)
            res[split][nm] = dict(control=ctl, seed=seed, ckpt=ck, sids=[r["sid"] for r in q],
                                  ssim=[r["ssim"] for r in q], psnr=[r["psnr"] for r in q])
            print(f"  {split} {nm:22s} SSIM {np.mean(res[split][nm]['ssim']):.3f}"
                  f"  PSNR {np.mean(res[split][nm]['psnr']):.2f}", flush=True)
            del net; torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/d3robust_ssim_ctl_T2.json", "w"), indent=1)


# ------------------------------------------------------------------ preregistration
def prereg():
    """Expected outcomes, written once, before any A3 run. summary() carries the block unchanged."""
    if os.path.exists(STORE) and "preregistration" in json.load(open(STORE)):
        print("preregistration already written; left unchanged"); return
    block = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        note="Written by `d3robust.py prereg` before any A3 model was trained or evaluated. "
             "Thresholds come from the plan (analysis plan, A3) and were not tuned.",
        thresholds=list(THRS), large_diam_mm=[2 * r for r in LARGE_RADII], large_r_vox=list(LARGE_RADII),
        picks="read from out_public/d3zoo_stats{,_ext_on_hfe,_ext_v5}_T2.json['selection'] "
              "(synthseg, ssim, ret); 3T-only cohorts use the internal-test picks",
        expected=dict(
            threshold="At every threshold the SynthSeg-Dice pick erases more lesions than the retention "
                      "leader (subject-bootstrap 95% CI of the gap above 0) and ranks in the lower half "
                      "(rank >= 7 of 12, 1 = least erased) on the internal test set.",
            large="Positive control: every seed-0 enhancer erases <= 5% of lesions (R < 0.25) at each of "
                  "16, 20 and 25 mm in every cohort (synthetic arm), and mean R rises with diameter from "
                  "12 to 25 mm. If not, investigate before reporting.",
            disc="Changing only the region definition (Gaussian with level-set regions -> Gaussian with "
                 "distance regions) moves the zoo-mean R by less than 0.03 and keeps the enhancer ranking "
                 "(Spearman >= 0.8). No direction is predicted for the sharp disc.",
            controls=dict(
                mismatched_pairs="low R and low SSIM: seed-averaged mean R (standard radii, Gaussian) and "
                                 "brain-masked SSIM on the acquired 64 mT input both below the lowest of the "
                                 "12 zoo enhancers on the internal test set",
                three_subjects="low R and low SSIM, by the same criterion",
                weaker_reading="below the zoo median",
                role="controls never enter ranking statistics; a control that scores well on R is "
                     "reported as such"),
        ))
    json.dump({"preregistration": block}, open(STORE, "w"), indent=1)
    print(json.dumps(block, indent=1))


# ------------------------------------------------------------------ summary
B_REP = 10000


def load_rows(mode, split):
    z = np.load(rows_path(mode, split), allow_pickle=False)
    return {k: z[k] for k in z.files}


def retention(rw, cond, key=None):
    """-> (valid mask, R (n_models, n_sites)) with d3arch's keep rule and clamp."""
    t = rw[f"tru_{cond}"]
    ok = (rw[f"ncore_{cond}"] >= 4) & (rw[f"nring_{cond}"] >= 8) & (np.abs(t) >= 0.25 * np.abs(rw["c"]))
    rec = rw[key or f"rec_{cond}"]
    R = np.clip(rec / np.where(np.abs(t) < 1e-6, 1e-6, t), -0.5, 1.5)
    return ok, R


def subj_matrix(rw, ok, V, sids):
    """Per (model, subject) mean of V over valid sites -> (n_models, n_subj)."""
    return np.stack([V[:, ok & (rw["sid"] == s)].mean(1) for s in sids], 1)


def enh_matrix(rw, X):
    """Average runs of the same enhancer -> names, (n_enh, n_subj)."""
    names = list(dict.fromkeys(rw["enhancer"].tolist()))
    return names, np.stack([X[rw["enhancer"] == e].mean(0) for e in names])


def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(n, size=(B_REP, n))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def rank_of(vals, i, lower_better=True):
    """1 = best; ties share the better rank."""
    v = vals if lower_better else -vals
    return int(np.sum(v < v[i]) + 1)


def picks_for(split):
    sfx = {"test": "", "ext_on_hfe": "_ext_on_hfe", "ext_v5": "_ext_v5"}.get(split, "")
    sel = json.load(open(f"{OUT}/d3zoo_stats{sfx}_T2.json"))["selection"]
    return dict(dice=sel["synthseg"]["pick"], ssim=sel["ssim"]["pick"], leader=sel["ret"]["pick"],
                source=f"d3zoo_stats{sfx}_T2.json" + ("" if split in ("test", "ext_on_hfe", "ext_v5")
                                                     else " (internal-test picks; no 64 mT here)"))


def sanity_std(rw, split):
    """Gaussian rows vs d3zoo per-subject retention and erased fraction."""
    zp = f"{OUT}/d3zoo_T2.json" if split == "test" else f"{OUT}/d3zoo_{split}_T2.json"
    z = json.load(open(zp))
    ok, R = retention(rw, "gauss_b")
    dmax, emax, n, cells, dr, bad = 0.0, 0.0, 0, 0, [], {}
    for i, nm in enumerate(rw["models"]):
        if nm not in z: continue
        for k, s in enumerate(z[nm]["sids"]):
            m = ok & (rw["sid"] == s)
            dr.append(abs(R[i, m].mean() - z[nm]["ret"][k]))
            de = abs((R[i, m] < 0.25).mean() - z[nm]["erased"][k])
            dmax, emax, cells = max(dmax, dr[-1]), max(emax, de), cells + 1
            if round(de * m.sum()):
                bad[z[nm]["enhancer"]] = bad.get(z[nm]["enhancer"], 0) + 1
        n += 1
    return dict(models_compared=n, max_abs_diff_ret=float(dmax), median_abs_diff_ret=float(np.median(dr)),
                max_abs_diff_erased=float(emax), subject_cells=cells, cells_with_erased_diff=bad,
                n_sites=int(ok.sum()), passed=bool(dmax < 0.01 and emax < 0.026 and n >= 36))


def threshold_split(rw, split):
    ok, R = retention(rw, "gauss_b")
    sids = sorted(set(rw["sid"][ok].tolist()))
    pk = picks_for(split)
    out = dict(picks=pk, n_sites=int(ok.sum()), n_subjects=len(sids), per_threshold={})
    bi = boot_idx(len(sids))
    names, Rm = enh_matrix(rw, subj_matrix(rw, ok, R, sids))
    zoo = [i for i, e in enumerate(names) if e != NOENH]
    iD, iS, iL = (names.index(pk[k]) for k in ("dice", "ssim", "leader"))
    out["mean_R"] = {e: float(Rm[i].mean()) for i, e in enumerate(names)}
    rz = Rm[zoo].mean(1)
    out["rank_by_mean_R"] = dict(dice=rank_of(rz, zoo.index(iD), False), ssim=rank_of(rz, zoo.index(iS), False),
                                 leader=rank_of(rz, zoo.index(iL), False), n=len(zoo))
    for thr in THRS:
        names, E = enh_matrix(rw, subj_matrix(rw, ok, (R < thr).astype(float), sids))
        ez = E[zoo]
        gap = E[iD] - E[iL]
        reps_gap = gap[bi].mean(1)
        reps_rank = np.array([rank_of(ez[:, b].mean(1), zoo.index(iD)) for b in bi])
        reps_rank_L = np.array([rank_of(ez[:, b].mean(1), zoo.index(iL)) for b in bi])
        em = ez.mean(1)
        out["per_threshold"][f"{thr:.2f}"] = dict(
            erased={e: float(E[i].mean()) for i, e in enumerate(names)},
            erased_ci={e: ci(E[i][bi].mean(1)) for i, e in enumerate(names)},
            gap_dice_minus_leader=float(gap.mean()), gap_ci=ci(reps_gap),
            gap_subjects_positive=int((gap > 0).sum()),
            gap_ssim_minus_leader=float((E[iS] - E[iL]).mean()),
            gap_ssim_ci=ci((E[iS] - E[iL])[bi].mean(1)),
            rank_dice=rank_of(em, zoo.index(iD)), rank_dice_ci=ci(reps_rank),
            rank_ssim=rank_of(em, zoo.index(iS)), rank_leader=rank_of(em, zoo.index(iL)),
            rank_leader_ci=ci(reps_rank_L),
            least_erased=names[zoo[int(np.argmin(em))]],
            holds=bool(gap.mean() > 0 and ci(reps_gap)[0] > 0 and rank_of(em, zoo.index(iD)) >= 7))
    # ECDF on a fixed grid, sites pooled over seeds
    grid = np.linspace(-0.5, 1.5, 81)
    ec = {}
    for k, e in (("dice", pk["dice"]), ("ssim", pk["ssim"]), ("leader", pk["leader"]), ("none", NOENH)):
        v = R[rw["enhancer"] == e][:, ok].ravel()
        ec[k] = dict(enhancer=e, n=int(v.size), cdf=[float((v <= g).mean()) for g in grid],
                     quartiles=[float(q) for q in np.percentile(v, [25, 50, 75])])
    out["ecdf"] = dict(grid=grid.tolist(), curves=ec)
    return out


def disc_split(rw, split):
    """Region-definition and profile effects, on sites valid under all three conditions."""
    from scipy.stats import spearmanr
    oks, Rs = zip(*(retention(rw, c) for c in CONDS))
    ok = oks[0] & oks[1] & oks[2]
    sids = sorted(set(rw["sid"][ok].tolist()))
    bi = boot_idx(len(sids))
    M = {}
    for c, R in zip(CONDS, Rs):
        names, M[c] = enh_matrix(rw, subj_matrix(rw, ok, R, sids))
        names, M[c + "_er"] = enh_matrix(rw, subj_matrix(rw, ok, (R < 0.25).astype(float), sids))
    zoo = [i for i, e in enumerate(names) if e != NOENH]
    pk = picks_for(split)
    iD, iL = names.index(pk["dice"]), names.index(pk["leader"])
    out = dict(n_sites_all3=int(ok.sum()), n_sites_each={c: int(o.sum()) for c, o in zip(CONDS, oks)},
               px_core={c: {str(r): float(rw[f"ncore_{c}"][rw["r_vox"] == r].mean()) for r in STD_RADII} for c in CONDS},
               px_ring={c: {str(r): float(rw[f"nring_{c}"][rw["r_vox"] == r].mean()) for r in STD_RADII} for c in CONDS},
               ring_width={str(r): float(ring_width(r)) for r in STD_RADII},
               mean_R={c: {e: float(M[c][i].mean()) for i, e in enumerate(names)} for c in CONDS},
               erased={c: {e: float(M[c + "_er"][i].mean()) for i, e in enumerate(names)} for c in CONDS},
               by_diam={})
    for a, b, lab in (("gauss_d", "gauss_b", "region_effect"), ("disc_d", "gauss_d", "profile_effect"),
                      ("disc_d", "gauss_b", "total_effect")):
        d = (M[a][zoo] - M[b][zoo]).mean(0)                  # zoo mean, per subject
        out[lab] = dict(delta_zoo_mean_R=float(d.mean()), ci=ci(d[bi].mean(1)),
                        per_enhancer={names[i]: float((M[a][i] - M[b][i]).mean()) for i in zoo},
                        no_enhancer=float((M[a][names.index(NOENH)] - M[b][names.index(NOENH)]).mean()))
        ra, rb = M[a][zoo], M[b][zoo]
        reps = [spearmanr(ra[:, q].mean(1), rb[:, q].mean(1)).statistic for q in bi[:2000]]
        out[lab]["rank_rho"] = float(spearmanr(ra.mean(1), rb.mean(1)).statistic)
        out[lab]["rank_rho_ci"] = ci(np.array(reps))
    for c in CONDS:
        em = M[c + "_er"][zoo].mean(1); rm = M[c][zoo].mean(1)
        gap = M[c + "_er"][iD] - M[c + "_er"][iL]
        out[c] = dict(rank_dice_by_R=rank_of(rm, zoo.index(iD), False), rank_dice_by_erased=rank_of(em, zoo.index(iD)),
                      leader_rank_by_R=rank_of(rm, zoo.index(iL), False), top_by_R=names[zoo[int(np.argmax(rm))]],
                      gap_dice_minus_leader_erased=float(gap.mean()), gap_ci=ci(gap[bi].mean(1)))
    for r in STD_RADII:
        m = ok & (rw["r_vox"] == r)
        out["by_diam"][f"{2 * r} mm"] = {c: float(R[[i for i, e in enumerate(rw["enhancer"]) if e != NOENH]][:, m].mean())
                                         for c, R in zip(CONDS, Rs)}
    return out


def large_split(rw, rs_std=None):
    ok, R = retention(rw, "gauss_b")
    paired = "rec_acq" in rw
    Ra = retention(rw, "gauss_b", "rec_acq")[1] if paired else None
    sids = sorted(set(rw["sid"][ok].tolist()))
    bi = boot_idx(len(sids))
    out = dict(n_sites=int(ok.sum()), n_subjects=len(sids), paired=paired, by_diam={},
               acquired_neighbourhood=dict(
                   radius_vox={f"{2 * r:g} mm": float(max(4 * r, 24) * np.sqrt(np.log(20) / 2)) for r in LARGE_RADII},
                   frac_clipped_by_slab={f"{2 * r:g} mm": float(rw["clip"][rw["r_vox"] == r].mean()) for r in LARGE_RADII},
                   D_response_energy_outside_median={f"{2 * r:g} mm": float(np.median(rw["leak"][rw["r_vox"] == r]))
                                                     for r in LARGE_RADII},
                   D_response_energy_outside_p95={f"{2 * r:g} mm": float(np.percentile(rw["leak"][rw["r_vox"] == r], 95))
                                                  for r in LARGE_RADII},
                   frac_in_brain_mean={f"{2 * r:g} mm": float(rw["near_in_brain"][rw["r_vox"] == r].mean())
                                       for r in LARGE_RADII}))
    for r in LARGE_RADII:
        m = ok & (rw["r_vox"] == r)
        d = {}
        for i, nm in enumerate(rw["models"]):
            per = np.array([R[i, m & (rw["sid"] == s)].mean() if (m & (rw["sid"] == s)).any() else np.nan for s in sids])
            pe = np.array([(R[i, m & (rw["sid"] == s)] < 0.25).mean() if (m & (rw["sid"] == s)).any() else np.nan
                           for s in sids])
            keep = ~np.isnan(per)
            e = dict(n=int(m.sum()), mean_R=float(R[i, m].mean()),
                     mean_R_ci=ci(np.nanmean(per[bi], 1)) if keep.sum() > 1 else None,
                     erased={f"{t:.2f}": float((R[i, m] < t).mean()) for t in THRS},
                     erased_ci=ci(np.nanmean(pe[bi], 1)) if keep.sum() > 1 else None)
            if paired:
                e["mean_R_acq"] = float(Ra[i, m].mean())
                e["erased_acq"] = {f"{t:.2f}": float((Ra[i, m] < t).mean()) for t in THRS}
            d[str(nm)] = e
        out["by_diam"][f"{2 * r:g} mm"] = dict(r_vox=r, diam_mm=2 * r, models=d)
    zoo = [str(nm) for nm in rw["models"] if nm != NOENH]
    out["max_erased_zoo"] = {k: max(v["models"][nm]["erased"]["0.25"] for nm in zoo) for k, v in out["by_diam"].items()}
    if paired:
        out["max_erased_zoo_acq"] = {k: max(v["models"][nm]["erased_acq"]["0.25"] for nm in zoo)
                                     for k, v in out["by_diam"].items()}
    if rs_std is not None:                                # R vs size, 4 to 25 mm, seed-0 models
        ok2, R2 = retention(rs_std, "gauss_b")
        out["std_by_diam"] = {}
        for r in STD_RADII:
            m = ok2 & (rs_std["r_vox"] == r)
            out["std_by_diam"][f"{2 * r} mm"] = {str(nm): float(R2[i, m].mean()) for i, nm in enumerate(rs_std["models"])
                                                 if nm in rw["models"]}
    return out


def controls_summary(splits, zoo_std, ctl_std):
    """Controls vs the zoo range; never added to a ranking."""
    out = {}
    ss = json.load(open(f"{OUT}/d3robust_ssim_ctl_T2.json")) if os.path.exists(f"{OUT}/d3robust_ssim_ctl_T2.json") else {}
    for split in splits:
        if split not in ctl_std: continue
        rz, rc = zoo_std[split], ctl_std[split]
        okz, Rz = retention(rz, "gauss_b"); okc, Rc = retention(rc, "gauss_b")
        assert np.array_equal(rz["cy"], rc["cy"]) and np.array_equal(rz["r_vox"], rc["r_vox"]), "control sites differ"
        n_mis = int((okz != okc).sum()); okz = okc = okz & okc
        sids = sorted(set(rz["sid"][okz].tolist()))
        names, Zr = enh_matrix(rz, subj_matrix(rz, okz, Rz, sids))
        _, Ze = enh_matrix(rz, subj_matrix(rz, okz, (Rz < 0.25).astype(float), sids))
        zoo = [i for i, e in enumerate(names) if e != NOENH]
        zr = Zr[zoo].mean(1)
        cn, Cr = enh_matrix(rc, subj_matrix(rc, okc, Rc, sids))
        _, Ce = enh_matrix(rc, subj_matrix(rc, okc, (Rc < 0.25).astype(float), sids))
        d = dict(keep_rule_mismatches=n_mis,
                 zoo_R=dict(min=float(zr.min()), median=float(np.median(zr)), max=float(zr.max())),
                 no_enhancer_R=float(Zr[names.index(NOENH)].mean()), controls={})
        bi = boot_idx(len(sids))
        zs = None
        if split in ("test", "ext_v5"):
            zp = f"{OUT}/d3zoo_T2.json" if split == "test" else f"{OUT}/d3zoo_{split}_T2.json"
            z = json.load(open(zp))
            ens = sorted({v["enhancer"] for v in z.values()})
            zs = np.array([np.mean([np.mean(v["ssim"]) for v in z.values() if v["enhancer"] == e]) for e in ens])
            d["zoo_SSIM"] = dict(min=float(zs.min()), median=float(np.median(zs)), max=float(zs.max()))
            idn = json.load(open(f"{OUT}/base_identity_T2.json"))
            if split == "test":
                d["no_enhancer_SSIM"] = float(np.mean([r["ssim"] for r in idn["test_rows"]]))
        for k, c in enumerate(cn):
            runs = [str(m) for m, e in zip(rc["models"], rc["enhancer"]) if e == c]
            e = dict(runs=runs, mean_R=float(Cr[k].mean()), mean_R_ci=ci(Cr[k][bi].mean(1)),
                     erased=float(Ce[k].mean()), erased_ci=ci(Ce[k][bi].mean(1)),
                     per_run_R={m: float(Rc[list(rc["models"]).index(m), okc].mean()) for m in runs},
                     R_rank_if_added=rank_of(np.append(zr, Cr[k].mean()), len(zr), False),
                     R_below_zoo_min=bool(Cr[k].mean() < zr.min()),
                     R_below_zoo_median=bool(Cr[k].mean() < np.median(zr)))
            if split in ss:
                sv = [np.mean(ss[split][m]["ssim"]) for m in runs if m in ss[split]]
                pv = [np.mean(ss[split][m]["psnr"]) for m in runs if m in ss[split]]
                if sv:
                    e.update(ssim=float(np.mean(sv)), psnr=float(np.mean(pv)),
                             per_run_ssim={m: float(np.mean(ss[split][m]["ssim"])) for m in runs if m in ss[split]})
                    if zs is not None:
                        e.update(SSIM_rank_if_added=rank_of(np.append(zs, np.mean(sv)), len(zs), False),
                                 SSIM_below_zoo_min=bool(np.mean(sv) < zs.min()),
                                 SSIM_below_zoo_median=bool(np.mean(sv) < np.median(zs)))
            if split == "test" and "ssim" in e:
                e["expected_low_R_and_low_SSIM_met"] = bool(e["R_below_zoo_min"] and e["SSIM_below_zoo_min"])
                e["weaker_reading_met"] = bool(e["R_below_zoo_median"] and e["SSIM_below_zoo_median"])
            d["controls"][c] = e
        out[split] = d
    return out


def figures(res, std, large_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    os.makedirs(FIG, exist_ok=True)
    COL = dict(dice="#2a78d6", ssim="#eb6834", leader="#1baf7a", none="#52514e", zoo="#b9b8b2")
    LAB = dict(dice="SynthSeg-Dice pick", ssim="SSIM pick", leader="retention leader", none="no enhancer")
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    # (a) ECDF, internal test
    th = res["threshold"]["test"]
    g = th["ecdf"]["grid"]
    for k in ("dice", "ssim", "leader", "none"):
        cv = th["ecdf"]["curves"][k]
        ax[0, 0].step(g, cv["cdf"], where="post", color=COL[k], lw=1.4, label=f"{LAB[k]}\n({cv['enhancer']})")
    for t in THRS:
        ax[0, 0].axvline(t, color="#8a8984", lw=0.6, ls=":")
    ax[0, 0].set(xlabel="Retention R", ylabel="Fraction of lesions with R ≤ x", xlim=(-0.5, 1.5), ylim=(0, 1))
    ax[0, 0].legend(frameon=False, fontsize=5.2, loc="lower right")
    ax[0, 0].set_title("a  ECDF of R, internal test", loc="left", fontsize=7.5, fontweight="bold")
    # (b, c) erased vs threshold
    for a, split, lab in ((ax[0, 1], "test", "b  OpenNeuro test"), (ax[0, 2], "ext_v5", "c  Zenodo")):
        t = res["threshold"][split]; pk = t["picks"]
        xs = [float(k) for k in t["per_threshold"]]
        ens = list(t["per_threshold"]["0.25"]["erased"])
        for e in ens:
            ys = [100 * t["per_threshold"][k]["erased"][e] for k in t["per_threshold"]]
            key = next((k for k in ("dice", "ssim", "leader") if pk[k] == e), "none" if e == NOENH else None)
            if key is None:
                a.plot(xs, ys, color=COL["zoo"], lw=0.8, marker="o", ms=2)
            else:
                a.plot(xs, ys, color=COL[key], lw=1.6, marker="o", ms=3.5, label=LAB[key], zorder=3)
        a.set(xlabel="Erasure threshold on R", ylabel="Lesions erased (%)", xticks=xs, ylim=(0, 100))
        a.set_title(lab + " (grey: other enhancers)", loc="left", fontsize=7.5, fontweight="bold")
        if split == "test":
            a.legend(frameon=False, fontsize=5.5, loc="upper left")
    # (d) R vs diameter, seed-0 models, internal test
    L = res["large"]["test"]
    a = ax[1, 0]
    pk = res["threshold"]["test"]["picks"]
    s0 = {str(m): str(e) for m, e in zip(large_rows["test"]["models"], large_rows["test"]["enhancer"])}
    for m, e in s0.items():
        xs = [2 * r for r in STD_RADII] + [2 * r for r in LARGE_RADII]
        ys = [L["std_by_diam"][f"{2 * r} mm"][m] for r in STD_RADII] + \
             [L["by_diam"][f"{2 * r:g} mm"]["models"][m]["mean_R"] for r in LARGE_RADII]
        key = next((k for k in ("dice", "ssim", "leader") if pk[k] == e), "none" if e == NOENH else None)
        a.plot(xs, ys, color=COL[key] if key else COL["zoo"], lw=1.5 if key else 0.8, marker="o",
               ms=3 if key else 2, zorder=3 if key else 1, label=LAB[key] if key else None)
    a.axhline(0.25, color="#8a8984", lw=0.6, ls=":")
    a.axvspan(14, 26, color="#f1f0ec", zorder=0)
    a.set(xlabel="Lesion diameter 2r (mm)", ylabel="Mean retention R", ylim=(0, None))
    a.set_title("d  Size, incl. 16-25 mm control", loc="left", fontsize=7.5, fontweight="bold")
    # (e) disc vs Gaussian, internal test
    a = ax[1, 1]
    dz = res["disc"]["test"]
    ens = [e for e in dz["mean_R"]["gauss_b"] if e != NOENH]
    xb = [dz["mean_R"]["gauss_b"][e] for e in ens]
    a.scatter(xb, [dz["mean_R"]["gauss_d"][e] for e in ens], s=14, facecolor="white", edgecolor=COL["dice"],
              lw=1.0, label="Gaussian, distance regions", zorder=3)
    a.scatter(xb, [dz["mean_R"]["disc_d"][e] for e in ens], s=14, marker="^", color=COL["ssim"],
              label="sharp disc, distance regions", zorder=3)
    lo = min(xb + [dz["mean_R"][c][e] for c in ("gauss_d", "disc_d") for e in ens]) - 0.02
    hi = max(xb + [dz["mean_R"][c][e] for c in ("gauss_d", "disc_d") for e in ens]) + 0.02
    a.plot([lo, hi], [lo, hi], color="#8a8984", lw=0.6, ls="--")
    a.set(xlabel="R, Gaussian with level-set regions (main)", ylabel="R, alternative definition", xlim=(lo, hi), ylim=(lo, hi))
    a.legend(frameon=False, fontsize=5.5, loc="upper left")
    a.set_title("e  Lesion profile and regions", loc="left", fontsize=7.5, fontweight="bold")
    # (f) controls
    a = ax[1, 2]
    c = res["controls"].get("test", {})
    z = json.load(open(f"{OUT}/d3zoo_T2.json"))
    rmean = res["threshold"]["test"]["mean_R"]
    ens = sorted({v["enhancer"] for v in z.values()})
    zs = [np.mean([np.mean(v["ssim"]) for v in z.values() if v["enhancer"] == e]) for e in ens]
    a.scatter(zs, [rmean[e] for e in ens], s=12, color=COL["zoo"], label="zoo enhancers", zorder=2)
    for (nm, v), col, mk in zip(c.get("controls", {}).items(), ("#e34948", "#6250d6"), ("s", "D")):
        if "ssim" in v:
            a.scatter([v["ssim"]], [v["mean_R"]], s=18, color=col, marker=mk, label=f"control: {nm}", zorder=3)
    if "no_enhancer_SSIM" in c:
        a.scatter([c["no_enhancer_SSIM"]], [c["no_enhancer_R"]], s=16, color=COL["none"], marker="x",
                  label="no enhancer", zorder=3)
    a.set(xlabel="SSIM on acquired 64 mT input", ylabel="Mean retention R")
    a.legend(frameon=False, fontsize=5.5, loc="best")
    a.set_title("f  Negative controls", loc="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/robust.pdf"); plt.close(fig)
    # per-cohort ECDFs
    sp = [s for s in ["test", "ext_on_hfe", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"] if s in res["threshold"]]
    fig, ax = plt.subplots(1, len(sp), figsize=(1.6 * len(sp), 1.9), sharey=True)
    for a, s in zip(np.atleast_1d(ax), sp):
        th = res["threshold"][s]
        for k in ("dice", "ssim", "leader", "none"):
            a.step(th["ecdf"]["grid"], th["ecdf"]["curves"][k]["cdf"], where="post", color=COL[k], lw=1.1, label=LAB[k])
        for t in THRS:
            a.axvline(t, color="#8a8984", lw=0.5, ls=":")
        a.set(xlim=(-0.5, 1.5), ylim=(0, 1), xlabel="R"); a.set_title(s, fontsize=7)
    np.atleast_1d(ax)[0].set_ylabel("ECDF"); np.atleast_1d(ax)[0].legend(frameon=False, fontsize=5)
    fig.tight_layout(); fig.savefig(f"{FIG}/robust_ecdf.pdf"); plt.close(fig)


def summary():
    store = json.load(open(STORE))
    assert "preregistration" in store, "run `prereg` first"
    res = {"preregistration": store["preregistration"],
           "size_convention": "r_vox is the 1/e^2 radius of b_r = exp(-2 d^2 / r^2) in 1 mm voxels; diam_mm = 2 r",
           "sanity": {}, "threshold": {}, "disc": {}, "large": {}, "controls": {}}
    std = {s: load_rows("std", s) for s in SPLITS if os.path.exists(rows_path("std", s))}
    for s, rw in std.items():
        res["sanity"][f"std_vs_d3zoo_{s}"] = sanity_std(rw, s)
        print(f"  sanity {s}: {res['sanity'][f'std_vs_d3zoo_{s}']}", flush=True)
        res["threshold"][s] = threshold_split(rw, s)
        if s == "test":                       # unit 2 shares unit 1's 3T, so its rows are test's
            res["threshold"]["ext_on_hfe"] = threshold_split(rw, "ext_on_hfe")
        res["disc"][s] = disc_split(rw, s)
    large_rows = {s: load_rows("large", s) for s in SPLITS if os.path.exists(rows_path("large", s))}
    for s, rw in large_rows.items():
        res["large"][s] = large_split(rw, std.get(s))
        L = res["large"][s]
        L["positive_control_passed"] = bool(max(L["max_erased_zoo"].values()) <= 0.05)
        if "std_by_diam" in L:
            mono = []
            for nm in L["std_by_diam"]["12 mm"]:
                seq = [L["std_by_diam"]["12 mm"][nm]] + [L["by_diam"][f"{2 * r:g} mm"]["models"][nm]["mean_R"]
                                                        for r in LARGE_RADII]
                mono.append(bool(seq[-1] > seq[0]))
            L["R_rises_12_to_25mm_all_models"] = bool(all(mono))
    ctl = {s: load_rows("stdctl", s) for s in SPLITS if os.path.exists(rows_path("stdctl", s))}
    for s, rw in ctl.items():
        res["sanity"][f"ctl_sites_match_{s}"] = bool(np.array_equal(rw["cy"], std[s]["cy"]) and
                                                    np.allclose(rw["tru_gauss_b"], std[s]["tru_gauss_b"]))
    res["controls"] = controls_summary(SPLITS, std, ctl)
    for k in ("std", "large", "stdctl"):
        res.setdefault("row_files", {})[k] = [os.path.basename(rows_path(k, s)) for s in SPLITS
                                              if os.path.exists(rows_path(k, s))]
    json.dump(res, open(STORE, "w"), indent=1)
    if "test" in res["threshold"] and "test" in res["large"] and "test" in res["disc"]:
        figures(res, std, large_rows)
    print_summary(res)


def print_summary(res):
    for s, t in res["threshold"].items():
        p = t["picks"]
        print(f"\n== threshold sweep {s}: Dice pick {p['dice']}, SSIM pick {p['ssim']}, leader {p['leader']}")
        for thr, d in t["per_threshold"].items():
            print(f"  R<{thr}: Dice {100 * d['erased'][p['dice']]:.1f}%  leader {100 * d['erased'][p['leader']]:.1f}%"
                  f"  gap {100 * d['gap_dice_minus_leader']:+.1f} [{100 * d['gap_ci'][0]:+.1f}, {100 * d['gap_ci'][1]:+.1f}]"
                  f"  Dice rank {d['rank_dice']} [{d['rank_dice_ci'][0]:.0f}-{d['rank_dice_ci'][1]:.0f}]"
                  f"  leader rank {d['rank_leader']}  no-enh {100 * d['erased'][NOENH]:.1f}%  holds {d['holds']}")
    for s, d in res["disc"].items():
        print(f"\n== disc {s}: region effect {d['region_effect']['delta_zoo_mean_R']:+.3f} {d['region_effect']['ci']}"
              f" rho {d['region_effect']['rank_rho']:.2f}; profile effect {d['profile_effect']['delta_zoo_mean_R']:+.3f}"
              f" {d['profile_effect']['ci']} rho {d['profile_effect']['rank_rho']:.2f}")
    for s, L in res["large"].items():
        print(f"\n== large {s}: max zoo erased {L['max_erased_zoo']}"
              + (f"  acq {L['max_erased_zoo_acq']}" if "max_erased_zoo_acq" in L else "")
              + f"  passed {L['positive_control_passed']}")
    for s, c in res["controls"].items():
        print(f"\n== controls {s}: zoo R {c['zoo_R']}  " + (f"zoo SSIM {c.get('zoo_SSIM')}" if "zoo_SSIM" in c else ""))
        for nm, v in c["controls"].items():
            print(f"  {nm:18s} R {v['mean_R']:.3f} {v['mean_R_ci']}  erased {100 * v['erased']:.1f}%"
                  + (f"  SSIM {v['ssim']:.3f}" if "ssim" in v else "") + f"  R rank if added {v['R_rank_if_added']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "std", "large", "diag_large", "train_lowdata", "ssim_ctl", "summary"])
    ap.add_argument("--models", default="zoo", choices=["zoo", "ctl"])
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute splits whose rows exist")
    ap.add_argument("--n_subj", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    a = ap.parse_args()
    if a.cmd == "prereg":
        prereg()
    elif a.cmd in ("std", "large"):
        assert os.path.exists(STORE), "run `prereg` first"
        mode = ("std" if a.models == "zoo" else "stdctl") if a.cmd == "std" else "large"
        reg = models(a.models if a.cmd == "std" else "zoo0")
        print(f"[{mode}] {len(reg)} models: {[r[0] for r in reg]}", flush=True)
        for split in a.splits:
            if os.path.exists(rows_path(mode, split)) and not a.force:
                print(f"  {rows_path(mode, split)} exists, skipped"); continue
            o = run_chunked(run_std if a.cmd == "std" else run_large, split, reg, a.chunk)
            np.savez(rows_path(mode, split), **o)
            print(f"  wrote {rows_path(mode, split)}: {len(o['sid'])} sites", flush=True)
    elif a.cmd == "train_lowdata":
        assert os.path.exists(STORE), "run `prereg` first"
        train_lowdata(a.n_subj, a.seed, a.steps)
    elif a.cmd == "diag_large":
        diag_large(tuple(s for s in a.splits if D.PairedCohort(s, "T2").ulf is not None))
    elif a.cmd == "ssim_ctl":
        ssim_ctl()
    else:
        summary()
