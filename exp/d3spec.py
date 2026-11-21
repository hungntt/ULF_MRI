"""Does an enhancer that keeps lesions also invent them? Specificity of the zoo.

Retention measures sensitivity: how much of an inserted lesion survives. A model can raise it
by hallucinating focal structure everywhere, so retention alone cannot rank enhancers. This
script measures the other side on lesion-free inputs, with three quantities (plan A2):

  1. Acquired residual blobs. On the acquired 64 mT scan x_real of paired cohorts the residual
     e = G(x_real) - y_3T is searched for focal blobs with a scale-normalised LoG at the inserted
     radii (r = 2, 3, 4, 6 voxels, i.e. diameters 4, 6, 8, 12 mm; sigma = r/2 for the profile
     exp(-2 d^2 / r^2)), with the blob contrast measured exactly as for retention (core b > 0.6
     minus ring 0.05 < b < 0.25) and a threshold |contrast| >= 0.12, the smallest inserted
     contrast. The blob disc (1.25 r) must lie inside the brain mask eroded by one voxel. A false
     lesion is a blob of e with no same-sign blob of the input residual x_real - y_3T within
     2 voxels. Reported per 100 slices and per 1000 cm^3, for positive (hyperintense, the polarity
     of the inserted lesions) and negative blobs, with the no-enhancer row (G = identity) as the
     reference. Because the identity row has no new blobs by construction, the total blob rate of
     e (new + inherited from the input) is reported next to the new-blob rate; the total rate of
     the identity row is the input's own noise floor.
  2. Test-retest. The five OpenNeuro test subjects were scanned on two Swoop units (HFC, HFE),
     registered to the same 3T. Blobs of G(x_HFC) without a same-sign blob of G(x_HFE) within
     2 voxels, and vice versa, are unstable; the same on the residuals to the 3T. Rates per
     100 slices, identity row as floor.
  3. Closed loop, from the A1 rows (d3stage): the fraction of lesion sites with R > 1 and R > 1.5
     and the spill-over (energy of dp outside the dilated footprint over the core energy).

The rule asked for by the plan: which enhancer has the highest retention subject to a
false-lesion rate no higher than the no-enhancer reference, and do pix2pix and LoHiResGAN stay
on top. Both readings of the reference are evaluated (new blobs; total blobs). Thresholds are
the plan's and are not tuned.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3spec.py [splits...]
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F_
sys.path.insert(0, ".")
from ulfkit import data as D, train as T
from exp.d3dfactor import RADII
from exp.d3stage import Ident, registry_all, NOENH, boot_subject_mean
from exp import d3zoo as Z

OUT = os.environ.get("ULF_OUT", "./out")
FIG = os.environ.get("ULF_FIG", "./fig")
THR_C = 0.12
TOL = 2
ACQ = ["test", "ext_on_hfe", "ext_v5"]


def gauss_k(sig):
    n = int(math.ceil(4 * sig)); x = torch.arange(-n, n + 1, device="cuda").float()
    k = torch.exp(-x ** 2 / (2 * sig ** 2)); return k / k.sum()


LAP = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda")[None, None]


def log_norm(img, sig):
    """Scale-normalised LoG, bright blobs positive: -sigma^2 * laplacian(G_sigma * img)."""
    k = gauss_k(sig); n = len(k) // 2
    g = F_.conv2d(F_.pad(img, (n, n, 0, 0), mode="reflect"), k[None, None, None, :])
    g = F_.conv2d(F_.pad(g, (0, 0, n, n), mode="reflect"), k[None, None, :, None])
    return -(sig ** 2) * F_.conv2d(F_.pad(g, (1, 1, 1, 1), mode="reflect"), LAP)


def kernels(r):
    """Core and ring disc kernels of the lesion profile b = exp(-2 d^2 / r^2), unit sum each."""
    h = int(math.ceil(1.25 * r)); a = torch.arange(-h, h + 1, device="cuda").float()
    d2 = a[:, None] ** 2 + a[None, :] ** 2
    b = torch.exp(-2 * d2 / r ** 2)
    core, ring = (b > 0.6).float(), ((b > 0.05) & (b < 0.25)).float()
    return (core / core.sum())[None, None], (ring / ring.sum())[None, None], h


KER = {r: kernels(r) for r in RADII}
DISK = None


def disk(rad):
    a = torch.arange(-rad, rad + 1, device="cuda").float()
    return ((a[:, None] ** 2 + a[None, :] ** 2) <= rad ** 2).float()[None, None]


DISK = disk(TOL)


def detect(img, mask):
    """img (B,1,H,W), mask (B,1,H,W) float brain mask (already eroded by 1 voxel).
    -> (pos, neg) float maps (B,1,H,W): 1 at the centre of each blob with |core-ring| >= 0.12."""
    S, C, IN = [], [], []
    for r in RADII:
        core, ring, h = KER[r]
        S.append(log_norm(img * mask, r / 2.0))
        C.append(F_.conv2d(F_.pad(img * mask, (h,) * 4), core) - F_.conv2d(F_.pad(img * mask, (h,) * 4), ring))
        IN.append((-F_.max_pool2d(-mask, 2 * h + 1, 1, h)) > 0.5)         # blob disc inside the mask
    S, C, IN = torch.cat(S, 1), torch.cat(C, 1), torch.stack(IN, 1)[:, :, 0]
    out = []
    for sgn in (1.0, -1.0):
        s = sgn * S
        nms = F_.max_pool2d(F_.max_pool2d(s, 3, 1, 1).amax(1, keepdim=True), 1)   # 3x3 and across scales
        peak = (s >= nms - 1e-9) & (s > 1e-4)
        hit = peak & IN & (sgn * C >= THR_C)
        out.append(hit.any(1, keepdim=True).float())
    return out[0], out[1]


def near(m):
    return F_.conv2d(F_.pad(m, (TOL,) * 4), DISK) > 0


def erode1(y):
    return -F_.max_pool2d(-(y > 0).float(), 3, 1, 1)


@torch.no_grad()
def run_split(split, nets, with_hfe=None):
    """-> per model: per-subject counts; plus test-retest if `with_hfe` (the HFE cohort)."""
    coh = D.PairedCohort(split, "T2")
    other = D.PairedCohort(with_hfe, "T2") if with_hfe else None
    res = {nm: [] for nm, _ in nets}
    for sid in coh.subjects:
        idx = coh.subject_rows(sid)
        acc = {nm: dict(slices=0, vol=0.0, pos=0, neg=0, fpos=0, fneg=0,
                        tpos_out=0, tneg_out=0, tpos_res=0, tneg_res=0, u_out=0, u_res=0) for nm, _ in nets}
        for i0 in range(0, len(idx), 16):
            j = idx[i0:i0 + 16]
            y = torch.from_numpy(coh.hf[j].astype(np.float32)).cuda()
            x = torch.from_numpy(coh.ulf[j].astype(np.float32)).cuda()
            xo = torch.from_numpy(other.ulf[j].astype(np.float32)).cuda() if other is not None else None
            M = erode1(y)
            ein = x[:, 1:2] - y
            ip, ineg = detect(ein, M)
            nip, nin = near(ip), near(ineg)
            for nm, net in nets:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    g = net(x).float().clamp(0, 1)
                e = g - y
                bp, bn = detect(e, M)
                a = acc[nm]
                a["slices"] += len(j); a["vol"] += float(M.sum()) / 1000.0
                a["pos"] += int(bp.sum()); a["neg"] += int(bn.sum())
                a["fpos"] += int((bp * ~nip).sum()); a["fneg"] += int((bn * ~nin).sum())
                if xo is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        go = net(xo).float().clamp(0, 1)
                    for tag, (A, B) in {"out": (g, go), "res": (g - y, go - y)}.items():
                        ap, an = detect(A, M); cp, cn = detect(B, M)
                        u = int((ap * ~near(cp)).sum() + (an * ~near(cn)).sum()
                                + (cp * ~near(ap)).sum() + (cn * ~near(an)).sum())
                        a["u_" + tag] += u
                        a["tpos_" + tag] += int(ap.sum() + cp.sum()); a["tneg_" + tag] += int(an.sum() + cn.sum())
        for nm, _ in nets:
            a = acc[nm]; a["sid"] = sid; res[nm].append(a)
        print(f"    {split} {sid} done", flush=True)
    return res


def rates(rows, key, per="slices"):
    """Per-subject rate per 100 slices (or per 1000 cm^3)."""
    return np.array([100.0 * r[key] / r["slices"] if per == "slices" else 1000.0 * r[key] / r["vol"] for r in rows])


def main(splits):
    T.seed_all(0)
    reg = registry_all()
    todo = [(NOENH, None)] + [(r[0], r) for r in reg]
    out = dict(meta=dict(
        lesion_convention="r_vox = 1/e^2 radius in 1 mm voxels; diam_mm = 2 r_vox",
        r_vox=list(RADII), diam_mm=[2 * r for r in RADII], contrast_threshold=THR_C, match_tolerance_vox=TOL,
        splits=splits,
        note="identity row has no new blobs by construction; its total blob rate is the input noise floor",
        negative_controls="absent from out_public; not trained in this round"), splits={})
    for split in splits:
        coh = D.PairedCohort(split, "T2")
        if coh.ulf is None:
            continue
        hfe = "ext_on_hfe" if split == "test" else None
        per = {}
        for c0 in range(0, len(todo), 6):
            part = todo[c0:c0 + 6]
            nets = [(NOENH, Ident().cuda()) if r is None else (r[0], Z.load(r[4], r[3])) for _, r in part]
            print(f"  [{split}] chunk {[n for n, _ in nets]}", flush=True)
            per.update(run_split(split, nets, hfe))
            del nets; torch.cuda.empty_cache()
        meta = {r[0]: (r[1], r[2]) for _, r in todo if r}
        meta[NOENH] = (NOENH, 0)
        sp = dict(subjects=coh.subjects, per_model={}, per_enhancer={})
        for nm, rows in per.items():
            d = dict(enhancer=meta[nm][0], seed=meta[nm][1], rows=rows)
            for per_u in ("slices", "vol"):
                u = "per100slices" if per_u == "slices" else "per1000cm3"
                for k_out, key in (("new_pos", "fpos"), ("new_neg", "fneg"), ("total_pos", "pos"), ("total_neg", "neg")):
                    d[f"{k_out}_{u}"] = rates(rows, key, per_u).tolist()
                d[f"new_abs_{u}"] = (rates(rows, "fpos", per_u) + rates(rows, "fneg", per_u)).tolist()
                d[f"total_abs_{u}"] = (rates(rows, "pos", per_u) + rates(rows, "neg", per_u)).tolist()
            if hfe:
                d["unstable_out_per100slices"] = rates(rows, "u_out").tolist()
                d["unstable_res_per100slices"] = rates(rows, "u_res").tolist()
                d["blobs_out_per100slices"] = ((rates(rows, "tpos_out") + rates(rows, "tneg_out")) / 2).tolist()
                d["blobs_res_per100slices"] = ((rates(rows, "tpos_res") + rates(rows, "tneg_res")) / 2).tolist()
            sp["per_model"][nm] = d
        # enhancer-level (seeds averaged), subject bootstrap
        keys = [k for k in sp["per_model"][NOENH] if k.endswith(("per100slices", "per1000cm3"))]
        for e in sorted({v["enhancer"] for v in sp["per_model"].values()}, key=lambda s: (s != NOENH, s)):
            nms = [n for n, v in sp["per_model"].items() if v["enhancer"] == e]
            ent = dict(n_models=len(nms))
            for k in keys:
                v = np.mean([sp["per_model"][n][k] for n in nms], 0)
                mu, ci = boot_subject_mean(v)
                ent[k] = dict(mean=mu, ci=ci, per_subject=v.tolist())
            sp["per_enhancer"][e] = ent
        out["splits"][split] = sp
        json.dump(out, open(f"{OUT}/d3spec_T2.json.partial", "w"))
    out["detector_calibration"] = calibrate()
    out["closed_loop"] = closed_loop()
    out["selection"] = selection(out)
    json.dump(out, open(f"{OUT}/d3spec_T2.json", "w"), indent=1)
    os.remove(f"{OUT}/d3spec_T2.json.partial")
    figure(out, f"{FIG}/spec_retention_flr.pdf")
    print_summary(out)


@torch.no_grad()
def calibrate(n=64, seed=0):
    """Detector recall for inserted lesions (c, r) on a zero residual at random interior sites, and
    the measured core-minus-ring contrast of each: what the fixed 0.12 threshold means in nominal c."""
    from exp.d3detect import site_pool, blobs
    coh = D.PairedCohort("test", "T2"); rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(coh.hf), 32, replace=False))
    y = torch.from_numpy(coh.hf[rows].astype(np.float32)).cuda(); ok = site_pool(y); M = erode1(y)
    sites = []
    for s in range(len(y)):
        cand = torch.nonzero(ok[s])
        if len(cand) >= 8:
            sites += [(s,) + tuple(cand[rng.integers(len(cand))].tolist()) for _ in range(2)]
    sites = sites[:n]
    si = torch.tensor([t[0] for t in sites], device="cuda")
    cy = torch.tensor([t[1] for t in sites], device="cuda"); cx = torch.tensor([t[2] for t in sites], device="cuda")
    out = {}
    for r in RADII:
        b = blobs(cy, cx, torch.full_like(cy, r)) * (y[si] > 0).float()
        core, ring = (b > 0.6).float(), ((b > 0.05) & (b < 0.25)).float()
        for c in (0.12, 0.25):
            img = c * b
            meas = ((img * core).sum((1, 2, 3)) / core.sum((1, 2, 3)) - (img * ring).sum((1, 2, 3)) / ring.sum((1, 2, 3)))
            pm, _ = detect(img, M[si])
            hit = torch.stack([(near(pm[k:k + 1])[0, 0, cy[k], cx[k]]) for k in range(len(sites))])
            out[f"diam{2 * r}mm_c{c}"] = dict(r_vox=r, diam_mm=2 * r, c=c, measured_contrast=float(meas.mean()),
                                              recall=float(hit.float().mean()))
    return out


def closed_loop():
    """Measure 3 from the A1 rows: R > 1, R > 1.5 and spill-over per enhancer, per cohort and operator."""
    p = f"{OUT}/d3stage_T2.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    cl = {}
    for split, sp in s["splits"].items():
        cl[split] = {o: {e: dict(frac_R_gt_1=d["frac_R_gt_1"], frac_R_gt_1p5=d["frac_R_gt_1p5"],
                                 spill_median=d["spill_median"], R=d["R"]["mean"])
                         for e, d in sp["per_enhancer"][o].items()} for o in sp["per_enhancer"]}
    return cl


def selection(out):
    """Max retention subject to FLR <= no-enhancer reference (two readings), on the primary cohort."""
    p = f"{OUT}/d3stage_T2.json"
    if not os.path.exists(p) or "test" not in out["splits"]:
        return None
    st = json.load(open(p))["splits"]["test"]["per_enhancer"]["HFC"]
    sp = out["splits"]["test"]["per_enhancer"]
    zoo = [e for e in sp if e != NOENH]
    res = {}
    for reading, key in (("new blobs (identity = 0 by construction)", "new_abs_per100slices"),
                         ("total blobs of the residual (identity = input noise floor)", "total_abs_per100slices")):
        floor = sp[NOENH][key]["mean"]
        ok = [e for e in zoo if sp[e][key]["mean"] <= floor + 1e-12]
        order = sorted(zoo, key=lambda e: -st[e]["R"]["mean"])
        win = max(ok, key=lambda e: st[e]["R"]["mean"]) if ok else None
        res[reading] = dict(floor=floor, admissible=ok, winner=win, retention_order=order,
                            unconstrained_leader=order[0],
                            flr={e: sp[e][key]["mean"] for e in zoo})
    return res


def figure(out, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = f"{OUT}/d3stage_T2.json"
    if not os.path.exists(p) or "test" not in out["splits"]:
        return
    st = json.load(open(p))["splits"]["test"]["per_enhancer"]["HFC"]
    sp = out["splits"]["test"]["per_enhancer"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, ttl in ((axes[0], "new_abs_per100slices", "new blobs (not in the input residual)"),
                         (axes[1], "total_abs_per100slices", "all residual blobs")):
        for e, d in sp.items():
            x = st[e]["R"]["mean"]; y = d[key]["mean"]; lo, hi = d[key]["ci"]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o" if e != NOENH else "s", ms=5,
                        color="k" if e == NOENH else "#0072B2", elinewidth=0.6, capsize=1.5)
            ax.annotate(e if e != NOENH else "no enhancer", (x, y), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
        ax.axhline(sp[NOENH][key]["mean"], color="grey", lw=0.6, ls=":")
        ax.set_xlabel("retention R (closed loop, HFC operator)", fontsize=8)
        ax.set_ylabel("false lesions per 100 slices", fontsize=8); ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def print_summary(out):
    for split, sp in out["splits"].items():
        print(f"\n== {split}: false lesions (|contrast| >= {THR_C}) per 100 slices, mean over subjects")
        print(f"  {'enhancer':28s} {'new':>8s} {'total':>8s} {'new+':>7s} {'new-':>7s}" +
              (f" {'unstab.out':>10s} {'unstab.res':>10s}" if 'unstable_out_per100slices' in sp['per_enhancer'][NOENH] else ""))
        for e, d in sp["per_enhancer"].items():
            line = (f"  {e:28s} {d['new_abs_per100slices']['mean']:8.2f} {d['total_abs_per100slices']['mean']:8.2f} "
                    f"{d['new_pos_per100slices']['mean']:7.2f} {d['new_neg_per100slices']['mean']:7.2f}")
            if "unstable_out_per100slices" in d:
                line += f" {d['unstable_out_per100slices']['mean']:10.2f} {d['unstable_res_per100slices']['mean']:10.2f}"
            print(line)
    if out.get("selection"):
        for k, v in out["selection"].items():
            print(f"\n  rule [{k}]: floor {v['floor']:.2f}; admissible {v['admissible']}; winner {v['winner']}; "
                  f"unconstrained leader {v['unconstrained_leader']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ACQ)
