"""Which ATLAS stroke lesions can be transplanted, and is the contralateral mirror a good counterfactual?

Step S1-S2 of the ATLAS analysis plan. ATLAS R3.0 (Liew et al. 2018, Sci Data 5:180011; Liew et al. 2022,
Sci Data 9:320; INDI data-use agreement) gives 1,453 chronic-stroke T1w sessions from 55 sites with
manual lesion masks, also resampled to MNI152NLin2009aSym (197 x 233 x 189, 1 mm, intensity 0-100).

Retention needs a lesion-free counterfactual for each real lesion. Inpainting failed its bias test on
the MS lesions (reports/review_C.md, median |bias| 0.24 of the contrast) and is worse for large
infarcts. The symmetric template offers a better one for unilateral stroke: the same subject's
contralateral hemisphere, x -> -x (voxel i -> 196 - i; x = 0 is voxel 98). This script
  index      lists the MNI sessions, their native voxel size (raw form) and metadata
  template   a lesion-excluded mean T1 over 300 small-lesion sessions, and mri_synthseg on it
             (tissue classes in MNI space: cortex, cerebral WM, deep GM, CSF)
  components lesion components (26-connectivity) per session, with size, side, tissue, distance to
             CSF, and whether the mirror region is lesion-free (the "mirror-clean" flag); and the
             core-minus-ring fractional contrast of y against its mirror
  qc         pseudo-lesions: each sampled lesion's core/ring masks applied to a control session
             where both the region and its mirror are lesion-free. The true contrast is 0, so the
             measured one is the counterfactual error. Pre-specified pass criterion (written to
             the JSON before any pseudo-lesion is measured): median |pseudo contrast| < 10% of the
             median |lesion contrast|, overall and in each size stratum.
  select     up to 150 mirror-clean components per stratum (4-12, 12-25, > 25 mm equivalent
             diameter), at most 4 per site per stratum and 2 per session, seed 0; writes each one's
             fractional contrast map m = (y - mirror(y)) / mirror(y) on its support (mask dilated
             2 voxels) and its core / ring masks, for the transplant (exp/d3atlas.py).

Fractional contrast is taken only where the mirror is tissue (> 20% of the template WM intensity);
elsewhere m = 0. Sizes are equivalent diameters of the component volume in the 1 mm MNI space.

    ULF_SCRATCH=datasets/scratch_public ULF_OUT=out_public python exp/prep_atlas.py index|template|components|qc|select
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "."
RAW = f"{ROOT}/datasets/raw/atlas/R3.0"
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch_public")
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
W = f"{SCR}/atlas"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
NX, MID = 197, 98                          # MNI sym grid: x = 0 at voxel 98, mirror i -> 196 - i
STRATA = [("4-12 mm", 4.0, 12.0), ("12-25 mm", 12.0, 25.0), (">25 mm", 25.0, 1e9)]
CORE_ERODE, SUP_DIL, RING = 1, 2, (2, 4)   # voxels (1 mm): core = mask eroded 1; support = mask dilated 2; ring 2-4 mm
MIRROR_MARGIN = 3                          # lesion-free margin (vox) required around the mirrored region
TISSUE_FRAC = 0.2                          # m only where the mirror > 20% of the template WM intensity
PER_STRATUM, PER_SITE, PER_SESSION, SEED = 150, 4, 2, 0
QC_CRIT = 0.10
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44, 72)
CORTEX, CWM = (3, 42), (2, 41)
DEEP_GM = (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)


def nib():
    import nibabel
    return nibabel


def load(p):
    return np.asarray(nib().load(p).dataobj, dtype=np.float32)


def mirror(a):
    return a[::-1].copy()              # axis 0 is x; NX = 197 and MID = 98, so index i -> 196 - i


# ------------------------------------------------------------------ index
def index():
    os.makedirs(W, exist_ok=True)
    rows = []
    for t in sorted(glob.glob(f"{RAW}/preprocessed/ATLAS3_Training_Preprocessed/*/sub-*/ses-1/anat/*desc-brainnorm_T1w.nii.gz")):
        d = os.path.dirname(t); sid = os.path.basename(t).split("_ses-1")[0]; site = t.split("/")[-5]
        m = [f for f in glob.glob(f"{d}/*label-lesion_desc-T1lesion_mask.nii.gz") if os.path.getsize(f) > 0]  # R3.0 ships one empty mask (sub-r001s005)
        raw = glob.glob(f"{RAW}/raw/ATLAS3_Training_Raw/{site}/{sid}/ses-1/anat/*desc-brain_T1w.nii.gz")
        meta = {}
        mc = glob.glob(f"{d}/*metadata.csv")
        if mc:
            lines = open(mc[0]).read().strip().splitlines()
            if len(lines) == 2: meta = dict(zip(lines[0].split(","), lines[1].split(",")))
        vox = [float(v) for v in nib().load(raw[0]).header.get_zooms()[:3]] if raw else None
        rows.append(dict(sid=sid, site=site, t1=t, mask=m[0] if m else None, native_vox=vox,
                         native_max=max(vox) if vox else None, days=meta.get("DAYS_POST_STROKE", ""),
                         chronicity=meta.get("CHRONICITY", "")))
    json.dump(rows, open(f"{W}/index.json", "w"), indent=1)
    hi = sum(1 for r in rows if r["native_max"] and r["native_max"] <= 1.5)
    print(f"{len(rows)} sessions, {len({r['site'] for r in rows})} sites; native max voxel <= 1.5 mm: {hi}")


# ------------------------------------------------------------------ template
def _lesion_vol(r):
    return r["sid"], float(load(r["mask"]).sum())


def template(n=300, workers=6):
    rows = [r for r in json.load(open(f"{W}/index.json")) if r["mask"] and r["native_max"] and r["native_max"] <= 1.5]
    with ProcessPoolExecutor(workers) as ex:
        vols = dict(ex.map(_lesion_vol, rows, chunksize=8))
    pick = sorted(rows, key=lambda r: vols[r["sid"]])[:n]
    acc = np.zeros((NX, 233, 189), np.float64); cnt = np.zeros_like(acc)
    for r in pick:
        y, m = load(r["t1"]), load(r["mask"]) > 0
        ok = (y > 0) & ~m
        acc[ok] += y[ok]; cnt[ok] += 1
    mean = np.where(cnt > n * 0.5, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    aff = nib().load(pick[0]["t1"]).affine
    nib().save(nib().Nifti1Image(mean, aff), f"{W}/template_T1.nii.gz")
    json.dump(dict(n=n, max_lesion_ml=vols[pick[-1]["sid"]] / 1000, sids=[r["sid"] for r in pick]),
              open(f"{W}/template_sessions.json", "w"), indent=1)
    env = dict(os.environ, FREESURFER_HOME=FS)
    cmd = [f"{FS}/bin/mri_synthseg", "--i", f"{W}/template_T1.nii.gz", "--o", f"{W}/template_seg.nii.gz",
           "--robust", "--threads", "4", "--cpu", "--resample", f"{W}/template_seg_input.nii.gz"]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: raise RuntimeError(r.stdout[-1500:])
    seg = np.asarray(nib().load(f"{W}/template_seg.nii.gz").dataobj)
    if seg.shape != mean.shape:                  # resample labels back to the MNI grid by the affines
        import scipy.ndimage as ndi
        simg = nib().load(f"{W}/template_seg.nii.gz")
        M = np.linalg.inv(simg.affine) @ aff
        seg = ndi.affine_transform(seg, M[:3, :3], M[:3, 3], output_shape=mean.shape, order=0)
        nib().save(nib().Nifti1Image(seg.astype(np.int16), aff), f"{W}/template_seg.nii.gz")
    wm = float(np.median(mean[np.isin(seg, CWM)]))
    sym = float(np.corrcoef(mean[(mean > 0) & (mirror(mean) > 0)], mirror(mean)[(mean > 0) & (mirror(mean) > 0)])[0, 1])
    json.dump(dict(wm_median=wm, lr_symmetry_corr=sym), open(f"{W}/template_stats.json", "w"))
    print(f"template from {n} sessions (lesion <= {vols[pick[-1]['sid']] / 1000:.2f} ml); WM median {wm:.1f}; "
          f"left-right symmetry r {sym:.4f}")


# ------------------------------------------------------------------ components
def _tpl():
    seg = np.asarray(nib().load(f"{W}/template_seg.nii.gz").dataobj).astype(np.int16)
    st = json.load(open(f"{W}/template_stats.json"))
    return seg, st["wm_median"]


def regions(comp, others, brain, csf, crop=None):
    """core, ring, support masks (bool, full volume) for one component."""
    import scipy.ndimage as ndi
    dist_out = ndi.distance_transform_edt(~comp)
    core = ndi.binary_erosion(comp, iterations=CORE_ERODE)
    if core.sum() < 4: core = comp.copy()
    sup = dist_out <= SUP_DIL
    ring = (dist_out > RING[0]) & (dist_out <= RING[1]) & brain & ~csf & ~others   # (2, 4] mm, as d3reallesion
    return core, ring, sup, dist_out <= RING[1]


def _box(comp, margin):
    idx = np.argwhere(comp); lo = np.maximum(idx.min(0) - margin, 0); hi = np.minimum(idx.max(0) + margin + 1, comp.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def contrast_frac(y, cf, core, ring, wm):
    """core-minus-ring mean of the fractional difference (y - cf) / cf, tissue voxels of cf only."""
    t = cf > TISSUE_FRAC * wm
    m = np.where(t, (y - cf) / np.maximum(cf, 1e-6), 0.0)
    c, r = core & t, ring & t
    if c.sum() < 4 or r.sum() < 8: return np.nan
    return float(m[c].mean() - m[r].mean())


def _components(r):
    import scipy.ndimage as ndi
    seg, wm = _tpl()
    brain = seg > 0; csf = np.isin(seg, CSF_LABELS)
    y, les = load(r["t1"]), load(r["mask"]) > 0
    cf = mirror(y)
    lab, n = ndi.label(les, structure=np.ones((3, 3, 3)))
    out = []
    if n == 0: return out
    sizes = np.bincount(lab.ravel())[1:]
    les_dil_m = mirror(ndi.binary_dilation(les, iterations=MIRROR_MARGIN))
    for k in range(1, n + 1):
        v = int(sizes[k - 1]); d = 2 * (3 * v / (4 * np.pi)) ** (1 / 3)
        if d < 4.0: continue
        comp = lab == k
        bx = _box(comp, RING[1] + SUP_DIL + MIRROR_MARGIN + 2)
        c_ = comp[bx]; others = (les & ~comp)[bx]; others = ndi.binary_dilation(others, iterations=1)
        core, ring, sup, reg = regions(c_, others, brain[bx], csf[bx])
        cidx = np.argwhere(comp).mean(0)
        x_mm = float(cidx[0] - MID)
        reg_full = np.zeros_like(comp); reg_full[bx] = reg
        mirror_clean = not bool((reg_full & les_dil_m).any())          # mirrored lesion + ring region hits no lesion
        midline = abs(x_mm) < 8 or bool(reg_full[MID - 2:MID + 3].any())
        segc = seg[bx][c_]
        frac = lambda labs: float(np.isin(segc, labs).mean())
        out.append(dict(sid=r["sid"], site=r["site"], comp=k, vox=v, diam=float(d), x_mm=x_mm,
                        centroid=[float(c) for c in cidx], side="L" if x_mm < 0 else "R",
                        in_brain=float(brain[bx][c_].mean()), cortex=frac(CORTEX), wm=frac(CWM),
                        deep_gm=frac(DEEP_GM), csf=frac(CSF_LABELS),
                        mirror_clean=mirror_clean and not midline, midline=midline,
                        n_core=int(core.sum()), n_ring=int(ring.sum()),
                        c_frac=contrast_frac(y[bx], cf[bx], core, ring, wm),
                        c_abs=float(y[bx][core].mean() - y[bx][ring].mean()) if ring.sum() >= 8 else np.nan,
                        native_max=r["native_max"]))
    return out


def components(workers=6):
    rows = [r for r in json.load(open(f"{W}/index.json")) if r["mask"]]
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(_components, rows, chunksize=4)):
            res += o
            if (i + 1) % 200 == 0: print(f"  {i + 1}/{len(rows)} sessions, {len(res)} components >= 4 mm", flush=True)
    json.dump(res, open(f"{W}/components.json", "w"))
    mc = [c for c in res if c["mirror_clean"]]
    print(f"{len(res)} components >= 4 mm from {len(rows)} sessions; mirror-clean {len(mc)}")
    for nm, lo, hi in STRATA:
        a = [c for c in res if lo <= c["diam"] < hi]; b = [c for c in a if c["mirror_clean"]]
        print(f"  {nm:9s} all {len(a):5d}  mirror-clean {len(b):5d}  median contrast (frac) "
              f"{np.nanmedian([c['c_frac'] for c in b]):+.3f}")


# ------------------------------------------------------------------ qc
def _qc_one(job):
    """Pseudo-lesion: the lesion's core/ring masks (in place) on a control session whose region and
    mirror are lesion-free; returns the fractional contrast of y against its mirror (truth 0)."""
    import scipy.ndimage as ndi
    lesion_row, ctl_row, comp_k = job
    seg, wm = _tpl()
    brain = seg > 0; csf = np.isin(seg, CSF_LABELS)
    les = load(lesion_row["mask"]) > 0
    lab, _ = ndi.label(les, structure=np.ones((3, 3, 3)))
    comp = lab == comp_k
    bx = _box(comp, RING[1] + SUP_DIL + MIRROR_MARGIN + 2)
    c_ = comp[bx]
    core, ring, sup, reg = regions(c_, np.zeros_like(c_), brain[bx], csf[bx])
    yc, lc = load(ctl_row["t1"]), load(ctl_row["mask"]) > 0
    lcd = ndi.binary_dilation(lc, iterations=MIRROR_MARGIN)
    reg_full = np.zeros_like(comp); reg_full[bx] = reg
    if (reg_full & (lcd | mirror(lcd))).any(): return None
    return contrast_frac(yc[bx], mirror(yc)[bx], core, ring, wm)


def qc(n_per_stratum=100, workers=6):
    comps = json.load(open(f"{W}/components.json"))
    idx = {r["sid"]: r for r in json.load(open(f"{W}/index.json"))}
    pre = dict(written=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
               criterion=f"median |pseudo-lesion contrast| < {QC_CRIT:.0%} of median |lesion contrast| (fractional, "
                         "core minus ring of (y - mirror(y)) / mirror(y)), overall and per stratum",
               n_per_stratum=n_per_stratum, seed=SEED)
    res = dict(prespec=pre)
    json.dump(res, open(f"{OUT}/atlas_mirror_qc.json", "w"), indent=1)       # written before measuring
    rng = np.random.default_rng(SEED)
    ctl_pool = [r for r in idx.values() if r["mask"] and r["native_max"] and r["native_max"] <= 1.5]
    allp, alll = [], []
    for nm, lo, hi in STRATA:
        cand = [c for c in comps if lo <= c["diam"] < hi and c["mirror_clean"] and c["in_brain"] >= 0.98
                and np.isfinite(c["c_frac"])]
        pick = [cand[i] for i in rng.permutation(len(cand))[:n_per_stratum]]
        jobs = []
        for c in pick:
            for _ in range(6):                     # up to 6 tries for a clean control
                ctl = ctl_pool[rng.integers(len(ctl_pool))]
                if ctl["sid"] != c["sid"]: jobs.append((idx[c["sid"]], ctl, c["comp"]))
        with ProcessPoolExecutor(workers) as ex:
            got = list(ex.map(_qc_one, jobs, chunksize=2))
        ps, seen = [], set()
        for (lr, ctl, k), g in zip(jobs, got):
            key = (lr["sid"], k)
            if g is None or not np.isfinite(g) or key in seen: continue
            seen.add(key); ps.append(g)
        lc = np.array([c["c_frac"] for c in pick]); ps = np.array(ps)
        ratio = float(np.median(np.abs(ps)) / np.median(np.abs(lc)))
        res[nm] = dict(n_lesions=len(pick), n_pseudo=len(ps), lesion_median=float(np.median(lc)),
                       lesion_median_abs=float(np.median(np.abs(lc))), pseudo_median=float(np.median(ps)),
                       pseudo_median_abs=float(np.median(np.abs(ps))),
                       pseudo_p95_abs=float(np.percentile(np.abs(ps), 95)), ratio=ratio, passed=ratio < QC_CRIT)
        allp += list(ps); alll += list(lc)
        print(f"  {nm:9s} lesions {len(pick):4d} median contrast {np.median(lc):+.3f}; pseudo n {len(ps):4d} median "
              f"{np.median(ps):+.4f} |median| {np.median(np.abs(ps)):.4f}  ratio {ratio:.3f}  "
              f"{'PASS' if ratio < QC_CRIT else 'FAIL'}", flush=True)
    allp, alll = np.array(allp), np.array(alll)
    ratio = float(np.median(np.abs(allp)) / np.median(np.abs(alll)))
    res["overall"] = dict(n_pseudo=len(allp), ratio=ratio, passed=ratio < QC_CRIT,
                          pseudo_median=float(np.median(allp)), pseudo_median_abs=float(np.median(np.abs(allp))))
    json.dump(res, open(f"{OUT}/atlas_mirror_qc.json", "w"), indent=1)
    print(f"  overall ratio {ratio:.3f} {'PASS' if ratio < QC_CRIT else 'FAIL'}")


# ------------------------------------------------------------------ select
def _export(job):
    import scipy.ndimage as ndi
    r, c = job
    seg, wm = _tpl()
    brain = seg > 0; csf = np.isin(seg, CSF_LABELS)
    y, les = load(r["t1"]), load(r["mask"]) > 0
    lab, _ = ndi.label(les, structure=np.ones((3, 3, 3)))
    comp = lab == c["comp"]
    bx = _box(comp, RING[1] + SUP_DIL + 4)
    c_ = comp[bx]; others = ndi.binary_dilation((les & ~comp)[bx], iterations=1)
    core, ring, sup, _ = regions(c_, others, brain[bx], csf[bx])
    cf = mirror(y)[bx]; yb = y[bx]
    t = cf > TISSUE_FRAC * wm
    m = np.where(sup & t, (yb - cf) / np.maximum(cf, 1e-6), 0.0).astype(np.float32)
    off = [s.start for s in bx]
    name = f"{c['sid']}_c{c['comp']}"
    np.savez_compressed(f"{W}/lesions/{name}.npz", m=m, comp=c_, core=core, ring=ring, support=sup,
                        offset=np.array(off), y=yb.astype(np.float32), mirror=cf.astype(np.float32))
    return name


def select(workers=6):
    comps = json.load(open(f"{W}/components.json"))
    idx = {r["sid"]: r for r in json.load(open(f"{W}/index.json"))}
    rng = np.random.default_rng(SEED)
    os.makedirs(f"{W}/lesions", exist_ok=True)
    chosen, why = [], {}
    for nm, lo, hi in STRATA:
        cand = [c for c in comps if lo <= c["diam"] < hi]
        why[nm] = dict(all=len(cand))
        cand = [c for c in cand if c["native_max"] and c["native_max"] <= 1.5]; why[nm]["native<=1.5mm"] = len(cand)
        cand = [c for c in cand if c["in_brain"] >= 0.98]; why[nm]["in_brain>=0.98"] = len(cand)
        cand = [c for c in cand if c["mirror_clean"]]; why[nm]["mirror_clean"] = len(cand)
        cand = [c for c in cand if np.isfinite(c["c_frac"]) and c["n_ring"] >= 8]; why[nm]["ring_ok"] = len(cand)
        per_site, per_sess, got = {}, {}, []
        for i in rng.permutation(len(cand)):
            c = cand[i]
            if per_site.get(c["site"], 0) >= PER_SITE or per_sess.get(c["sid"], 0) >= PER_SESSION: continue
            per_site[c["site"]] = per_site.get(c["site"], 0) + 1; per_sess[c["sid"]] = per_sess.get(c["sid"], 0) + 1
            got.append(dict(c, stratum=nm))
            if len(got) >= PER_STRATUM: break
        why[nm]["selected"] = len(got); why[nm]["sites"] = len(per_site)
        chosen += got
    with ProcessPoolExecutor(workers) as ex:
        names = list(ex.map(_export, [(idx[c["sid"]], c) for c in chosen], chunksize=2))
    for c, n in zip(chosen, names): c["file"] = f"lesions/{n}.npz"
    json.dump(dict(selection=chosen, funnel=why, rules=dict(per_stratum=PER_STRATUM, per_site=PER_SITE,
                                                            per_session=PER_SESSION, seed=SEED)),
              open(f"{W}/selected.json", "w"), indent=1)
    for nm, w in why.items(): print(f"  {nm:9s} {w}")
    print(f"{len(chosen)} lesions exported to {W}/lesions")


if __name__ == "__main__":
    {"index": index, "template": template, "components": components, "qc": qc, "select": select}[sys.argv[1]]()
