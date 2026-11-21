"""Do real chronic-stroke lesions behave like the framework's Gaussians, and does retention track a
released model's documented lesion behaviour? ATLAS R3.0 lesions transplanted into healthy 3T T1.

Steps S3-S6 of the ATLAS analysis plan; the lesions come from exp/prep_atlas.py (S1-S2): 435 components
in three size strata (4-12, 12-25, > 25 mm equivalent diameter) from 1,452 ATLAS sessions, each with
its fractional T1 contrast m = (y - mirror(y)) / mirror(y) against the subject's own contralateral
hemisphere in MNI152NLin2009aSym space (the counterfactual; pseudo-lesion check in
out_public/atlas_mirror_qc.json: random error ~10% of the contrast, bias <= 3%).

  register  SynthMorph joint (affine + deformable) from the lesion-free ATLAS template
            (datasets/scratch_public/atlas/template_T1.nii.gz) to each healthy target's 1 mm 3T T1
            (the targets of exp/prep_lesion.py: 5 OpenNeuro test subjects, which also have acquired
            64 mT T1, and the 15 MRI-PairedField 3T-only subjects; Zenodo is excluded because D_T1
            does not transfer to its T1 protocol, reports/review_B.md). The warp is a displacement
            field on the target grid: MNI RAS = target RAS + d (checked: identical to
            `mri_synthmorph apply` on the template segmentation). QC: Dice of the warped template
            SynthSeg against the target's SynthSeg.
  place     Each lesion goes to one OpenNeuro test target and one 3T-only target (round robin over a
            seed-0 permutation within each stratum). m, the component mask and the support (mask
            dilated by 2 mm) are pulled from MNI into the target (linear for m, linear > 0.5 for
            masks), then into the target's band rows (exactly its datasets/npy_public rows).
            y_lesion = clip(y_target * (1 + m), 0, 1) on the support; the counterfactual is y_target.
            Central slice = largest axial cross-section within the band. Kept when (pre-specified):
            >= 95% of the warped mask inside the target brain; central slice >= 5 rows from either
            band end; equivalent radius of the cross-section r_eq >= 2 voxels; core >= 4 px; ring
            >= 8 px. Core = cross-section eroded by 1 px (all of it if that empties it); ring = 2-4 mm
            shell outside it, in brain, excluding CSF (target SynthSeg).
  prereg    Expectations written to out_public/d3atlas_T1.json before any lesion is scored.
  run       2D, central slice (the framework's unit): D_T1 (out_public/d1_forward_T1.pt), then
            no enhancer and route A (the T1 -> T1 absolute-output U-Net, base_unet_noresid_T1). Per
            lesion a Gaussian at the cross-section centroid, r = r_eq, with its contrast scaled so
            that its core-minus-ring input contrast equals the real lesion's (d3reallesion
            convention). Acquired-input arm on the OpenNeuro targets (x_acq + [D(y_l) - D(y)] near).
  released  3D, OpenNeuro targets: lesions packed into slab volumes (supports >= 20 mm apart),
            D_T1 slice-wise, Hyperfine-native 1.5 x 1.5 x 5 mm geometry (d3synthsr.to_native), then
            SynthSR low-field on the T1 (FreeSurfer 8.2; inpaints lesions by design) and SuperSynth
            T1 from the T1 (does not inpaint; exp/d3pretrained.py wrapper). Outputs back on the
            1 mm slab; R on the lesion's central slice with the symmetric intensity map
            (sd_y / sd_p, i.e. tp * sqrt(a / k) / ty with a, k from d3pretrained.lsq on the brain),
            least-squares a and slope k conventions stored beside it. The 64 mT input in 3D is the
            no-enhancer reference.
  summary   R and erased fraction (R < 0.25) per model and stratum, real vs Gaussian (lesion-level
            CCC, Bland-Altman; ATLAS-session cluster bootstrap), route A / no enhancer by size,
            acquired vs synthetic arm, cortical vs non-cortical lesions, and the pre-registered checks.

Pre-specified: erased = R < 0.25; R clamped to [-0.5, 1.5]; a lesion enters R only if |input
core-minus-ring contrast| >= 0.03 (d3reallesion KEEP_MIN). Sizes are equivalent diameters (MNI, 1 mm).

    ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_SCRATCH=datasets/scratch_public \
    ULF_FIG=fig_public python -I exp/d3atlas.py {register,place,prereg,run,released,summary}
"""
import json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, ".")
ROOT = "."
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch")
FIG = os.environ.get("ULF_FIG", f"{ROOT}/fig")
W = f"{SCR}/atlas"
RL = f"{SCR}/reallesion"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
SPLITS = ("test", "pub3t_unc", "pub3t_penn", "pub3t_hip")
ERASED, R_LO, R_HI, KEEP_MIN = 0.25, -0.5, 1.5, 0.03
IN_BRAIN, BAND_MARGIN, R_EQ_MIN, CORE_MIN, RING_MIN = 0.95, 5, 2.0, 4, 8
RING_IN, RING_OUT, S_DIL, PACK_SEP = 2.0, 4.0, 2.0, 20.0
CSF = (4, 5, 14, 15, 24, 43, 44, 72)
CORTEX, CWM = (3, 42), (2, 41)
DEEP_GM = (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
STRATA = ("4-12 mm", "12-25 mm", ">25 mm")
N_BOOT = 10000


def nib():
    import nibabel
    return nibabel


def load(p, dt=np.float32):
    return np.asarray(nib().load(p).dataobj, dtype=dt)


def targets():
    t = [tuple(x) for x in json.load(open(f"{RL}/targets.json"))]
    return [(sp, sid) for sp, sid in t if sp in SPLITS]


# ================================================================== register (CPU)
def _reg(job):
    sp, sid = job
    od = f"{W}/reg/{sid}"
    if os.path.exists(f"{od}/qc.json"):
        return sid, "cached"
    os.makedirs(od, exist_ok=True)
    env = dict(os.environ, FREESURFER_HOME=FS)
    if not os.path.exists(f"{od}/warp.nii.gz"):
        cmd = [f"{FS}/bin/mri_synthmorph", "register", "-m", "joint", "-j", "4", "-o", f"{od}/moved.nii.gz",
               "-t", f"{od}/warp.nii.gz", f"{W}/template_T1.nii.gz", f"{RL}/targets/{sid}/t1.nii.gz"]
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode or not os.path.exists(f"{od}/warp.nii.gz"):
            return sid, "FAILED " + r.stdout[-300:]
    S = load(f"{W}/template_seg.nii.gz", np.int16)
    src = source_coords(sid)
    tool = ndi.map_coordinates(S, [src[..., k] for k in range(3)], order=0, mode="constant")
    T = load(f"{RL}/synthseg/targets_out/{sid}_synthseg.nii.gz", np.int16)
    qc = {}
    for nm, labs in (("WM", CWM), ("cortex", CORTEX), ("ventricles", (4, 43)), ("deep GM", DEEP_GM)):
        a, b = np.isin(T, labs), np.isin(tool, labs)
        qc[nm] = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    json.dump(dict(split=sp, target=sid, dice=qc), open(f"{od}/qc.json", "w"))
    return sid, "ok " + " ".join(f"{k} {v:.2f}" for k, v in qc.items())


def source_coords(sid):
    """MNI voxel coordinates of every target voxel (float32, target grid x 3)."""
    w = nib().load(f"{W}/reg/{sid}/warp.nii.gz")
    d = np.asarray(w.dataobj, dtype=np.float32).reshape(*w.shape[:3], 3)
    Am = nib().load(f"{W}/template_T1.nii.gz").affine
    iAm = np.linalg.inv(Am)
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in d.shape[:3]], indexing="ij"), -1)
    ras = g @ w.affine[:3, :3].T.astype(np.float32) + w.affine[:3, 3].astype(np.float32) + d
    return (ras @ iAm[:3, :3].T.astype(np.float32) + iAm[:3, 3].astype(np.float32)).astype(np.float32)


def register(workers=3):
    jobs = targets()
    print(f"{len(jobs)} targets", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for sid, msg in ex.map(_reg, jobs):
            print(f"  {sid}: {msg}", flush=True)


# ================================================================== place (CPU)
def core_ring(cs, brain, excl):
    core = ndi.binary_erosion(cs)
    if not core.any(): core = cs
    d = ndi.distance_transform_edt(~cs)
    ring = (d > RING_IN) & (d <= RING_OUT) & brain & ~excl
    return core, ring


def assign():
    sel = json.load(open(f"{W}/selected.json"))["selection"]
    tg = targets()
    test = [s for sp, s in tg if sp == "test"]
    other = [(sp, s) for sp, s in tg if sp != "test"]
    rng = np.random.default_rng(0)
    other = [other[i] for i in rng.permutation(len(other))]
    jobs = []
    for st in STRATA:
        L = [c for c in sel if c["stratum"] == st]
        for i, j in enumerate(rng.permutation(len(L))):
            c = L[j]
            jobs.append((c, "test", test[i % len(test)]))
            jobs.append((c, *other[i % len(other)]))
    return jobs


def _place_target(args):
    sp, sid, items = args
    from exp.d3reallesion import target_rows
    od = f"{W}/placed/{sid}"
    os.makedirs(od, exist_ok=True)
    src = source_coords(sid)
    fr, rows = target_rows(sid)
    y = rows(load(f"{RL}/targets/{sid}/t1.nii.gz"))
    brain = y > 0
    seg = rows(load(f"{RL}/synthseg/targets_out/{sid}_synthseg.nii.gz", np.int16))
    csf = np.isin(seg, CSF)
    nz = y.shape[0]
    recs = []
    for c in items:
        name = f"{c['sid']}_c{c['comp']}"
        if os.path.exists(f"{od}/{name}.json"):
            recs.append(json.load(open(f"{od}/{name}.json"))); continue
        z = np.load(f"{W}/{c['file']}")
        off = z["offset"].astype(np.float32); shp = np.array(z["m"].shape, np.float32)
        rel = src - off
        inside = np.all((rel >= -1) & (rel <= shp), axis=-1)
        why = []
        if not inside.any():
            why.append("outside target")
        else:
            idx = np.argwhere(inside); lo, hi = idx.min(0), idx.max(0) + 1
            bx = tuple(slice(a, b) for a, b in zip(lo, hi))
            co = [rel[bx][..., k] for k in range(3)]
            M = np.zeros(src.shape[:3], np.float32); C = np.zeros_like(M); S_ = np.zeros_like(M)
            M[bx] = ndi.map_coordinates(z["m"].astype(np.float32), co, order=1, mode="constant")
            C[bx] = ndi.map_coordinates(z["comp"].astype(np.float32), co, order=1, mode="constant")
            S_[bx] = ndi.map_coordinates(z["support"].astype(np.float32), co, order=1, mode="constant")
            m, comp, sup = rows(M), rows(C) >= 0.5, rows(S_) >= 0.5
            m = np.where(sup, m, 0.0)
            vol_full = float((C >= 0.5).sum())
            vol_band = float(comp.sum())
            if vol_band == 0:
                why.append("not in band")
        rec = dict(lesion=name, atlas_sid=c["sid"], site=c["site"], comp=c["comp"], stratum=c["stratum"],
                   diam=c["diam"], c_frac_mni=c["c_frac"], tpl_cortex=c["cortex"], tpl_wm=c["wm"],
                   tpl_deep_gm=c["deep_gm"], split=sp, target=sid, why=why)
        if not why:
            in_brain = float(brain[comp].mean())
            area = comp.sum((1, 2))
            kc = int(np.argmax(area))
            cs = comp[kc]
            r_eq = float(np.sqrt(cs.sum() / np.pi))
            core, ring = core_ring(cs, brain[kc], csf[kc])
            yl = np.clip(y[kc] * (1 + m[kc]), 0, 1) * brain[kc]
            tru = float((yl - y[kc])[core].mean() - (yl - y[kc])[ring].mean()) if core.any() and ring.any() else np.nan
            segc = seg[comp]
            rec.update(vox_target=vol_full, vox_mni=c["vox"], frac_in_band=vol_band / max(vol_full, 1),
                       in_brain=in_brain, kc=kc, r_eq2d=r_eq, diam2d=2 * r_eq, n_core=int(core.sum()),
                       n_ring=int(ring.sum()), tru=tru,
                       tgt_cortex=float(np.isin(segc, CORTEX).mean()), tgt_wm=float(np.isin(segc, CWM).mean()),
                       tgt_deep_gm=float(np.isin(segc, DEEP_GM).mean()), tgt_csf=float(np.isin(segc, CSF).mean()),
                       cy=float(np.argwhere(cs).mean(0)[0]), cx=float(np.argwhere(cs).mean(0)[1]))
            if in_brain < IN_BRAIN: why.append(f"in brain {in_brain:.2f}")
            if kc < BAND_MARGIN or kc > nz - 1 - BAND_MARGIN: why.append("central slice near band end")
            if r_eq < R_EQ_MIN: why.append(f"r_eq {r_eq:.2f}")
            if core.sum() < CORE_MIN: why.append("core small")
            if ring.sum() < RING_MIN: why.append("ring small")
            if not why:
                zz = np.where(sup.any((1, 2)))[0]
                k0, k1 = int(max(0, zz.min() - 5)), int(min(nz, zz.max() + 6))
                np.savez_compressed(f"{od}/{name}.npz", k0=k0, k1=k1, kc=kc, m=m[k0:k1].astype(np.float16),
                                    comp=comp[k0:k1], sup=sup[k0:k1], core=core, ring=ring)
        rec["kept"] = not why
        json.dump(rec, open(f"{od}/{name}.json", "w"), default=float)
        recs.append(rec)
    return sid, recs


def place(workers=3):
    jobs = assign()
    by = {}
    for c, sp, sid in jobs:
        by.setdefault((sp, sid), []).append(c)
    args = [(sp, sid, items) for (sp, sid), items in by.items()]
    allr = []
    with ProcessPoolExecutor(workers) as ex:
        for sid, recs in ex.map(_place_target, args):
            k = sum(r["kept"] for r in recs)
            print(f"  {sid}: {k}/{len(recs)} kept", flush=True)
            allr += recs
    json.dump(allr, open(f"{W}/placed.json", "w"), default=float)
    from collections import Counter
    why = Counter(w.split(" ")[0] + (" " + w.split(" ")[1] if w.startswith(("in", "central", "not", "outside")) else "")
                  for r in allr for w in r["why"])
    print(f"kept {sum(r['kept'] for r in allr)} of {len(allr)} placements; reasons {dict(why)}")
    for st in STRATA:
        a = [r for r in allr if r["stratum"] == st]
        print(f"  {st:9s} kept {sum(r['kept'] for r in a)}/{len(a)}  (test {sum(r['kept'] for r in a if r['split'] == 'test')})")


# ================================================================== prereg
def prereg():
    p = f"{OUT}/d3atlas_T1.json"
    res = json.load(open(p)) if os.path.exists(p) else {}
    if "prereg" in res:
        print("pre-registration already written", res["prereg"]["written"]); return
    res["prereg"] = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        cutoffs=dict(erased=ERASED, R_clamp=[R_LO, R_HI], keep_min=KEEP_MIN, in_brain=IN_BRAIN,
                     band_margin=BAND_MARGIN, r_eq_min=R_EQ_MIN, core_min=CORE_MIN, ring_min=RING_MIN,
                     ring_mm=[RING_IN, RING_OUT], pack_sep_mm=PACK_SEP),
        expected={
            "size": "No-enhancer R rises with stratum (4-12 < 12-25 <= >25 mm). Route A / no enhancer is "
                    "about the same in 12-25 and >25 mm (|difference| < 0.10), as found for T2 Gaussians.",
            "documented_behaviour": "SynthSR low-field (inpaints lesions by design) erases > 80% of lesions in "
                                    "each stratum >= 12 mm; SuperSynth T1 (does not inpaint) erases fewer, "
                                    "with the subject-cluster bootstrap CI of the difference above 0.",
            "real_vs_gauss": "Matched Gaussians under-state real-lesion retention (bias Gauss - real < 0), "
                             "as for MS (reports/review_C.md).",
            "acquired_arm": "R_acq agrees with R_syn on the OpenNeuro targets (CCC > 0.6)."},
        primary="all kept lesions (decision 2026-11-09: proceed despite the narrow mirror-QC fail; "
                "cortical vs non-cortical reported as a breakdown)")
    json.dump(res, open(p, "w"), indent=1)
    print("pre-registration written")


# ================================================================== run (GPU, 2D)
def kept_placements():
    P = [r for r in json.load(open(f"{W}/placed.json")) if r["kept"]]
    return P


def run():
    import torch
    from ulfkit import data as D, train as T
    from exp.d3detect import operator, blobs
    from exp.d3qa import enhancer as abs_unet
    from exp.d3reallesion import run_net, Ident
    T.seed_all(0)
    P = kept_placements()
    coh = {sp: D.PairedCohort(sp, "T1") for sp in SPLITS}
    f1 = operator("T1", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    cu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().cuda()
    eA = abs_unet(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
    models = [("no enhancer", Ident()), ("route A (T1 U-Net)", eA)]
    rows = []
    with torch.no_grad():
        for r in P:
            z = np.load(f"{W}/placed/{r['target']}/{r['lesion']}.npz")
            c = coh[r["split"]]; rr = c.subject_rows(r["target"])
            kc = int(z["kc"]); k0 = int(z["k0"])
            y = c.hf[rr[kc], 0].astype(np.float32)
            br = y > 0
            m = z["m"][kc - k0].astype(np.float32)
            core, ring = z["core"], z["ring"]
            yl = np.clip(y * (1 + m), 0, 1) * br
            tru = float((yl - y)[core].mean() - (yl - y)[ring].mean())
            cy, cx, rq = int(round(r["cy"])), int(round(r["cx"])), float(r["r_eq2d"])
            b = blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
                      torch.tensor([rq], device="cuda"))[0, 0].cpu().numpy() * br
            gc, gr = b > 0.6, (b > 0.05) & (b < 0.25)
            unit = float(b[gc].mean() - b[gr].mean()) if gc.any() and gr.any() else np.nan
            cg = tru / unit if np.isfinite(unit) and unit != 0 else 0.0
            yg = np.clip(y + cg * b, 0, 1) * br
            tru_g = float((yg - y)[gc].mean() - (yg - y)[gr].mean()) if gc.any() and gr.any() else np.nan
            Y = cu(np.stack([y, yl, yg]))[:, None]
            X = f1(Y, gen)[:, :1]                          # D's centre slice
            con = lambda d, a, bb: float(d[a].mean() - d[bb].mean()) if a.any() and bb.any() else np.nan
            dd = (X[1, 0] - X[0, 0]).cpu().numpy(); ddg = (X[2, 0] - X[0, 0]).cpu().numpy()
            rec = dict(r, tru=tru, tru_g=tru_g, gauss_c=cg, dd_real=con(dd, core, ring), dd_g=con(ddg, gc, gr), out={})
            acq = None
            if c.ulf is not None:
                x0 = cu(c.ulf[rr[kc]].astype(np.float32))[None]
                near = (blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
                              torch.clamp(torch.tensor([rq * 4], device="cuda"), min=24)) > 0.05).float()
                acq = (x0, x0 + ((X[1:2] - X[0:1]) * near).repeat(1, 3, 1, 1), x0 + ((X[2:3] - X[0:1]) * near).repeat(1, 3, 1, 1))
            for nm, net in models:
                p = run_net(net, X.repeat(1, 3, 1, 1))[:, 0].cpu().numpy()
                o = dict(real=con(p[1] - p[0], core, ring), gauss=con(p[2] - p[0], gc, gr),
                         gauss_m=con(p[2] - p[0], core, ring))
                if acq is not None:
                    a = run_net(net, torch.cat(acq))[:, 0].cpu().numpy()
                    o.update(acq_real=con(a[1] - a[0], core, ring), acq_gauss=con(a[2] - a[0], gc, gr))
                rec["out"][nm] = o
            rows.append(rec)
    json.dump(dict(n=len(rows), rows=rows), open(f"{OUT}/d3atlas_rows_2d.json", "w"), default=float)
    print(f"  wrote {len(rows)} lesion rows (2D)", flush=True)


# ================================================================== released (3D)
def released(workers=3):
    import torch
    from ulfkit import data as D
    from exp.d3detect import operator
    from exp.d3synthsr import to_native
    from exp import d3pretrained as PT
    P = [r for r in kept_placements() if r["split"] == "test"]
    c = D.PairedCohort("test", "T1")
    f1 = operator("T1", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    R3 = f"{W}/released"
    os.makedirs(R3, exist_ok=True)

    def degrade(vol):                                   # (slice, row, col) -> same, slice-wise
        out = []
        for i in range(0, len(vol), 16):
            yb = torch.from_numpy(np.ascontiguousarray(vol[i:i + 16, None])).float().cuda()
            with torch.no_grad():
                out.append(f1(yb, gen)[:, 0].float().cpu().numpy())
        return np.concatenate(out)

    plan = {}
    for sid in sorted({r["target"] for r in P}):
        rr = c.subject_rows(sid)
        y = c.hf[rr, 0].astype(np.float32); br = y > 0
        mine = [r for r in P if r["target"] == sid]
        vols = []                                       # greedy packing, supports >= PACK_SEP apart
        for r in sorted(mine, key=lambda r: -r["vox_target"]):
            z = np.load(f"{W}/placed/{sid}/{r['lesion']}.npz")
            sup = np.zeros(y.shape, bool); sup[int(z["k0"]):int(z["k1"])] = z["sup"]
            far = ndi.distance_transform_edt(~sup) <= PACK_SEP
            for v in vols:
                if not (v["occ"] & far).any():
                    v["occ"] |= sup; v["les"].append(r["lesion"]); break
            else:
                vols.append(dict(occ=sup.copy(), les=[r["lesion"]]))
        d = f"{R3}/{sid}"; os.makedirs(d, exist_ok=True)
        x0 = degrade(y) * br
        PT.save(PT.to_ras(y), f"{d}/3t.nii.gz"); PT.save(PT.to_ras(x0), f"{d}/sim_clean.nii.gz")
        for vi, v in enumerate(vols):
            yl = y.copy()
            for ln in v["les"]:
                z = np.load(f"{W}/placed/{sid}/{ln}.npz"); k0, k1 = int(z["k0"]), int(z["k1"])
                m = np.zeros(y.shape, np.float32); m[k0:k1] = z["m"].astype(np.float32)
                yl = np.clip(yl * (1 + m), 0, 1)
            yl *= br
            PT.save(PT.to_ras(degrade(yl) * br), f"{d}/sim_v{vi}.nii.gz")
        for f in ["sim_clean"] + [f"sim_v{vi}" for vi in range(len(vols))]:
            to_native(f"{d}/{f}.nii.gz", f"{d}/{f}_native.nii.gz")
        plan[sid] = [v["les"] for v in vols]
        print(f"  {sid}: {len(mine)} lesions in {len(vols)} volumes", flush=True)
    json.dump(plan, open(f"{R3}/plan.json", "w"), indent=1)
    # SuperSynth (GPU): T1 synthesis from the 64 mT T1
    nets = PT.ss_load()
    for sid, vols in plan.items():
        d = f"{R3}/{sid}"
        for f in ["sim_clean"] + [f"sim_v{vi}" for vi in range(len(vols))]:
            o = f"{d}/ss_{f}_T1.nii.gz"
            if os.path.exists(o): continue
            t1, t2, aff, imax = PT.ss_infer(nets, f"{d}/{f}_native.nii.gz")
            PT.save(t1, o + ".tmp.nii.gz", aff); os.replace(o + ".tmp.nii.gz", o)
    del nets; torch.cuda.empty_cache()
    print("  SuperSynth done", flush=True)
    # SynthSR low-field (CPU)
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for sid, vols in plan.items():
        d = f"{R3}/{sid}"
        for f in ["sim_clean"] + [f"sim_v{vi}" for vi in range(len(vols))]:
            o = f"{d}/sr_{f}.nii.gz"
            if not os.path.exists(o):
                jobs.append([f"{FS}/bin/mri_synthsr", "--lowfield", "--i", f"{d}/{f}_native.nii.gz", "--o", o,
                             "--threads", "4", "--cpu"])
    env = PT.fs_env()

    def go(cmd):
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode: print("  FAILED", cmd[-6], r.stderr[-300:], flush=True)
    print(f"  {len(jobs)} SynthSR runs", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))
    score_released()


def score_released():
    from exp import d3pretrained as PT
    R3 = f"{W}/released"
    plan = json.load(open(f"{R3}/plan.json"))
    P = {r["lesion"] + "@" + r["target"]: r for r in kept_placements()}
    models = {"64 mT input (no enhancer, 3D)": lambda d, f: f"{d}/{f}.nii.gz",
              "SynthSR low-field (T1 in)": lambda d, f: f"{d}/sr_{f}.nii.gz",
              "SuperSynth T1 (T1 in)": lambda d, f: f"{d}/ss_{f}_T1.nii.gz"}
    rows = []
    for sid, vols in plan.items():
        d = f"{R3}/{sid}"
        y = PT.from_ras(PT.load(f"{d}/3t.nii.gz")); br = y > 0
        for nm, pf in models.items():
            ref = lambda f: PT.from_ras(PT.on_ref(pf(d, f), PT.to_ras(y).shape)) if "no enhancer" not in nm \
                else PT.from_ras(PT.load(pf(d, f)))
            P0 = ref("sim_clean")
            a, b_, k = PT.lsq(y, P0, br)
            for vi, les in enumerate(vols):
                dp = ref(f"sim_v{vi}") - P0
                for ln in les:
                    r = P[f"{ln}@{sid}"]
                    z = np.load(f"{W}/placed/{sid}/{ln}.npz"); kc = int(z["kc"])
                    core, ring = z["core"], z["ring"]
                    tp = float(dp[kc][core].mean() - dp[kc][ring].mean())
                    rows.append(dict(model=nm, lesion=ln, target=sid, atlas_sid=r["atlas_sid"], stratum=r["stratum"],
                                     diam=r["diam"], tgt_cortex=r["tgt_cortex"], tru=r["tru"], tp=tp, a=a, k=k))
    json.dump(dict(rows=rows), open(f"{OUT}/d3atlas_rows_3d.json", "w"), default=float)
    print(f"  wrote {len(rows)} released-model rows", flush=True)


# ================================================================== summary
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3: return np.nan
    return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def cboot(clusters, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(clusters); idx = {k: np.where(clusters == k)[0] for k in u}
    vals = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        ii = np.concatenate([idx[k] for k in pick])
        v = fn(ii)
        if np.isfinite(v): vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def Rv(num, den):
    return np.clip(np.asarray(num, float) / np.asarray(den, float), R_LO, R_HI)


def summary():
    res = json.load(open(f"{OUT}/d3atlas_T1.json"))
    pl = json.load(open(f"{W}/placed.json"))
    res["placement"] = {st: dict(placements=sum(r["stratum"] == st for r in pl),
                                 kept=sum(r["kept"] and r["stratum"] == st for r in pl)) for st in STRATA}
    res["registration_qc"] = {sid: json.load(open(f"{W}/reg/{sid}/qc.json"))["dice"] for _, sid in targets()
                              if os.path.exists(f"{W}/reg/{sid}/qc.json")}
    R2 = json.load(open(f"{OUT}/d3atlas_rows_2d.json"))["rows"]
    R2 = [r for r in R2 if abs(r["tru"]) >= KEEP_MIN]
    out2 = {}
    for nm in R2[0]["out"]:
        o = {}
        for st in STRATA + ("all",):
            L = [r for r in R2 if st == "all" or r["stratum"] == st]
            if not L: continue
            cl = np.array([r["atlas_sid"] for r in L])
            rr = Rv([r["out"][nm]["real"] for r in L], [r["tru"] for r in L])
            rg = Rv([r["out"][nm]["gauss"] for r in L], [r["tru_g"] for r in L])
            d = dict(n=len(L), R_real=float(np.nanmean(rr)), R_real_ci=cboot(cl, lambda i: np.nanmean(rr[i])),
                     R_gauss=float(np.nanmean(rg)), erased_real=float(np.mean(rr < ERASED)),
                     erased_real_ci=cboot(cl, lambda i: np.mean(rr[i] < ERASED)), erased_gauss=float(np.mean(rg < ERASED)),
                     ccc=ccc(rg, rr), bias=float(np.nanmean(rg - rr)))
            for tag, sel in (("non-cortical", lambda r: r["tgt_cortex"] <= 0.3), ("cortical", lambda r: r["tgt_cortex"] > 0.3)):
                ii = np.array([sel(r) for r in L])
                if ii.sum() >= 5:
                    d[tag] = dict(n=int(ii.sum()), R_real=float(np.nanmean(rr[ii])), erased_real=float(np.mean(rr[ii] < ERASED)))
            acq = [r for r in L if "acq_real" in r["out"][nm]]
            if acq:
                ra = Rv([r["out"][nm]["acq_real"] for r in acq], [r["tru"] for r in acq])
                rs = Rv([r["out"][nm]["real"] for r in acq], [r["tru"] for r in acq])
                d["acquired"] = dict(n=len(acq), R_acq=float(np.nanmean(ra)), R_syn=float(np.nanmean(rs)), ccc=ccc(ra, rs))
            o[st] = d
        out2[nm] = o
    res["two_d"] = out2
    ratio = {}
    for st in STRATA:
        a, b = out2["route A (T1 U-Net)"][st]["R_real"], out2["no enhancer"][st]["R_real"]
        ratio[st] = a / b if b else np.nan
    res["routeA_over_noenh"] = ratio
    if os.path.exists(f"{OUT}/d3atlas_rows_3d.json"):
        R3 = json.load(open(f"{OUT}/d3atlas_rows_3d.json"))["rows"]
        out3 = {}
        for nm in sorted({r["model"] for r in R3}):
            o = {}
            for st in STRATA + ("all",):
                L = [r for r in R3 if r["model"] == nm and abs(r["tru"]) >= KEEP_MIN and (st == "all" or r["stratum"] == st)]
                if not L: continue
                cl = np.array([r["atlas_sid"] for r in L])
                conv = {"symmetric": np.array([r["tp"] * np.sqrt(max(r["a"], 1e-9) / max(r["k"], 1e-9)) for r in L]),
                        "lsq a": np.array([r["tp"] * r["a"] for r in L]), "slope k": np.array([r["tp"] / r["k"] for r in L])}
                if "no enhancer" in nm: conv = {"symmetric": np.array([r["tp"] for r in L])}
                tru = np.array([r["tru"] for r in L])
                d = dict(n=len(L))
                for cn, tp in conv.items():
                    rr = Rv(tp, tru)
                    d[cn] = dict(R=float(np.nanmean(rr)), erased=float(np.mean(rr < ERASED)),
                                 erased_ci=cboot(cl, lambda i: np.mean(rr[i] < ERASED)))
                o[st] = d
            out3[nm] = o
        res["three_d"] = out3
        # pre-registered documented-behaviour check: SynthSR vs SuperSynth erased, paired by lesion
        chk = {}
        for st in STRATA:
            key = lambda r: (r["lesion"], r["target"])
            A = {key(r): r for r in R3 if r["model"].startswith("SynthSR") and r["stratum"] == st and abs(r["tru"]) >= KEEP_MIN}
            B = {key(r): r for r in R3 if r["model"].startswith("SuperSynth") and r["stratum"] == st and abs(r["tru"]) >= KEEP_MIN}
            ks = sorted(set(A) & set(B))
            if not ks: continue
            sym = lambda r: r["tp"] * np.sqrt(max(r["a"], 1e-9) / max(r["k"], 1e-9)) / r["tru"]
            ea = np.array([Rv(sym(A[k]), 1) < ERASED for k in ks], float); eb = np.array([Rv(sym(B[k]), 1) < ERASED for k in ks], float)
            cl = np.array([A[k]["atlas_sid"] for k in ks])
            chk[st] = dict(n=len(ks), synthsr_erased=float(ea.mean()), supersynth_erased=float(eb.mean()),
                           diff=float(ea.mean() - eb.mean()), diff_ci=cboot(cl, lambda i: ea[i].mean() - eb[i].mean()))
        res["documented_behaviour"] = chk
    json.dump(res, open(f"{OUT}/d3atlas_T1.json", "w"), indent=1, default=float)
    for nm, o in out2.items():
        print(f"\n== 2D {nm}")
        for st, d in o.items():
            print(f"  {st:9s} n {d['n']:4d}  R real {d['R_real']:.3f} [{d['R_real_ci'][0]:.3f},{d['R_real_ci'][1]:.3f}]  "
                  f"gauss {d['R_gauss']:.3f}  erased real {d['erased_real']:.2f} gauss {d['erased_gauss']:.2f}  "
                  f"CCC {d['ccc']:.2f} bias {d['bias']:+.3f}" + (f"  acq {d['acquired']['R_acq']:.3f} vs syn {d['acquired']['R_syn']:.3f} (CCC {d['acquired']['ccc']:.2f})" if "acquired" in d else ""))
    print("\nroute A / no enhancer:", {k: round(v, 3) for k, v in ratio.items()})
    if "three_d" in res:
        for nm, o in res["three_d"].items():
            print(f"\n== 3D {nm}")
            for st, d in o.items():
                print(f"  {st:9s} n {d['n']:4d}  " + "  ".join(f"{cn}: R {v['R']:.3f} erased {v['erased']:.2f}" for cn, v in d.items() if cn != "n"))
        print("\ndocumented behaviour:", json.dumps(res["documented_behaviour"], default=float))


if __name__ == "__main__":
    {"register": register, "place": place, "prereg": prereg, "run": run, "released": released,
     "score_released": score_released, "summary": summary}[sys.argv[1]]()
