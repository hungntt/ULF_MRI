"""Is a Gaussian blob a fair stand-in for a real lesion? Real MS lesions through D and G.

Every retention number in the paper comes from Gaussian blobs inserted into healthy 3T slices.
The standing objection is that a blob is not a lesion: real lesions have irregular outlines,
flat cores, soft shoulders and texture. Here real MS lesions (Muslim et al. 2022, 1.5 T, consensus
T2 masks; built by exp/prep_lesion.py) are transplanted into the same healthy public 3T targets
the framework uses, passed through the learned operator D and every enhancer G, and compared
lesion by lesion with a Gaussian at the same site, of the same equivalent radius and contrast.

Method (phase C).
  lesions   Per patient, each 3D component of the T2 mask is a lesion. Both contrasts are inpainted
            slice by slice with d3realret.inpaint (feathered fill), fed with a lesion-excluded
            15 x 15 median background (three passes that replace the dilated lesion by the current
            estimate), giving the lesion-free patient y0. Fractional contrast
            m = (y - y0) / y0 inside the mask dilated by 2 voxels (0 outside, clipped to [-0.9, 3]),
            for T1 and T2. Polarity of each lesion is measured directly on the patient:
            mean(core) - mean(ring) of the 1.5 T image, core = mask eroded by 1 voxel, ring =
            2-4 mm shell outside the mask, excluding other lesions.
  register  Each patient goes to two targets: one OpenNeuro test subject (round robin, so the
            acquired 64 mT arm runs) and one of the 26 others (Zenodo 11, 5 per MRI-PairedField
            cohort; seed-0 permutation, round robin). SynthMorph joint (affine + deformable),
            moving = inpainted patient T1, fixed = target T1; m and the lesion labels follow
            (linear / nearest).
  place     In the target's band rows (exactly its datasets/npy_public rows): a lesion is kept when
            its mask and 4 mm surround lie inside the band and the brain, >= 90% of its mask is
            SynthSeg cerebral white matter, >= 60% of its ring is white matter, and its central
            axial cross-section (largest area) has equivalent radius r_eq = sqrt(A / pi) >= 2
            voxels (diameter >= 4 mm, the framework's smallest lesion) with a ring of >= 8 px.
            Lesions whose patient-side inpainting window (15 x 15 px at the central slice) holds
            > 20% CSF are excluded (median fill biased by CSF). y_l = y_target * (1 + m) for T1 and
            T2 jointly; the counterfactual is y_target.
  run       Per kept lesion, on its central target slice (the framework's 2D unit, since D works
            slice-wise and G sees D's centre slice repeated):
              real    only this lesion inserted;
              gauss   d3detect.blobs at the cross-section centroid with r = r_eq, contrast scaled
                      so its core-minus-ring input contrast equals the real lesion's; core b > 0.6,
                      ring 0.05 < b < 0.25 (the standard definitions);
              all     every kept lesion of the placement inserted together (co-occurrence),
                      scored on the central slice and over the whole lesion in 3D.
            Real core = cross-section eroded by 1 px (whole cross-section if that empties it),
            ring = 2-4 mm shell outside it, in brain, excluding other kept lesions. Stored per
            lesion and model: input, D-stage and output core-minus-ring contrasts (so R, R_D, R_G|D
            can be recomputed), for the 12 seed-0 T2 enhancers (the three Mean-Teacher ones of the
            15-configuration registry need the withdrawn unpaired 64 mT pool) and G = identity;
            the acquired-input arm on OpenNeuro test targets (d3detreal.lesion_input); routes A-C
            and the three retrained route-C checkpoints on T1 (real and Gaussian with matched
            signed T1 and T2 contrasts); the retention map f_theta (d3detect_T2 and seeds s1-s3)
            queried with r_eq and the real input contrast.
  sanity    (i) a Gaussian inserted through this transplant code path must reproduce the standard
            pipeline: m = c b / y (exactly additive) must match d3detect's R within 0.01 at every
            site; m = c b / ring-mean(y) (fractional) is reported, pre-specified median |dR| < 0.05.
  inplace   Patients' own 1.5 T T2 through D (off-domain for D; a check only): R of each lesion
            against the inpainted image; and inpainting bias on pseudo-lesions: each real lesion's
            central cross-section moved to healthy white matter of the same patient and inpainted
            the same way; reported as input bias tru_pseudo / tru_real and residual output contrast
            dp_pseudo / tru_real.
  summary   lesion-level CCC and Bland-Altman of R_gauss vs R_real, subject-cluster bootstrap
            (target subjects; 10,000 replicates); enhancer ranking Spearman (real vs Gaussian, and
            real vs the zoo's random-site Gaussians); map AUROC for real-lesion erasure; T1/T2 sign
            distribution and route A-C inversion on real lesions.

Pre-specified (fixed before any result was seen): erased = R < 0.25; R clamped to [-0.5, 1.5]
as in d3detect; a lesion enters R only if |input core-minus-ring contrast| >= 0.03 (0.25 x the
smallest synthetic contrast 0.12); sign patterns use |direct contrast| >= 0.03, else 'iso'; the
primary comparison is real vs Gaussian on the central slice with one lesion per image, synthetic
arm through D_HFC. Lesion size: the code's r is the 1/e^2 radius of b = exp(-2 d^2 / r^2) in 1 mm
voxels; Gaussians are reported by diameter 2r, real lesions by equivalent diameter 2 r_eq.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_SCRATCH=datasets/scratch_public \
    ULF_FIG=fig_public python exp/d3reallesion.py {lesions,register,place,sanity,run,inplace,summary}
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, ".")
ROOT = "."
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch")
FIG = os.environ.get("ULF_FIG", f"{ROOT}/fig")
RL = f"{SCR}/reallesion"
NPY_MS = f"{ROOT}/datasets/npy_public_ms"

# ---- pre-specified cutoffs
ERASED, R_LO, R_HI, KEEP_MIN, SIGN_MIN = 0.25, -0.5, 1.5, 0.03, 0.03
S_DIL, RING_IN, RING_OUT = 2.0, 2.0, 4.0
WM_MIN, RING_WM_MIN, R_EQ_MIN, RING2_MIN, CSF_WIN_MAX = 0.90, 0.60, 2.0, 8, 0.20
WM, VENT, CORTEX = (2, 41), (4, 43), (3, 42)
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)
NEAR_MM, N_BOOT = 3.0, 10000
MAPS = ["d3detect_T2", "d3detect_T2_s1", "d3detect_T2_s2", "d3detect_T2_s3"]
ROUTES = ["A_T1in", "B_T2in", "C_both", "C_bothtoT1_ctl", "C_bothtoT1_aug", "C_bothtoT1_augw"]
NOENH = "no enhancer"
MAP_ENH = "U-Net absolute s0"            # the enhancer the retention map was trained for


def _nib():
    import nibabel as nib
    return nib


def load_nii(p):
    return np.asarray(_nib().load(p).dataobj)


# ================================================================== stage: lesions (CPU)
def bg_excluding(y, excl, size=15, iters=3):
    """Median background with the (dilated) lesions replaced by the current estimate."""
    b = ndi.median_filter(y, size=size)
    for _ in range(iters):
        b = ndi.median_filter(np.where(excl, b, y), size=size)
    return b


def inpaint_slices(y, mask):
    """Slice-wise (axial = axis 2) lesion removal with d3realret.inpaint."""
    from exp.d3realret import inpaint
    y0 = y.astype(np.float32).copy()
    for z in np.where(mask.any((0, 1)))[0]:
        fm = mask[:, :, z]
        b = bg_excluding(y[:, :, z], ndi.binary_dilation(fm, iterations=2))
        y0[:, :, z] = inpaint(y[:, :, z], fm, b)
    return y0


def ball(r):
    g = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1, -int(r):int(r) + 1]
    return (g ** 2).sum(0) <= r * r + 1e-6


def frac_contrast(y, y0, support):
    m = np.where(support & (y0 > 0), (y - y0) / np.maximum(y0, 0.02), 0.0)
    return np.clip(m, -0.9, 3.0).astype(np.float32)


def core_ring(mask, brain, others=None, ring=(RING_IN, RING_OUT)):
    """Core = mask eroded by 1 voxel (whole mask if that empties it); ring = (2, 4] mm shell
    outside the mask, inside the brain, excluding `others`. Works in 2D and 3D."""
    core = ndi.binary_erosion(mask)
    if not core.any():
        core = mask
    d = ndi.distance_transform_edt(~mask)
    rg = (d > ring[0]) & (d <= ring[1]) & brain
    if others is not None:
        rg &= ~others
    return core, rg


def contrast(a, core, ring):
    return float(a[core].mean() - a[ring].mean()) if core.any() and ring.any() else float("nan")


def lesion_job(key):
    nib = _nib()
    d = f"{RL}/patients/{key}"
    if os.path.exists(f"{d}/lesions.json"):
        return key, "cached"
    img = nib.load(f"{d}/t1.nii.gz"); A = img.affine
    y1 = np.asarray(img.dataobj, np.float32); y2 = load_nii(f"{d}/t2.nii.gz").astype(np.float32)
    les = load_nii(f"{d}/les_t2.nii.gz") > 0
    brain = load_nii(f"{d}/brain.nii.gz") > 0
    seg = load_nii(f"{RL}/synthseg/patients_out/{key}_synthseg.nii.gz").astype(np.int32) \
        if os.path.exists(f"{RL}/synthseg/patients_out/{key}_synthseg.nii.gz") else None
    lab, n = ndi.label(les, structure=np.ones((3, 3, 3)))
    y01, y02 = inpaint_slices(y1, les), inpaint_slices(y2, les)
    sup = ndi.binary_dilation(les, ball(S_DIL))
    m1, m2 = frac_contrast(y1, y01, sup), frac_contrast(y2, y02, sup)
    for nm, arr, dt in (("y0_t1", y01, np.float32), ("y0_t2", y02, np.float32), ("m_t1", m1, np.float32),
                        ("m_t2", m2, np.float32), ("labels", lab, np.int16)):
        nib.save(nib.Nifti1Image(arr.astype(dt), A), f"{d}/{nm}.nii.gz")
    recs = []
    deep = ndi.binary_erosion(brain, iterations=3)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        pad = 7
        box = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad)) for s, dim in zip(sl, lab.shape))
        L = lab[box]; mk = L == i
        others = ndi.binary_dilation((L > 0) & ~mk, ball(S_DIL))
        core, ring = core_ring(mk, brain[box], others)
        zc = int(np.argmax(mk.sum((0, 1))))
        cs = mk[:, :, zc]
        c2core, c2ring = core_ring(cs, brain[box][:, :, zc], others[:, :, zc])
        r = dict(lesion=i, vox=int(mk.sum()), r_eq2d=float(np.sqrt(cs.sum() / np.pi)),
                 in_deep_brain=bool(deep[box][mk].all()),
                 t1_dir=contrast(y1[box], core, ring), t2_dir=contrast(y2[box], core, ring),
                 t1_dir2d=contrast(y1[box][:, :, zc], c2core, c2ring),
                 t2_dir2d=contrast(y2[box][:, :, zc], c2core, c2ring),
                 t1_inp=contrast((y1 - y01)[box], core, ring), t2_inp=contrast((y2 - y02)[box], core, ring),
                 m1_core=float(m1[box][core].mean()), m2_core=float(m2[box][core].mean()))
        # CSF share of the 15 x 15 inpainting window at the central slice (patient SynthSeg)
        if seg is not None:
            cz = box[2].start + zc
            cy, cx = [int(round(v)) for v in np.argwhere(cs).mean(0)]
            cy += box[0].start; cx += box[1].start
            w = seg[max(0, cy - 7):cy + 8, max(0, cx - 7):cx + 8, cz]
            r["csf_win"] = float(np.isin(w, CSF_LABELS).mean())
            r["wm_frac_patient"] = float(np.isin(seg[box][mk], WM).mean())
        recs.append(r)
    json.dump(dict(key=key, n=int(n), lesions=recs), open(f"{d}/lesions.json", "w"), indent=1)
    return key, f"{n} lesions"


def kept_patients():
    out = []
    for f in sorted(glob.glob(f"{RL}/patients/MS-*/meta.json")):
        m = json.load(open(f))
        if not m["exclude"]:
            out.append(m["key"])
    return out


def lesions(workers=3):
    keys = kept_patients()
    with ProcessPoolExecutor(workers) as ex:
        for k, msg in ex.map(lesion_job, keys):
            print(f"  {k}: {msg}", flush=True)


# ================================================================== stage: register (CPU)
def pairs():
    tl = [tuple(t) for t in json.load(open(f"{RL}/targets.json"))]
    test = [s for sp, s in tl if sp == "test"]
    ext = [t for t in tl if t[0] != "test"]
    ext = [ext[i] for i in np.random.default_rng(0).permutation(len(ext))]
    out = []
    for i, k in enumerate(kept_patients()):
        out.append((k, "test", test[i % len(test)]))
        out.append((k, *ext[i % len(ext)]))
    return out


def reg_job(p):
    from exp.prep_lesion import synthmorph, synthmorph_apply
    k, split, sid = p
    od = f"{RL}/pairs/{k}__{sid}"
    if os.path.exists(f"{od}/reg.json"):
        return f"{k}->{sid}", "cached"
    os.makedirs(od, exist_ok=True)
    pd, td = f"{RL}/patients/{k}", f"{RL}/targets/{sid}"
    try:
        synthmorph(f"{pd}/y0_t1.nii.gz", f"{td}/t1.nii.gz", f"{od}/moved_t1.nii.gz",
                   f"{od}/warp.nii.gz", model="joint")
        synthmorph_apply(f"{od}/warp.nii.gz", [(f"{pd}/m_t1.nii.gz", f"{od}/m_t1.nii.gz"),
                                               (f"{pd}/m_t2.nii.gz", f"{od}/m_t2.nii.gz")])
        synthmorph_apply(f"{od}/warp.nii.gz", [(f"{pd}/labels.nii.gz", f"{od}/labels.nii.gz")],
                         method="nearest", dtype="int16")
        a = load_nii(f"{od}/moved_t1.nii.gz").astype(np.float32); b = load_nii(f"{td}/t1.nii.gz").astype(np.float32)
        br = load_nii(f"{td}/brain.nii.gz") > 0
        ncc = float(np.corrcoef(a[br], b[br])[0, 1])
        json.dump(dict(patient=k, split=split, target=sid, ncc=ncc), open(f"{od}/reg.json", "w"))
        os.remove(f"{od}/warp.nii.gz")
        return f"{k}->{sid}", f"ok brain NCC {ncc:.3f}"
    except Exception as e:  # noqa: BLE001
        return f"{k}->{sid}", f"FAILED {type(e).__name__}: {str(e)[-200:]}"


def register(workers=3):
    ps = pairs()
    json.dump(ps, open(f"{RL}/pairs.json", "w"), indent=1)
    print(f"{len(ps)} patient->target registrations", flush=True)
    with ProcessPoolExecutor(min(workers, 3)) as ex:
        for nm, msg in ex.map(reg_job, ps):
            print(f"  {nm}: {msg}", flush=True)


# ================================================================== stage: place (CPU)
def target_rows(sid):
    """Band rows of a target (identical to its npy_public rows) for every map we need."""
    from ulfkit import mri_preprocess as mp
    from exp.prep_lesion import band_rows
    fr = json.load(open(f"{RL}/targets/{sid}/frame.json"))
    z0, z1, cy, cx = fr["frame"]
    frame = mp.FrameSpec(centroid_xy=(cy, cx), z_start=int(z0), z_end=int(z1), mask_z_bbox=(0, 0))
    return fr, (lambda vol: band_rows(vol, frame, fr["z_T2"]))


def place_job(p):
    k, split, sid = p
    od = f"{RL}/pairs/{k}__{sid}"
    if os.path.exists(f"{od}/placed.json"):
        return f"{k}->{sid}", "cached"
    if not os.path.exists(f"{od}/reg.json"):
        return f"{k}->{sid}", "no registration"
    fr, rows = target_rows(sid)
    y1 = rows(load_nii(f"{RL}/targets/{sid}/t1.nii.gz").astype(np.float32))
    y2 = rows(load_nii(f"{RL}/targets/{sid}/t2.nii.gz").astype(np.float32))
    brain = (y1 > 0) & (y2 > 0)
    seg = rows(load_nii(f"{RL}/synthseg/targets_out/{sid}_synthseg.nii.gz").astype(np.int16))
    m1 = rows(load_nii(f"{od}/m_t1.nii.gz").astype(np.float32))
    m2 = rows(load_nii(f"{od}/m_t2.nii.gz").astype(np.float32))
    lab = rows(load_nii(f"{od}/labels.nii.gz").astype(np.int16))
    # rows are (z, x, y): z = band row index, contiguous 1 mm slices
    wm = np.isin(seg, WM); vent = np.isin(seg, VENT); ctx = np.isin(seg, CORTEX)
    dvent = ndi.distance_transform_edt(~vent) if vent.any() else np.full(seg.shape, 99.0)
    dctx = ndi.distance_transform_edt(~ctx) if ctx.any() else np.full(seg.shape, 99.0)
    pl = json.load(open(f"{RL}/patients/{k}/lesions.json"))
    prec = {r["lesion"]: r for r in pl["lesions"]}
    n_rows = lab.shape[0]
    allsup = ndi.binary_dilation(lab > 0, ball(S_DIL))
    recs, kept = [], []
    objs = ndi.find_objects(lab)
    pad = int(RING_OUT) + 2
    for i in [int(v) for v in np.unique(lab) if v > 0]:
        box = tuple(slice(max(0, sl.start - pad), min(dim, sl.stop + pad)) for sl, dim in zip(objs[i - 1], lab.shape))
        z_off = box[0].start
        mk = lab[box] == i
        zz = np.where(mk.any((1, 2)))[0] + z_off
        why = []
        d = ndi.distance_transform_edt(~mk)
        sur = d <= RING_OUT
        if zz.min() - RING_OUT < 0 or zz.max() + RING_OUT > n_rows - 1:
            why.append("outside band")
        if not brain[box][sur].all():
            why.append("surround leaves brain")
        others = allsup[box] & ~(d <= S_DIL)
        core3, ring3 = core_ring(mk, brain[box], others)
        wmf = float(wm[box][mk].mean()); rwm = float(wm[box][ring3].mean()) if ring3.any() else 0.0
        if wmf < WM_MIN: why.append(f"WM {wmf:.2f}")
        if rwm < RING_WM_MIN: why.append(f"ring WM {rwm:.2f}")
        kc = int(zz[np.argmax(mk[zz - z_off].sum((1, 2)))])
        cs = mk[kc - z_off]
        r2 = float(np.sqrt(cs.sum() / np.pi))
        _, ring2 = core_ring(cs, brain[box][kc - z_off], others[kc - z_off])
        if r2 < R_EQ_MIN: why.append(f"r_eq {r2:.2f} < {R_EQ_MIN}")
        if ring2.sum() < RING2_MIN: why.append("ring2 small")
        pr = prec.get(i, {})
        if pr.get("csf_win", 0.0) > CSF_WIN_MAX: why.append(f"CSF in fill window {pr['csf_win']:.2f}")
        cyx = np.argwhere(cs).mean(0) + [box[1].start, box[2].start]
        dv, dc = float(dvent[box][mk].min()), float(dctx[box][mk].min())
        loc = "periventricular" if dv <= NEAR_MM else ("juxtacortical" if dc <= NEAR_MM else "deep WM")
        rec = dict(lesion=i, kc=kc, cy=float(cyx[0]), cx=float(cyx[1]), area=int(cs.sum()), r_eq2d=r2,
                   vox=int(mk.sum()), r_eq3d=float((3 * mk.sum() / (4 * np.pi)) ** (1 / 3)),
                   wm_frac=wmf, ring_wm=rwm, d_vent=dv, d_ctx=dc, location=loc, why=why,
                   patient=pr)
        recs.append(rec)
        if not why:
            kept.append(i)
    # insert only the kept lesions: support = within 2 mm of a kept mask, assigned to the nearest
    keep_m = np.isin(lab, kept)
    if kept:
        dk, ind = ndi.distance_transform_edt(~keep_m, return_indices=True)
        slab = np.where(dk <= S_DIL, lab[tuple(ind)], 0).astype(np.int16)
    else:
        slab = np.zeros_like(lab)
    sup = slab > 0
    touched = np.where(sup.any((1, 2)))[0]
    res = dict(patient=k, split=split, target=sid, n_rows=n_rows, n_lesions=len(recs), kept=kept,
               reg=json.load(open(f"{od}/reg.json")), lesions=recs)
    if len(touched):
        # the 3D ring reaches 4 mm beyond the mask: keep those rows too
        k0, k1 = max(0, int(touched.min()) - pad), min(n_rows, int(touched.max()) + 1 + pad)
        lab_k = np.where(keep_m, lab, 0).astype(np.int16)
        np.savez_compressed(f"{od}/placed.npz", k0=k0, k1=k1,
                            m1=np.where(sup, m1, 0)[k0:k1].astype(np.float16),
                            m2=np.where(sup, m2, 0)[k0:k1].astype(np.float16),
                            lab=lab_k[k0:k1], slab=slab[k0:k1])
        for r in recs:                                    # input contrasts on the central slice
            if r["lesion"] in kept:
                i, kc = r["lesion"], r["kc"]
                cs = lab_k[kc] == i
                oth = (slab[kc] > 0) & (slab[kc] != i)
                core, ring = core_ring(cs, brain[kc], oth)
                s_i = slab[kc] == i
                for c, y, m in (("t1", y1, m1), ("t2", y2, m2)):
                    yl = np.clip(y[kc] * (1 + m[kc] * s_i), 0, 1)
                    r[f"tru_{c}"] = contrast(yl - y[kc], core, ring)
    json.dump(res, open(f"{od}/placed.json", "w"), indent=1, default=float)
    return f"{k}->{sid}", f"{len(kept)}/{len(recs)} lesions kept"


def place(workers=3):
    ps = [tuple(p) for p in json.load(open(f"{RL}/pairs.json"))]
    with ProcessPoolExecutor(min(workers, 3)) as ex:
        for nm, msg in ex.map(place_job, ps):
            print(f"  {nm}: {msg}", flush=True)


# ================================================================== GPU helpers
def gpu_setup():
    import torch
    from exp.d3detect import operator, blobs
    return torch, operator, blobs


def batched(fn, x, bs=16):
    import torch
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)]) if len(x) else x


def run_net(net, x, bs=16):
    import torch
    def f(z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(z).float().clamp(0, 1)
    return batched(f, x, bs)


def enhancers(seeds=False):
    """-> [(run name, enhancer, seed, module)] lazily: list of (name, enhancer, seed, kind, ckpt)."""
    from exp import d3zoo as Z
    reg = [r for r in Z.registry() if (r[2] != 0 if seeds else r[2] == 0)]
    return [r for r in reg if r[4] == "diff" or os.path.exists(f"{OUT}/{r[3]}.pt")]


class Ident:
    def __call__(self, x):
        return x[:, 1:2]
    def eval(self):
        return self


def cr_t(a, core, ring):
    """Core-minus-ring contrast of a (H,W) torch tensor with numpy masks."""
    import torch
    c = torch.from_numpy(core).to(a.device); r = torch.from_numpy(ring).to(a.device)
    return float(a[c].mean() - a[r].mean())


def gauss_masks(b):
    return b > 0.6, (b > 0.05) & (b < 0.25)


# ================================================================== stage: run (GPU)
def load_pairs():
    out = []
    for p in json.load(open(f"{RL}/pairs.json")):
        od = f"{RL}/pairs/{p[0]}__{p[2]}"
        if os.path.exists(f"{od}/placed.npz"):
            pj = json.load(open(f"{od}/placed.json"))
            if pj["kept"]:
                out.append((tuple(p), pj, od))
    return out


def prepare_pair(pj, od, coh2, coh1):
    """-> dict of numpy arrays for one placement in target band-row space."""
    z = np.load(f"{od}/placed.npz")
    sid = pj["target"]
    rows = coh2.subject_rows(sid)
    rows1 = coh1.subject_rows(sid)
    k0, k1 = int(z["k0"]), int(z["k1"])
    y2 = coh2.hf[rows[k0:k1], 0].astype(np.float32)
    y1 = coh1.hf[rows1[k0:k1], 0].astype(np.float32)
    ulf = coh2.ulf[rows[k0:k1]].astype(np.float32) if coh2.ulf is not None else None
    return dict(k0=k0, k1=k1, y1=y1, y2=y2, ulf=ulf, m1=z["m1"].astype(np.float32),
                m2=z["m2"].astype(np.float32), lab=z["lab"], slab=z["slab"],
                brain=(y1 > 0) & (y2 > 0))


def lesion_geometry(P, r):
    """Masks for one kept lesion on its central slice (2D) and over the slab (3D)."""
    i, kc = r["lesion"], r["kc"] - P["k0"]
    oth2 = (P["slab"][kc] > 0) & (P["slab"][kc] != i)
    cs = P["lab"][kc] == i
    core2, ring2 = core_ring(cs, P["brain"][kc], oth2)
    mk3 = P["lab"] == i
    oth3 = (P["slab"] > 0) & (P["slab"] != i)
    core3, ring3 = core_ring(mk3, P["brain"], oth3)
    return dict(kc=kc, s2=(P["slab"][kc] == i), core2=core2, ring2=ring2, core3=core3, ring3=ring3,
                cy=int(round(r["cy"])), cx=int(round(r["cx"])), r=float(r["r_eq2d"]))


def run(seeds=False):
    torch, operator, blobs = gpu_setup()
    from ulfkit import data as D, models as Mo, train as T
    from exp import d3zoo as Z
    from exp.d3pipe import d2_model, pack
    from exp.d3qa import enhancer as abs_unet
    T.seed_all(0)
    coh2, coh1 = {}, {}
    plist = load_pairs()
    for (_, split, _), _, _ in plist:
        if split not in coh2:
            coh2[split], coh1[split] = D.PairedCohort(split, "T2"), D.PairedCohort(split, "T1")
    f2, f1 = operator("T2", "learned"), operator("T1", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    cu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()
    # D is deterministic and slice-wise and its output is the centre slice repeated three times:
    # keep one channel on the CPU (float32) and expand it on use.
    d1c = lambda f, y: torch.cat([f(y[i:i + 16], gen)[:, :1] for i in range(0, len(y), 16)]).cpu()
    g3 = lambda x: x.cuda().repeat(1, 3, 1, 1)

    prep = []
    for (pp, pj, od) in plist:
        split = pj["split"]
        P = prepare_pair(pj, od, coh2[split], coh1[split])
        recs = [r for r in pj["lesions"] if r["lesion"] in pj["kept"]]
        G = [lesion_geometry(P, r) for r in recs]
        kc = [g["kc"] for g in G]
        with torch.no_grad():
            y2 = cu(P["y2"])[:, None]; y1 = cu(P["y1"])[:, None]
            m2 = cu(P["m2"])[:, None]; m1 = cu(P["m1"])[:, None]
            sup = cu((P["slab"] > 0).astype(np.float32))[:, None]
            y2_all = (y2 * (1 + m2 * sup)).clamp(0, 1)
            s_i = torch.stack([cu(g["s2"].astype(np.float32)) for g in G])[:, None]
            y2_real = (y2[kc] * (1 + m2[kc] * s_i)).clamp(0, 1)
            y1_real = (y1[kc] * (1 + m1[kc] * s_i)).clamp(0, 1)
            bmask = cu(P["brain"][kc].astype(np.float32))[:, None]
            b = blobs(torch.tensor([g["cy"] for g in G], device="cuda"),
                      torch.tensor([g["cx"] for g in G], device="cuda"),
                      torch.tensor([g["r"] for g in G], device="cuda")) * bmask
            bn = b.cpu().numpy()[:, 0]
            gc, gr = gauss_masks(bn)
            unit = np.array([float(bn[j][gc[j]].mean() - bn[j][gr[j]].mean()) if gc[j].any() and gr[j].any()
                             else np.nan for j in range(len(G))])
            dy2 = (y2_real - y2[kc]).cpu().numpy()[:, 0]; dy1 = (y1_real - y1[kc]).cpu().numpy()[:, 0]
            tru2 = np.array([contrast(dy2[j], G[j]["core2"], G[j]["ring2"]) for j in range(len(G))])
            tru1 = np.array([contrast(dy1[j], G[j]["core2"], G[j]["ring2"]) for j in range(len(G))])
            c2 = torch.tensor(np.nan_to_num(tru2 / unit), device="cuda", dtype=torch.float32)[:, None, None, None]
            c1 = torch.tensor(np.nan_to_num(tru1 / unit), device="cuda", dtype=torch.float32)[:, None, None, None]
            y2_g = (y2[kc] + c2 * b).clamp(0, 1); y1_g = (y1[kc] + c1 * b).clamp(0, 1)
            X = dict(x2_base=d1c(f2, y2), x2_all=d1c(f2, y2_all), x2_real=d1c(f2, y2_real), x2_g=d1c(f2, y2_g),
                     x1_base=d1c(f1, y1), x1_real=d1c(f1, y1_real), x1_g=d1c(f1, y1_g))
            dg2 = (y2_g - y2[kc]).cpu().numpy()[:, 0]; dg1 = (y1_g - y1[kc]).cpu().numpy()[:, 0]
            da = (y2_all - y2).cpu().numpy()[:, 0]
            acq = None
            if P["ulf"] is not None:                      # acquired 64 mT arm (d3detreal.lesion_input)
                rr = torch.tensor([g["r"] for g in G], device="cuda")
                near = (blobs(torch.tensor([g["cy"] for g in G], device="cuda"),
                              torch.tensor([g["cx"] for g in G], device="cuda"),
                              torch.clamp(rr * 4, min=24)) > 0.05).float().cpu()
                acq = dict(x0=torch.from_numpy(P["ulf"][kc]), near=near)
        ddr = (X["x2_real"] - X["x2_base"][kc]).numpy()[:, 0]; ddg = (X["x2_g"] - X["x2_base"][kc]).numpy()[:, 0]
        dda = (X["x2_all"] - X["x2_base"]).numpy()[:, 0]
        L = []
        for j, (r, g) in enumerate(zip(recs, G)):
            L.append(dict(patient=pj["patient"], target=pj["target"], split=split, lesion=r["lesion"],
                          r_eq2d=r["r_eq2d"], diam_mm=2 * r["r_eq2d"], r_eq3d=r["r_eq3d"], vox=r["vox"],
                          area=r["area"], location=r["location"], wm_frac=r["wm_frac"], d_vent=r["d_vent"],
                          tru_t2=float(tru2[j]), tru_t1=float(tru1[j]),
                          gauss_c_t2=float(c2[j]), gauss_c_t1=float(c1[j]),
                          tru_g_t2=contrast(dg2[j], gc[j], gr[j]), tru_g_t1=contrast(dg1[j], gc[j], gr[j]),
                          tru_gm_t2=contrast(dg2[j], g["core2"], g["ring2"]),
                          tru_gm_t1=contrast(dg1[j], g["core2"], g["ring2"]),
                          tru_all3d_t2=contrast(da, g["core3"], g["ring3"]),
                          dd_real=contrast(ddr[j], g["core2"], g["ring2"]), dd_g=contrast(ddg[j], gc[j], gr[j]),
                          dd_all3d=contrast(dda, g["core3"], g["ring3"]),
                          patient_t1_dir=r["patient"].get("t1_dir"), patient_t2_dir=r["patient"].get("t2_dir"),
                          csf_win=r["patient"].get("csf_win"), out={}))
        prep.append(dict(G=G, kc=kc, gc=gc, gr=gr, X=X, acq=acq, L=L))
        del y2, y1, m2, m1, y2_all, y2_real, y1_real, y2_g, y1_g, b
    torch.cuda.empty_cache()
    print(f"  {len(prep)} placements, {sum(len(p['L']) for p in prep)} lesions prepared", flush=True)

    def score(name, net, routes=False):
        for q in prep:
            X, G, kc, gc, gr = q["X"], q["G"], q["kc"], q["gc"], q["gr"]
            if routes:                                    # name is a route, net its model
                if name == "A_T1in":
                    fn = lambda x1, x2: net(x1)
                elif name == "B_T2in":
                    fn = lambda x1, x2: net(pack(x2=x2))
                else:
                    fn = lambda x1, x2: net(pack(x1, x2))
                def r_(x1, x2):
                    out = []
                    for i in range(0, len(x1), 12):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out.append(fn(g3(x1[i:i + 12]), g3(x2[i:i + 12])).float().clamp(0, 1))
                    return torch.cat(out).cpu().numpy()[:, 0]
                p0 = r_(X["x1_base"][kc], X["x2_base"][kc])
                pr, pg = r_(X["x1_real"], X["x2_real"]) - p0, r_(X["x1_g"], X["x2_g"]) - p0
                for j, g in enumerate(G):
                    q["L"][j]["out"][f"T1|{name}"] = dict(
                        real=contrast(pr[j], g["core2"], g["ring2"]), gauss=contrast(pg[j], gc[j], gr[j]),
                        gauss_m=contrast(pg[j], g["core2"], g["ring2"]))
                continue
            run_ = lambda x: torch.cat([run_net(net, g3(x[i:i + 16])) for i in range(0, len(x), 16)]).cpu().numpy()[:, 0]
            p0 = run_(X["x2_base"])
            pr = run_(X["x2_real"]) - p0[kc]
            pg = run_(X["x2_g"]) - p0[kc]
            pa = run_(X["x2_all"]) - p0
            if q["acq"] is not None:
                x0 = q["acq"]["x0"]; nr = q["acq"]["near"]
                dlr = (X["x2_real"] - X["x2_base"][kc]) * nr; dlg = (X["x2_g"] - X["x2_base"][kc]) * nr
                f3 = lambda x: torch.cat([run_net(net, x[i:i + 16].cuda()) for i in range(0, len(x), 16)]).cpu().numpy()[:, 0]
                a0 = f3(x0)
                ar, ag = f3(x0 + dlr.repeat(1, 3, 1, 1)) - a0, f3(x0 + dlg.repeat(1, 3, 1, 1)) - a0
            for j, g in enumerate(G):
                o = dict(real=contrast(pr[j], g["core2"], g["ring2"]), gauss=contrast(pg[j], gc[j], gr[j]),
                         gauss_m=contrast(pg[j], g["core2"], g["ring2"]),
                         all2d=contrast(pa[g["kc"]], g["core2"], g["ring2"]),
                         all3d=contrast(pa, g["core3"], g["ring3"]))
                if q["acq"] is not None:
                    o.update(acq_real=contrast(ar[j], g["core2"], g["ring2"]), acq_gauss=contrast(ag[j], gc[j], gr[j]))
                q["L"][j]["out"][name] = o

    with torch.no_grad():
        if not seeds:
            score(NOENH, Ident())
            # retention map, queried at each lesion with r_eq and the real input contrast
            base = abs_unet(); base.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T2.pt")); base.eval()
            for mname in MAPS:
                det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
                det.load_state_dict(torch.load(f"{OUT}/{mname}.pt")); det.eval()
                for q in prep:
                    x0 = g3(q["X"]["x2_base"][q["kc"]])
                    p0 = run_net(base, x0)
                    for j, g in enumerate(q["G"]):
                        rr = int(round(g["r"])); l = q["L"][j]
                        inp = torch.cat([x0[j:j + 1], p0[j:j + 1], torch.full_like(p0[j:j + 1], g["r"] / 6.0),
                                         torch.full_like(p0[j:j + 1], l["tru_t2"] / 0.25)], 1)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = det(inp).float()
                        sl = (slice(max(0, g["cy"] - rr), min(256, g["cy"] + rr + 1)),
                              slice(max(0, g["cx"] - rr), min(256, g["cx"] + rr + 1)))
                        l.setdefault("map", {})[mname] = float(v[0, 0][sl].mean())
                del det
            del base
            print("  map scored", flush=True)
            eA = abs_unet(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
            score("A_T1in", eA, routes=True)
            score("B_T2in", d2_model("T2toT1"), routes=True)
            score("C_both", d2_model("bothtoT1"), routes=True)
            for t in ("bothtoT1_ctl", "bothtoT1_aug", "bothtoT1_augw"):
                if os.path.exists(f"{OUT}/d2_{t}.pt"):
                    score(f"C_{t}", d2_model(t), routes=True)
            torch.cuda.empty_cache()
            print("  routes scored", flush=True)
        for nm, enh, seed, ck, kind in enhancers(seeds):
            net = Z.load(kind, ck)
            score(nm, net)
            print(f"  scored {nm}", flush=True)
            del net; torch.cuda.empty_cache()
    rows = [l for q in prep for l in q["L"]]
    tag = "seeds" if seeds else "s0"
    json.dump(dict(cutoffs=cutoffs(), n=len(rows), rows=rows), open(f"{OUT}/d3reallesion_rows_{tag}.json", "w"),
              default=float)
    print(f"  wrote {len(rows)} lesion rows", flush=True)


def cutoffs():
    return dict(erased=ERASED, R_clamp=[R_LO, R_HI], keep_min=KEEP_MIN, sign_min=SIGN_MIN,
                support_dil_vox=S_DIL, ring_mm=[RING_IN, RING_OUT], wm_min=WM_MIN, ring_wm_min=RING_WM_MIN,
                r_eq_min_vox=R_EQ_MIN, ring2_min_px=RING2_MIN, csf_window_max=CSF_WIN_MAX,
                near_mm=NEAR_MM, n_boot=N_BOOT, primary="central slice, one lesion per image, D_HFC")


# ================================================================== stage: sanity (GPU)
def sanity():
    """(i) Gaussians through the transplant code path vs the standard d3detect computation."""
    torch, operator, blobs = gpu_setup()
    from ulfkit import data as D, train as T
    from exp import d3zoo as Z
    from exp.d3detect import site_pool
    T.seed_all(0)
    coh = D.PairedCohort("test", "T2")
    f2 = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    rng = np.random.default_rng(3)
    nets = [(nm, Z.load(kind, ck)) for nm, _, _, ck, kind in enhancers()[:3]]
    rows = []
    with torch.no_grad():
        for sid in coh.subjects:
            idx = coh.subject_rows(sid)
            sel = idx[np.linspace(len(idx) // 5, 4 * len(idx) // 5, 4).astype(int)]
            y = torch.from_numpy(coh.hf[sel].astype(np.float32)).cuda()
            ok = site_pool(y)
            for s in range(len(sel)):
                cand = torch.nonzero(ok[s])
                for _ in range(4):
                    cy, cx = cand[rng.integers(len(cand))].tolist()
                    r = float(rng.choice([2, 3, 4, 6])); c = float(rng.choice([0.12, 0.25]))
                    y0 = y[s:s + 1]
                    b = blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
                              torch.tensor([r], device="cuda")) * (y0 > 0).float()
                    core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                    # (a) standard pipeline (d3detect.make_labels arithmetic)
                    yl = (y0 + c * b).clamp(0, 1)
                    # (b) transplant path, exactly additive m;  (c) fractional m
                    cn, rn = core[0, 0].cpu().numpy(), ring[0, 0].cpu().numpy()
                    ybar = float(y0[0, 0][ring[0, 0]].mean())
                    m_add = torch.where(y0 > 0, c * b / y0.clamp(min=1e-6), torch.zeros_like(y0))
                    m_frac = (c / ybar) * b
                    yb, yc = (y0 * (1 + m_add)).clamp(0, 1), (y0 * (1 + m_frac)).clamp(0, 1)
                    xs = f2(torch.cat([y0, yl, yb, yc]), gen)
                    for nm, net in nets:
                        p = run_net(net, xs)
                        tru = float((yl - y0)[core].mean() - (yl - y0)[ring].mean())
                        Ra = float(np.clip((float((p[1] - p[0])[0][core[0, 0]].mean() - (p[1] - p[0])[0][ring[0, 0]].mean())) / tru, R_LO, R_HI))
                        ret = lambda pl, yy: float(np.clip(cr_t((pl - p[0])[0], cn, rn) / cr_t((yy - y0)[0, 0], cn, rn), R_LO, R_HI))
                        # (d) the same Gaussian scored with the real-lesion definitions: mask = b > e^-2
                        # (d < r), core = mask eroded by 1 px, ring = 2-4 mm shell outside the mask
                        mk = (b[0, 0] > np.exp(-2)).cpu().numpy()
                        mc, mr = core_ring(mk, (y0[0, 0] > 0).cpu().numpy())
                        Rm = float(np.clip(cr_t((p[1] - p[0])[0], mc, mr) / cr_t((yl - y0)[0, 0], mc, mr), R_LO, R_HI))
                        rows.append(dict(sid=sid, r=r, c=c, net=nm, R_standard=Ra, R_transplant_add=ret(p[2], yb),
                                         R_transplant_frac=ret(p[3], yc), R_maskdef=Rm))
    d_add = np.array([abs(x["R_standard"] - x["R_transplant_add"]) for x in rows])
    d_frac = np.array([abs(x["R_standard"] - x["R_transplant_frac"]) for x in rows])
    res = dict(n=len(rows), max_abs_diff_additive=float(d_add.max()), median_abs_diff_fractional=float(np.median(d_frac)),
               p95_abs_diff_fractional=float(np.percentile(d_frac, 95)),
               pass_additive=bool(d_add.max() < 0.01), pass_fractional=bool(np.median(d_frac) < 0.05),
               definition_effect=dict(median_abs=float(np.median([abs(x["R_standard"] - x["R_maskdef"]) for x in rows])),
                                      mean_signed=float(np.mean([x["R_maskdef"] - x["R_standard"] for x in rows])),
                                      by_r={str(rr): float(np.mean([x["R_maskdef"] - x["R_standard"] for x in rows if x["r"] == rr]))
                                            for rr in (2.0, 3.0, 4.0, 6.0)}),
               criteria="additive max|dR| < 0.01; fractional median|dR| < 0.05 (pre-specified)", rows=rows)
    json.dump(res, open(f"{OUT}/d3reallesion_sanity.json", "w"), indent=1)
    print(f"  {len(rows)} sites x nets: additive max |dR| {d_add.max():.4f} "
          f"({'PASS' if res['pass_additive'] else 'FAIL'}); fractional median |dR| {np.median(d_frac):.4f} "
          f"p95 {np.percentile(d_frac, 95):.4f} ({'PASS' if res['pass_fractional'] else 'FAIL'})")


# ================================================================== stage: inplace (GPU)
def inplace():
    """Patients' own T2 through D and G against the inpainted image; pseudo-lesion bias."""
    torch, operator, blobs = gpu_setup()
    from ulfkit import train as T
    from exp import d3zoo as Z
    from exp.d3realret import inpaint
    from ulfkit import mri_preprocess as mp
    from exp.prep_lesion import band_rows
    T.seed_all(0)
    rng = np.random.default_rng(0)
    f2 = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    man = json.load(open(f"{NPY_MS}/manifest_ms_T2.json"))["subjects"]
    H = np.load(f"{NPY_MS}/paired_ms_hf_T2.npy")
    items = []                                            # (y, y0, core, ring, kind, meta)
    for s in man:
        key = s["subject_id"]; d = f"{RL}/patients/{key}"
        meta = json.load(open(f"{d}/meta.json"))
        z0, z1, cy, cx = meta["frame"]
        frame = mp.FrameSpec(centroid_xy=(cy, cx), z_start=int(z0), z_end=int(z1), mask_z_bbox=(0, 0))
        rows = lambda v: band_rows(v, frame, meta["z"])
        y = H[s["slice_start"]:s["slice_start"] + s["slice_count"], 0].astype(np.float32)
        y0 = rows(load_nii(f"{d}/y0_t2.nii.gz").astype(np.float32)) * (y > 0)
        lab = rows(load_nii(f"{d}/labels.nii.gz").astype(np.int16))
        seg = rows(load_nii(f"{RL}/synthseg/patients_out/{key}_synthseg.nii.gz").astype(np.int16))
        brain = y > 0
        wm = np.isin(seg, WM)
        allsup = ndi.binary_dilation(lab > 0, ball(S_DIL))
        far = ~ndi.binary_dilation(lab > 0, ball(5))
        pl = {r["lesion"]: r for r in json.load(open(f"{d}/lesions.json"))["lesions"]}
        for i in [int(v) for v in np.unique(lab) if v > 0]:
            mk = lab == i
            zz = np.where(mk.any((1, 2)))[0]
            kc = int(zz[np.argmax(mk[zz].sum((1, 2)))])
            cs = mk[kc]
            r2 = float(np.sqrt(cs.sum() / np.pi))
            if r2 < R_EQ_MIN or pl.get(i, {}).get("csf_win", 0) > CSF_WIN_MAX:
                continue
            oth = allsup[kc] & ~ndi.binary_dilation(cs, iterations=2)
            core, ring = core_ring(cs, brain[kc], oth)
            if ring.sum() < RING2_MIN or np.isin(seg[kc][cs], WM).mean() < WM_MIN:
                continue
            tru = contrast(y[kc] - y0[kc], core, ring)
            items.append((y[kc], y0[kc], core, ring, "real", dict(patient=key, lesion=i, r_eq2d=r2, tru=tru)))
            # pseudo-lesion: same shape moved to healthy WM of the same patient, inpainted the same way
            ys, xs = np.nonzero(cs); sh = np.stack([ys - int(ys.mean()), xs - int(xs.mean())], 1)
            for _ in range(200):
                kz = int(rng.integers(2, len(y) - 2))
                cand = np.argwhere(wm[kz] & far[kz] & far[kz - 2] & far[kz + 2])
                if not len(cand): continue
                py, px = cand[rng.integers(len(cand))]
                pts = sh + [py, px]
                if pts.min() < 8 or pts.max() > 247: continue
                pm = np.zeros_like(cs); pm[pts[:, 0], pts[:, 1]] = True
                pc, prg = core_ring(pm, brain[kz])
                surround = ndi.distance_transform_edt(~pm) <= RING_OUT
                if not (wm[kz][surround].mean() >= 0.9 and far[kz][surround].all() and prg.sum() >= RING2_MIN):
                    continue
                b = bg_excluding(y[kz], ndi.binary_dilation(pm, iterations=2))
                yp0 = inpaint(y[kz], pm, b) * (y[kz] > 0)
                items.append((y[kz], yp0.astype(np.float32), pc, prg, "pseudo",
                              dict(patient=key, lesion=i, r_eq2d=r2, tru=contrast(y[kz] - yp0, pc, prg),
                                   tru_real=tru, m_core=float(((y[kz] - yp0) / np.maximum(yp0, 0.02))[pc].mean()),
                                   m_core_real=pl.get(i, {}).get("m2_core"))))
                break
    print(f"  {sum(1 for x in items if x[4] == 'real')} real, {sum(1 for x in items if x[4] == 'pseudo')} pseudo", flush=True)
    Y = torch.from_numpy(np.stack([x[0] for x in items])).cuda()[:, None]
    Y0 = torch.from_numpy(np.stack([x[1] for x in items])).cuda()[:, None]
    with torch.no_grad():
        X, X0 = batched(lambda a: f2(a, gen), Y), batched(lambda a: f2(a, gen), Y0)
        out = [dict(kind=x[4], **x[5], out={}) for x in items]
        for j, x in enumerate(items):
            out[j]["dd"] = cr_t((X[j] - X0[j])[1], x[2], x[3])
        for nm, enh, seed, ck, kind in enhancers():
            net = Z.load(kind, ck)
            dp = run_net(net, X) - run_net(net, X0)
            for j, x in enumerate(items):
                out[j]["out"][nm] = cr_t(dp[j, 0], x[2], x[3])
            del net; torch.cuda.empty_cache()
            print(f"  scored {nm}", flush=True)
    json.dump(dict(cutoffs=cutoffs(), rows=out), open(f"{OUT}/d3reallesion_inplace_rows.json", "w"), default=float)
    json.dump(summ_inplace(out), open(f"{OUT}/d3reallesion_inplace.json", "w"), indent=1, default=float)


# ================================================================== statistics
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def cluster_boot(clusters, fn, n=N_BOOT, seed=0):
    """Percentile CI of fn(idx) resampling clusters (lesions nested)."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(clusters); us = sorted(set(cl.tolist())); ix = {u: np.where(cl == u)[0] for u in us}
    reps = []
    for _ in range(n):
        idx = np.concatenate([ix[u] for u in rng.choice(us, len(us))])
        try:
            v = fn(idx)
        except Exception:  # noqa: BLE001 - degenerate resample
            continue
        if v is not None and np.all(np.isfinite(v)):
            reps.append(v)
    reps = np.array(reps)
    return np.percentile(reps, 2.5, axis=0).tolist(), np.percentile(reps, 97.5, axis=0).tolist()


def ratio(out, tru):
    return float(np.clip(out / tru, R_LO, R_HI)) if np.isfinite(out) and abs(tru) > 1e-9 else float("nan")


def summ_inplace(rows):
    real = [r for r in rows if r["kind"] == "real" and abs(r["tru"]) >= KEEP_MIN]
    ps = [r for r in rows if r["kind"] == "pseudo" and abs(r.get("tru_real", 0)) >= KEEP_MIN]
    res = dict(n_real=len(real), n_pseudo=len(ps), per_enhancer={})
    cl = [r["patient"] for r in real]
    for nm in real[0]["out"] if real else []:
        R = np.array([ratio(r["out"][nm], r["tru"]) for r in real])
        res["per_enhancer"][nm] = dict(R=float(np.nanmean(R)), erased=float(np.nanmean(R < ERASED)),
                                       R_ci=cluster_boot(cl, lambda i: np.nanmean(R[i]), n=2000))
    RD = np.array([ratio(r["dd"], r["tru"]) for r in real])
    res["R_D"] = float(np.nanmean(RD))
    bias = np.array([r["tru"] / r["tru_real"] for r in ps])
    mb = np.array([r["m_core"] / r["m_core_real"] for r in ps if r.get("m_core_real")])
    res["pseudo_input_bias"] = dict(median=float(np.median(bias)), iqr=np.percentile(bias, [25, 75]).tolist(),
                                    median_abs=float(np.median(np.abs(bias))),
                                    pass_10pct=bool(np.median(np.abs(bias)) <= 0.10),
                                    m_core_ratio_median=float(np.median(mb)) if len(mb) else None)
    res["pseudo_residual_R"] = {}
    for nm in ps[0]["out"] if ps else []:
        v = np.array([r["out"][nm] / r["tru_real"] for r in ps])
        res["pseudo_residual_R"][nm] = dict(median=float(np.median(v)), mean=float(np.mean(v)),
                                           p95_abs=float(np.percentile(np.abs(v), 95)))
    return res


def summary():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    S = json.load(open(f"{OUT}/d3reallesion_rows_s0.json"))
    rows = S["rows"]
    names = [r[0] for r in enhancers()]
    res = dict(cutoffs=cutoffs(), data={}, transplant={}, agreement={}, ranking={}, map={}, polarity={},
               routes={}, acquired={}, size={}, sanity=None, inplace=None)
    # ---- data description
    pj = [json.load(open(f)) for f in sorted(glob.glob(f"{RL}/pairs/*/placed.json"))]
    allles = [l for p in pj for l in p["lesions"]]
    why = {}
    for l in allles:
        for w in l["why"]:
            k = w.split(" ")[0] if not w.startswith("CSF") else "CSF window"
            why[k] = why.get(k, 0) + 1
    excl = json.load(open(f"{RL}/exclusions.json"))
    pats = sorted({l["patient"] for l in rows}); tg = sorted({l["target"] for l in rows})
    res["data"] = dict(patients_total=60, patients_excluded=excl["excluded"], patients_prepped=len(kept_patients()),
                       placements=len(pj), placements_with_lesions=len({(l['patient'], l['target']) for l in rows}),
                       warped_lesions=len(allles), kept_lesions=len(rows), patients_contributing=len(pats),
                       targets_contributing=len(tg), rejection_reasons=why,
                       reg_ncc_median=float(np.median([p["reg"]["ncc"] for p in pj])),
                       reg_ncc_min=float(np.min([p["reg"]["ncc"] for p in pj])),
                       diam_mm=dict(median=float(np.median([l["diam_mm"] for l in rows])),
                                    iqr=np.percentile([l["diam_mm"] for l in rows], [25, 75]).tolist(),
                                    max=float(np.max([l["diam_mm"] for l in rows]))),
                       tru_t2=dict(median=float(np.median([l["tru_t2"] for l in rows])),
                                   iqr=np.percentile([l["tru_t2"] for l in rows], [25, 75]).tolist()),
                       location={k: sum(l["location"] == k for l in rows) for k in ("periventricular", "juxtacortical", "deep WM")},
                       by_split={s: sum(l["split"] == s for l in rows) for s in sorted({l["split"] for l in rows})})
    fin = lambda l: all(np.isfinite(l["out"][nm][k]) for nm in [NOENH] + names for k in ("real", "gauss", "all2d", "all3d"))
    ok = [l for l in rows if abs(l["tru_t2"]) >= KEEP_MIN and abs(l["tru_g_t2"]) >= KEEP_MIN and fin(l)]
    res["data"]["dropped_low_contrast"] = sum(abs(l["tru_t2"]) < KEEP_MIN for l in rows)
    cl = np.array([l["target"] for l in ok]); clp = np.array([l["patient"] for l in ok])
    res["data"]["lesions_in_R"] = len(ok)
    Rr = {nm: np.array([ratio(l["out"][nm]["real"], l["tru_t2"]) for l in ok]) for nm in [NOENH] + names}
    Rg = {nm: np.array([ratio(l["out"][nm]["gauss"], l["tru_g_t2"]) for l in ok]) for nm in [NOENH] + names}
    Ra2 = {nm: np.array([ratio(l["out"][nm]["all2d"], l["tru_t2"]) for l in ok]) for nm in names}
    Ra3 = {nm: np.array([ratio(l["out"][nm]["all3d"], l["tru_all3d_t2"]) for l in ok]) for nm in names}
    Rgm = {nm: np.array([ratio(l["out"][nm]["gauss_m"], l["tru_gm_t2"]) for l in ok]) for nm in [NOENH] + names}
    RDr = np.array([ratio(l["dd_real"], l["tru_t2"]) for l in ok]); RDg = np.array([ratio(l["dd_g"], l["tru_g_t2"]) for l in ok])
    mean_ci = lambda v, c: (float(np.nanmean(v)), cluster_boot(c, lambda i: np.nanmean(v[i])))
    for nm in [NOENH] + names:
        a, b = Rr[nm], Rg[nm]
        m_r, ci_r = mean_ci(a, cl); m_g, ci_g = mean_ci(b, cl)
        e_r, eci_r = mean_ci((a < ERASED).astype(float), cl); e_g, eci_g = mean_ci((b < ERASED).astype(float), cl)
        d = b - a
        res["transplant"][nm] = dict(R_real=m_r, R_real_ci=ci_r, erased_real=e_r, erased_real_ci=eci_r,
                                     R_gauss=m_g, R_gauss_ci=ci_g, erased_gauss=e_g, erased_gauss_ci=eci_g,
                                     R_real_patient_ci=cluster_boot(clp, lambda i: np.nanmean(a[i]), n=2000))
        gm = Rgm[nm]; okm = np.isfinite(gm)
        res["transplant"][nm].update(R_gauss_maskdef=float(np.nanmean(gm)),
                                     ccc_maskdef=ccc(gm[okm], a[okm]) if okm.sum() > 2 else None)
        if nm != NOENH:
            res["transplant"][nm].update(R_all2d=float(np.nanmean(Ra2[nm])), R_all3d=float(np.nanmean(Ra3[nm])),
                                         erased_all3d=float(np.nanmean(Ra3[nm] < ERASED)))
        kap = None
        try:
            from sklearn.metrics import cohen_kappa_score
            kap = float(cohen_kappa_score(a < ERASED, b < ERASED))
        except Exception:  # noqa: BLE001
            pass
        res["agreement"][nm] = dict(ccc=ccc(b, a), ccc_ci=cluster_boot(cl, lambda i: ccc(b[i], a[i])),
                                    pearson=float(np.corrcoef(a, b)[0, 1]), ba_bias=float(d.mean()),
                                    ba_bias_ci=cluster_boot(cl, lambda i: d[i].mean()),
                                    ba_loa=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                                    kappa_erased=kap)
    res["transplant"]["R_D"] = dict(real=float(np.nanmean(RDr)), real_ci=cluster_boot(cl, lambda i: np.nanmean(RDr[i])),
                                    gauss=float(np.nanmean(RDg)), gauss_ci=cluster_boot(cl, lambda i: np.nanmean(RDg[i])))
    # pooled over the 12 enhancers (each lesion contributes one pair per enhancer)
    A = np.stack([Rr[nm] for nm in names], 1); B = np.stack([Rg[nm] for nm in names], 1)
    res["agreement"]["pooled"] = dict(ccc=ccc(B.ravel(), A.ravel()),
                                      ccc_ci=cluster_boot(cl, lambda i: ccc(B[i].ravel(), A[i].ravel())),
                                      ba_bias=float((B - A).mean()),
                                      ba_loa=[float((B - A).mean() - 1.96 * (B - A).std(ddof=1)),
                                              float((B - A).mean() + 1.96 * (B - A).std(ddof=1))])
    # subject level (target means), as in stats_agree A4
    ut = sorted(set(cl.tolist()))
    sa = np.array([[np.nanmean(Rr[nm][cl == u]) for nm in names] for u in ut])
    sb = np.array([[np.nanmean(Rg[nm][cl == u]) for nm in names] for u in ut])
    res["agreement"]["subject_level_pooled"] = dict(n_targets=len(ut), ccc=ccc(sb.ravel(), sa.ravel()))
    # ---- ranking
    mr = lambda R, i: np.array([np.nanmean(R[nm][i]) for nm in names])
    allidx = np.arange(len(ok))
    rho = float(spearmanr(mr(Rr, allidx), mr(Rg, allidx)).statistic)
    res["ranking"]["real_vs_gauss"] = dict(rho=rho, ci=cluster_boot(cl, lambda i: spearmanr(mr(Rr, i), mr(Rg, i)).statistic))
    er = lambda R, i: np.array([np.nanmean(R[nm][i] < ERASED) for nm in names])
    res["ranking"]["erased_real_vs_gauss"] = dict(rho=float(spearmanr(er(Rr, allidx), er(Rg, allidx)).statistic),
                                                  ci=cluster_boot(cl, lambda i: spearmanr(er(Rr, i), er(Rg, i)).statistic))
    zoo = json.load(open(f"{OUT}/d3zoo_T2.json"))
    zr = np.array([np.mean(zoo[nm]["ret"]) for nm in names])
    res["ranking"]["real_vs_zoo_random_sites"] = dict(rho=float(spearmanr(mr(Rr, allidx), zr).statistic),
                                                      ci=cluster_boot(cl, lambda i: spearmanr(mr(Rr, i), zr).statistic),
                                                      zoo_ret={nm: float(v) for nm, v in zip(names, zr)})
    res["ranking"]["order_real"] = [names[i] for i in np.argsort(-mr(Rr, allidx))]
    res["ranking"]["order_gauss"] = [names[i] for i in np.argsort(-mr(Rg, allidx))]
    # ---- size bins (equivalent diameter, mm)
    dm = np.array([l["diam_mm"] for l in ok])
    for lo, hi in ((4, 6), (6, 8), (8, 12), (12, 99)):
        sel = (dm >= lo) & (dm < hi)
        if sel.sum() >= 5:
            res["size"][f"{lo}-{hi} mm"] = dict(n=int(sel.sum()),
                                                R_real=float(np.nanmean(A[sel])), R_gauss=float(np.nanmean(B[sel])),
                                                erased_real=float(np.nanmean(A[sel] < ERASED)),
                                                erased_gauss=float(np.nanmean(B[sel] < ERASED)))
    # ---- retention map AUROC for real-lesion erasure (enhancer of the map)
    sc = {m: np.array([l["map"][m] for l in ok]) for m in MAPS}
    rad = np.array([l["r_eq2d"] for l in ok])
    for tag, R in (("real", Rr[MAP_ENH]), ("gauss", Rg[MAP_ENH])):
        e = (R < ERASED).astype(int)
        d = dict(n=int(len(e)), erased=float(e.mean()))
        if 0 < e.sum() < len(e):
            def au(s, i): return roc_auc_score(e[i], -s[i]) if len(set(e[i].tolist())) == 2 else np.nan
            for m in MAPS:
                d[m] = [float(roc_auc_score(e, -sc[m]))] + list(cluster_boot(cl, lambda i: au(sc[m], i), n=2000))
            d["radius"] = [float(roc_auc_score(e, rad))]
            a4 = [d[m][0] for m in MAPS]
            d["map_mean"], d["map_sd"] = float(np.mean(a4)), float(np.std(a4, ddof=1))
            d["spearman_map_vs_R"] = float(spearmanr(sc[MAPS[0]], R).statistic)
        res["map"][tag] = d
    # ---- acquired arm (OpenNeuro test targets)
    acq = [j for j, l in enumerate(ok) if "acq_real" in l["out"][names[0]]]
    if acq:
        ai = np.array(acq)
        for nm in names:
            ar = np.array([ratio(ok[j]["out"][nm]["acq_real"], ok[j]["tru_t2"]) for j in acq])
            ag = np.array([ratio(ok[j]["out"][nm]["acq_gauss"], ok[j]["tru_g_t2"]) for j in acq])
            res["acquired"][nm] = dict(n=len(acq), R_acq_real=float(np.nanmean(ar)), R_syn_real=float(np.nanmean(Rr[nm][ai])),
                                       ccc_acq_vs_syn_real=ccc(ar, Rr[nm][ai]), R_acq_gauss=float(np.nanmean(ag)),
                                       ccc_acq_real_vs_acq_gauss=ccc(ag, ar))
    # ---- polarity on real MS lesions (patient-space direct contrast, lesions >= 4 mm)
    allp = []
    for k in kept_patients():
        for l in json.load(open(f"{RL}/patients/{k}/lesions.json"))["lesions"]:
            allp.append(dict(l, patient=k))
    def pat(t1, t2):
        s = lambda v: "iso" if not np.isfinite(v) or abs(v) < SIGN_MIN else ("bright" if v > 0 else "dark")
        return f"T1-{s(t1)}/T2-{s(t2)}"
    wmr = polarity_wmring()
    for tag, sel, k1, k2 in (("r_eq>=2", [l for l in allp if l["r_eq2d"] >= R_EQ_MIN], "t1_dir", "t2_dir"),
                             ("all", allp, "t1_dir", "t2_dir"),
                             ("r_eq>=2, inpainting-based", [l for l in allp if l["r_eq2d"] >= R_EQ_MIN], "t1_inp", "t2_inp"),
                             ("r_eq>=2, WM-only ring", wmr, "t1", "t2")):
        P = [pat(l[k1], l[k2]) for l in sel]
        cats = sorted(set(P)); cp = np.array([l["patient"] for l in sel])
        d = dict(n=len(sel), n_patients=len(set(cp.tolist())))
        for c in cats:
            v = np.array([p == c for p in P], float)
            d[c] = [float(v.mean())] + list(cluster_boot(cp, lambda i: v[i].mean(), n=2000))
        t1 = np.array([l[k1] for l in sel]); t2 = np.array([l[k2] for l in sel])
        both = (np.abs(t1) >= SIGN_MIN) & (np.abs(t2) >= SIGN_MIN)
        d["same_sign_of_signed"] = float(np.mean(np.sign(t1[both]) == np.sign(t2[both]))) if both.any() else None
        d["opposite_sign_of_signed"] = float(np.mean(np.sign(t1[both]) != np.sign(t2[both]))) if both.any() else None
        d["n_signed_both"] = int(both.sum())
        res["polarity"][tag] = d
    # ---- routes A-C on the transplanted lesions (T1 contrast measurable)
    t1ok = [l for l in rows if abs(l["tru_t1"]) >= KEEP_MIN and abs(l["tru_g_t1"]) >= KEEP_MIN]
    clt = np.array([l["target"] for l in t1ok])
    pats_t = [pat(l["tru_t1"], l["tru_t2"]) for l in t1ok]
    res["routes"]["n_T1_measurable"] = len(t1ok)
    res["routes"]["n_kept"] = len(rows)
    res["routes"]["patterns"] = {c: pats_t.count(c) for c in sorted(set(pats_t))}
    for rt in ROUTES:
        key = f"T1|{rt}"
        if not t1ok or key not in t1ok[0]["out"]:
            continue
        R1 = np.array([ratio(l["out"][key]["real"], l["tru_t1"]) for l in t1ok])
        G1 = np.array([ratio(l["out"][key]["gauss"], l["tru_g_t1"]) for l in t1ok])
        inv, invg = (R1 < 0).astype(float), (G1 < 0).astype(float)
        d = dict(R_real=float(np.nanmean(R1)), inverted_real=[float(inv.mean())] + list(cluster_boot(clt, lambda i: inv[i].mean(), n=2000)),
                 R_gauss=float(np.nanmean(G1)), inverted_gauss=[float(invg.mean())] + list(cluster_boot(clt, lambda i: invg[i].mean(), n=2000)),
                 erased_real=float(np.mean(R1 < ERASED)), erased_gauss=float(np.mean(G1 < ERASED)),
                 ccc=ccc(G1, R1), by_pattern={})
        for c in sorted(set(pats_t)):
            s = np.array([p == c for p in pats_t])
            if s.sum() >= 3:
                d["by_pattern"][c] = dict(n=int(s.sum()), inverted_real=float(inv[s].mean()), inverted_gauss=float(invg[s].mean()))
        res["routes"][rt] = d
    # ---- sanity and in-place
    if os.path.exists(f"{OUT}/d3reallesion_sanity.json"):
        s = json.load(open(f"{OUT}/d3reallesion_sanity.json")); s.pop("rows", None); res["sanity"] = s
    if os.path.exists(f"{OUT}/d3reallesion_inplace.json"):
        res["inplace"] = json.load(open(f"{OUT}/d3reallesion_inplace.json"))
    if os.path.exists(f"{OUT}/d3reallesion_rows_seeds.json"):
        res["seeds"] = seed_summary(names)
    json.dump(res, open(f"{OUT}/d3reallesion.json", "w"), indent=1, default=float)
    figure(res, ok, Rr, Rg, names, sc)
    print_summary(res, names)


def polarity_wmring():
    """Sensitivity for the polarity table: direct core-minus-ring contrast with the ring restricted
    to patient SynthSeg white matter (the plain ring mixes in grey matter and CSF, which are darker
    than white matter on T1 and brighter on T2). Lesions with central r_eq >= 2 and >= 10 WM ring voxels."""
    out = []
    for key in kept_patients():
        d = f"{RL}/patients/{key}"
        y1 = load_nii(f"{d}/t1.nii.gz").astype(np.float32); y2 = load_nii(f"{d}/t2.nii.gz").astype(np.float32)
        lab = load_nii(f"{d}/labels.nii.gz").astype(np.int32); brain = load_nii(f"{d}/brain.nii.gz") > 0
        wm = np.isin(load_nii(f"{RL}/synthseg/patients_out/{key}_synthseg.nii.gz"), WM)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None: continue
            box = tuple(slice(max(0, q.start - 7), min(dim, q.stop + 7)) for q, dim in zip(sl, lab.shape))
            L = lab[box]; mk = L == i
            if np.sqrt(mk[:, :, int(np.argmax(mk.sum((0, 1))))].sum() / np.pi) < R_EQ_MIN: continue
            others = ndi.binary_dilation((L > 0) & ~mk, ball(S_DIL))
            core, ring = core_ring(mk, brain[box] & wm[box], others)
            if ring.sum() < 10: continue
            out.append(dict(patient=key, lesion=i, t1=contrast(y1[box], core, ring), t2=contrast(y2[box], core, ring)))
    return out


def seed_summary(names0):
    rows = json.load(open(f"{OUT}/d3reallesion_rows_seeds.json"))["rows"]
    ok = [l for l in rows if abs(l["tru_t2"]) >= KEEP_MIN and abs(l["tru_g_t2"]) >= KEEP_MIN]
    out = {}
    for nm in ok[0]["out"]:
        a = np.array([ratio(l["out"][nm]["real"], l["tru_t2"]) for l in ok])
        b = np.array([ratio(l["out"][nm]["gauss"], l["tru_g_t2"]) for l in ok])
        out[nm] = dict(R_real=float(np.nanmean(a)), R_gauss=float(np.nanmean(b)), ccc=ccc(b, a),
                       erased_real=float(np.mean(a < ERASED)), erased_gauss=float(np.mean(b < ERASED)))
    return out


def figure(res, ok, Rr, Rg, names, sc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
    A = np.stack([Rr[nm] for nm in names], 1); B = np.stack([Rg[nm] for nm in names], 1)
    a = ax[0]
    a.scatter(B.ravel(), A.ravel(), s=2, alpha=0.25, color="#2a78d6", lw=0)
    a.plot([-0.5, 1.5], [-0.5, 1.5], color="0.5", lw=0.6, ls="--")
    a.axhline(ERASED, color="0.75", lw=0.5); a.axvline(ERASED, color="0.75", lw=0.5)
    pc = res["agreement"]["pooled"]
    a.set_title(f"a  lesion level, 12 enhancers\nCCC {pc['ccc']:.2f} [{pc['ccc_ci'][0]:.2f}, {pc['ccc_ci'][1]:.2f}]", loc="left")
    a.set_xlabel("R, matched Gaussian"); a.set_ylabel("R, real MS lesion")
    a = ax[1]
    for j, nm in enumerate(names):
        t = res["transplant"][nm]
        a.errorbar(t["R_gauss"], t["R_real"], xerr=[[t["R_gauss"] - t["R_gauss_ci"][0]], [t["R_gauss_ci"][1] - t["R_gauss"]]],
                   yerr=[[t["R_real"] - t["R_real_ci"][0]], [t["R_real_ci"][1] - t["R_real"]]], fmt="o", ms=2.5,
                   lw=0.6, color="#2a78d6")
    lo = min(min(res["transplant"][nm]["R_gauss_ci"][0], res["transplant"][nm]["R_real_ci"][0]) for nm in names) - 0.03
    hi = max(max(res["transplant"][nm]["R_gauss_ci"][1], res["transplant"][nm]["R_real_ci"][1]) for nm in names) + 0.03
    a.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, ls="--")
    rk = res["ranking"]["real_vs_gauss"]
    a.set_title(f"b  enhancer means\nSpearman {rk['rho']:.2f} [{rk['ci'][0]:.2f}, {rk['ci'][1]:.2f}]", loc="left")
    a.set_xlabel("mean R, Gaussian"); a.set_ylabel("mean R, real")
    a = ax[2]
    keys = [k for k in res["size"]]
    x = np.arange(len(keys))
    a.bar(x - 0.18, [res["size"][k]["erased_real"] * 100 for k in keys], 0.36, color="#2a78d6", label="real")
    a.bar(x + 0.18, [res["size"][k]["erased_gauss"] * 100 for k in keys], 0.36, color="#eb6834", label="Gaussian")
    a.set_xticks(x); a.set_xticklabels([k.replace(" mm", "").replace("12-99", ">12") for k in keys])
    a.set_xlabel("equivalent diameter (mm)"); a.set_ylabel("% erased (12 enhancers)"); a.legend(frameon=False)
    a.set_title("c  erasure by size", loc="left")
    a = ax[3]
    for tag, col in (("real", "#2a78d6"), ("gauss", "#eb6834")):
        R = (Rr if tag == "real" else Rg)[MAP_ENH]
        e = (R < ERASED).astype(int)
        if 0 < e.sum() < len(e):
            f, t, _ = roc_curve(e, -sc[MAPS[0]])
            a.plot(f, t, color=col, lw=1, label=f"{tag}: AUROC {res['map'][tag][MAPS[0]][0]:.2f}")
    a.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--")
    a.set_xlabel("false-positive rate"); a.set_ylabel("true-positive rate"); a.legend(frameon=False, loc="lower right")
    a.set_title("d  retention map, erasure", loc="left")
    plt.tight_layout()
    plt.savefig(f"{FIG}/reallesion.pdf"); plt.savefig(f"{FIG}/reallesion.png", dpi=200)
    plt.close()


def print_summary(res, names):
    d = res["data"]
    print(f"\n== real MS lesions: {d['kept_lesions']} kept ({d['lesions_in_R']} with measurable contrast) from "
          f"{d['patients_contributing']} patients in {d['targets_contributing']} targets; diameter median "
          f"{d['diam_mm']['median']:.1f} mm")
    print(f"  {'enhancer':28s} {'R real':>16s} {'R gauss':>16s} {'erased real':>12s} {'erased gauss':>12s} {'CCC':>6s} {'bias':>7s}")
    for nm in [NOENH] + names:
        t, g = res["transplant"][nm], res["agreement"][nm]
        print(f"  {nm:28s} {t['R_real']:6.3f} [{t['R_real_ci'][0]:.2f},{t['R_real_ci'][1]:.2f}] "
              f"{t['R_gauss']:6.3f} [{t['R_gauss_ci'][0]:.2f},{t['R_gauss_ci'][1]:.2f}] {t['erased_real']*100:11.0f}% "
              f"{t['erased_gauss']*100:11.0f}% {g['ccc']:6.2f} {g['ba_bias']:+7.3f}")
    print(f"  pooled CCC {res['agreement']['pooled']['ccc']:.3f} {res['agreement']['pooled']['ccc_ci']}")
    print(f"  ranking real vs gauss rho {res['ranking']['real_vs_gauss']['rho']:.3f} {res['ranking']['real_vs_gauss']['ci']}; "
          f"real vs zoo {res['ranking']['real_vs_zoo_random_sites']['rho']:.3f}")
    for tag in ("real", "gauss"):
        m = res["map"].get(tag, {})
        if MAPS[0] in m:
            print(f"  map AUROC ({tag} erasure, {m['erased']*100:.0f}% erased): {m[MAPS[0]][0]:.3f} "
                  f"[{m[MAPS[0]][1]:.3f},{m[MAPS[0]][2]:.3f}], 4 maps {m['map_mean']:.3f}+-{m['map_sd']:.3f}, radius {m['radius'][0]:.3f}")
    print("  polarity:", {k: (round(v[0], 3) if isinstance(v, list) else v) for k, v in res["polarity"]["r_eq>=2"].items()})
    for rt in ROUTES:
        if rt in res["routes"]:
            r = res["routes"][rt]
            print(f"  route {rt:16s} inverted real {r['inverted_real'][0]*100:5.1f}% gauss {r['inverted_gauss'][0]*100:5.1f}%  "
                  f"R real {r['R_real']:.3f} gauss {r['R_gauss']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lesions", "register", "place", "sanity", "run", "inplace", "summary"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", action="store_true", help="run: seeds 1-4 of the seeded architectures")
    a = ap.parse_args()
    if a.stage == "lesions": lesions(min(a.workers, 3))
    elif a.stage == "register": register(a.workers)
    elif a.stage == "place": place(a.workers)
    elif a.stage == "sanity": sanity()
    elif a.stage == "run": run(seeds=a.seeds)
    elif a.stage == "inplace": inplace()
    else: summary()
