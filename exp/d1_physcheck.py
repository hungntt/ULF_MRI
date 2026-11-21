"""Does the learned degradation D attenuate real small structures the way the 64 mT scanner does?

Every absolute retention number of the framework (R_D = 0.30 / 0.27, the erased fractions) rests on
how strongly D attenuates a small 3T structure. The agreement check against acquired 64 mT data
cannot test this, because its acquired arm is x_real + [D(y_l) - D(y)] and takes the lesion signal
from D in both arms. This script is the direct test: anatomy that already exists in the 3T scan is
measured in the 3T y, in the acquired 64 mT x of the same subject, and in D(y), with identical masks.

Subjects: paired subjects that did not train D. OpenNeuro ds006557 dev-val (3; used only for D's
checkpoint selection) and test (5), each with two Swoop units on the same GE 3T (HFC, HFE), and
Zenodo 20281403 (11, Philips 3T, a third Swoop). Operators: D_HFC (out_public/d1_forward_T2.pt),
D_HFE (out_public_hfe/d1_forward_T2.pt), each against the unit it was trained on, crossed as a
sensitivity check, and D_HFC T1 (out_public/d1_forward_T1.pt) for the T1 rows. D is deterministic
(d3detect.operator ignores the generator), so no noise draws are averaged.

Structures, defined on the 3T only, on the pipeline's 1 mm axial slice stack:
  sulcal CSF   SynthSeg (FreeSurfer 8.2) on the 3T T1: CSF (label 24) within 3 mm of cortex;
  vessels      3D Frangi vesselness, dark tubes, on the 3T T2, outside WM / ventricles / plexus;
  PVS          3D Frangi, bright tubes, on the 3T T2, inside eroded cerebral WM;
  ventricle    lateral-ventricle edge: core >= 3 mm inside, ring 3-6 mm outside (large control).
Size = 3D local thickness (largest inscribed ball, on a 2x upsampled grid of the mask smoothed
with sigma 0.5 mm), binned 1-2, 2-3, 3-5, 5-8 mm. In-plane (2D) thickness is a secondary binning.
An instance = one structure class and size bin inside one 15 x 15 mm tile of one slice:
core = the class voxels of that bin; ring = brain voxels 1 < d <= 3 mm (in-plane) from the core,
> 1 mm from any other voxel of the class, outside all CSF, vessel and PVS masks. Core and ring
voxels lie >= 3 mm inside the 3T brain mask; an instance is dropped for a 64 mT unit when any of
its voxels is zero there.

    C = mean(core) - mean(ring);  T_real = C(x) / C(y);  T_D = C(D(y)) / C(y)

Pre-specified before any T value was computed (PRESPEC below): instances need >= 4 core and >= 8
ring voxels and |C(y)| >= 0.05 (0.4x the smallest inserted lesion contrast; 0.03 and 0.10 as
sensitivity); the subject-level transfer is sum(C(x) sgn C(y)) / sum|C(y)| (the ratio-of-means
form of R_D in d3dfactor.py); the calibration factor k = median over instances of T_real / T_D;
lesion diameters 2r = 4, 6, 8, 12 mm map to size bins by the FWHM of the Gaussian profile
(1.18 r = 2.4, 3.5, 4.7, 7.1 mm -> 2-3, 3-5, 3-5, 5-8), by 2r as sensitivity; k is taken from the
sulcal-CSF rows (bright on darker tissue in T2, like the main map's lesions). Registration: as
stored, and after a per-tile integer shift (+-2 voxels) maximising the NCC with the 3T y, applied
to x and, symmetrically, to D(y). CIs: subject-cluster bootstrap (10,000 replicates, instances
nested by Poisson weights).

The erasure band: d3dfactor.run's lesions (identical seed and sites) are re-run with D's local
lesion response scaled by k, x_k = D(y) + k [D(y_l) - D(y)], through each seed-0 enhancer, so no
linearity of G is assumed; R = R_D * R_G|D from out_public/d3dfactor_T2.json scaled by k is the
mean-level approximation, reported beside it.

  prep     3T T1 stacks as NIfTI                          (datasets/scratch_public/physcheck)
  seg      mri_synthseg --robust on CPU, 3 workers x 4 threads
  struct   structure masks and local thickness per 3T subject
  measure  D(y) on GPU, instance contrasts                -> physcheck/rows.npz
  erase    k-grid lesion retention for seed-0 enhancers   -> physcheck/erase_{split}.npz
  summary  statistics, figures                            -> out_public/d1_physcheck.json,
                                                             fig_public/physcheck*.pdf

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d1_physcheck.py <stage>
"""
import glob, json, os, subprocess, sys
import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch")
FIG = os.environ.get("ULF_FIG", f"{ROOT}/fig")
DATA = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy")
HFE_DATA = f"{ROOT}/datasets/npy_public_hfe"
WORK = f"{SCR}/physcheck"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")

PRESPEC = dict(
    c_min=0.05, c_min_sensitivity=[0.03, 0.10], min_core=4, min_ring=8,
    tile_mm=15, shift_search_vox=2, ring_mm=[1.0, 3.0], edge_margin_mm=3.0,
    size_bins_mm=[1, 2, 3, 5, 8], thickness="3D local thickness, 2x grid, mask smoothed sigma 0.5 mm",
    sulcal_csf="SynthSeg label 24 within 3 mm (3D) of cortex (3/42), on the 3T T1",
    vessel="3D Frangi (skimage, black ridges, sigma 1-2.5 mm) on 3T T2, top 1% of eligible voxels, "
           "components >= 10 voxels; eligible = >= 3 mm inside brain, not WM/ventricle/plexus",
    pvs="3D Frangi (bright ridges, sigma 0.5-1 mm) on 3T T2, top 1% of WM eroded by 1 voxel, "
        "components >= 3 voxels",
    ventricle="SynthSeg 4/43: core in-plane depth >= 3 mm, ring 3-6 mm outside, non-CSF",
    subject_transfer="sum(C(x) sgn C(y)) / sum|C(y)|",
    k="median over instances of T_real / T_D = C(x) / C(D(y)); sulcal-CSF rows",
    lesion_to_bin_primary="FWHM 1.18 r: r=2->2-3, 3->3-5, 4->3-5, 6->5-8 mm",
    lesion_to_bin_sensitivity="diameter 2r: r=2->3-5, 3->5-8, 4->5-8, 6->5-8 mm",
    bootstrap="10000 subject-cluster replicates, Poisson(1) instance weights within subject",
    stochastic_D="none: d3detect.operator / d3dfactor.load_op are deterministic",
)
BINS = PRESPEC["size_bins_mm"]
BIN_NAMES = [f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:])]
CLASSES = ["sulcal_csf", "vessel", "pvs", "ventricle"]
LESION_BIN_FWHM = {2: "2-3", 3: "3-5", 4: "3-5", 6: "5-8"}
LESION_BIN_DIAM = {2: "3-5", 3: "5-8", 4: "5-8", 6: "5-8"}


# ------------------------------------------------------------------ data
def _manifest(d, split, c):
    return json.load(open(f"{d}/manifest_{split}_{c}.json"))["subjects"]


def subjects():
    """-> [(sid, cohort, split)] for the 3T subjects in the check; never a D-training subject."""
    sp = json.load(open(f"{DATA}/split_public.json"))
    train_ids = {x["id"] for x in sp["dev_train"]}
    out = [(x["id"], "ON", "train") for x in sp["dev_val"]] + [(x["id"], "ON", "test") for x in sp["test"]]
    out += [(s["subject_id"], "V5", "ext_v5") for s in _manifest(DATA, "ext_v5", "T2")]
    assert not train_ids & {s for s, _, _ in out}, "a D-training subject entered the check"
    return out


_CACHE = {}


def _arr(d, name):
    k = (d, name)
    if k not in _CACHE:
        _CACHE[k] = np.load(f"{d}/{name}.npy")
    return _CACHE[k]


def stack(sid, split, c, kind, d=None):
    """-> (Z,256,256) float32 3T ('hf') or 64 mT centre slice ('ulf') for one subject."""
    d = d or DATA
    man = _manifest(d, split, c)
    s = next(x for x in man if x["subject_id"] == sid)
    z = np.array(s["z_indices"]); assert np.all(np.diff(z) == 1), f"{sid}: slices not contiguous"
    a = _arr(d, f"paired_{split}_{kind}_{c}")[s["slice_start"]:s["slice_start"] + s["slice_count"]]
    return (a[:, 0] if kind == "hf" else a[:, 1]).astype(np.float32)


def units(cohort):
    """64 mT units with real scans for a cohort: name -> data dir."""
    return {"HFC": DATA, "HFE": HFE_DATA} if cohort == "ON" else {"V5": DATA}


# ------------------------------------------------------------------ prep / seg
def prep():
    import nibabel as nib
    os.makedirs(f"{WORK}/in", exist_ok=True)
    for sid, coh, split in subjects():
        y1, y2 = stack(sid, split, "T1", "hf"), stack(sid, split, "T2", "hf")
        assert y1.shape == y2.shape
        if coh == "ON":     # HFC and HFE share the GE 3T arrays byte for byte
            assert np.array_equal(y2, stack(sid, split, "T2", "hf", HFE_DATA))
        # d3synthseg.py orientation: (row, col=posterior->anterior, slice=inferior->superior)
        nib.save(nib.Nifti1Image(np.transpose(y1, (1, 2, 0)), np.eye(4)), f"{WORK}/in/{sid}.nii.gz")
        print(f"  {sid} ({coh}, {split}): {y1.shape[0]} slices")


def seg(workers=3, threads=4):
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, FREESURFER_HOME=FS, FS_LICENSE=f"{FS}/license.txt",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(threads))
    os.makedirs(f"{WORK}/seg", exist_ok=True)
    todo = [s for s, _, _ in subjects() if not os.path.exists(f"{WORK}/seg/{s}.nii.gz")]
    def go(sid):
        cmd = [f"{FS}/bin/mri_synthseg", "--i", f"{WORK}/in/{sid}.nii.gz", "--o",
               f"{WORK}/seg/{sid}.nii.gz", "--robust", "--cpu", "--threads", str(threads)]
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"  {sid}: exit {r.returncode}", flush=True)
        if r.returncode:
            print(r.stdout[-1500:], flush=True)
        return r.returncode
    with ThreadPoolExecutor(workers) as ex:
        rc = list(ex.map(go, todo))
    assert not any(rc), "SynthSeg failed on some subjects"


def load_seg(sid):
    """SynthSeg labels on the array grid (Z,H,W): resampled nearest-neighbour by the affines."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    ref = nib.load(f"{WORK}/in/{sid}.nii.gz")
    s = nib.load(f"{WORK}/seg/{sid}.nii.gz")
    if s.shape != ref.shape or not np.allclose(s.affine, ref.affine):
        s = resample_from_to(s, ref, order=0)
    return np.transpose(np.asarray(s.dataobj).astype(np.int16), (2, 0, 1))


# ------------------------------------------------------------------ structures
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]        # ventricles, 3rd/4th ventricle, extracerebral CSF
PLEXUS = [31, 63]


def _bbox(m, pad=2):
    idx = np.nonzero(m)
    return tuple(slice(max(int(i.min()) - pad, 0), int(i.max()) + pad + 1) for i in idx)


def local_thickness(mask, dims=3, kmax=12):
    """Sphere-fitting local thickness in mm (integer) on a 2x nearest-upsampled grid.

    On that grid a sheet of w voxels has inscribed-ball EDT w sub-voxels, so the thickness of a
    voxel is the largest k such that it lies in a ball of radius < k sub-voxels centred where the
    EDT >= k (checked on synthetic sheets of 1-8 voxels: exact; tubes of diameter w: w or w - 1).
    dims=2 computes it slice by slice (in-plane thickness)."""
    from scipy import ndimage as ndi
    out = np.zeros(mask.shape, np.uint8)
    if not mask.any():
        return out
    if dims == 2:
        for z in range(mask.shape[0]):
            out[z] = local_thickness(mask[z][None], 3, kmax)[0] if mask[z].any() else 0
        return out
    bb = _bbox(mask, 2)
    m = mask[bb]
    if m.shape[0] == 1 and mask.ndim == 3:       # one slice: 2D thickness
        m2 = m[0].repeat(2, 0).repeat(2, 1)
    else:
        m2 = m.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    e = ndi.distance_transform_edt(m2)
    t = np.zeros(m2.shape, np.uint8)
    for k in range(1, kmax + 1):
        c = e >= k
        if not c.any():
            break
        t[(ndi.distance_transform_edt(~c) < k) & m2] = k
    if m2.ndim == 2:
        t = t.reshape(t.shape[0] // 2, 2, t.shape[1] // 2, 2).max((1, 3))[None]
    else:
        t = t.reshape(t.shape[0] // 2, 2, t.shape[1] // 2, 2, t.shape[2] // 2, 2).max((1, 3, 5))
    out[bb] = np.where(m, np.maximum(t, 1), 0)
    return out


def _clean(m, min_vox):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        return m
    sz = np.bincount(lab.ravel())
    keep = sz >= min_vox; keep[0] = False
    return keep[lab]


def struct_one(job):
    """-> datasets/scratch_public/physcheck/struct/{sid}.npz: thickness maps per class, masks."""
    sid, coh, split = job
    from scipy import ndimage as ndi
    from skimage.filters import frangi
    dst = f"{WORK}/struct/{sid}.npz"
    if os.path.exists(dst):
        return sid, "cached"
    y1, y2 = stack(sid, split, "T1", "hf"), stack(sid, split, "T2", "hf")
    sg = load_seg(sid)
    assert sg.shape == y2.shape, (sg.shape, y2.shape)
    brain = (y1 > 0) & (y2 > 0)
    inner = ndi.distance_transform_edt(brain) >= PRESPEC["edge_margin_mm"]
    cortex, wm = np.isin(sg, [3, 42]), np.isin(sg, [2, 41])
    vent, plexus, csf = np.isin(sg, [4, 43]), np.isin(sg, PLEXUS), np.isin(sg, CSF_LABELS)
    # Frangi on the 3T T2, outside-brain filled with the brain median so the strip edge is no ridge
    bb = _bbox(brain, 4)
    img = y2.copy(); img[~brain] = np.median(y2[brain])
    vd = np.zeros_like(y2); vb = np.zeros_like(y2)
    vd[bb] = frangi(img[bb], sigmas=(1.0, 1.5, 2.0, 2.5), black_ridges=True)
    vb[bb] = frangi(img[bb], sigmas=(0.5, 0.75, 1.0), black_ridges=False)
    el_v = inner & ~wm & ~ndi.binary_dilation(vent | plexus)
    ves = _clean((vd >= np.percentile(vd[el_v], 99)) & (vd > 0) & el_v, 10)
    el_p = ndi.binary_erosion(wm) & inner
    pvs = _clean((vb >= np.percentile(vb[el_p], 99)) & (vb > 0) & el_p & ~ves, 3)
    sulc = (sg == 24) & (ndi.distance_transform_edt(~cortex) <= 3) & brain & ~ves
    res = dict(inner=inner, brain=brain, vent=vent, plexus=plexus, csf=csf)
    for nm, m in (("sulcal_csf", sulc), ("vessel", ves), ("pvs", pvs)):
        res[f"t3_{nm}"] = local_thickness(m, 3)
        res[f"t2_{nm}"] = local_thickness(m, 2)
    os.makedirs(f"{WORK}/struct", exist_ok=True)
    np.savez_compressed(dst, **res)
    n = {k: int((v > 0).sum()) for k, v in res.items() if k.startswith("t3_")}
    return sid, f"ok {n}"


def struct(workers=3):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        for sid, msg in ex.map(struct_one, subjects()):
            print(f"  {sid}: {msg}", flush=True)
    qc_struct()


def qc_struct():
    """Montage: one slice per subject, 3T T2 with sulcal CSF coloured by size bin, vessels, PVS
    and the ventricle edge core/ring. 3T only, no outcome."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    subs = subjects()
    f, ax = plt.subplots(4, 5, figsize=(25, 20))
    cols = np.array([[1, 0, 0], [1, .6, 0], [1, 1, 0], [0, 1, 0]])
    for a, (sid, coh, split) in zip(ax.ravel(), subs):
        S = np.load(f"{WORK}/struct/{sid}.npz")
        y2 = stack(sid, split, "T2", "hf"); z = int(y2.shape[0] * 0.7)
        rgb = np.repeat(np.clip(y2[z], 0, 1)[..., None], 3, -1) * 0.8
        tb = np.digitize(S["t3_sulcal_csf"][z], BINS) - 1
        for b in range(4):
            rgb[(S["t3_sulcal_csf"][z] > 0) & (tb == b)] = cols[b]
        rgb[S["t3_vessel"][z] > 0] = [0, .4, 1]
        rgb[S["t3_pvs"][z] > 0] = [1, 0, 1]
        a.imshow(rgb[40:216, 30:226]); a.set_title(f"{sid} z={z}", fontsize=9); a.axis("off")
    for a in ax.ravel()[len(subs):]:
        a.axis("off")
    f.suptitle("sulcal CSF 1-2 red, 2-3 orange, 3-5 yellow, 5-8 green; vessels blue; PVS magenta")
    plt.savefig(f"{WORK}/qc_struct.png", dpi=55, bbox_inches="tight"); plt.close()
    print(f"  montage {WORK}/qc_struct.png")


# ------------------------------------------------------------------ measurement
TILE = PRESPEC["tile_mm"]
NT = 256 // TILE                       # 17 tiles per axis; row/col 255 falls outside every tile
SHIFTS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
S0 = SHIFTS.index((0, 0))
KEYS = ["x_HFC", "x_HFE", "x_V5", "D_HFC", "D_HFE"]          # T1 rows: D_HFC = T1 operator
REGS = ("noreg", "reg", "glob")
COLS = (["sub", "contrast", "cls", "binning", "bin", "z", "tile", "n_core", "n_ring", "Cy"]
        + [f"{k}_{r}" for k in KEYS for r in REGS])


def global_offset(img, ref, m):
    """Post-hoc diagnostic (added after D(y) was found displaced from its own input): one
    in-plane translation per subject and image, the NCC peak over the whole stack at integer
    shifts +-2 refined by a parabola per axis. -> (dy, dx) with img[i+dy, j+dx] ~ ref[i, j]."""
    a = ref[m].astype(float)
    def ncc(dy, dx):
        b = np.pad(img, ((0, 0), (2, 2), (2, 2)))[:, 2 + dy:258 + dy, 2 + dx:258 + dx][m]
        return np.corrcoef(a, b)[0, 1]
    c = {(dy, dx): ncc(dy, dx) for dy, dx in SHIFTS}
    by, bx = max(c, key=c.get)
    def par(lo, mid, hi):
        den = lo - 2 * mid + hi
        return 0.0 if den >= 0 else float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))
    oy = by + (par(c[(by - 1, bx)], c[(by, bx)], c[(by + 1, bx)]) if abs(by) < 2 else 0.0)
    ox = bx + (par(c[(by, bx - 1)], c[(by, bx)], c[(by, bx + 1)]) if abs(bx) < 2 else 0.0)
    return oy, ox


def _tile_ids():
    r = np.arange(256) // TILE
    r[r >= NT] = -1
    return np.where((r[:, None] >= 0) & (r[None] >= 0), r[:, None] * NT + r[None], -1)


def _shifted(img):
    """(H,W) -> (25,H,W) with out[s][i,j] = img[i+dy, j+dx] (zero outside)."""
    p = np.pad(img, 2)
    return np.stack([p[2 + dy:258 + dy, 2 + dx:258 + dx] for dy, dx in SHIFTS])


def _best_shift(ref, mov_s, m, tid):
    """Per tile: the shift maximising NCC(ref, shifted mov) over mask m; zero shift when the
    tile has < 50 mask voxels."""
    nt = NT * NT
    idx = np.nonzero(m & (tid >= 0)); t = tid[idx]
    n = np.bincount(t, minlength=nt).astype(float); nn = np.maximum(n, 1)
    a = ref[idx].astype(float)
    sa, saa = np.bincount(t, a, nt), np.bincount(t, a * a, nt)
    ncc = np.empty((len(SHIFTS), nt))
    for s in range(len(SHIFTS)):
        b = mov_s[s][idx].astype(float)
        sb, sbb, sab = np.bincount(t, b, nt), np.bincount(t, b * b, nt), np.bincount(t, a * b, nt)
        va, vb = saa - sa ** 2 / nn, sbb - sb ** 2 / nn
        ncc[s] = (sab - sa * sb / nn) / np.sqrt(np.maximum(va * vb, 1e-12))
    best = ncc.argmax(0)
    best[n < 50] = S0
    return best


def _masks(S, z):
    """-> [(cls, binning, bin, core, ring)] for slice z: 2D boolean masks."""
    from scipy import ndimage as ndi
    inner, csf, vent, plexus = S["inner"][z], S["csf"][z], S["vent"][z], S["plexus"][z]
    ves, pvs = S["t3_vessel"][z] > 0, S["t3_pvs"][z] > 0
    excl = {"sulcal_csf": csf | ves | pvs | plexus, "vessel": vent | plexus | pvs,
            "pvs": csf | ves | plexus}
    lo, hi = PRESPEC["ring_mm"]
    out = []
    for ci, cls in enumerate(CLASSES[:3]):
        for bi, nb in ((0, 3), (1, 2)):
            th = S[f"t{nb}_{cls}"][z]
            cm = th > 0
            if not cm.any():
                continue
            bn = np.digitize(th, BINS) - 1           # 1->0, 2->1, 3-4->2, 5-7->3, 8+->4
            bn[th == 8] = 3
            for b in range(len(BIN_NAMES)):
                core = cm & (bn == b)
                if not (core & inner).any():
                    continue
                other = cm & ~core
                dc = ndi.distance_transform_edt(~core)
                do = ndi.distance_transform_edt(~other) if other.any() else np.full(core.shape, 99.)
                ring = (dc > lo) & (dc <= hi) & (do > lo) & ~excl[cls] & ~cm & inner
                out.append((ci, bi, b, core & inner, ring))
    if vent.any():
        din, dout = ndi.distance_transform_edt(vent), ndi.distance_transform_edt(~vent)
        core = vent & (din >= 3) & inner
        ring = (dout >= 3) & (dout <= 6) & ~csf & ~ves & ~pvs & ~plexus & inner
        if core.any():
            out.append((3, 2, -1, core, ring))
    return out


def _contrasts(imgs, best, masks, tid, mn=(PRESPEC["min_core"], PRESPEC["min_ring"])):
    """-> rows (list of arrays) of tile-level contrasts for one slice and contrast."""
    nt = NT * NT
    rows = []
    for ci, bi, b, core, ring in masks:
        ic, ir = np.nonzero(core & (tid >= 0)), np.nonzero(ring & (tid >= 0))
        tc, tr = tid[ic], tid[ir]
        nc, nr = np.bincount(tc, minlength=nt), np.bincount(tr, minlength=nt)
        ok = np.nonzero((nc >= mn[0]) & (nr >= mn[1]))[0]
        if not len(ok):
            continue
        def con(v_c, v_r):
            return (np.bincount(tc, v_c, nt) / np.maximum(nc, 1)
                    - np.bincount(tr, v_r, nt) / np.maximum(nr, 1))[ok]
        def zero(v_c, v_r):
            return (np.bincount(tc, v_c == 0, nt) + np.bincount(tr, v_r == 0, nt))[ok] > 0
        y = imgs["y"]
        R = np.full((len(ok), len(COLS)), np.nan, np.float32)
        R[:, 2:10] = np.stack([np.full(len(ok), ci), np.full(len(ok), bi), np.full(len(ok), b),
                               np.zeros(len(ok)), ok, nc[ok], nr[ok], con(y[ic], y[ir])], 1)
        for k in KEYS:
            if k not in imgs:
                continue
            a = imgs[k]
            j = COLS.index(f"{k}_noreg")
            c0 = con(a[ic], a[ir]); c0[zero(a[ic], a[ir])] = np.nan
            sh = best[k]
            vc, vr = imgs[k + "_s"][sh[tc], ic[0], ic[1]], imgs[k + "_s"][sh[tr], ir[0], ir[1]]
            c1 = con(vc, vr); c1[zero(vc, vr)] = np.nan
            g = imgs[k + "_g"]
            c2 = con(g[ic], g[ir]); c2[zero(a[ic], a[ir])] = np.nan
            R[:, j], R[:, j + 1], R[:, j + 2] = c0, c1, c2
        rows.append(R)
    return rows


def measure():
    import torch
    from exp.d3dfactor import load_op
    ops = {"T2": {"D_HFC": load_op(f"{OUT}/d1_forward_T2.pt"), "D_HFE": load_op(f"{OUT}_hfe/d1_forward_T2.pt")},
           "T1": {"D_HFC": load_op(f"{OUT}/d1_forward_T1.pt")}}
    tid = _tile_ids()
    allrows, subs, offsets = [], subjects(), {}
    for si, (sid, coh, split) in enumerate(subs):
        S = np.load(f"{WORK}/struct/{sid}.npz")
        S = {k: S[k] for k in S.files}
        for cc, c in enumerate(("T2", "T1")):
            y = stack(sid, split, c, "hf")
            imgs = {f"x_{u}": stack(sid, split, c, "ulf", d) for u, d in units(coh).items()}
            with torch.no_grad():
                yt = torch.from_numpy(y[:, None]).cuda()
                for k, f in ops[c].items():
                    imgs[k] = torch.cat([f(yt[i:i + 16])[:, :1] for i in range(0, len(y), 16)])[:, 0].cpu().numpy()
            from scipy import ndimage as ndi
            for k in list(imgs):
                oy, ox = global_offset(imgs[k], y, S["brain"])
                offsets[f"{sid}|{c}|{k}"] = (oy, ox)
                imgs[k + "_g"] = ndi.shift(imgs[k], (0, -oy, -ox), order=3, mode="constant")
            if sid == "ON-HYPE03" and c == "T2":
                np.savez_compressed(f"{WORK}/example_{sid}.npz", y=y, **{k: v for k, v in imgs.items() if not k.endswith("_g")})
            nrow = 0
            for z in range(len(y)):
                sl = {"y": y[z]}
                best = {}
                for k in [k for k in imgs if not k.endswith("_g")]:
                    sl[k] = imgs[k][z]; sl[k + "_s"] = _shifted(imgs[k][z]); sl[k + "_g"] = imgs[k + "_g"][z]
                    best[k] = _best_shift(y[z], sl[k + "_s"], S["brain"][z], tid)
                for R in _contrasts(sl, best, _masks(S, z), tid):
                    R[:, 0], R[:, 1], R[:, 5] = si, cc, z
                    allrows.append(R); nrow += len(R)
            print(f"  {sid} {c}: {nrow} instances", flush=True)
    rows = np.concatenate(allrows)
    np.savez_compressed(f"{WORK}/rows.npz", rows=rows, cols=np.array(COLS),
                        subjects=np.array([s for s, _, _ in subs]),
                        cohort=np.array([c for _, c, _ in subs]), split=np.array([p for _, _, p in subs]))
    json.dump(offsets, open(f"{WORK}/global_offsets.json", "w"), indent=1)
    print(f"  {len(rows)} instances -> {WORK}/rows.npz")


# ------------------------------------------------------------------ erasure under k-scaled D
KGRID = np.array([0.5, 0.8, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0])     # R(k) interpolated linearly between
IDENTITY = "no enhancer (identity)"


def erase(splits=("test", "ext_v5")):
    """d3dfactor.run's lesions (same rows, sites, radii, contrasts: seed 11, 96 rows, 4 sites),
    with D's local lesion response scaled by k before the enhancer:
        x_k = clamp(D(y) + k [D(y_l) - D(y)], 0, 1) * brain,   p_k = G(x_k).
    Stores per lesion: r, c, dy (3T core-ring), dd (D core-ring), dp[k] (output core-ring)."""
    import torch
    from ulfkit import data as D, train as T
    from exp.d3detect import site_pool, blobs
    from exp import d3dfactor as F, d3zoo as Z
    T.seed_all(0)
    ops = {o: F.load_op(p) for o, p in F.OPS.items()}
    reg = [r for r in Z.registry() if r[2] == 0 and r[4] != "diff" and os.path.exists(f"{OUT}/{r[3]}.pt")]
    kt = torch.tensor(KGRID, device="cuda", dtype=torch.float32)
    for split in splits:
        dst = f"{WORK}/erase_{split}.npz"
        if os.path.exists(dst):
            print(f"  {split}: cached"); continue
        nets = [(r[1], Z.load(r[4], r[3])) for r in reg] + [(IDENTITY, lambda x: x[:, 1:2])]
        coh = D.PairedCohort(split, "T2")
        rng = np.random.default_rng(11)                       # == d3dfactor.run(seed=11)
        rows = np.sort(rng.choice(len(coh.hf), min(96, len(coh.hf)), replace=False))
        rec = {(o, nm): dict(r=[], c=[], dy=[], dd=[], dp=[]) for o in ops for nm, _ in nets}
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                y = torch.from_numpy(coh.hf[rows[i:i + 8]].astype(np.float32)).cuda()
                ok = site_pool(y)
                jobs = []
                for s in range(len(y)):
                    cand = torch.nonzero(ok[s])
                    if len(cand) < 8: continue
                    for _ in range(4):
                        cy, cx = cand[rng.integers(len(cand))].tolist()
                        jobs.append((s, cy, cx, int(rng.choice(F.RADII)), float(rng.choice(F.CS))))
                if not jobs: continue
                si = torch.tensor([j[0] for j in jobs], device="cuda")
                b = blobs(torch.tensor([j[1] for j in jobs], device="cuda"),
                          torch.tensor([j[2] for j in jobs], device="cuda"),
                          torch.tensor([j[3] for j in jobs], device="cuda")) * (y[si] > 0).float()
                c = torch.tensor([j[4] for j in jobs], device="cuda")[:, None, None, None]
                y0 = y[si]; yl = (y0 + c * b).clamp(0, 1)
                core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
                good = [k for k in range(len(jobs)) if core[k].sum() >= 4 and ring[k].sum() >= 8]
                con = lambda d, k: float(d[k][core[k]].mean() - d[k][ring[k]].mean())
                for o, f in ops.items():
                    x0, xl = f(y0), f(yl)
                    dx = xl - x0
                    for nm, net in nets:
                        if nm == IDENTITY:
                            p0 = x0[:, 1:2]
                        else:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                p0 = torch.cat([net(x0[k0:k0 + 12]).float().clamp(0, 1)
                                                for k0 in range(0, len(jobs), 12)])
                        n = len(jobs)
                        xk = ((x0[None] + kt[:, None, None, None, None] * dx[None]).clamp(0, 1)
                              * (y0 > 0)[None]).reshape(-1, *x0.shape[1:])          # (K*n,3,H,W)
                        if nm == IDENTITY:
                            pk = xk[:, 1:2]
                        else:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pk = torch.cat([net(xk[k0:k0 + 24]).float().clamp(0, 1)
                                                for k0 in range(0, len(xk), 24)])
                        dps = list((pk.reshape(len(kt), n, *pk.shape[1:]) - p0[None]).unbind(0))
                        R = rec[(o, nm)]
                        for k in good:
                            R["r"].append(jobs[k][3]); R["c"].append(jobs[k][4])
                            R["dy"].append(con(yl - y0, k)); R["dd"].append(con(dx[:, 1:2], k))
                            R["dp"].append([con(d, k) for d in dps])
                print(f"  {split}: rows {i + 8}/{len(rows)}", flush=True)
        out = {}
        for (o, nm), R in rec.items():
            for f_ in ("r", "c", "dy", "dd", "dp"):
                out[f"{o}|{nm}|{f_}"] = np.array(R[f_], np.float32)
        np.savez_compressed(dst, kgrid=KGRID, **out)
        print(f"  {split}: {len(rec[(o, nm)]['r'])} lesions per (D, G) -> {dst}", flush=True)
        del nets; torch.cuda.empty_cache()


# ------------------------------------------------------------------ statistics
# (unit with real scans, operator column): matched = the operator trained on that unit
COMPARE = {"T2": {"HFC|D_HFC": "matched", "HFE|D_HFE": "matched", "V5|D_HFC": "external",
                  "V5|D_HFE": "external", "HFE|D_HFC": "cross", "HFC|D_HFE": "cross"},
           "T1": {"HFC|D_HFC": "matched", "HFE|D_HFC": "cross", "V5|D_HFC": "external"}}
POPULATED = dict(min_subjects=3, min_instances=50)     # a size bin with fewer is treated as empty
PRESPEC["populated_bin"] = POPULATED
PRESPEC["empty_bin_rule"] = "a lesion size whose bin is empty takes k of the nearest populated bin below"
PRESPEC["primary"] = "T2, 3D binning, per-tile shift refinement, c_min 0.05, sulcal CSF for k"
POST_HOC = ("'glob' alignment (one sub-voxel in-plane translation per subject and image, NCC peak "
            "over the whole stack, cubic-spline shift) was added after D(y) was found displaced from "
            "its own input by ~0.5 voxel along the anterior-posterior axis; it is a diagnostic beside "
            "the pre-specified per-tile refinement ('reg') and the stored alignment ('noreg').")
GROUPS = ([(ci, 0, b) for ci in range(3) for b in range(4)] + [(3, 2, -1)])        # primary: 3D
GROUPS_2D = [(ci, 1, b) for ci in range(3) for b in range(4)]


def _gname(ci, bi, b):
    return CLASSES[ci] if ci == 3 else f"{CLASSES[ci]}|{'3d' if bi == 0 else '2d'}|{BIN_NAMES[b]}"


def _ccc(x, y, w=None):
    w = np.ones_like(x) if w is None else w
    sw = w.sum(-1, keepdims=True)
    mx, my = (w * x).sum(-1, keepdims=True) / sw, (w * y).sum(-1, keepdims=True) / sw
    vx, vy = (w * (x - mx) ** 2).sum(-1) / sw[..., 0], (w * (y - my) ** 2).sum(-1) / sw[..., 0]
    cov = (w * (x - mx) * (y - my)).sum(-1) / sw[..., 0]
    return 2 * cov / (vx + vy + (mx - my)[..., 0] ** 2)


def stats(sub, cy, cx, cd, B=10000, seed=0, chunk=250):
    """Point estimates and subject-cluster bootstrap CIs (subjects resampled with replacement,
    instances nested by Poisson(1) weights) for one comparison and instance group."""
    us, inv = np.unique(sub, return_inverse=True)
    S, N = len(us), len(sub)
    sg = np.sign(cy)
    ay, ax_, ad = np.abs(cy), cx * sg, cd * sg
    ratio = cx / cd
    M = np.zeros((N, S), np.float32); M[np.arange(N), inv] = 1
    sy, sx, sd = ay @ M, ax_ @ M, ad @ M
    tr_s, td_s = sx / sy, sd / sy
    d = tr_s - td_s
    pt = dict(n=int(N), n_subj=int(S), T_real=float(ax_.sum() / ay.sum()), T_D=float(ad.sum() / ay.sum()),
              k_median=float(np.median(ratio)), C_y_median=float(np.median(cy)),
              subj=dict(ids=[str(u) for u in us], T_real=tr_s.tolist(), T_D=td_s.tolist(),
                        n=np.bincount(inv).tolist()))
    pt["k_ratio"] = pt["T_real"] / pt["T_D"]
    if S >= 2:
        pt["ccc"] = float(_ccc(tr_s, td_s)); pt["ba_bias"] = float(d.mean())
        sdd = float(d.std(ddof=1)); pt["ba_loa"] = [pt["ba_bias"] - 1.96 * sdd, pt["ba_bias"] + 1.96 * sdd]
    if B == 0 or S < 3:
        return pt
    rng = np.random.default_rng(seed)
    order = np.argsort(ratio); rs = ratio[order]
    acc = {k: [] for k in ("T_real", "T_D", "k_ratio", "k_median", "ccc", "ba_bias")}
    for c0 in range(0, B, chunk):
        b = min(chunk, B - c0)
        cnt = rng.multinomial(S, np.full(S, 1.0 / S), size=b).astype(np.float32)        # (b,S)
        P = rng.poisson(1.0, size=(b, N)).astype(np.float32)                              # (b,N)
        py, px, pd_ = (P * ay) @ M, (P * ax_) @ M, (P * ad) @ M                           # (b,S)
        Ty, Tx, Td = (cnt * py).sum(1), (cnt * px).sum(1), (cnt * pd_).sum(1)
        acc["T_real"].append(Tx / Ty); acc["T_D"].append(Td / Ty); acc["k_ratio"].append(Tx / Td)
        ok = py > 0
        w = cnt * ok
        trb, tdb = np.where(ok, px / np.where(ok, py, 1), 0), np.where(ok, pd_ / np.where(ok, py, 1), 0)
        acc["ccc"].append(_ccc(trb, tdb, w))
        acc["ba_bias"].append((w * (trb - tdb)).sum(1) / w.sum(1))
        W = (cnt[:, inv] * P)[:, order]
        cs = np.cumsum(W, 1)
        acc["k_median"].append(rs[np.minimum((cs < cs[:, -1:] / 2).sum(1), N - 1)])
    for k, v in acc.items():
        v = np.concatenate(v)
        pt[f"{k}_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return pt


def _select(rows, cols, cc, ci, bi, b, comp, reg, cmin, subs_ok=None):
    u, o = comp.split("|")
    j = cols.index
    cx, cd = rows[:, j(f"x_{u}_{reg}")], rows[:, j(f"{o}_{reg}")]
    m = ((rows[:, 1] == cc) & (rows[:, 2] == ci) & (rows[:, 3] == bi) & (rows[:, 4] == b)
         & (np.abs(rows[:, 9]) >= cmin) & np.isfinite(cx) & np.isfinite(cd))
    if subs_ok is not None:
        m &= np.isin(rows[:, 0], subs_ok)
    return rows[m, 0].astype(int), rows[m, 9].astype(float), cx[m].astype(float), cd[m].astype(float)


def summary_stats():
    d = np.load(f"{WORK}/rows.npz")
    rows, cols = d["rows"], [str(c) for c in d["cols"]]
    sids, split = [str(s) for s in d["subjects"]], [str(s) for s in d["split"]]
    test_only = [i for i, s in enumerate(split) if s in ("test", "ext_v5")]   # drops dev-val (selection)
    cmin = PRESPEC["c_min"]
    res = dict(prespec=PRESPEC, post_hoc=POST_HOC, subjects=sids, analyses={})
    off = f"{WORK}/global_offsets.json"
    if os.path.exists(off):
        o = json.load(open(off))
        res["global_offsets"] = o
        for key in ("D_HFC", "D_HFE", "x_HFC", "x_HFE", "x_V5"):
            for c in ("T2", "T1"):
                v = np.array([o[k] for k in o if k.endswith(f"|{c}|{key}")])
                if len(v):
                    res.setdefault("offset_summary", {})[f"{c}|{key}"] = dict(
                        n=len(v), mean_dy=float(v[:, 0].mean()), mean_dx=float(v[:, 1].mean()),
                        sd_dx=float(v[:, 1].std(ddof=1)), min_dx=float(v[:, 1].min()), max_dx=float(v[:, 1].max()))
    res["rd_offset_loss"] = rd_offset_loss(res.get("offset_summary", {}).get("T2|D_HFC", {}).get("mean_dx", 0.5))
    def put(key, *a, B=10000, **kw):
        s = _select(rows, cols, *a, **kw)
        if len(s[0]) == 0:
            return
        res["analyses"][key] = stats(*s, B=B)
        r = res["analyses"][key]
        print(f"  {key:62s} n={r['n']:6d} S={r['n_subj']:2d} T_real={r['T_real']:.3f} T_D={r['T_D']:.3f} "
              f"k={r['k_median']:.3f}", flush=True)
    for c, cc in (("T2", 0), ("T1", 1)):
        for comp, kind in COMPARE[c].items():
            primary_comp = c == "T2" and kind in ("matched", "external")
            for ci, bi, b in GROUPS:
                for reg in REGS:
                    full = primary_comp or reg == "reg"
                    put(f"{c}|{comp}|{_gname(ci, bi, b)}|{reg}|c{cmin}", cc, ci, bi, b, comp, reg, cmin,
                        B=10000 if full else 2000)
                if primary_comp:
                    for cm in PRESPEC["c_min_sensitivity"]:
                        put(f"{c}|{comp}|{_gname(ci, bi, b)}|reg|c{cm}", cc, ci, bi, b, comp, "reg", cm, B=2000)
                    if comp.startswith(("HFC", "HFE")):
                        put(f"{c}|{comp}|{_gname(ci, bi, b)}|reg|c{cmin}|testonly", cc, ci, bi, b, comp,
                            "reg", cmin, B=2000, subs_ok=test_only)
            if primary_comp:
                for ci, bi, b in GROUPS_2D:
                    put(f"{c}|{comp}|{_gname(ci, bi, b)}|reg|c{cmin}", cc, ci, bi, b, comp, "reg", cmin, B=2000)
    return res


def rd_offset_loss(dx):
    """Fraction of a Gaussian lesion's core-minus-ring contrast that remains when its rendering is
    displaced by dx voxels (the masks stay at the 3T position), per radius: what D's learned
    displacement alone costs R_D."""
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[:64, :64].astype(float)
    out = {"dx": float(dx)}
    for r in (2, 3, 4, 6):
        b = np.exp(-2 * ((yy - 32) ** 2 + (xx - 32) ** 2) / r ** 2)
        core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
        bs = ndi.shift(b, (0, dx), order=3)
        c0 = b[core].mean() - b[ring].mean(); c1 = bs[core].mean() - bs[ring].mean()
        out[str(r)] = float(c1 / c0)
    return out


def k_for_lesions(res, comp, mapping, binning="3d", reg="reg", stat="k_median"):
    """-> {r: (k, lo, hi, bin used)} from the sulcal-CSF rows of a T2 comparison."""
    A = res["analyses"]
    def get(bn):
        a = A.get(f"T2|{comp}|sulcal_csf|{binning}|{bn}|{reg}|c{PRESPEC['c_min']}")
        if a and a["n_subj"] >= POPULATED["min_subjects"] and a["n"] >= POPULATED["min_instances"]:
            return a
    out = {}
    for r, bn in mapping.items():
        i = BIN_NAMES.index(bn)
        while i >= 0 and get(BIN_NAMES[i]) is None:
            i -= 1
        a = get(BIN_NAMES[i])
        out[r] = (a[stat], *a[f"{stat}_ci"], BIN_NAMES[i])
    return out


def erasure_band(res):
    """Erased fraction (R < 0.25) of the seed-0 enhancers when D's lesion response is scaled by
    the per-size k of the matching 64 mT unit; linear interpolation on the k grid per lesion."""
    fac = json.load(open(f"{OUT}/d3dfactor_T2.json"))
    unit = {("test", "HFC"): "HFC|D_HFC", ("test", "HFE"): "HFE|D_HFE",
            ("ext_v5", "HFC"): "V5|D_HFC", ("ext_v5", "HFE"): "V5|D_HFE"}
    out = dict(note=("x_k = D(y) + k [D(y_l) - D(y)] through each enhancer (no linearity of G "
                     "assumed); 'approx' is the mean-level R_D * R_G|D * k from d3dfactor_T2.json "
                     "with k averaged over the four lesion sizes, which assumes R_G|D unchanged "
                     "when the 64 mT lesion contrast scales by k."), splits={})
    for split in ("test", "ext_v5"):
        f = f"{WORK}/erase_{split}.npz"
        if not os.path.exists(f):
            continue
        E = np.load(f); kg = E["kgrid"]
        names = sorted({k.split("|")[1] for k in E.files if "|" in k})
        S = out["splits"][split] = {}
        for op in ("HFC", "HFE"):
            comp = unit[(split, op)]
            ks = {m: k_for_lesions(res, comp, mp) for m, mp in
                  (("fwhm", LESION_BIN_FWHM), ("diameter", LESION_BIN_DIAM))}
            ks["fwhm_kratio"] = k_for_lesions(res, comp, LESION_BIN_FWHM, stat="k_ratio")
            ks["fwhm_noreg"] = k_for_lesions(res, comp, LESION_BIN_FWHM, reg="noreg")
            ks["fwhm_glob"] = k_for_lesions(res, comp, LESION_BIN_FWHM, reg="glob")
            S[op] = dict(k_source=comp, k={m: {str(r): v for r, v in kk.items()} for m, kk in ks.items()},
                         enhancers={})
            for nm in names:
                r, dy, dp = E[f"{op}|{nm}|r"], E[f"{op}|{nm}|dy"], E[f"{op}|{nm}|dp"]
                Rk = dp / dy[:, None]                       # (n, K), unclamped like d3dfactor
                def er(kmap, j):
                    kv = np.array([kmap[int(x)][j] for x in r])
                    Ri = np.array([np.interp(kv[i], kg, Rk[i]) for i in range(len(r))])
                    return float(np.mean(Ri < 0.25)), float(np.mean(Ri)), \
                        {str(rr): float(np.mean(Ri[r == rr] < 0.25)) for rr in (2, 3, 4, 6)}
                one = {int(x): (1.0, 1.0, 1.0) for x in (2, 3, 4, 6)}
                e1 = er(one, 0)
                row = dict(erased_k1=e1[0], R_k1=e1[1], erased_k1_by_r=e1[2])
                for m in ks:
                    pe, a1, a2 = er(ks[m], 0), er(ks[m], 1), er(ks[m], 2)    # k point, CI low, CI high
                    row[m] = dict(erased=pe[0], erased_band=sorted([a1[0], a2[0]]), R=pe[1],
                                  R_band=sorted([a1[1], a2[1]]), erased_by_r=pe[2],
                                  erased_by_r_band={q: sorted([a1[2][q], a2[2][q]]) for q in pe[2]})
                fk = fac.get(split, {}).get("per", {}).get(f"{op}|{nm}")
                if fk:
                    kbar = np.mean([ks["fwhm"][x][0] for x in (2, 3, 4, 6)])
                    row["d3dfactor"] = dict(R=fk["R"], erased=fk["erased"], R_D=fk["R_D"], R_GD=fk["R_GD"],
                                            approx_R_k=fk["R_D"] * fk["R_GD"] * kbar)
                    row["reproduces_d3dfactor"] = bool(abs(fk["erased"] - e1[0]) < 0.02)
                S[op]["enhancers"][nm] = row
    return out


def _logk(a):
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    a.set_yscale("log"); a.set_ylim(0.45, 3.4)
    a.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.25, 1.5, 2, 3]))
    a.yaxis.set_major_formatter(FormatStrFormatter("%g")); a.yaxis.set_minor_locator(NullLocator())


def figures(res, er):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    col = {"HFC": "#2a78d6", "HFE": "#eb6834", "V5": "#1baf7a"}
    lab = {"HFC": "OpenNeuro HFC (D_HFC)", "HFE": "OpenNeuro HFE (D_HFE)", "V5": "Zenodo (D_HFC)"}
    comps = {"HFC": "HFC|D_HFC", "HFE": "HFE|D_HFE", "V5": "V5|D_HFC"}
    A, cm = res["analyses"], PRESPEC["c_min"]
    xs = BIN_NAMES + ["ventricle"]
    def key(comp, g, reg="reg", c="T2"):
        return f"{c}|{comp}|{g}|{reg}|c{cm}"
    def gk(i):
        return "ventricle" if i == 4 else f"sulcal_csf|3d|{BIN_NAMES[i]}"
    f, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    # (a) transfer vs size, real (filled, solid) vs D (open, dashed)
    a = ax[0, 0]
    for j, (u, comp) in enumerate(comps.items()):
        off = (j - 1) * 0.12
        for which, ls, mfc in (("T_real", "-", col[u]), ("T_D", "--", "white")):
            pts = [(i + off, A[key(comp, gk(i))]) for i in range(5) if key(comp, gk(i)) in A]
            x = [p[0] for p in pts]; v = [p[1][which] for p in pts]
            ci = np.array([p[1].get(f"{which}_ci", [np.nan, np.nan]) for p in pts])
            a.errorbar(x, v, yerr=np.abs(ci.T - v), color=col[u], ls=ls, lw=1.2, marker="o", ms=4,
                       mfc=mfc, mew=1, capsize=0, elinewidth=0.8,
                       label=f"{lab[u]}, {'acquired' if which == 'T_real' else 'D(y)'}")
    a.set_xticks(range(5)); a.set_xticklabels([f"{x}\nmm" if x != "ventricle" else "ventricle\nedge" for x in xs])
    a.set_ylabel("contrast transfer T = C(64 mT) / C(3T)"); a.set_title("a  Sulcal CSF (T2) by thickness", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=5.5, loc="upper left")
    # (b) k vs size
    a = ax[0, 1]
    a.axhline(1, color="#888", lw=0.6)
    a.axvspan(0.5, 3.5, color="#eee", lw=0)
    a.text(2.0, 0.02, "lesion scale (FWHM 2.4-7.1 mm)", ha="center", va="bottom", fontsize=6, color="#555",
           transform=a.get_xaxis_transform())
    for j, (u, comp) in enumerate(comps.items()):
        off = (j - 1) * 0.12
        pts = [(i + off, A[key(comp, gk(i))]) for i in range(5) if key(comp, gk(i)) in A]
        x = [p[0] for p in pts]; v = [p[1]["k_median"] for p in pts]
        ci = np.array([p[1].get("k_median_ci", [np.nan, np.nan]) for p in pts])
        a.errorbar(x, v, yerr=np.abs(ci.T - v), color=col[u], lw=1.2, marker="o", ms=4, capsize=0,
                   elinewidth=0.8, label=lab[u])
    a.set_xticks(range(5)); a.set_xticklabels([f"{x}\nmm" if x != "ventricle" else "ventricle\nedge" for x in xs])
    _logk(a)
    a.set_ylabel("k = median T_real / T_D, log scale\n(k > 1: D attenuates more than the scanner)", fontsize=6)
    a.set_title("b  Calibration factor k", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=6)
    # (c) small structures: vessels and PVS (3D, 1-2 and 2-3 mm), real vs D
    a = ax[1, 0]
    grp = [(f"vessel|3d|{BIN_NAMES[0]}", "vessel 1-2"), (f"vessel|3d|{BIN_NAMES[1]}", "vessel 2-3"),
           (f"pvs|3d|{BIN_NAMES[0]}", "PVS 1-2"), (f"pvs|3d|{BIN_NAMES[1]}", "PVS 2-3")]
    a.axhline(0, color="#888", lw=0.6)
    for j, (u, comp) in enumerate(comps.items()):
        for gi, (g, _) in enumerate(grp):
            r_ = A.get(key(comp, g))
            if not r_: continue
            for w, (which, mfc) in enumerate((("T_real", col[u]), ("T_D", "white"))):
                xx = gi + (j - 1) * 0.22 + (w - 0.5) * 0.08
                ci = r_.get(f"{which}_ci", [np.nan, np.nan])
                a.errorbar([xx], [r_[which]], yerr=[[r_[which] - ci[0]], [ci[1] - r_[which]]], color=col[u],
                           marker="o", ms=4, mfc=mfc, mew=1, lw=0, elinewidth=0.8)
    a.set_xticks(range(len(grp))); a.set_xticklabels([g[1].replace(" ", "\n") + " mm" for g in grp])
    a.set_ylabel("contrast transfer T"); a.set_title("c  Vessels and PVS (T2; filled acquired, open D)",
                                                     loc="left", fontweight="bold")
    # (d) erasure band, test split, D_HFC
    a = ax[1, 1]
    S = er["splits"].get("test", {}).get("HFC")
    if S:
        names = sorted(S["enhancers"], key=lambda n: S["enhancers"][n]["erased_k1"])
        for i, nm in enumerate(names):
            r_ = S["enhancers"][nm]
            a.plot([r_["erased_k1"] * 100], [i], "o", ms=4, color="#888", mfc="white", mew=1)
            fw = r_["fwhm"]
            a.plot([fw["erased_band"][0] * 100, fw["erased_band"][1] * 100], [i, i], color=col["HFC"], lw=1.5)
            a.plot([fw["erased"] * 100], [i], "o", ms=4, color=col["HFC"])
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=5.5)
        a.set_xlabel("lesions erased (R < 0.25), %")
        a.plot([], [], "o", color="#888", mfc="white", label="as published (k = 1)")
        a.plot([], [], "o-", color=col["HFC"], label="k-scaled D (95% CI band)")
        a.legend(frameon=False, fontsize=6, loc="lower right")
    a.set_title("d  Erasure under k-scaled D (test, D_HFC)", loc="left", fontweight="bold")
    f.tight_layout()
    f.savefig(f"{FIG}/physcheck.pdf"); f.savefig(f"{WORK}/physcheck.png", dpi=110); plt.close(f)
    # supplementary: registration, binning, T1, cross operators, subject-level Bland-Altman
    f, ax = plt.subplots(2, 3, figsize=(10, 5.6))
    def kplot(a, items, title):
        a.axhline(1, color="#888", lw=0.6)
        for j, (lbl, comp, g_fn, reg, c, colr, ls) in enumerate(items):
            pts = [(i + (j - len(items) / 2) * 0.08, A.get(f"{c}|{comp}|{g_fn(i)}|{reg}|c{cm}")) for i in range(5)]
            pts = [p for p in pts if p[1]]
            v = [p[1]["k_median"] for p in pts]
            ci = np.array([p[1].get("k_median_ci", [np.nan, np.nan]) for p in pts])
            a.errorbar([p[0] for p in pts], v, yerr=np.abs(ci.T - v), color=colr, ls=ls, marker="o", ms=3,
                       lw=1, elinewidth=0.7, label=lbl)
        a.set_xticks(range(5)); a.set_xticklabels(["1-2 mm", "2-3", "3-5", "5-8", "ventricle"])
        a.set_title(title, loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=5, loc="upper left")
        _logk(a); a.set_ylabel("k (log scale)")
    g3 = gk
    g2 = lambda i: "ventricle" if i == 4 else f"sulcal_csf|2d|{BIN_NAMES[i]}"
    kplot(ax[0, 0], [("HFC refined", "HFC|D_HFC", g3, "reg", "T2", col["HFC"], "-"),
                     ("HFC as stored", "HFC|D_HFC", g3, "noreg", "T2", col["HFC"], "--"),
                     ("HFE refined", "HFE|D_HFE", g3, "reg", "T2", col["HFE"], "-"),
                     ("HFE as stored", "HFE|D_HFE", g3, "noreg", "T2", col["HFE"], "--"),
                     ("V5 refined", "V5|D_HFC", g3, "reg", "T2", col["V5"], "-"),
                     ("V5 as stored", "V5|D_HFC", g3, "noreg", "T2", col["V5"], "--"),
                     ("HFC global shift", "HFC|D_HFC", g3, "glob", "T2", col["HFC"], ":"),
                     ("HFE global shift", "HFE|D_HFE", g3, "glob", "T2", col["HFE"], ":"),
                     ("V5 global shift", "V5|D_HFC", g3, "glob", "T2", col["V5"], ":")],
          "a  Alignment variants")
    kplot(ax[0, 1], [("HFC 3D thickness", "HFC|D_HFC", g3, "reg", "T2", col["HFC"], "-"),
                     ("HFC in-plane", "HFC|D_HFC", g2, "reg", "T2", col["HFC"], "--"),
                     ("HFE 3D", "HFE|D_HFE", g3, "reg", "T2", col["HFE"], "-"),
                     ("HFE in-plane", "HFE|D_HFE", g2, "reg", "T2", col["HFE"], "--"),
                     ("V5 3D", "V5|D_HFC", g3, "reg", "T2", col["V5"], "-"),
                     ("V5 in-plane", "V5|D_HFC", g2, "reg", "T2", col["V5"], "--")], "b  3D vs in-plane size")
    kplot(ax[0, 2], [("HFC, D_HFC (matched)", "HFC|D_HFC", g3, "reg", "T2", col["HFC"], "-"),
                     ("HFC, D_HFE (cross)", "HFC|D_HFE", g3, "reg", "T2", col["HFC"], "--"),
                     ("HFE, D_HFE (matched)", "HFE|D_HFE", g3, "reg", "T2", col["HFE"], "-"),
                     ("HFE, D_HFC (cross)", "HFE|D_HFC", g3, "reg", "T2", col["HFE"], "--"),
                     ("V5, D_HFE", "V5|D_HFE", g3, "reg", "T2", col["V5"], "--")], "c  Operator vs unit")
    kplot(ax[1, 0], [("HFC T1, D_HFC T1", "HFC|D_HFC", g3, "reg", "T1", col["HFC"], "-"),
                     ("HFE T1, D_HFC T1", "HFE|D_HFC", g3, "reg", "T1", col["HFE"], "--"),
                     ("V5 T1, D_HFC T1", "V5|D_HFC", g3, "reg", "T1", col["V5"], "-")], "d  T1 (CSF dark)")
    for a, gname, ttl in ((ax[1, 1], "sulcal_csf|3d|2-3", "e  Bland-Altman, sulcal CSF 2-3 mm"),
                          (ax[1, 2], "sulcal_csf|3d|3-5", "f  Bland-Altman, sulcal CSF 3-5 mm")):
        a.axhline(0, color="#888", lw=0.6)
        for u, comp in comps.items():
            r_ = A.get(key(comp, gname))
            if not r_: continue
            tr, td = np.array(r_["subj"]["T_real"]), np.array(r_["subj"]["T_D"])
            a.plot((tr + td) / 2, tr - td, "o", ms=3.5, color=col[u], label=lab[u])
            a.axhline(r_["ba_bias"], color=col[u], lw=0.8)
            for l in r_["ba_loa"]:
                a.axhline(l, color=col[u], lw=0.5, ls=":")
        a.set_xlabel("(T_real + T_D) / 2 per subject"); a.set_ylabel("T_real - T_D")
        a.set_title(ttl, loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=5.5)
    f.tight_layout(); f.savefig(f"{FIG}/physcheck_supp.pdf"); f.savefig(f"{WORK}/physcheck_supp.png", dpi=110)
    plt.close(f)
    example_figure()


def example_figure():
    """One OpenNeuro test slice: 3T, acquired HFC / HFE, D_HFC(y), D_HFE(y), sulcal masks."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f_ = f"{WORK}/example_ON-HYPE03.npz"
    if not os.path.exists(f_):
        return
    E = np.load(f_); S = np.load(f"{WORK}/struct/ON-HYPE03.npz")
    z = int(E["y"].shape[0] * 0.7); win = (slice(60, 140), slice(40, 140))
    ims = [("3T y", E["y"]), ("64 mT HFC", E["x_HFC"]), ("D_HFC(y)", E["D_HFC"]),
           ("64 mT HFE", E["x_HFE"]), ("D_HFE(y)", E["D_HFE"])]
    f, ax = plt.subplots(1, 6, figsize=(12, 2.4))
    for a, (t, im) in zip(ax, ims):
        a.imshow(im[z][win], cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=8); a.axis("off")
    th = S["t3_sulcal_csf"][z]; inner = S["inner"][z]
    rgb = np.repeat(np.clip(E["y"][z], 0, 1)[..., None], 3, -1) * 0.7
    cols = np.array([[.89, .29, .28], [.92, .41, .2], [.93, .63, 0], [0, .51, 0]])
    tb = np.digitize(th, BINS) - 1
    for b in range(4):
        rgb[(th > 0) & (tb == b) & inner] = cols[b]
    ax[5].imshow(rgb[win]); ax[5].set_title("sulcal CSF by size bin", fontsize=8); ax[5].axis("off")
    f.tight_layout(); f.savefig(f"{FIG}/physcheck_example.pdf"); f.savefig(f"{WORK}/physcheck_example.png", dpi=110)
    plt.close(f)


def summary():
    res = summary_stats()
    json.dump(res, open(f"{OUT}/d1_physcheck.json", "w"), indent=1)
    er = erasure_band(res)
    json.dump(er, open(f"{OUT}/d1_physcheck_erasure.json", "w"), indent=1)
    for split, S in er["splits"].items():
        for op, s in S.items():
            print(f"\n== erasure, {split}, D_{op}; k from {s['k_source']}: {s['k']['fwhm']}")
            for nm, r_ in s["enhancers"].items():
                fw = r_["fwhm"]
                print(f"  {nm:30s} erased k=1 {r_['erased_k1']:.3f} (d3dfactor {r_.get('d3dfactor', {}).get('erased', np.nan):.3f})"
                      f"  k-scaled {fw['erased']:.3f} [{fw['erased_band'][0]:.3f}, {fw['erased_band'][1]:.3f}]")
    figures(res, er)


if __name__ == "__main__":
    stage = sys.argv[1]
    {"prep": prep, "seg": seg, "struct": struct, "qc": qc_struct, "measure": measure,
     "erase": erase, "summary": summary, "figures": lambda: figures(
         json.load(open(f"{OUT}/d1_physcheck.json")), json.load(open(f"{OUT}/d1_physcheck_erasure.json")))
     }[stage]()
