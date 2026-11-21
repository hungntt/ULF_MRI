"""Can real MS lesions enter the public pipeline the way the healthy 3T cohorts do?

Phase C answers "a Gaussian blob is not a lesion" with real
lesions. The source is Muslim et al. 2022 (Mendeley Data 10.17632/8bctsm8jz7.1, CC BY 4.0):
60 MS patients, 1.5 T from 20 centres (Baghdad Teaching Hospital MS clinic, 2019-2020), 2D axial
T1w / T2w / FLAIR with consensus manual lesion masks drawn per sequence. This script builds their
arrays as exp/prep_public.py builds the 3T-only cohorts, and builds the healthy 3T targets that
exp/d3reallesion.py transplants the lesions into.

Geometry repair. The released NIfTI files carry no geometry: qform and sform codes are 0, every
voxel is labelled 1 x 1 x 1 mm, and matrices run from 224 x 224 to 1024 x 1024 with 15-33 slices.
Before any FreeSurfer tool can run, each volume gets a new affine:
  orientation  rows anterior->posterior, columns right->left (radiological), slices
               inferior->superior, checked visually for all 120 T1/T2 volumes. Eight volumes store
               their slices superior->inferior and are flipped (REVERSED). Left and right cannot be
               told apart on these images; the radiological convention is assumed.
  slice step   'Spacing Between Slices' from the T1 column of the dataset's supplementary Table 2
               (slices 3.5-6 mm thick, steps 4.7-8.7 mm, i.e. with gaps). The T2 columns of that
               table repeat the T1 values (T2 'TR 400 ms, TE 8.7 ms'), so the T2 starts from the T1
               step and the T2->T1 affine absorbs the error.
  pixel size   recorded nowhere. Estimated per volume from the head's in-plane extent on its
               largest axial section: s = (185 / AP_px + 148 / LR_px) / 2 mm (adult head length and
               breadth with scalp). Within a patient the T1/T2 estimates agree with the matrix
               ratio wherever the two share a field of view. Any global scale error is absorbed by
               the patient->target affine of the transplant, so lesion sizes are reported in the
               target's 1 mm space, not the patient's.

Pipeline per patient, mirroring prep_external.py / prep_public.py: RAS -> SynthStrip -> N4 in
native geometry -> T1 to 1 mm; the T2 is registered to the 1 mm T1 with SynthMorph AFFINE
(12 dof, not rigid: the pixel sizes are estimates and the T2 slice step is unknown); lesion masks
follow their image with nearest-neighbour resampling; intensities are scaled like every public
cohort (brain-mask 0.5/99.5 percentiles of the T1-framed middle 45% band, to [0, 1]), here applied
to the whole 1 mm volume as well as to the band rows. Lesions are the T2 consensus mask; the T1
mask (the T1-visible subset) is carried along for reference.

Pre-specified exclusions (reasons in {ULF_SCRATCH}/reallesion/exclusions.json): a tool fails;
the T2 lesion mask is empty after resampling; the T2->T1 affine scales any axis by more than 40%;
the brain-masked correlation of T1 and T2 gradient magnitudes after registration is below 0.25.

  run       per patient -> {ULF_SCRATCH}/reallesion/patients/MS-XX/
  pack      -> datasets/npy_public_ms/paired_ms_{hf,les}_{T1,T2}.npy + manifests (3T-only format)
  qc        one montage per patient, overview montages, exclusions.json
  targets   healthy 3T targets (5 OpenNeuro test, 11 Zenodo, 5 per MRI-PairedField cohort):
            full 1 mm volumes scaled exactly like their datasets/npy_public rows, checked row by row
  synthseg  mri_synthseg --robust on the target and patient T1 (white matter for placement)

    python exp/prep_lesion.py run --raw datasets/raw/ms_muslim/ex [--workers 3]
    python exp/prep_lesion.py {pack,qc,targets,synthseg}
"""
import argparse, glob, json, os, re, subprocess, sys, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, ".")
ROOT = "."
NPY = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy_public")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch_public")
RL = f"{SCR}/reallesion"
NPY_MS = f"{ROOT}/datasets/npy_public_ms"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
os.environ.setdefault("FREESURFER_HOME", FS)
os.environ.setdefault("SYNTHSTRIP_BIN", f"{FS}/bin/mri_synthstrip")
THREADS = 4

# (patient, contrast) volumes whose slices run superior->inferior (visual check of all 120)
REVERSED = {(16, "T1"), (16, "T2"), (32, "T2"), (46, "T1"), (46, "T2"), (48, "T2"),
            (57, "T1"), (57, "T2")}
HEAD_AP_MM, HEAD_LR_MM = 185.0, 148.0
MAX_SCALE_DEV, MIN_GRAD_CORR = 0.40, 0.25          # exclusion cutoffs, fixed before running
TARGET_SPLITS = ("test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip")
N_PUB3T = 5


def pid(k):
    return f"MS-{int(k):02d}"


# ------------------------------------------------------------------ dataset tables
def read_xlsx(path):
    """First sheet of an .xlsx as a list of rows (stdlib only: zip + XML)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            ss.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    def col(ref):
        n = 0
        for ch in re.match(r"[A-Z]+", ref).group(0):
            n = n * 26 + ord(ch) - 64
        return n - 1
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is not None:
                vals[col(c.get("r"))] = ss[int(v.text)] if c.get("t") == "s" else v.text
        if vals:
            rows.append([vals.get(i, "") for i in range(max(vals) + 1)])
    return rows


def seq_table(raw):
    """-> {patient: dict(t1_thick, t1_step, flair_step)} from supplementary Table 2."""
    f = glob.glob(f"{raw}/Supplementary Table 2*.xlsx")[0]
    out = {}
    for r in read_xlsx(f)[2:]:
        if r and r[0].strip().isdigit():
            out[int(r[0])] = dict(flair_thick=float(r[3]), flair_step=float(r[4]),
                                  t1_tr=float(r[5]), t1_te=float(r[6]), t1_thick=float(r[7]),
                                  t1_step=float(r[8]), t2_tr=float(r[9]), t2_te=float(r[10]),
                                  t2_thick=float(r[11]), t2_step=float(r[12]))
    return out


def patient_table(raw):
    f = glob.glob(f"{raw}/Supplementary Table 1*.xlsx")[0]
    rows = read_xlsx(f)
    head = rows[1]
    return {int(r[0]): dict(zip(head, r)) for r in rows[2:] if r and r[0].strip().isdigit()}


# ------------------------------------------------------------------ geometry repair
def head_mask_slices(v):
    import scipy.ndimage as ndi
    from skimage.filters import threshold_otsu
    t = threshold_otsu(v[v > 0]) if (v > 0).any() else 0.0
    m = v > 0.5 * t
    hm = np.zeros_like(m)
    for z in range(v.shape[2]):
        lab, n = ndi.label(m[:, :, z])
        if n:
            hm[:, :, z] = ndi.binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
    return hm


def pixel_size(v):
    """In-plane pixel size (mm) from the head's extent on its largest axial section."""
    hm = head_mask_slices(v)
    z = int(np.argmax(hm.sum((0, 1))))
    e0 = np.ptp(np.where(hm[:, :, z].any(1))[0]) + 1
    e1 = np.ptp(np.where(hm[:, :, z].any(0))[0]) + 1
    return 0.5 * (HEAD_AP_MM / e0 + HEAD_LR_MM / e1), int(e0), int(e1)


def pls_affine(shape, s, dz):
    """Axis 0 -> posterior, axis 1 -> left, axis 2 -> superior; origin at the volume centre."""
    A = np.zeros((4, 4))
    A[1, 0] = -s          # i: anterior -> posterior  (RAS y decreases)
    A[0, 1] = -s          # j: right -> left          (RAS x decreases)
    A[2, 2] = dz          # k: inferior -> superior
    A[3, 3] = 1
    c = (np.array(shape[:3]) - 1) / 2.0
    A[:3, 3] = -A[:3, :3] @ c
    return A


def repaired(raw, k, c, step):
    """-> (image, mask, geometry record) with the new affine; mask is the consensus lesion mask."""
    import nibabel as nib
    d = f"{raw}/Patient-{k}"
    v = np.asarray(nib.load(f"{d}/{k}-{c}.nii").dataobj, np.float32)
    m = np.asarray(nib.load(f"{d}/{k}-LesionSeg-{c}.nii").dataobj) > 0
    assert v.shape == m.shape, (k, c, v.shape, m.shape)
    if (k, c) in REVERSED:
        v, m = v[:, :, ::-1].copy(), m[:, :, ::-1].copy()
    s, e0, e1 = pixel_size(v)
    A = pls_affine(v.shape, s, step)
    img = nib.Nifti1Image(np.clip(v, 0, None), A); img.set_qform(A, 1); img.set_sform(A, 1)
    msk = nib.Nifti1Image(m.astype(np.uint8), A); msk.set_qform(A, 1); msk.set_sform(A, 1)
    return img, msk, dict(shape=list(v.shape), pixel_mm=round(float(s), 4), step_mm=step,
                          head_px=[e0, e1], reversed=(k, c) in REVERSED)


# ------------------------------------------------------------------ registration helpers
def synthmorph(moving, fixed, out, trans, model="affine", threads=THREADS):
    env = dict(os.environ, FREESURFER_HOME=FS)
    cmd = [f"{FS}/bin/mri_synthmorph", "register", "-m", model, "-j", str(threads),
           "-o", out, "-t", trans, moving, fixed]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode or not os.path.exists(trans):
        raise RuntimeError(f"synthmorph {model} failed: {r.stderr[-400:]}")


def synthmorph_apply(trans, pairs, method="linear", dtype="float32"):
    env = dict(os.environ, FREESURFER_HOME=FS)
    cmd = [f"{FS}/bin/mri_synthmorph", "apply", "-m", method, "-t", dtype, trans]
    for a, b in pairs:
        cmd += [a, b]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode or not all(os.path.exists(b) for _, b in pairs):
        raise RuntimeError(f"synthmorph apply failed: {r.stderr[-400:]}")


def lta_matrix(path):
    """4x4 RAS-to-RAS matrix of an LTA file (first transform)."""
    lines = open(path).read().splitlines()
    i = next(n for n, l in enumerate(lines) if l.strip().startswith("1 4 4"))
    return np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(4)])


def grad_corr(a, b, m):
    import scipy.ndimage as ndi
    ga = np.sqrt(sum(ndi.sobel(a, ax) ** 2 for ax in range(3)))
    gb = np.sqrt(sum(ndi.sobel(b, ax) ** 2 for ax in range(3)))
    return float(np.corrcoef(ga[m], gb[m])[0, 1])


# ------------------------------------------------------------------ intensity scaling
def scale_like_public(v, m, frame):
    """The public cohorts' brain-mask 0.5/99.5 percentile scaling (stats on the framed middle
    band, as in mri_preprocess.frame_and_normalize), applied to the whole unframed volume."""
    from ulfkit import mri_preprocess as mp
    vf = mp.pad_crop_inplane_about_center(v, frame.centroid_xy)
    mf = mp.pad_crop_inplane_about_center(m.astype(np.float32), frame.centroid_xy) > 0.5
    z1 = min(frame.z_end, vf.shape[2]); z0 = min(max(0, frame.z_start), z1)
    vals = vf[:, :, z0:z1][mf[:, :, z0:z1]]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    n = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return n * (n > 0), float(lo), float(hi)


def band_rows(vol, frame, z_idx):
    """Framed rows (z_indices relative to the band start), like the npy arrays."""
    from ulfkit import mri_preprocess as mp
    vf = mp.pad_crop_inplane_about_center(vol, frame.centroid_xy)
    return np.stack([vf[:, :, frame.z_start + z] for z in z_idx])


def band_z(frame, depth):
    """z indices kept by extract_paired_slices(middle_fraction=1.0): slab-safe rows of the band."""
    from ulfkit.slices import extract_paired_slices
    n = min(frame.z_end, depth) - frame.z_start
    dummy = np.zeros((2, 2, n), np.float32)
    _, _, _, z = extract_paired_slices(dummy, dummy, slice_selection="middle",
                                       middle_fraction=1.0, filter_empty=False)
    return [int(x) for x in z]


# ------------------------------------------------------------------ per-patient pipeline
def process(job):
    k, raw, seq = job
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from ulfkit import mri_preprocess as mp
    from exp.prep_external import strip_n4
    key = pid(k)
    od = f"{RL}/patients/{key}"
    if os.path.exists(f"{od}/meta.json"):
        return key, "cached"
    os.makedirs(od, exist_ok=True)
    meta = dict(key=key, patient=k)
    try:
        geo = {}
        for c in ("T1", "T2"):
            img, msk, g = repaired(raw, k, c, seq["t1_step"])
            nib.save(img, f"{od}/raw_{c}.nii.gz"); nib.save(msk, f"{od}/raw_les_{c}.nii.gz")
            geo[c] = g
        meta["geometry"] = geo
        # strip + N4 in native geometry, then the 1 mm T1 reference grid
        strip_n4(f"{od}/raw_T1.nii.gz", f"{od}/T1_native.nii.gz")
        ref = mp.resample_to_isotropic(nib.load(f"{od}/T1_native.nii.gz"), 1.0, order=1)
        nib.save(ref, f"{od}/T1_1mm.nii.gz")
        les1 = resample_from_to(mp.reorient_to_ras(nib.load(f"{od}/raw_les_T1.nii.gz")), ref, order=0)
        strip_n4(f"{od}/raw_T2.nii.gz", f"{od}/T2_native.nii.gz")
        synthmorph(f"{od}/T2_native.nii.gz", f"{od}/T1_1mm.nii.gz", f"{od}/T2_1mm.nii.gz",
                   f"{od}/T2_to_T1.lta", "affine")
        synthmorph_apply(f"{od}/T2_to_T1.lta", [(f"{od}/raw_les_T2.nii.gz", f"{od}/les_T2_1mm.nii.gz")],
                         method="nearest", dtype="uint8")
        M = lta_matrix(f"{od}/T2_to_T1.lta")
        sv = np.linalg.svd(M[:3, :3], compute_uv=False)
        meta["T2_to_T1_singular_values"] = [round(float(x), 4) for x in sv]
        vols = {"T1": np.asarray(ref.dataobj, np.float32),
                "T2": np.asarray(nib.load(f"{od}/T2_1mm.nii.gz").dataobj, np.float32)}
        les = {"T1": np.asarray(les1.dataobj) > 0,
               "T2": np.asarray(nib.load(f"{od}/les_T2_1mm.nii.gz").dataobj) > 0}
        assert vols["T2"].shape == vols["T1"].shape == les["T2"].shape, "grid mismatch"
        masks = {}
        for c in ("T1", "T2"):
            v = np.clip(np.nan_to_num(vols[c]), 0, None); vols[c] = v
            masks[c] = v > 1e-3 * np.percentile(v[v > 0], 99)
        _, _, frame = mp.frame_and_normalize(vols["T1"], masks["T1"])
        normed, lohi = {}, {}
        for c in ("T1", "T2"):
            normed[c], lo, hi = scale_like_public(vols[c], masks[c], frame)
            lohi[c] = [lo, hi]
        brain = (normed["T1"] > 0) & (normed["T2"] > 0)
        for c in ("T1", "T2"):
            les[c] &= brain
        meta["grad_corr_T1_T2"] = grad_corr(normed["T1"], normed["T2"], brain)
        meta["les_vox"] = {c: int(les[c].sum()) for c in ("T1", "T2")}
        meta["brain_cm3"] = float(brain.sum() / 1000.0)
        A = ref.affine
        for nm, arr, dt in (("t1", normed["T1"], np.float32), ("t2", normed["T2"], np.float32),
                            ("les_t1", les["T1"], np.uint8), ("les_t2", les["T2"], np.uint8),
                            ("brain", brain, np.uint8)):
            nib.save(nib.Nifti1Image(arr.astype(dt), A), f"{od}/{nm}.nii.gz")
        z = band_z(frame, normed["T1"].shape[2])
        np.savez(f"{od}/arrays.npz",
                 hf_T1=band_rows(normed["T1"], frame, z).astype(np.float16),
                 hf_T2=band_rows(normed["T2"] * brain, frame, z).astype(np.float16),
                 les_T1=band_rows(les["T1"].astype(np.uint8), frame, z),
                 les_T2=band_rows(les["T2"].astype(np.uint8), frame, z))
        meta.update(frame=[frame.z_start, frame.z_end, *frame.centroid_xy], z=z, lohi=lohi,
                    shape_1mm=list(normed["T1"].shape))
        reasons = []
        if meta["les_vox"]["T2"] == 0:
            reasons.append("T2 lesion mask empty after resampling")
        if np.any(np.abs(sv - 1) > MAX_SCALE_DEV):
            reasons.append(f"T2->T1 affine scale {sv.round(2).tolist()} beyond +-{MAX_SCALE_DEV:.0%}")
        if meta["grad_corr_T1_T2"] < MIN_GRAD_CORR:
            reasons.append(f"T1/T2 gradient correlation {meta['grad_corr_T1_T2']:.2f} < {MIN_GRAD_CORR}")
        meta["exclude"] = reasons
        for f in glob.glob(f"{od}/*_native.nii.gz"):
            os.remove(f)
        json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
        return key, ("EXCLUDED " + "; ".join(reasons)) if reasons else \
            f"ok {len(z)} rows, lesion {meta['les_vox']['T2']} vox, gradcorr {meta['grad_corr_T1_T2']:.2f}"
    except Exception as e:  # noqa: BLE001 - report and continue with the other patients
        meta["exclude"] = [f"FAILED {type(e).__name__}: {str(e)[-300:]}"]
        json.dump(meta, open(f"{od}/failed.json", "w"), indent=1)
        return key, meta["exclude"][0]


def run(raw, workers=3, only=None):
    seq = seq_table(raw)
    ks = sorted(int(os.path.basename(d).split("-")[1]) for d in glob.glob(f"{raw}/Patient-*"))
    jobs = [(k, raw, seq[k]) for k in ks if only is None or k in only]
    print(f"{len(jobs)} patients", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for key, msg in ex.map(process, jobs):
            print(f"  {key}: {msg}", flush=True)


def kept_patients():
    out = []
    for f in sorted(glob.glob(f"{RL}/patients/MS-*/meta.json")):
        m = json.load(open(f))
        if not m["exclude"]:
            out.append(m["key"])
    return out


def pack():
    os.makedirs(NPY_MS, exist_ok=True)
    keys = kept_patients()
    for c in ("T1", "T2"):
        H, L, man, off = [], [], [], 0
        for key in keys:
            a = np.load(f"{RL}/patients/{key}/arrays.npz"); m = json.load(open(f"{RL}/patients/{key}/meta.json"))
            H.append(a[f"hf_{c}"][:, None]); L.append(a[f"les_{c}"][:, None])
            n = len(a[f"hf_{c}"])
            man.append(dict(subject_id=key, split="ms", contrast=c, slice_start=off, slice_count=n,
                            volume_depth=int(m["frame"][1] - m["frame"][0]), z_indices=m["z"],
                            raw=f"Patient-{m['patient']}"))
            off += n
        np.save(f"{NPY_MS}/paired_ms_hf_{c}.npy", np.concatenate(H))
        np.save(f"{NPY_MS}/paired_ms_les_{c}.npy", np.concatenate(L))
        json.dump(dict(contrast=c, split="ms", source="Muslim et al. 2022, Mendeley 10.17632/8bctsm8jz7.1, "
                       "CC BY 4.0; 1.5 T; lesion arrays = consensus masks (T2 mask defines lesions)",
                       total_slices=off, num_subjects=len(man), subjects=man),
                  open(f"{NPY_MS}/manifest_ms_{c}.json", "w"), indent=1)
        print(f"  ms {c}: {len(man)} patients, {off} rows")


def qc():
    import nibabel as nib
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import scipy.ndimage as ndi
    os.makedirs(f"{RL}/qc", exist_ok=True)
    excl, geo = {}, {}
    for d in sorted(glob.glob(f"{RL}/patients/MS-*")):
        key = os.path.basename(d)
        f = f"{d}/meta.json" if os.path.exists(f"{d}/meta.json") else f"{d}/failed.json"
        m = json.load(open(f))
        geo[key] = dict(geometry=m.get("geometry"), sv=m.get("T2_to_T1_singular_values"),
                        grad_corr=m.get("grad_corr_T1_T2"), les_vox=m.get("les_vox"),
                        brain_cm3=m.get("brain_cm3"))
        if m["exclude"]:
            excl[key] = m["exclude"]
        if not os.path.exists(f"{d}/t1.nii.gz"):
            continue
        t1 = nib.load(f"{d}/t1.nii.gz").get_fdata(); t2 = nib.load(f"{d}/t2.nii.gz").get_fdata()
        l2 = nib.load(f"{d}/les_t2.nii.gz").get_fdata() > 0; l1 = nib.load(f"{d}/les_t1.nii.gz").get_fdata() > 0
        zs = [int(np.argmax(l2.sum((0, 1)))) if l2.any() else t1.shape[2] // 2, t1.shape[2] // 2]
        fig, ax = plt.subplots(2, 4, figsize=(14, 7))
        for r, z in enumerate(zs):
            e = np.hypot(ndi.sobel(t2[:, :, z], 0), ndi.sobel(t2[:, :, z], 1))
            ax[r, 0].imshow(np.rot90(t1[:, :, z]), cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"{key} T1 z{z}")
            ax[r, 1].imshow(np.rot90(t2[:, :, z]), cmap="gray", vmin=0, vmax=1); ax[r, 1].set_title("T2 (affine to T1)")
            ax[r, 2].imshow(np.rot90(t1[:, :, z]), cmap="gray", vmin=0, vmax=1)
            ax[r, 2].contour(np.rot90(e > np.percentile(e, 92)), [0.5], colors="c", linewidths=0.4)
            ax[r, 2].set_title(f"T1 + T2 edges  r_grad={m.get('grad_corr_T1_T2', 0):.2f}")
            ax[r, 3].imshow(np.rot90(t2[:, :, z]), cmap="gray", vmin=0, vmax=1)
            ax[r, 3].contour(np.rot90(l2[:, :, z]), [0.5], colors="r", linewidths=0.5)
            ax[r, 3].contour(np.rot90(l1[:, :, z]), [0.5], colors="y", linewidths=0.5)
            ax[r, 3].set_title("T2 + masks (red T2, yellow T1)")
        for a in ax.ravel(): a.axis("off")
        plt.tight_layout(); plt.savefig(f"{RL}/qc/{key}.png", dpi=55); plt.close()
    json.dump(dict(cutoffs=dict(max_scale_dev=MAX_SCALE_DEV, min_grad_corr=MIN_GRAD_CORR),
                   excluded=excl, per_patient=geo), open(f"{RL}/exclusions.json", "w"), indent=1)
    print(f"  {len(excl)} excluded: {excl}")
    gc = [g["grad_corr"] for g in geo.values() if g["grad_corr"] is not None]
    print(f"  gradient correlation T1/T2: median {np.median(gc):.2f}, min {np.min(gc):.2f}")


# ------------------------------------------------------------------ healthy 3T targets
def target_list():
    rng = np.random.default_rng(0)
    out = []
    for split in TARGET_SPLITS:
        sids = sorted(s["subject_id"] for s in json.load(open(f"{NPY}/manifest_{split}_T2.json"))["subjects"])
        if split.startswith("pub3t"):
            sids = sorted(rng.choice(sids, N_PUB3T, replace=False).tolist())
        out += [(split, s) for s in sids]
    return out


def target_src(split, sid):
    if split == "test":
        return f"{ROOT}/datasets/scratch/external/ext_on_hfc/{sid}"
    if split == "ext_v5":
        return f"{ROOT}/datasets/scratch/external/ext_v5/{sid}"
    return f"{ROOT}/datasets/scratch/public/{split}/{sid}"


def build_target(job):
    """Full 1 mm target volumes scaled like their npy rows; rows re-derived and compared."""
    split, sid = job
    import nibabel as nib
    from ulfkit import mri_preprocess as mp
    od = f"{RL}/targets/{sid}"
    if os.path.exists(f"{od}/frame.json"):
        return sid, "cached"
    os.makedirs(od, exist_ok=True)
    src = target_src(split, sid)
    meta = json.load(open(f"{src}/meta.json"))
    z0, z1, cy, cx = meta["frame"]
    frame = mp.FrameSpec(centroid_xy=(cy, cx), z_start=int(z0), z_end=int(z1), mask_z_bbox=(0, 0))
    out, rec = {}, dict(split=split, sid=sid, frame=[int(z0), int(z1), cy, cx], src=src)
    aff = None
    for c in ("T1", "T2"):
        if split.startswith("pub3t"):
            img = mp.resample_to_isotropic(mp.reorient_to_ras(nib.load(meta["paths"][f"{c}W"])), 1.0, order=1)
        else:
            img = nib.load(f"{src}/hf_{c}.nii.gz")
        v = np.clip(np.nan_to_num(np.asarray(img.dataobj, np.float32)), 0, None)
        m = v > 1e-3 * np.percentile(v[v > 0], 99)
        n, lo, hi = scale_like_public(v, m, frame)
        man = json.load(open(f"{NPY}/manifest_{split}_{c}.json"))
        s = next(x for x in man["subjects"] if x["subject_id"] == sid)
        hf = np.load(f"{NPY}/paired_{split}_hf_{c}.npy")[s["slice_start"]:s["slice_start"] + s["slice_count"], 0]
        rows = band_rows(n, frame, s["z_indices"])
        rec[f"maxdiff_{c}"] = float(np.abs(rows - hf.astype(np.float32)).max())
        rec[f"z_{c}"] = s["z_indices"]; rec[f"row0_{c}"] = s["slice_start"]; rec[f"lohi_{c}"] = [lo, hi]
        out[c] = n
        if aff is None:
            aff = img.affine
        else:
            assert np.allclose(aff, img.affine, atol=1e-3) and n.shape == out["T1"].shape, "T1/T2 grids differ"
    assert rec["z_T1"] == rec["z_T2"], "T1/T2 rows differ"
    assert max(rec["maxdiff_T1"], rec["maxdiff_T2"]) < 2e-3, rec
    nib.save(nib.Nifti1Image(out["T1"], aff), f"{od}/t1.nii.gz")
    nib.save(nib.Nifti1Image(out["T2"], aff), f"{od}/t2.nii.gz")
    nib.save(nib.Nifti1Image(((out["T1"] > 0) & (out["T2"] > 0)).astype(np.uint8), aff), f"{od}/brain.nii.gz")
    rec["shape"] = list(out["T1"].shape)
    json.dump(rec, open(f"{od}/frame.json", "w"), indent=1)
    return sid, f"ok rows match npy (max |diff| {max(rec['maxdiff_T1'], rec['maxdiff_T2']):.1e})"


def targets(workers=3):
    tl = target_list()
    print(f"{len(tl)} targets", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for sid, msg in ex.map(build_target, tl):
            print(f"  {sid}: {msg}", flush=True)
    json.dump(tl, open(f"{RL}/targets.json", "w"), indent=1)


def synthseg():
    """mri_synthseg --robust on the 1 mm T1 of every target and kept patient (CPU): three
    processes (targets, two halves of the patients), THREADS threads each; output keeps the
    input grid. Outputs land in synthseg/{targets,patients}_out/<key>_synthseg.nii.gz."""
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, FREESURFER_HOME=FS)
    tg = [f"{RL}/targets/{s}/t1.nii.gz" for _, s in json.load(open(f"{RL}/targets.json"))]
    pt = [f"{RL}/patients/{k}/t1.nii.gz" for k in kept_patients()]
    groups = []
    for g, src, outd in (("targets", tg, "targets_out"), ("patients_a", pt[0::2], "patients_out"),
                         ("patients_b", pt[1::2], "patients_out")):
        ind, outd = f"{RL}/synthseg/{g}_in", f"{RL}/synthseg/{outd}"
        os.makedirs(ind, exist_ok=True); os.makedirs(outd, exist_ok=True)
        for f in src:
            name = f.split("/")[-2] + ".nii.gz"
            if not os.path.exists(f"{outd}/{name[:-7]}_synthseg.nii.gz") and not os.path.lexists(f"{ind}/{name}"):
                os.symlink(f, f"{ind}/{name}")
        for name in os.listdir(ind):                      # done ones are not re-run
            if os.path.exists(f"{outd}/{name[:-7]}_synthseg.nii.gz"):
                os.remove(f"{ind}/{name}")
        if os.listdir(ind):
            groups.append((g, ind, outd))
    def go(t):
        g, ind, outd = t
        cmd = [f"{FS}/bin/mri_synthseg", "--i", ind, "--o", outd, "--robust", "--keepgeom",
               "--threads", str(THREADS), "--cpu"]
        print(f"  synthseg {g}: {len(os.listdir(ind))} volumes", flush=True)
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"  synthseg {g}: exit {r.returncode} {r.stdout[-300:] if r.returncode else ''}", flush=True)
        return r.returncode
    with ThreadPoolExecutor(3) as ex:
        codes = list(ex.map(go, groups))
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "pack", "qc", "targets", "synthseg"])
    ap.add_argument("--raw", default=f"{ROOT}/datasets/raw/ms_muslim/ex")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    os.makedirs(f"{RL}/patients", exist_ok=True); os.makedirs(f"{RL}/targets", exist_ok=True)
    if a.stage == "run": run(a.raw, min(a.workers, 3), a.only)
    elif a.stage == "pack": pack()
    elif a.stage == "qc": qc()
    elif a.stage == "targets": targets(min(a.workers, 3))
    else: synthseg()
