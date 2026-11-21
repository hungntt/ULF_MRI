"""Do public pretrained 64 mT enhancers keep lesions, and where do they land among ours?

Every enhancer in the zoo was trained in this programme. Three released models take a 64 mT
scan without any training on our data, and are run here unchanged (never fine-tuned):
  SuperSynth  (FreeSurfer 8.2 mri_super_synth, weights SuperSynth_August_2025.pth): any
              contrast/resolution -> 1 mm T1w, T2w, FLAIR. Not described as inpainting lesions.
  SynthSR     (mri_synthsr --lowfield, mri_synthsr_hyperfine): -> MPRAGE-like T1. Inpaints
              lesions by design (FreeSurfer wiki), so it is a known-erasure reference.
  GAMBAS      (Baljer et al., MIDL 2025, release v1.0): paediatric 64 mT T2 -> 3T T2, 3D;
              off-domain for adults. Runs in its own venv (datasets/raw/gambas/.venv).

Protocol (3D, as d3synthsr.py). Per locked test subject (61-66 contiguous 1 mm axial slices):
  T2 benchmark  N_VOL lesioned copies of the 3T T2, each with N_LES well-separated 3D Gaussian
                lesions, bright contrasts c in (0.12, 0.25), radii r in (2, 3, 4, 6) vox
                (diameter 2r = 4-12 mm), balanced over sites. Synthetic arm: the learned operator
                D slice-wise on the lesioned and lesion-free 3T. Acquired arm (d3detreal): the
                acquired 64 mT T2 plus D's local lesion response. Every T2-output model, the 36
                zoo models (slice-wise, 2.5D slabs from the volume) and SuperSynth/GAMBAS see the
                same lesions at the same sites.
  T1 table      d3synthsr.py's lesion volumes (read only): r = 4, |c| = 0.25, the four T1/T2
                sign patterns of d3lesiontype.py, both contrasts through D. SynthSR (3 variants),
                SuperSynth T1 from the 64 mT T1 and from the 64 mT T2, and routes A-C slice-wise.
Retention R = (core-minus-ring change of the output) / (that of the 3T), core b > 0.6, ring
0.05 < b < 0.25 (d3detect.py). A released model has its own intensity scale, so its output is
put in 3T units by the least-squares line y_3T ~ a*p + b fitted on its lesion-free output over
the brain (the map an MSE-trained enhancer learns; a zoo model has a ~ 1), and R uses a*dp.
d3synthsr.py's convention (divide by the slope of p on y) is kept as R_k. Erased: R < 0.25.
Inverted: the output change has the opposite sign of the 3T change (scale-free).
On the ACQUIRED lesion-free inputs: SSIM/PSNR vs 3T after the same linear map (ulfkit.metrics,
brain mask), and SynthSeg Dice against SynthSeg on the 3T (d3synthseg.py conventions).

  make      (GPU) simulated and acquired 64 mT volumes -> {ULF_SCRATCH}/pretrained/in
  ssynth    (GPU) SuperSynth, official preprocessing, float outputs (the CLI writes uint8)
  ss_check  (CPU) the official CLI on one volume, to show the float wrapper reproduces it
  synthsr   (CPU) SynthSR variants: rerun on d3synthsr's inputs (sanity), acquired inputs
  zoo       (GPU) the 36 zoo models and routes A-C on the same 3D volumes
  gambas_prep / gambas_run (venv) / gambas_back   GAMBAS in template space
  synthseg  (CPU) SynthSeg on every model's output on the acquired input
  score     -> out_public/d3pretrained.json (+ _rows, _zoo_rows)
  fig       -> fig_public/pretrained.pdf

    ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_SCRATCH=datasets/scratch_public \
    ULF_FIG=fig_public python -I exp/d3pretrained.py <stage>
"""
import glob, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, ".")

ROOT = "."
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch")
FIG = os.environ.get("ULF_FIG", f"{ROOT}/fig")
SPLIT = os.environ.get("ULF_SPLIT", "test")
SFX = "" if SPLIT == "test" else f"_{SPLIT}"
WORK = f"{SCR}/pretrained" + SFX
SRIN = f"{SCR}/synthsr/in"                 # d3synthsr.py's T1-table volumes (test), read only
SROUT = f"{SCR}/synthsr/out"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
SS_W = f"{ROOT}/datasets/raw/supersynth/SuperSynth_August_2025.pth"
GB = f"{ROOT}/datasets/raw/gambas"
RADII, CS = (2, 3, 4, 6), (0.12, 0.25)
N_VOL, N_LES, MIN_SEP, ZPAD = 4, 12, 28, 8
QUADS = [("t1d_t2b", -1, +1), ("t1b_t2d", +1, -1), ("t1b_t2b", +1, +1), ("t1d_t2d", -1, -1)]
C_T1 = 0.25                                # d3synthsr.py: |c| for every T1-table lesion
THR = 0.25


def to_ras(v):
    """(slice,row,col) -> (row, col, slice), the d3synthsr/d3synthseg NIfTI layout."""
    return np.transpose(v, (1, 2, 0))


def from_ras(v):
    return np.transpose(v, (2, 0, 1))


def save(v, path, aff=None):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.asarray(v, np.float32), np.eye(4) if aff is None else aff), path)


def load(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def fs_env():
    return dict(os.environ, FREESURFER_HOME=FS, FS_LICENSE=f"{FS}/license.txt",
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4")


# ----------------------------------------------------------------------------- lesions
def sites(y, rng, n=N_LES):
    """Interior T2 sites (eroded brain, 0.12 < y < 0.45 as d3detect.site_pool), >= MIN_SEP apart,
    >= ZPAD slices from the slab ends so the r = 6 ring (1.22 r) stays inside."""
    from scipy.ndimage import binary_erosion
    ok = binary_erosion(y > 0, iterations=6) & (y > 0.12) & (y < 0.45)
    ok[:, :, :ZPAD] = ok[:, :, -ZPAD:] = False
    cand = np.argwhere(ok)
    rng.shuffle(cand)
    picked = []
    for c in cand[:40000]:
        if all(np.linalg.norm(c - p) >= MIN_SEP for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return np.array(picked)


def lesion_maps(shape, cen, rr, cc, mask):
    """-> profile b (max over lesions), label map (b_k > 0.05), signed contrast map sum c_k b_k,
    and the acquired-arm neighbourhood (d3detreal.lesion_input: radius 1.22 max(4r, 24))."""
    zz = np.indices(shape, dtype=np.float32)
    b = np.zeros(shape, np.float32); cm = np.zeros(shape, np.float32)
    lab = np.zeros(shape, np.int16); near = np.zeros(shape, bool)
    for k, (c, r, cv) in enumerate(zip(cen, rr, cc)):
        d2 = sum((zz[i] - c[i]) ** 2 for i in range(3))
        bk = np.exp(-2.0 * d2 / float(r) ** 2)
        b = np.maximum(b, bk); cm += cv * bk
        lab[bk > 0.05] = k + 1
        near |= d2 <= (1.2238 * max(4 * r, 24)) ** 2
    return b * mask, lab, cm * mask, near


def region_labels(lab, b):
    return np.where(b > 0.6, lab, 0).astype(np.int32), \
        np.where((b > 0.05) & (b < 0.25), lab, 0).astype(np.int32)


def contrasts(d, core, ring, n):
    """core-minus-ring mean of d for lesions 1..n, and the voxel counts."""
    nc = np.bincount(core.ravel(), minlength=n + 1)[1:]
    nr = np.bincount(ring.ravel(), minlength=n + 1)[1:]
    sc = np.bincount(core.ravel(), weights=d.ravel().astype(np.float64), minlength=n + 1)[1:]
    sr = np.bincount(ring.ravel(), weights=d.ravel().astype(np.float64), minlength=n + 1)[1:]
    return sc / np.maximum(nc, 1) - sr / np.maximum(nr, 1), nc, nr


def lsq(y, p, m):
    """y ~ a p + b on mask m -> (a, b); and d3synthsr's slope k of p on y."""
    a, b = np.polyfit(p[m].astype(np.float64), y[m].astype(np.float64), 1)
    k = np.polyfit(y[m].astype(np.float64), p[m].astype(np.float64), 1)[0]
    return float(a), float(b), float(k)


def subjects():
    return sorted(os.listdir(f"{WORK}/in")) if os.path.isdir(f"{WORK}/in") else []


# ----------------------------------------------------------------------------- make
def make():
    import torch
    from ulfkit import data as D
    from exp.d3detect import operator
    from exp.d3synthsr import to_native
    mc = D.MultiContrastCohort(SPLIT)
    f1, f2 = operator("T1", "learned"), operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    rng = np.random.default_rng(7)

    def degrade(f, vol):                                   # RAS volume -> RAS volume, slice-wise
        v, out = from_ras(vol), []
        for i in range(0, len(v), 16):
            y = torch.from_numpy(np.ascontiguousarray(v[i:i + 16, None])).float().cuda()
            with torch.no_grad():
                out.append(f(y, gen)[:, 0].float().cpu().numpy())
        return to_ras(np.concatenate(out))

    sr_meta = json.load(open(f"{SRIN}/../meta.json")) if SPLIT == "test" else {}
    meta, check = {}, {}
    for sid in sorted(set(mc.sid)):
        pos = np.where(mc.sid == sid)[0]
        ulf, hf = mc.batch(pos)
        y1, y2 = to_ras(hf[:, 0].astype(np.float32)), to_ras(hf[:, 1].astype(np.float32))
        a1 = to_ras(ulf[:, 0].astype(np.float32)) * (y1 > 0)
        a2 = to_ras(ulf[:, 1].astype(np.float32)) * (y2 > 0)
        d = f"{WORK}/in/{sid}"; os.makedirs(d, exist_ok=True)
        save(y1, f"{d}/3t_t1.nii.gz"); save(y2, f"{d}/3t_t2.nii.gz")
        save(a1, f"{d}/acq_t1.nii.gz"); save(a2, f"{d}/acq_t2.nii.gz")
        x2 = degrade(f2, y2) * (y2 > 0)
        save(x2, f"{d}/sim_t2_clean.nii.gz")
        meta[sid] = []
        for v in range(N_VOL):
            cen = sites(y2, rng)
            rr = [RADII[k % 4] for k in range(len(cen))]
            cc = [CS[(k // 4 + v) % 2] for k in range(len(cen))]
            b, lab, cm, near = lesion_maps(y2.shape, cen, rr, cc, (y2 > 0))
            y2l = np.clip(y2 + cm, 0, 1)
            x2l = degrade(f2, y2l) * (y2 > 0)
            save(x2l, f"{d}/sim_t2_v{v}.nii.gz")
            save(a2 + (x2l - x2) * near, f"{d}/acq_t2_v{v}.nii.gz")
            np.savez_compressed(f"{d}/les_v{v}.npz", b=b, lab=lab, cm=cm, cen=cen,
                                r=np.array(rr), c=np.array(cc))
            meta[sid].append(dict(n=len(cen), r=rr, c=cc, centres=cen.tolist()))
        if sid in sr_meta:                                 # T1 table, acquired arm
            s = f"{SRIN}/{sid}"
            prof = load(f"{s}/lesion_profile.nii.gz")
            x1 = degrade(f1, y1) * (y1 > 0)
            check[sid] = dict(
                t1_vs_d3synthsr=float(np.abs(x1 - load(f"{s}/ulf_t1_clean.nii.gz")).max()),
                t2_vs_d3synthsr=float(np.abs(x2 - load(f"{s}/ulf_t2_clean.nii.gz")).max()),
                y1_vs_d3synthsr=float(np.abs(y1 - load(f"{s}/3t_t1_clean.nii.gz")).max()))
            cen = np.array(sr_meta[sid]["centres"])
            _, _, _, near = lesion_maps(y1.shape, cen, [4.0] * len(cen), [0.0] * len(cen), y1 > 0)
            for q, s1, s2 in QUADS:
                x1l = degrade(f1, np.clip(y1 + s1 * C_T1 * prof, 0, 1)) * (y1 > 0)
                x2l = degrade(f2, np.clip(y2 + s2 * C_T1 * prof, 0, 1)) * (y2 > 0)
                save(a1 + (x1l - x1) * near, f"{d}/acq_t1_{q}.nii.gz")
                save(a2 + (x2l - x2) * near, f"{d}/acq_t2_{q}.nii.gz")
        for f in glob.glob(f"{d}/*.nii.gz"):               # Hyperfine-native 1.5 x 1.5 x 5 mm
            n = os.path.basename(f)[:-7]
            if n.startswith(("sim_", "acq_")) and not n.endswith("_native"):
                to_native(f, f"{d}/{n}_native.nii.gz")
        print(f"  {sid}: {[m['n'] for m in meta[sid]]} lesions, check {check.get(sid)}", flush=True)
    json.dump(dict(meta=meta, check_vs_d3synthsr=check), open(f"{WORK}/meta.json", "w"), indent=1)


# ----------------------------------------------------------------------------- SuperSynth
def ss_load(dev="cuda"):
    sys.path.insert(0, f"{FS}/python/packages/SuperSynth")
    from SuperSynth.frugal_models import frugal_models
    import torch
    assert torch.__version__ >= "2.6", "torch.load must default to weights_only=True"
    return frugal_models(SS_W, f"{FS}/python/packages/SuperSynth/atlas/atlas.qc_seg.nii.gz", dev)


def ss_infer(nets, path, dev="cuda"):
    """scripts/inference.py of SuperSynth up to the synthesis outputs, with test-time flipping
    (the CLI passes --test_time_flipping); returns 100*T1, 100*T2 as float (the CLI writes them
    clipped to uint8), the output affine and the input max used for normalisation."""
    import torch
    from SuperSynth.utils import MRIread, torch_resize, align_volume_to_ref
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        im, aff = MRIread(path, im_only=False, dtype="float")
        im = torch.tensor(np.squeeze(im), dtype=torch.float32, device=dev)
        while len(im.shape) > 3:
            im = im.mean(dim=-1)
        im[im.isnan()] = 0
        im, aff = torch_resize(im, aff, 1.0, dev)
        im, aff = align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        imax = float(im.max()); im /= im.max()
        W = (np.ceil(np.array(im.shape) / 16.0) * 16).astype("int")
        i0 = np.floor((W - im.shape) / 2).astype("int")
        sl = tuple(slice(i0[k], i0[k] + im.shape[k]) for k in range(3))
        S = torch.zeros(*W, dtype=torch.float32, device=dev)
        S[sl] = im
        pred = nets.ssynth_inference(S[None, None])
        T1, T2 = pred["T1"][0, 0][sl], pred["T2"][0, 0][sl]
        del pred
        S = torch.flip(S, [0])
        pred = nets.ssynth_inference(S[None, None])
        T1 = 0.5 * T1 + 0.5 * torch.flip(pred["T1"][0, 0], [0])[sl]
        T2 = 0.5 * T2 + 0.5 * torch.flip(pred["T2"][0, 0], [0])[sl]
        out = ((100 * T1).float().cpu().numpy(), (100 * T2).float().cpu().numpy(), aff, imax)
        del pred, S, im
        torch.cuda.empty_cache()
    return out


def ss_jobs():
    jobs = []
    for sid in subjects():
        d, o = f"{WORK}/in/{sid}", f"{WORK}/out/{sid}"
        for f in sorted(glob.glob(f"{d}/*.nii.gz")):
            n = os.path.basename(f)[:-7]
            if n.startswith(("sim_", "acq_")):
                jobs.append((f, f"{o}/ss_{n}"))
        if SPLIT == "test":
            for f in sorted(glob.glob(f"{SRIN}/{sid}/ulf_t*.nii.gz")):
                jobs.append((f, f"{o}/ss_{os.path.basename(f)[:-7]}"))
    return jobs


def ssynth():
    nets = ss_load()
    log_p = f"{WORK}/ss_imax.json"
    log = json.load(open(log_p)) if os.path.exists(log_p) else {}
    todo = [(f, o) for f, o in ss_jobs() if not os.path.exists(f"{o}_T2.nii.gz")]
    print(f"  {len(todo)} SuperSynth runs", flush=True)
    for i, (f, o) in enumerate(todo):
        os.makedirs(os.path.dirname(o), exist_ok=True)
        t1, t2, aff, imax = ss_infer(nets, f)
        for c, v in (("T1", t1), ("T2", t2)):         # atomic: a killed job never leaves a partial file
            save(v, f"{o}_{c}.tmp.nii.gz", aff)
            os.replace(f"{o}_{c}.tmp.nii.gz", f"{o}_{c}.nii.gz")
        log[o] = imax
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} {o}", flush=True)
            json.dump(log, open(log_p, "w"))
    json.dump(log, open(log_p, "w"))


def ss_check():
    """The official CLI (fspython, CPU) on one input vs the float wrapper's output."""
    sid = subjects()[0]
    f = f"{WORK}/in/{sid}/acq_t2_native.nii.gz"
    o = f"{WORK}/ss_check/{sid}"
    os.makedirs(o, exist_ok=True)
    if not os.path.exists(f"{o}/SynthT2.mgz"):
        cmd = [f"{FS}/bin/fspython", f"{FS}/python/packages/SuperSynth/scripts/inference.py",
               "--i", f, "--o", o, "--mode", "invivo", "--device", "cpu", "--threads", "4",
               "--model_file", SS_W, "--test_time_flipping"]
        subprocess.run(cmd, env=fs_env(), check=True, stdout=subprocess.DEVNULL)
    import nibabel as nib
    res = {}
    for c in ("T1", "T2"):
        ref = np.asarray(nib.load(f"{o}/Synth{c}.mgz").dataobj, np.float32)
        # the CLI casts (100 * pred).clip(0, 255) to uint8, i.e. truncates
        mine = np.floor(np.clip(load(f"{WORK}/out/{sid}/ss_acq_t2_native_{c}.nii.gz"), 0, 255))
        d = np.abs(ref - mine)
        m = ref > 0
        res[c] = dict(max_abs=float(d.max()), mean_abs_brain=float(d[m].mean()),
                      frac_gt1=float((d[m] > 1.01).mean()),
                      corr=float(np.corrcoef(ref[m], mine[m])[0, 1]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/d3pretrained_sscheck{SFX}.json", "w"), indent=1)


# ----------------------------------------------------------------------------- SynthSR
def synthsr(workers=3, threads=4):
    """mri_synthsr --lowfield (T1 alone, T2 alone) and mri_synthsr_hyperfine (T1 + T2):
    (a) rerun on d3synthsr.py's native inputs, the reproduction check; (b) the acquired lesion-free
    inputs (native and 1 mm); (c) the acquired arm of the T1 table (native)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = []

    def add(t1, t2, o, x):
        os.makedirs(o, exist_ok=True)
        for tag, cmd in (("sr_t1", ["mri_synthsr", "--lowfield", "--i", t1]),
                         ("sr_t2", ["mri_synthsr", "--lowfield", "--i", t2]),
                         ("sr_hf", ["mri_synthsr_hyperfine", "--t1", t1, "--t2", t2])):
            dst = f"{o}/{tag}_{x}.nii.gz"
            if not os.path.exists(dst):
                jobs.append([f"{FS}/bin/{cmd[0]}"] + cmd[1:] + ["--o", dst, "--threads", str(threads), "--cpu"])

    for sid in subjects():
        d, o = f"{WORK}/in/{sid}", f"{WORK}/out/{sid}"
        if SPLIT == "test" and os.path.isdir(f"{SRIN}/{sid}"):
            for q in ["clean"] + [q for q, _, _ in QUADS]:
                add(f"{SRIN}/{sid}/ulf_t1_{q}_native.nii.gz", f"{SRIN}/{sid}/ulf_t2_{q}_native.nii.gz",
                    f"{WORK}/sr_rerun/{sid}", f"{q}_native")
        for suf in ("_native", ""):
            add(f"{d}/acq_t1{suf}.nii.gz", f"{d}/acq_t2{suf}.nii.gz", o, f"acq{suf}")
        for q, _, _ in QUADS:
            if os.path.exists(f"{d}/acq_t1_{q}_native.nii.gz"):
                add(f"{d}/acq_t1_{q}_native.nii.gz", f"{d}/acq_t2_{q}_native.nii.gz", o, f"acq_{q}_native")
    print(f"  {len(jobs)} SynthSR runs", flush=True)
    env = fs_env()

    def go(cmd):
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            print("  FAILED", " ".join(cmd[-6:]), r.stderr[-300:], flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))


# ----------------------------------------------------------------------------- zoo + routes
def enhance(net, vol, mode="slab", pack_fn=None, bs=16):
    """RAS volume (or (vol1, vol2) for the T1 routes) -> RAS output, slice-wise.
    mode 'slab': 2.5D input from neighbouring slices (edge replicated), as the enhancers were
    trained; 'repeat': the centre slice three times, as d3arch.py's closed loop."""
    import torch

    def slabs(v):
        s = from_ras(v)
        if mode == "repeat":
            return np.repeat(s[:, None], 3, 1)
        p = np.concatenate([s[:1], s, s[-1:]])
        return np.stack([p[:-2], p[1:-1], p[2:]], 1)
    xs = [slabs(v) for v in (vol if isinstance(vol, tuple) else (vol,))]
    out = []
    for i in range(0, len(xs[0]), bs):
        t = [torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).float().cuda() for x in xs]
        inp = pack_fn(*t) if pack_fn else t[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net(inp).float().clamp(0, 1)[:, 0].cpu().numpy())
    return to_ras(np.concatenate(out))


def t2_lesions(sid, y):
    """-> per lesioned volume: dict(core, ring, n, r, c, ty, nc, nr) for the T2 benchmark."""
    L = []
    for v in range(N_VOL):
        z = np.load(f"{WORK}/in/{sid}/les_v{v}.npz")
        core, ring = region_labels(z["lab"], z["b"])
        n = int(z["lab"].max())
        ty, nc, nr = contrasts(np.clip(y + z["cm"], 0, 1) - y, core, ring, n)
        L.append(dict(core=core, ring=ring, n=n, r=z["r"], c=z["c"], ty=ty, nc=nc, nr=nr))
    return L


def t1_lesions(sid):
    """d3synthsr.py's T1 table: per quadrant, the 3T T1 change; labels and profile shared."""
    s = f"{SRIN}/{sid}"
    lab, b = load(f"{s}/lesion_labels.nii.gz").astype(np.int32), load(f"{s}/lesion_profile.nii.gz")
    core, ring = region_labels(lab, b)
    n = int(lab.max())
    y1 = load(f"{s}/3t_t1_clean.nii.gz")
    out = dict(core=core, ring=ring, n=n, y1=y1)
    for q, _, _ in QUADS:
        out[q] = contrasts(load(f"{s}/3t_t1_{q}.nii.gz") - y1, core, ring, n)
    return out


def zoo():
    import torch
    from ulfkit import data as D, metrics as M
    from exp import d3zoo as Z
    from exp.d3pipe import d2_model, pack
    from exp.d3qa import enhancer
    subs = subjects()
    V = {}
    for sid in subs:
        d = f"{WORK}/in/{sid}"
        y = load(f"{d}/3t_t2.nii.gz")
        V[sid] = dict(y=y, L=t2_lesions(sid, y),
                      sim=[load(f"{d}/sim_t2_clean.nii.gz")] + [load(f"{d}/sim_t2_v{v}.nii.gz") for v in range(N_VOL)],
                      acq=[load(f"{d}/acq_t2.nii.gz")] + [load(f"{d}/acq_t2_v{v}.nii.gz") for v in range(N_VOL)])
    coh = D.PairedCohort(SPLIT, "T2")
    rows, qual = [], {}

    def t2_rows(name, fam, arm, mode, sid, P, extra):
        y, m = V[sid]["y"], V[sid]["y"] > 0
        a, b_, k = lsq(y, P[0], m)
        for v, L in enumerate(V[sid]["L"]):
            tp, _, _ = contrasts(P[v + 1] - P[0], L["core"], L["ring"], L["n"])
            for j in range(L["n"]):
                if L["nc"][j] < 4 or L["nr"][j] < 8 or abs(L["ty"][j]) < 0.25 * L["c"][j]:
                    continue
                rows.append(dict(model=name, family=fam, arm=arm, mode=mode, sid=sid, v=v, les=j + 1,
                                 r_vox=int(L["r"][j]), diam_mm=2 * int(L["r"][j]), c=float(L["c"][j]),
                                 ty=float(L["ty"][j]), tp=float(tp[j]), a=a, k=k, **extra))

    reg = [r for r in Z.registry() if r[4] == "diff" or os.path.exists(f"{OUT}/{r[3]}.pt")]
    for nm, enh, seed, ck, kind in reg:
        net = Z.load(kind, ck)
        qual[nm] = dict(enhancer=enh, seed=seed, sids=[], ssim=[], psnr=[], ssim_cal=[], psnr_cal=[], a=[])
        for sid in subs:
            for arm, mode in (("sim", "slab"), ("sim", "repeat"), ("acq", "slab")):
                P = [enhance(net, x, mode) for x in V[sid][arm]]
                t2_rows(nm, "zoo", arm, mode, sid, P, dict(enhancer=enh, seed=seed))
            # acquired lesion-free input, the true 2.5D slabs: raw (= d3zoo) and calibrated scores
            idx = coh.subject_rows(sid)
            x = coh.ulf[idx].astype(np.float32)
            p = []
            for i in range(0, len(idx), 16):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p.append(net(torch.from_numpy(x[i:i + 16]).cuda()).float().clamp(0, 1)[:, 0].cpu().numpy())
            p = np.concatenate(p)
            q = quality(p, coh.hf[idx][:, 0].astype(np.float32), x[:, 1])
            qual[nm]["sids"].append(sid)
            for kk in ("ssim", "psnr", "ssim_cal", "psnr_cal", "a"):
                qual[nm][kk].append(q[kk])
        print(f"  {nm:30s} SSIM {np.mean(qual[nm]['ssim']):.3f} (cal {np.mean(qual[nm]['ssim_cal']):.3f})", flush=True)
        del net; torch.cuda.empty_cache()
    json.dump(dict(rows=rows, quality=qual), open(f"{OUT}/d3pretrained_zoo_rows{SFX}.json", "w"))

    if SPLIT != "test":
        return
    # routes A-C (d3lesiontype.py) on d3synthsr.py's 3D lesions and on the acquired inputs
    eA = enhancer(); eA.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_T1.pt")); eA.eval()
    routes = [("A_T1in", eA, "t1", None), ("B_T2in", d2_model("T2toT1"), "t2", lambda x2: pack(x2=x2)),
              ("C_both", d2_model("bothtoT1"), "both", pack)] + \
             [(f"C_{t}", d2_model(t), "both", pack) for t in ("bothtoT1_ctl", "bothtoT1_aug", "bothtoT1_augw")]
    rrows, rqual = [], {}
    c1 = D.PairedCohort(SPLIT, "T1")
    for name, net, inp, pk in routes:
        rqual[name] = dict(sids=[], ssim=[], psnr=[], ssim_cal=[], psnr_cal=[], a=[])
        for sid in subs:
            L = t1_lesions(sid)
            s, d = f"{SRIN}/{sid}", f"{WORK}/in/{sid}"
            for arm in ("sim", "acq"):
                def vols(q):
                    if arm == "sim":
                        v1, v2 = load(f"{s}/ulf_t1_{q}.nii.gz"), load(f"{s}/ulf_t2_{q}.nii.gz")
                    else:
                        v1 = load(f"{d}/acq_t1.nii.gz" if q == "clean" else f"{d}/acq_t1_{q}.nii.gz")
                        v2 = load(f"{d}/acq_t2.nii.gz" if q == "clean" else f"{d}/acq_t2_{q}.nii.gz")
                    return v1 if inp == "t1" else (v2 if inp == "t2" else (v1, v2))
                P0 = enhance(net, vols("clean"), pack_fn=pk)
                a, b_, k = lsq(L["y1"], P0, L["y1"] > 0)
                for q, _, _ in QUADS:
                    tp, _, _ = contrasts(enhance(net, vols(q), pack_fn=pk) - P0, L["core"], L["ring"], L["n"])
                    ty, nc, nr = L[q]
                    for j in range(L["n"]):
                        if nc[j] < 8 or nr[j] < 16 or abs(ty[j]) < 0.25 * C_T1:
                            continue
                        rrows.append(dict(model=name, family="route", arm=arm, sid=sid, quad=q, les=j + 1,
                                          ty=float(ty[j]), tp=float(tp[j]), a=a, k=k))
            # acquired lesion-free: quality vs 3T T1 and the output for SynthSeg
            idx = c1.subject_rows(sid)
            x1 = c1.ulf[idx].astype(np.float32); x2 = coh.ulf[coh.subject_rows(sid)].astype(np.float32)
            p = []
            for i in range(0, len(idx), 16):
                t1_, t2_ = torch.from_numpy(x1[i:i + 16]).cuda(), torch.from_numpy(x2[i:i + 16]).cuda()
                inp_t = t1_ if inp == "t1" else (pk(t2_) if inp == "t2" else pk(t1_, t2_))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p.append(net(inp_t).float().clamp(0, 1)[:, 0].cpu().numpy())
            p = np.concatenate(p)
            hf1 = c1.hf[idx][:, 0].astype(np.float32)
            q = quality(p, hf1, x1[:, 1])
            rqual[name]["sids"].append(sid)
            for kk in ("ssim", "psnr", "ssim_cal", "psnr_cal", "a"):
                rqual[name][kk].append(q[kk])
            os.makedirs(f"{WORK}/out/{sid}", exist_ok=True)
            save(to_ras(p * (hf1 > 0)), f"{WORK}/out/{sid}/route_{name}_acq.nii.gz")
        print(f"  route {name}: SSIM {np.mean(rqual[name]['ssim']):.3f}", flush=True)
    json.dump(dict(rows=rrows, quality=rqual), open(f"{OUT}/d3pretrained_route_rows{SFX}.json", "w"))


def quality(p, hf, ulf):
    """Brain-masked SSIM/PSNR (ulfkit.metrics.volume_metrics) of p (N,H,W) as is, and after the
    least-squares line to the 3T on the brain (the calibration given to the released models)."""
    import torch
    from ulfkit import metrics as M
    m = (hf > 0) | (ulf > 0)
    a, b, _ = lsq(hf, p, m)
    t = lambda v: torch.from_numpy(np.ascontiguousarray(v[:, None])).float().cuda()
    r = M.volume_metrics(t(np.clip(p, 0, 1)), t(hf), t(ulf))
    c = M.volume_metrics(t(np.clip(a * p + b, 0, 1)), t(hf), t(ulf))
    return dict(ssim=r["ssim"], psnr=r["psnr"], ssim_cal=c["ssim"], psnr_cal=c["psnr"], a=a, b=b)


# ----------------------------------------------------------------------------- released models
def t2_stem(arm, v):
    if v is None:
        return "sim_t2_clean" if arm == "sim" else "acq_t2"
    return f"{arm}_t2_v{v}"


def t1_stem(arm, inp, q):
    """inp 't1' / 't2': the 64 mT input contrast; q None = lesion-free."""
    if arm == "sim":
        return f"ulf_{inp}_{q or 'clean'}"
    return f"acq_{inp}" if q is None else f"acq_{inp}_{q}"


def t2_models():
    """T2-output released models: name -> (path(sid, arm, v), sharpen)."""
    o = lambda sid: f"{WORK}/out/{sid}"
    m = {"SuperSynth T2": (lambda sid, arm, v: f"{o(sid)}/ss_{t2_stem(arm, v)}_native_T2.nii.gz", False),
         "SuperSynth T2 (1 mm input)": (lambda sid, arm, v: f"{o(sid)}/ss_{t2_stem(arm, v)}_T2.nii.gz", False),
         "SuperSynth T2, sharpened": (lambda sid, arm, v: f"{o(sid)}/ss_{t2_stem(arm, v)}_native_T2.nii.gz", True),
         "GAMBAS": (lambda sid, arm, v: f"{WORK}/gambas/back/{sid}/{t2_stem(arm, v)}.nii.gz", False),
         "64 mT input, GAMBAS registration round trip":
             (lambda sid, arm, v: f"{WORK}/gambas/roundtrip/{sid}/{t2_stem(arm, v)}.nii.gz", False)}
    return m


def t1_models():
    """T1-output released models: name -> path(sid, arm, q). Native 1.5 x 1.5 x 5 mm inputs
    (primary, as d3synthsr.py); '(1 mm input)' rows use the 1 mm inputs, synthetic arm only."""
    o = lambda sid: f"{WORK}/out/{sid}"
    def sr(tag, geom="native"):
        def f(sid, arm, q):
            x = (q or "clean") if arm == "sim" else ("acq" if q is None else f"acq_{q}")
            if arm == "sim":
                return f"{WORK}/sr_rerun/{sid}/{tag}_{x}_native.nii.gz" if geom == "native" \
                    else f"{SROUT}/{sid}/{tag}_{x}.nii.gz"
            return f"{o(sid)}/{tag}_{x}_native.nii.gz"
        return f
    def ss(inp, geom="native"):
        g = "_native" if geom == "native" else ""
        return lambda sid, arm, q: f"{o(sid)}/ss_{t1_stem(arm, inp, q)}{g}_T1.nii.gz"
    return {"SuperSynth T1 (T1 in)": ss("t1"), "SuperSynth T1 (T2 in)": ss("t2"),
            "SynthSR low-field (T1 in)": sr("sr_t1"), "SynthSR low-field (T2 in)": sr("sr_t2"),
            "SynthSR Hyperfine (T1+T2 in)": sr("sr_hf"),
            "SuperSynth T1 (T1 in, 1 mm input)": ss("t1", "1mm"), "SuperSynth T1 (T2 in, 1 mm input)": ss("t2", "1mm"),
            "SynthSR low-field (T1 in, 1 mm input)": sr("sr_t1", "1mm"),
            "SynthSR low-field (T2 in, 1 mm input)": sr("sr_t2", "1mm"),
            "SynthSR Hyperfine (T1+T2 in, 1 mm input)": sr("sr_hf", "1mm")}


_REF = {}


def on_ref(path, shape):
    """A released model's output, trilinearly resampled onto the 1 mm slab grid (identity affine)."""
    import nibabel as nib
    from exp.d3synthsr import resample_to
    if shape not in _REF:
        _REF[shape] = nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4))
    return resample_to(nib.load(path), _REF[shape]).astype(np.float32)


def usm(p):
    """SuperSynth's --sharpen_synths: unsharp masking, amount 0.75, sigma 1.5 vox (linear)."""
    from scipy.ndimage import gaussian_filter
    return p + 0.75 * (p - gaussian_filter(p, 1.5))


def score_released():
    """Per-lesion rows for the released models (T2 benchmark and T1 table), and their quality on
    the acquired lesion-free inputs."""
    rows, qual = [], {}
    for name, (pf, sharp) in t2_models().items():
        for sid in subjects():
            if not os.path.exists(pf(sid, "acq", N_VOL - 1)):
                continue
            y = load(f"{WORK}/in/{sid}/3t_t2.nii.gz"); m = y > 0
            L = t2_lesions(sid, y)
            fam = "reference" if "round trip" in name else "released"
            cover = None                           # GAMBAS: lesions inside the template's field of view
            if "GAMBAS" in name or "round trip" in name:
                rt = load(f"{WORK}/gambas/roundtrip/{sid}/sim_t2_clean.nii.gz") > 0
                cover = [contrasts(rt.astype(np.float32), Lv["core"], np.zeros_like(Lv["ring"]), Lv["n"])[0]
                         for Lv in L]
            for arm in ("sim", "acq"):
                rd = lambda v: (usm if sharp else (lambda z: z))(on_ref(pf(sid, arm, v), y.shape))
                P0 = rd(None)
                a, b, k = lsq(y, P0, m)
                for v in range(N_VOL):
                    tp, _, _ = contrasts(rd(v) - P0, L[v]["core"], L[v]["ring"], L[v]["n"])
                    for j in range(L[v]["n"]):
                        if L[v]["nc"][j] < 4 or L[v]["nr"][j] < 8 or abs(L[v]["ty"][j]) < 0.25 * L[v]["c"][j]:
                            continue
                        if cover is not None and cover[v][j] < 1:
                            continue
                        rows.append(dict(model=name, family=fam, bench="T2", arm=arm, sid=sid, v=v,
                                         les=j + 1, r_vox=int(L[v]["r"][j]), diam_mm=2 * int(L[v]["r"][j]),
                                         c=float(L[v]["c"][j]), ty=float(L[v]["ty"][j]), tp=float(tp[j]), a=a, k=k))
                if arm == "acq":
                    q = quality(from_ras(P0), from_ras(y), from_ras(load(f"{WORK}/in/{sid}/acq_t2.nii.gz")))
                    qual.setdefault(name, dict(bench="T2", sids=[], ssim_cal=[], psnr_cal=[], a=[], b=[]))
                    for kk in ("ssim_cal", "psnr_cal", "a", "b"):
                        qual[name][kk].append(q[kk])
                    qual[name]["sids"].append(sid)
        print(f"  {name}: {sum(r['model'] == name for r in rows)} lesion rows", flush=True)
    # the 64 mT input itself (no enhancer): R = R_D in 3D, a reference row
    for sid in subjects():
        y = load(f"{WORK}/in/{sid}/3t_t2.nii.gz"); L = t2_lesions(sid, y)
        for arm in ("sim", "acq"):
            P0 = load(f"{WORK}/in/{sid}/{t2_stem(arm, None)}.nii.gz")
            a, b, k = lsq(y, P0, y > 0)
            for v in range(N_VOL):
                tp, _, _ = contrasts(load(f"{WORK}/in/{sid}/{t2_stem(arm, v)}.nii.gz") - P0,
                                     L[v]["core"], L[v]["ring"], L[v]["n"])
                for j in range(L[v]["n"]):
                    if L[v]["nc"][j] < 4 or L[v]["nr"][j] < 8 or abs(L[v]["ty"][j]) < 0.25 * L[v]["c"][j]:
                        continue
                    rows.append(dict(model="64 mT input (no enhancer)", family="reference", bench="T2", arm=arm,
                                     sid=sid, v=v, les=j + 1, r_vox=int(L[v]["r"][j]), diam_mm=2 * int(L[v]["r"][j]),
                                     c=float(L[v]["c"][j]), ty=float(L[v]["ty"][j]), tp=float(tp[j]), a=a, k=k))
    if SPLIT == "test":
        for name, pf in t1_models().items():
            for sid in subjects():
                if not os.path.exists(pf(sid, "sim", QUADS[-1][0])):
                    continue
                L = t1_lesions(sid)
                y1 = load(f"{WORK}/in/{sid}/3t_t1.nii.gz"); m = y1 > 0
                arms = ("sim",) if "1 mm" in name else ("sim", "acq")
                for arm in arms:
                    if not os.path.exists(pf(sid, arm, QUADS[-1][0])):
                        continue
                    P0 = on_ref(pf(sid, arm, None), y1.shape)
                    a, b, k = lsq(y1, P0, m)
                    for q, _, _ in QUADS:
                        tp, _, _ = contrasts(on_ref(pf(sid, arm, q), y1.shape) - P0, L["core"], L["ring"], L["n"])
                        ty, nc, nr = L[q]
                        for j in range(L["n"]):
                            if nc[j] < 8 or nr[j] < 16 or abs(ty[j]) < 0.25 * C_T1:
                                continue
                            rows.append(dict(model=name, family="released", bench="T1", arm=arm, sid=sid, quad=q,
                                             les=j + 1, ty=float(ty[j]), tp=float(tp[j]), a=a, k=k))
                    if arm == "acq":
                        q_ = quality(from_ras(P0), from_ras(y1), from_ras(load(f"{WORK}/in/{sid}/acq_t1.nii.gz")))
                        qual.setdefault(name, dict(bench="T1", sids=[], ssim_cal=[], psnr_cal=[], a=[], b=[]))
                        for kk in ("ssim_cal", "psnr_cal", "a", "b"):
                            qual[name][kk].append(q_[kk])
                        qual[name]["sids"].append(sid)
            print(f"  {name}: {sum(r['model'] == name for r in rows)} lesion rows", flush=True)
        # d3synthsr.py's own outputs through this scorer: the reproduction check
        for tag in ("sr_t1", "sr_t2", "sr_hf"):
            for geom, sfx in (("native", "_native"), ("1mm", "")):
                for sid in subjects():
                    L = t1_lesions(sid); y1 = L["y1"]
                    P0 = on_ref(f"{SROUT}/{sid}/{tag}_clean{sfx}.nii.gz", y1.shape)
                    a, b, k = lsq(y1, P0, y1 > 0)
                    for q, _, _ in QUADS:
                        tp, _, _ = contrasts(on_ref(f"{SROUT}/{sid}/{tag}_{q}{sfx}.nii.gz", y1.shape) - P0,
                                             L["core"], L["ring"], L["n"])
                        ty, nc, nr = L[q]
                        for j in range(L["n"]):
                            if nc[j] < 8 or nr[j] < 16 or abs(ty[j]) < 0.25 * C_T1:
                                continue
                            rows.append(dict(model=f"d3synthsr {tag} {geom}", family="check", bench="T1", arm="sim",
                                             sid=sid, quad=q, les=j + 1, ty=float(ty[j]), tp=float(tp[j]), a=a, k=k))
    json.dump(dict(rows=rows, quality=qual), open(f"{OUT}/d3pretrained_rows{SFX}.json", "w"))
    return rows, qual


# ----------------------------------------------------------------------------- GAMBAS
# GAMBAS expects a whole head rigidly registered to its Template.nii.gz (144 x 256 x 256, 1 mm).
# Our slabs are 61-66 axial slices of a framed, band-normalised volume (prep_external.py), so the
# slab is put back into its subject's full-FOV volume: the framing and the intensity line of
# prep_external.process are rebuilt (and checked against the stored arrays), the slab rows are
# replaced by this study's inputs, the volume is registered rigidly to the template (SynthMorph),
# GAMBAS runs there, and its output is mapped back and cut to the slab. Outside the slab the
# acquired arm keeps the acquired 64 mT, the synthetic arm D applied to the 3T, so both arms are
# seamless. The two resamplings blur a lesion, so the 64 mT input sent through the same round
# trip is scored as a reference row.
GDIR = f"{WORK}/gambas"


def ext_dir(sid):
    return f"{ROOT}/datasets/scratch/external/{'ext_on_hfc' if SPLIT == 'test' else SPLIT}/{sid}"


def full_frame(sid):
    """-> {hf_T2, lf_T2: framed full-depth volumes on prep_external's intensity line}, the affine of
    the framed grid, the full-volume z of every slab row, and the max error vs arrays.npz."""
    import nibabel as nib
    from ulfkit import mri_preprocess as mp
    e = ext_dir(sid)
    vols, masks, imgs = {}, {}, {}
    for k in ("hf_T1", "hf_T2", "lf_T2"):
        img = nib.load(f"{e}/{k}.nii.gz")
        v = np.asarray(img.dataobj, dtype=np.float32)
        v[~np.isfinite(v)] = 0
        v = np.clip(v, 0, None)
        vols[k], masks[k], imgs[k] = v, v > 1e-3 * np.percentile(v[v > 0], 99), img
    _, _, frame = mp.frame_and_normalize(vols["hf_T1"], masks["hf_T1"])
    out = {}
    for k in ("hf_T2", "lf_T2"):
        band, _, _ = mp.frame_and_normalize(vols[k], masks[k], frame=frame)
        vf = mp.pad_crop_inplane_about_center(vols[k], frame.centroid_xy, frame.target_inplane)
        mf = mp.pad_crop_inplane_about_center(masks[k].astype(np.float32), frame.centroid_xy,
                                              frame.target_inplane) > 0.5
        z0, z1 = frame.z_start, min(frame.z_end, vf.shape[2])
        bm = mf[:, :, z0:z1]
        if not bm.any():
            bm = mp.soft_foreground_mask(vf[:, :, z0:z1])
        lo, hi = np.percentile(vf[:, :, z0:z1][bm], list(mp.DEFAULT_PERCENTILES))
        full = ((np.clip(vf, lo, hi) - lo) / (hi - lo)).astype(np.float32)
        assert np.abs(full[:, :, z0:z1] - band).max() < 1e-5
        out[k] = full
    meta = json.load(open(f"{e}/meta.json"))
    zs = [frame.z_start + z for z in meta["z_T2"]]
    arr = np.load(f"{e}/arrays.npz")
    err = dict(hf=float(np.abs(out["hf_T2"][:, :, zs] - to_ras(arr["hf_T2"][:, 0].astype(np.float32))).max()),
               lf=float(np.abs((out["lf_T2"] * (out["lf_T2"] > 0))[:, :, zs]
                               - to_ras(arr["ulf_T2"][:, 1].astype(np.float32))).max()))
    cy, cx = frame.centroid_xy
    th, tw = frame.target_inplane
    y0, x0 = int(round(cy - (th - 1) / 2.0)), int(round(cx - (tw - 1) / 2.0))
    aff = imgs["hf_T2"].affine.copy()
    aff[:3, 3] = aff[:3, :3] @ np.array([y0, x0, 0.0]) + aff[:3, 3]
    return out, aff, zs, err


def gambas_prep():
    """Full-FOV inputs for both arms, rigid SynthMorph to the template, inputs moved to it."""
    import torch
    from exp.d3detect import operator
    f2 = operator("T2", "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    tpl = f"{GB}/repo/Template.nii.gz"
    info = {}
    for sid in subjects():
        d, o = f"{WORK}/in/{sid}", f"{GDIR}/{sid}"
        os.makedirs(f"{o}/full", exist_ok=True); os.makedirs(f"{o}/tpl", exist_ok=True)
        vol, aff, zs, err = full_frame(sid)
        y = load(f"{d}/3t_t2.nii.gz")
        assert np.abs(vol["hf_T2"][:, :, zs] - y).max() < 2e-3, "slab is not the framed volume"
        M = vol["hf_T2"] > 0
        v = from_ras(vol["hf_T2"]); dd = []
        for i in range(0, len(v), 16):
            with torch.no_grad():
                dd.append(f2(torch.from_numpy(np.ascontiguousarray(v[i:i + 16, None])).float().cuda(), gen)[:, 0]
                          .float().cpu().numpy())
        base = {"sim": to_ras(np.concatenate(dd)) * M, "acq": vol["lf_T2"] * (vol["lf_T2"] > 0) * M}
        stems = [t2_stem(a, v_) for a in ("sim", "acq") for v_ in [None] + list(range(N_VOL))]
        for s in stems:
            g = base[s[:3]].copy()
            g[:, :, zs] = load(f"{d}/{s}.nii.gz")
            save(g, f"{o}/full/{s}.nii.gz", aff)
        cmd = [f"{FS}/bin/mri_synthmorph", "register", "-m", "rigid", "-j", "4", "-t", f"{o}/fwd.lta",
               "-T", f"{o}/inv.lta", "-o", f"{o}/acq_moved.nii.gz", f"{o}/full/acq_t2.nii.gz", tpl]
        if not os.path.exists(f"{o}/inv.lta"):
            subprocess.run(cmd, env=fs_env(), check=True, stdout=subprocess.DEVNULL)
        args = []
        for s in stems:
            args += [f"{o}/full/{s}.nii.gz", f"{o}/tpl/{s}.nii.gz"]
        subprocess.run([f"{FS}/bin/mri_synthmorph", "apply", f"{o}/fwd.lta"] + args, env=fs_env(), check=True,
                       stdout=subprocess.DEVNULL)
        info[sid] = dict(slab_vs_arrays_max_err=err, z_full=[zs[0], zs[-1]], full_shape=list(vol["hf_T2"].shape),
                         sim_slab_vs_D_full=float(np.abs(base["sim"][:, :, zs] - load(f"{d}/sim_t2_clean.nii.gz")).max()))
        print(f"  {sid}: {info[sid]}", flush=True)
    json.dump(info, open(f"{GDIR}/prep.json", "w"), indent=1)
    gambas_qc()
    open(f"{GDIR}/prep_done", "w").write("ok\n")


def gambas_qc():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = load(f"{GB}/repo/Template.nii.gz")
    subs = subjects()
    fig, ax = plt.subplots(len(subs) + 1, 3, figsize=(7, 2.4 * (len(subs) + 1)))
    for i, (nm, v) in enumerate([("template", t)] + [(s, load(f"{GDIR}/{s}/tpl/acq_t2.nii.gz")) for s in subs]):
        c = np.array(v.shape) // 2
        for j, sl in enumerate((v[c[0]], v[:, c[1]], v[:, :, c[2]])):
            ax[i, j].imshow(np.rot90(sl), cmap="gray"); ax[i, j].axis("off")
        ax[i, 0].set_title(nm, fontsize=7, loc="left")
    plt.tight_layout(); os.makedirs(FIG, exist_ok=True)
    plt.savefig(f"{FIG}/pretrained_gambas_qc{SFX}.pdf"); plt.close()


def gambas_net():
    """GAMBAS generator from the cloned repo: only models/path_generate.py and
    models/mamba_modules3D.py are executed, loaded by explicit path; weights with weights_only."""
    import importlib.util, types, torch
    pkg = types.ModuleType("gambas_models"); pkg.__path__ = [f"{GB}/repo/models"]
    sys.modules["gambas_models"] = pkg
    for m in ("path_generate", "mamba_modules3D"):
        spec = importlib.util.spec_from_file_location(f"gambas_models.{m}", f"{GB}/repo/models/{m}.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    net = sys.modules["gambas_models.mamba_modules3D"].GAMBAS(input_dim=1, img_size=(128, 128, 128), output_dim=1)
    sd = torch.load(f"{GB}/weights/latest_net_G.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    return net.cuda().eval()


def gambas_infer(net, path, out_path, P=128, stride=32):
    """test.py inference(): sitk z-score then 0-255 rescale, 128^3 patches at stride 32 scaled to
    [-1, 1], overlap-averaged, +0.01; written on the input's grid."""
    import SimpleITK as sitk, torch
    img = sitk.ReadImage(path)
    img = sitk.NormalizeImageFilter().Execute(img)
    rs = sitk.RescaleIntensityImageFilter(); rs.SetOutputMaximum(255); rs.SetOutputMinimum(0)
    img = sitk.Cast(rs.Execute(img), sitk.sitkFloat32)
    a = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    assert min(a.shape) >= P and a.shape[2] % 2 == 0, a.shape
    lab = np.zeros(a.shape, np.float32); w = np.zeros(a.shape, np.float32)
    starts = [sorted({min(i * stride, n - P) for i in range(int(np.ceil((n - P) / stride)) + 1)}) for n in a.shape]
    for i in starts[0]:
        for j in starts[1]:
            for k in starts[2]:
                x = torch.from_numpy(((a[i:i + P, j:j + P, k:k + P] - 127.5) / 127.5)[None, None]).float().cuda()
                with torch.no_grad():
                    p = net(x).squeeze().float().cpu().numpy()
                lab[i:i + P, j:j + P, k:k + P] += p * 127.5 + 127.5
                w[i:i + P, j:j + P, k:k + P] += 1
    lab = lab / w + 0.01
    o = sitk.GetImageFromArray(np.transpose(lab, (2, 1, 0)))
    o.SetOrigin(img.GetOrigin()); o.SetDirection(img.GetDirection()); o.SetSpacing(img.GetSpacing())
    sitk.WriteImage(o, out_path)


def gambas_run():
    """Runs in datasets/raw/gambas/.venv (torch 2.2.2 + mamba_ssm 2.2.2 wheels)."""
    import time
    net = gambas_net()
    for sid in subjects():
        os.makedirs(f"{GDIR}/{sid}/tpl_out", exist_ok=True)
        for f in sorted(glob.glob(f"{GDIR}/{sid}/tpl/*.nii.gz")):
            o = f.replace("/tpl/", "/tpl_out/")
            if os.path.exists(o):
                continue
            t0 = time.time()
            gambas_infer(net, f, o)
            print(f"  {o} {time.time() - t0:.0f}s", flush=True)
    open(f"{GDIR}/run_done", "w").write("ok\n")


def gambas_back():
    """Template-space outputs (and, as the round-trip reference, the moved inputs) back to the
    framed full volume with the inverse rigid transform, then cut to the slab rows."""
    import nibabel as nib
    for sid in subjects():
        o = f"{GDIR}/{sid}"
        zs = None
        for src, dst in (("tpl_out", f"{GDIR}/back/{sid}"), ("tpl", f"{GDIR}/roundtrip/{sid}")):
            os.makedirs(f"{o}/{src}_full", exist_ok=True); os.makedirs(dst, exist_ok=True)
            args = []
            for f in sorted(glob.glob(f"{o}/{src}/*.nii.gz")):
                args += [f, f"{o}/{src}_full/{os.path.basename(f)}"]
            subprocess.run([f"{FS}/bin/mri_synthmorph", "apply", f"{o}/inv.lta"] + args, env=fs_env(), check=True,
                           stdout=subprocess.DEVNULL)
            if zs is None:
                _, aff, zs, _ = full_frame(sid)
            for f in sorted(glob.glob(f"{o}/{src}_full/*.nii.gz")):
                img = nib.load(f)
                assert np.allclose(img.affine, aff, atol=1e-3), (img.affine, aff)
                save(np.asarray(img.dataobj, np.float32)[:, :, zs], f"{dst}/{os.path.basename(f)}")
        print(f"  {sid}", flush=True)
    open(f"{GDIR}/back_done", "w").write("ok\n")


# ----------------------------------------------------------------------------- SynthSeg
SEG_T2 ={"SuperSynth T2": "SuperSynth_T2", "SuperSynth T2 (1 mm input)": "SuperSynth_T2_1mm",
          "GAMBAS": "GAMBAS"}
SEG_T1 = {"SuperSynth T1 (T1 in)": "SuperSynth_T1_T1in", "SuperSynth T1 (T2 in)": "SuperSynth_T1_T2in",
          "SynthSR low-field (T1 in)": "SynthSR_lowfield_T1in", "SynthSR low-field (T2 in)": "SynthSR_lowfield_T2in",
          "SynthSR Hyperfine (T1+T2 in)": "SynthSR_hyperfine",
          "A_T1in": "route_A_T1in", "B_T2in": "route_B_T2in", "C_both": "route_C_both"}


def synthseg_prepare():
    """Every model's output on the ACQUIRED lesion-free input, in 3T units (released models: the
    least-squares line), clamped to [0, 1] and masked by the 3T brain as d3zoo.export_nii:
      synthseg/in/<model>/<sid>.nii.gz  (row, col, slice), the d3synthseg.py layout
      flr/<T2|T1>/<model>/<sid>.nii.gz  (slice, row, col), the d3zoo zoo_nii layout, for d3spec.py"""
    t2m, t1m = t2_models(), t1_models()
    for sid in subjects():
        d = f"{WORK}/in/{sid}"
        for bench, ymap, names in (("T2", "3t_t2", SEG_T2), ("T1", "3t_t1", SEG_T1)):
            if bench == "T1" and SPLIT != "test":
                continue
            y = load(f"{d}/{ymap}.nii.gz"); m = y > 0
            if bench == "T1":
                for nm, v in (("_3T_T1_reference", y), ("_64mT_T1_input", load(f"{d}/acq_t1.nii.gz") * m)):
                    os.makedirs(f"{WORK}/synthseg/in/{nm}", exist_ok=True)
                    save(v, f"{WORK}/synthseg/in/{nm}/{sid}.nii.gz")
            for name, folder in names.items():
                if name in t2m:
                    f = t2m[name][0](sid, "acq", None)
                elif name in t1m:
                    f = t1m[name](sid, "acq", None)
                else:
                    f = f"{WORK}/out/{sid}/route_{name}_acq.nii.gz"
                if not os.path.exists(f):
                    continue
                p = on_ref(f, y.shape)
                if not name.startswith(("A_", "B_", "C_")):
                    a, b, _ = lsq(y, p, m | (load(f"{d}/acq_{bench.lower()}.nii.gz") > 0))
                    p = a * p + b
                p = np.clip(p, 0, 1) * m
                for dd, arr in ((f"{WORK}/synthseg/in/{folder}", p), (f"{WORK}/flr/{bench}/{folder}", from_ras(p))):
                    os.makedirs(dd, exist_ok=True)
                    save(arr, f"{dd}/{sid}.nii.gz")


def synthseg(workers=3, threads=4, final="1"):
    workers, threads = int(workers), int(threads)
    from concurrent.futures import ThreadPoolExecutor
    synthseg_prepare()
    todo = []
    for d in sorted(glob.glob(f"{WORK}/synthseg/in/*")):
        o = d.replace("/in/", "/seg/")
        if os.path.exists(o) and len(os.listdir(o)) >= len(os.listdir(d)):
            continue
        os.makedirs(o, exist_ok=True)
        todo.append((d, o))
    print(f"  {len(todo)} SynthSeg folders", flush=True)
    env = fs_env()

    def go(t):
        d, o = t
        cmd = [f"{FS}/bin/mri_synthseg", "--i", d, "--o", o, "--robust", "--threads", str(threads), "--cpu"]
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print("  ", os.path.basename(d), "rc", r.returncode, r.stderr[-200:] if r.returncode else "", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, todo))
    if final:
        open(f"{WORK}/synthseg/seg_done", "w").write("ok\n")


def dice_scores():
    """-> {model: [Dice per subject]} for T2 (vs SynthSeg on the 3T T2, d3synthseg's reference)
    and T1 (vs SynthSeg on the 3T T1), left/right merged, labels present in the reference."""
    from exp.d3synthseg import merged
    out = {"T2": {}, "T1": {}}
    ref2 = f"{SCR}/synthseg{SFX}/seg/_3T_reference"
    ref1 = f"{WORK}/synthseg/seg/_3T_T1_reference"
    for bench, names, refd in (("T2", SEG_T2, ref2), ("T1", dict(SEG_T1, **{"64 mT T1 input": "_64mT_T1_input"}), ref1)):
        for name, folder in names.items():
            d = f"{WORK}/synthseg/seg/{folder}"
            if not os.path.isdir(d):
                continue
            dice = []
            for sid in subjects():
                f, r = glob.glob(f"{d}/{sid}*.nii.gz"), glob.glob(f"{refd}/{sid}*.nii.gz")
                if not f or not r:
                    break
                s, rr = merged(load(f[0])), merged(load(r[0]))
                labs = [l for l in np.unique(rr) if l != 0]
                dice.append(float(np.mean([2 * ((s == l) & (rr == l)).sum() / ((s == l).sum() + (rr == l).sum())
                                           for l in labs])))
            if len(dice) == len(subjects()):
                out[bench][name] = dice
    return out


# ----------------------------------------------------------------------------- summary
def ret_of(r, conv="a"):
    """Retention of one lesion row: released models in 3T units via the least-squares line (a),
    zoo/routes/the 64 mT input as is (they already output 3T units); conv='k': d3synthsr's slope."""
    if conv == "k":
        return r["tp"] / r["k"] / r["ty"]
    return (r["a"] if r["family"] in ("released",) else 1.0) * r["tp"] / r["ty"]


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def per_subject(R, sids, conv="a"):
    """-> per-subject mean clamped retention (d3arch convention), erased and inverted fractions."""
    out = dict(ret=[], erased=[], inverted=[], ret_raw_median=[])
    for s in sids:
        x = [r for r in R if r["sid"] == s]
        rv = np.array([ret_of(r, conv) for r in x])
        out["ret"].append(float(np.clip(rv, -0.5, 1.5).mean()))
        out["erased"].append(float((rv < THR).mean()))
        out["inverted"].append(float((rv < 0).mean()))
        out["ret_raw_median"].append(float(np.median(rv)))
    return out


def describe(R, sids, conv="a", by=None):
    ps = per_subject(R, sids, conv)
    d = dict(n=len(R), sids=sids, **ps)
    for k in ("ret", "erased", "inverted"):
        d[f"{k}_mean"] = float(np.mean(ps[k])); d[f"{k}_ci"] = boot(ps[k])
    d["ret_median"] = float(np.median([ret_of(r, conv) for r in R]))
    if by:
        d["by_diam_mm"] = {}
        for g in sorted({r[by] for r in R}):
            Rg = [r for r in R if r[by] == g]
            pg = per_subject(Rg, sids, conv)
            d["by_diam_mm"][int(g)] = dict(n=len(Rg), ret_mean=float(np.mean(pg["ret"])), ret_ci=boot(pg["ret"]),
                                           erased_mean=float(np.mean(pg["erased"])), erased_ci=boot(pg["erased"]))
    return d


def score():
    from exp import d3zoo_stats as Zs
    rows, qual = score_released()
    zz = json.load(open(f"{OUT}/d3pretrained_zoo_rows{SFX}.json"))
    zrows, zqual = zz["rows"], zz["quality"]
    dice = dice_scores()
    sids = subjects()
    res = dict(split=SPLIT, sids=sids, protocol=dict(
        radii_vox=list(RADII), diam_mm=[2 * r for r in RADII], contrasts=list(CS), n_vol=N_VOL, n_les=N_LES,
        min_sep_vox=MIN_SEP, t1_table=dict(r_vox=4, diam_mm=8, c=C_T1, source="d3synthsr.py inputs"),
        retention="released: a*tp/ty with y_3T ~ a p + b on the lesion-free output; zoo: tp/ty; clamp [-0.5,1.5]"))
    meta = json.load(open(f"{WORK}/meta.json"))
    res["sanity"] = dict(operator_inputs_vs_d3synthsr=meta.get("check_vs_d3synthsr"))
    if os.path.exists(f"{OUT}/d3pretrained_sscheck{SFX}.json"):
        res["sanity"]["supersynth_cli_vs_wrapper"] = json.load(open(f"{OUT}/d3pretrained_sscheck{SFX}.json"))
    stored = json.load(open(Zs_path := (f"{OUT}/d3zoo_T2.json" if SPLIT == "test" else f"{OUT}/d3zoo_{SPLIT}_T2.json")))
    ssd = {}
    for nm, q in zqual.items():
        if nm in stored:
            ssd[nm] = float(np.max(np.abs(np.array(q["ssim"]) - np.array([stored[nm]["ssim"][stored[nm]["sids"].index(s)]
                                                                          for s in q["sids"]]))))
    res["sanity"]["zoo_ssim_vs_d3zoo_max_abs"] = max(ssd.values()) if ssd else None
    if os.path.exists(f"{WORK}/ss_imax.json"):           # SuperSynth divides by the input maximum
        im = json.load(open(f"{WORK}/ss_imax.json"))
        rel = []
        for o, v in im.items():
            n = os.path.basename(o)
            for pat, clean in (("sim_t2_v", "sim_t2_clean"), ("acq_t2_v", "acq_t2")):
                if n.startswith(f"ss_{pat}"):
                    c = o.replace(n, f"ss_{clean}" + ("_native" if n.endswith("_native") else ""))
                    if c in im:
                        rel.append(abs(v / im[c] - 1))
        res["sanity"]["supersynth_input_max_change_by_lesions"] = dict(n=len(rel), max_rel=float(max(rel)) if rel else None)

    # ---- T2 benchmark: per model, arm and mode
    t2 = {}
    sel = lambda R, **kw: [r for r in R if all(r.get(k) == v for k, v in kw.items())]
    zoo_names = sorted({r["model"] for r in zrows})
    for nm in zoo_names:
        R = sel(zrows, model=nm)
        t2[nm] = dict(family="zoo", enhancer=R[0]["enhancer"], seed=R[0]["seed"],
                      sim=describe(sel(R, arm="sim", mode="slab"), sids, by="diam_mm"),
                      sim_repeat=describe(sel(R, arm="sim", mode="repeat"), sids),
                      acq=describe(sel(R, arm="acq", mode="slab"), sids),
                      a_mean=float(np.mean([r["a"] for r in R])),
                      quality={k: zqual[nm][k] for k in ("ssim", "psnr", "ssim_cal", "psnr_cal", "a")})
    for nm in list(t2_models()) + ["64 mT input (no enhancer)"]:
        R = sel(rows, model=nm, bench="T2")
        if not R:
            continue
        fam = R[0]["family"]
        t2[nm] = dict(family=fam, sim=describe(sel(R, arm="sim"), sids, by="diam_mm"),
                      acq=describe(sel(R, arm="acq"), sids),
                      sim_k=describe(sel(R, arm="sim"), sids, conv="k"),
                      a_mean=float(np.mean([r["a"] for r in R if r["arm"] == "sim"])))
        if nm in qual:
            t2[nm]["quality"] = {k: qual[nm][k] for k in ("ssim_cal", "psnr_cal", "a", "b")}
        if nm in dice["T2"]:
            t2[nm]["dice"] = dice["T2"][nm]
    res["t2_models"] = t2

    # ---- ranking: zoo enhancers (seed-averaged) with and without the released T2 models
    ssj = f"{OUT}/d3zoo_synthseg_T2.json" if SPLIT == "test" else f"{OUT}/d3zoo_synthseg_{SPLIT}_T2.json"
    zs = json.load(open(ssj))["runs"] if os.path.exists(ssj) else {}
    z = {}
    for nm in zoo_names:
        e = t2[nm]
        z[nm] = dict(enhancer=e["enhancer"], seed=e["seed"], sids=sids, ret=e["sim"]["ret"], erased=e["sim"]["erased"],
                     ret_acq=e["acq"]["ret"], erased_acq=e["acq"]["erased"],
                     ssim=e["quality"]["ssim_cal"], psnr=e["quality"]["psnr_cal"],
                     ssim_raw=e["quality"]["ssim"], psnr_raw=e["quality"]["psnr"])
        if nm in zs:
            z[nm]["synthseg"] = zs[nm]["dice"]
    rel = [nm for nm in ("SuperSynth T2", "GAMBAS") if nm in t2 and "quality" in t2[nm]]
    zr = dict(z)
    for nm in rel:
        e = t2[nm]
        zr[nm] = dict(enhancer=nm, seed=0, sids=sids, ret=e["sim"]["ret"], erased=e["sim"]["erased"],
                      ret_acq=e["acq"]["ret"], erased_acq=e["acq"]["erased"],
                      ssim=e["quality"]["ssim_cal"], psnr=e["quality"]["psnr_cal"],
                      ssim_raw=e["quality"]["ssim_cal"], psnr_raw=e["quality"]["psnr_cal"])
        if "dice" in e:
            zr[nm]["synthseg"] = e["dice"]
    Zs.B = 5000
    rank = {}
    for tag, zd in (("zoo_only", z), ("with_released", zr)):
        keys = [k for k in ("ssim", "psnr", "synthseg") if all(k in v for v in zd.values())]
        st = {f"{k}~{t}": Zs.boot_rho(zd, k, t, "enhancer") for k in keys for t in ("ret", "erased")}
        st.update({f"{k}~ret_acq": Zs.boot_rho(zd, k, "ret_acq", "enhancer") for k in keys})
        st["selection"] = Zs.selection(zd, keys)
        st["n_enhancers"] = len({v["enhancer"] for v in zd.values()})
        # enhancer-level means and ranks (1 = best)
        names = sorted({v["enhancer"] for v in zd.values()})
        means = {e: {k: float(np.mean([np.mean(v[k]) for v in zd.values() if v["enhancer"] == e and k in v]))
                     for k in ("ret", "erased", "ret_acq", "ssim", "psnr", "ssim_raw") + (("synthseg",) if "synthseg" in keys else ())}
                 for e in names}
        ranks = {}
        for k, desc in (("ret", True), ("erased", False), ("ret_acq", True), ("ssim", True), ("psnr", True),
                        ("synthseg", True)):
            if k not in means[names[0]]:
                continue
            order = sorted(names, key=lambda e: -means[e][k] if desc else means[e][k])
            for e in names:
                ranks.setdefault(e, {})[k] = order.index(e) + 1
        st["means"], st["ranks"] = means, ranks
        rank[tag] = st
    # the 3D protocol vs the stored 2D zoo (d3zoo.py) ranking, zoo enhancers only
    from scipy.stats import spearmanr
    e2d = {}
    for nm, v in stored.items():
        e2d.setdefault(v["enhancer"], []).append(np.mean(v["ret"]))
    common = sorted(set(e2d) & set(rank["zoo_only"]["means"]))
    rank["zoo_3d_vs_2d_spearman"] = float(spearmanr([np.mean(e2d[e]) for e in common],
                                                    [rank["zoo_only"]["means"][e]["ret"] for e in common]).statistic)
    em = {e: np.mean([t2[n]["sim_repeat"]["ret_mean"] for n in zoo_names if t2[n]["enhancer"] == e]) for e in common}
    rank["zoo_3d_slab_vs_repeat_spearman"] = float(spearmanr([em[e] for e in common],
                                                             [rank["zoo_only"]["means"][e]["ret"] for e in common]).statistic)
    rank["zoo_sim_vs_acq_spearman"] = float(spearmanr([rank["zoo_only"]["means"][e]["ret"] for e in common],
                                                      [rank["zoo_only"]["means"][e]["ret_acq"] for e in common]).statistic)
    res["ranking"] = rank

    # ---- T1 table
    if SPLIT == "test":
        t1 = {}
        rq = json.load(open(f"{OUT}/d3pretrained_route_rows.json"))
        allrows = rows + [dict(r, bench="T1") for r in rq["rows"]]
        names = list(t1_models()) + [r for r in ("A_T1in", "B_T2in", "C_both", "C_bothtoT1_ctl", "C_bothtoT1_aug",
                                                 "C_bothtoT1_augw")] + \
            [f"d3synthsr {t} {g}" for t in ("sr_t1", "sr_t2", "sr_hf") for g in ("native", "1mm")]
        for nm in names:
            R = [r for r in allrows if r["model"] == nm and r.get("bench") == "T1"]
            if not R:
                continue
            fam = R[0]["family"]
            conv = "k" if fam == "check" else "a"
            t1[nm] = dict(family=fam)
            for arm in ("sim", "acq"):
                Ra = [r for r in R if r["arm"] == arm]
                if not Ra:
                    continue
                t1[nm][arm] = {}
                for q, _, _ in QUADS:
                    Rq = [r for r in Ra if r["quad"] == q]
                    d = describe(Rq, sids, conv)
                    d["ret_median_k"] = float(np.median([ret_of(r, "k") for r in Rq]))
                    t1[nm][arm][q] = d
            if nm in qual:
                t1[nm]["quality"] = {k: qual[nm][k] for k in ("ssim_cal", "psnr_cal", "a")}
            if nm in rq["quality"]:
                t1[nm]["quality"] = {k: rq["quality"][nm][k] for k in ("ssim", "psnr", "ssim_cal", "psnr_cal", "a")}
            if nm in dice["T1"]:
                t1[nm]["dice"] = dice["T1"][nm]
        if "64 mT T1 input" in dice["T1"]:
            t1["64 mT T1 input"] = dict(family="reference", dice=dice["T1"]["64 mT T1 input"])
        t1["2D d3lesiontype (reused)"] = json.load(open(f"{OUT}/d3lesiontype.json"))
        # reproduction check against out_public/d3synthsr{,_native}.json (d3synthsr.score's statistics)
        chk = {}
        for geom, f in (("native", "d3synthsr_native.json"), ("1mm", "d3synthsr.json")):
            ref = json.load(open(f"{OUT}/{f}"))
            for tag in ("sr_t1", "sr_t2", "sr_hf"):
                for q, _, _ in QUADS:
                    R = [r for r in rows if r["model"] == f"d3synthsr {tag} {geom}" and r["quad"] == q]
                    inv = np.mean([np.mean([r["tp"] * r["ty"] < 0 for r in R if r["sid"] == s]) for s in sids])
                    med = float(np.median([ret_of(r, "k") for r in R]))
                    o = ref[f"{tag}/{q}"]
                    chk[f"{geom}/{tag}/{q}"] = dict(inverted=float(inv), stored=o["inverted"], n=len(R), n_stored=o["n"],
                                                    ret=med, ret_stored=o["ret"])
            if geom == "native":                            # the rerun (sr_rerun) through the same scorer
                for tag, nm in (("sr_t1", "SynthSR low-field (T1 in)"), ("sr_t2", "SynthSR low-field (T2 in)"),
                                ("sr_hf", "SynthSR Hyperfine (T1+T2 in)")):
                    for q, _, _ in QUADS:
                        R = [r for r in rows if r["model"] == nm and r["arm"] == "sim" and r["quad"] == q]
                        inv = np.mean([np.mean([r["tp"] * r["ty"] < 0 for r in R if r["sid"] == s]) for s in sids])
                        chk[f"rerun/{tag}/{q}"] = dict(inverted=float(inv), stored=ref[f"{tag}/{q}"]["inverted"])
        ok = all(abs(v["inverted"] - v["stored"]) < 1e-9 for v in chk.values()) and \
            all(abs(v["ret"] - v["ret_stored"]) < 1e-6 for v in chk.values() if "ret" in v)
        res["sanity"]["synthsr_reproduction"] = dict(PASS=bool(ok), detail=chk)
        res["t1_table"] = t1
    json.dump(res, open(f"{OUT}/d3pretrained{SFX}.json", "w"), indent=1, default=float)
    report(res)


def report(res):
    print(f"\n== {res['split']}: {len(res['sids'])} subjects")
    print("  sanity:", {k: (v if not isinstance(v, dict) or "PASS" not in v else v["PASS"])
                        for k, v in res["sanity"].items()})
    print(f"\n  T2 benchmark, synthetic arm (3D lesions, {N_VOL} x {N_LES} per subject), mean over subjects [95% CI]")
    print(f"  {'model':34s} {'R':>20s} {'erased %':>20s} {'R acq':>8s} {'a':>6s}")
    for nm, d in sorted(res["t2_models"].items(), key=lambda kv: -kv[1]["sim"]["ret_mean"]):
        s = d["sim"]
        print(f"  {nm:34s} {s['ret_mean']:6.3f} [{s['ret_ci'][0]:.3f},{s['ret_ci'][1]:.3f}] "
              f"{s['erased_mean']*100:6.1f} [{s['erased_ci'][0]*100:.0f},{s['erased_ci'][1]*100:.0f}] "
              f"{d['acq']['ret_mean']:8.3f} {d.get('a_mean', 1):6.2f}")
    for tag in ("zoo_only", "with_released"):
        st = res["ranking"][tag]
        print(f"\n  ranking ({tag}, {st['n_enhancers']} enhancers): Spearman rho [CI over enhancers]")
        for k, v in st.items():
            if "~" in k:
                print(f"    {k:22s} {v['rho']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]")
        print("    selection:", {k: (v["pick"], v["ret_rank"]) for k, v in st["selection"].items()})
        for e, r in st["ranks"].items():
            if e in ("SuperSynth T2", "GAMBAS"):
                print(f"    rank of {e}: {r}")
    if "t1_table" in res:
        print("\n  T1 table: % wrong sign (synthetic arm | acquired arm)")
        for nm, d in res["t1_table"].items():
            if "sim" not in d:
                continue
            cells = "  ".join(f"{q}: {d['sim'][q]['inverted_mean']*100:3.0f}"
                              + (f"|{d['acq'][q]['inverted_mean']*100:3.0f}" if "acq" in d else "")
                              for q, _, _ in QUADS)
            print(f"    {nm:42s} {cells}")


def fig():
    """(a, b) 3D retention vs calibrated SSIM and SynthSeg Dice: zoo enhancers (seed mean +- sd),
    released models (diamonds, labelled), the 64 mT input; (c) T1 wrong-sign rate per sign pattern."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(f"{OUT}/d3pretrained{SFX}.json"))
    st = res["ranking"]["with_released"]
    t2 = res["t2_models"]
    ZOO, REL, REF, INK = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b"})
    has_t1 = "t1_table" in res
    fig, ax = plt.subplots(1, 3 if has_t1 else 2, figsize=(7.2 if has_t1 else 5.0, 2.5),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.25] if has_t1 else [1, 1]))
    for i, (k, lab) in enumerate((("ssim", "SSIM, acquired input (calibrated)"),
                                  ("synthseg", "SynthSeg Dice, acquired input"))):
        a = ax[i]
        for e, m in st["means"].items():
            if k not in m:
                continue
            runs = [n for n, d in t2.items() if d.get("family") == "zoo" and d["enhancer"] == e]
            if runs:
                rv = [t2[n]["sim"]["ret_mean"] for n in runs]
                kv = [np.mean(t2[n]["quality"]["ssim_cal"]) for n in runs] if k == "ssim" else None
                a.errorbar(m[k], m["ret"], yerr=np.std(rv, ddof=1) if len(rv) > 1 else None,
                           xerr=(np.std(kv, ddof=1) if kv and len(kv) > 1 else None), fmt="o", ms=4,
                           color=ZOO, ecolor=ZOO, elinewidth=0.8, capsize=0, zorder=3,
                           mfc=ZOO if len(runs) > 1 else "white")
            else:
                a.plot(m[k], m["ret"], "D", ms=5, color=REL, zorder=4)
                a.annotate(e, (m[k], m["ret"]), xytext=(4, 3), textcoords="offset points", color="#0b0b0b",
                           fontsize=6)
        ref = t2.get("64 mT input (no enhancer)")
        if ref is not None and k == "ssim":
            a.axhline(ref["sim"]["ret_mean"], color=REF, lw=0.8, ls="--", zorder=1)
            a.text(a.get_xlim()[0], ref["sim"]["ret_mean"], " 64 mT input (no enhancer)", va="bottom",
                   fontsize=6, color=INK)
        a.set_xlabel(lab); a.set_ylabel("retention R, 3D lesions" if i == 0 else "")
        a.set_title("ab"[i], loc="left", fontweight="bold")
    ax[0].plot([], [], "o", color=ZOO, ms=4, label="zoo enhancer (filled: 5 seeds)")
    ax[0].plot([], [], "D", color=REL, ms=5, label="released, pretrained")
    ax[0].legend(frameon=False, fontsize=6, loc="lower right")
    if has_t1:
        t1 = res["t1_table"]
        rows = [n for n in ("A_T1in", "B_T2in", "C_both", "SynthSR low-field (T1 in)", "SynthSR low-field (T2 in)",
                            "SynthSR Hyperfine (T1+T2 in)", "SuperSynth T1 (T1 in)", "SuperSynth T1 (T2 in)")
                if n in t1 and "sim" in t1[n]]
        names = {"A_T1in": "route A (T1 in)", "B_T2in": "route B (T2 in)", "C_both": "route C (T1+T2 in)"}
        qs = [q for q, _, _ in QUADS]
        M = np.array([[t1[n]["sim"][q]["inverted_mean"] * 100 for q in qs] for n in rows])
        a = ax[2]
        a.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                a.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=6,
                       color="white" if M[r, c] > 55 else "#0b0b0b")
        a.set_xticks(range(len(qs)), ["T1-dark\nT2-bright", "T1-bright\nT2-dark", "T1-bright\nT2-bright",
                                      "T1-dark\nT2-dark"], fontsize=6)
        a.set_yticks(range(len(rows)), [names.get(n, n) for n in rows], fontsize=6)
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title("c  % wrong sign in synthesised T1 (8 mm lesions)", loc="left", fontsize=7)
        a.title.set_fontweight("bold")
    plt.tight_layout(); os.makedirs(FIG, exist_ok=True)
    plt.savefig(f"{FIG}/pretrained{SFX}.pdf"); plt.savefig(f"{WORK}/pretrained_fig.png", dpi=200); plt.close()


if __name__ == "__main__":
    stage = sys.argv[1]
    globals()[stage](*sys.argv[2:])
