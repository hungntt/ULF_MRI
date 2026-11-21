"""Does the enhancer ranking depend on our own degradation operator? H2LO as a third-party D.

Every retention number in this programme passes through D, the 3T -> 64 mT operator we trained
ourselves (d1.py), and a natural objection is that the framework may measure D's prior rather
than the enhancers. D_HFC and D_HFE (d3dfactor.py) answer half of it: two units, one architecture,
one training recipe, ranking rho 0.96. This script adds an operator we did not design or train:
H2LO (Gao et al., MICCAI 2026, github.com/Z7Gao/H2Lo-release, MIT, commit pinned below), a DeepONet
whose branch is a 3D CNN on the whole 3T volume and whose trunk is a SIREN on the normalised voxel
coordinate. Its released fold-0 T2w weights were trained on OpenNeuro ds006557 HFC subjects
HYPE00-13 (val 14-15, test 16-22), so three of our five test subjects (03, 06, 08) were in its
training set; results are also split by that.

Method (identical lesions to d3dfactor.run: seed 11, 96 test rows, 4 sites per row, r in {2,3,4,6},
c in {0.12, 0.25}, the same core/ring regions; D_HFC and D_HFE are rerun in the same loop, which
must reproduce out_public/d3dfactor_T2.json):
  * 3T input: the subject's 3T T2 on its 3T-T1 1 mm grid (prep_external: SynthStrip, N4, rigid),
    the volume our test slab was cut from. H2LO max-normalises it, as in its own loader. The slab
    row is a framed (in-plane pad/crop about the brain centroid) slice of this volume, rescaled by
    a per-subject linear map a*V + b that is fitted on the brain (checked: r > 0.999).
  * A lesion c*b_r inserted into a slab row is mapped to the volume (divide by a, then by the
    volume max), H2LO is run on the clean and the lesioned volume, and the 64 mT response is the
    difference. The branch CNN has a 11-voxel receptive field and the trunk is pointwise, so the
    difference is computed exactly on a 52 x 52 x 25 crop around the lesion (encoder on the crop,
    trunk on the full-volume coordinates of the crop voxels; verified against the repository's
    predict_full_volume).
  * H2LO's output is in max-normalised low-field units. It is mapped to our 64 mT slab scale by a
    per-subject linear fit of the acquired 64 mT slab on H2LO's clean output over the brain
    (alpha, beta); the lesion response is scaled by alpha. The clean slab alpha*H2LO + beta is the
    synthetic-arm enhancer input, built like d3dfactor.load_op (centre slice repeated to the three
    input channels, times the 3T brain mask). A secondary variant ("H2LO 3-slice") gives the
    enhancer H2LO's true adjacent slices, since H2LO is 3D.
  * Acquired arm as d3detreal.lesion_input: x_real + response.

Approximations, stated in the output: N4 is applied to H2LO's input (its training data were not
bias-corrected); the grid is the subject's 3T-T1 grid, a few mm from the native T2 grid H2LO saw;
the lesion lives in one 1 mm slice (the framework's 2D lesion) and H2LO spreads it through z.
Only the OpenNeuro test split is used: H2LO's coordinate trunk is tied to the ds006557 field of view.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/d3h2lo.py [check|run]
"""
import importlib.util, json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = "."
sys.path.insert(0, ROOT)
from ulfkit import data as D, train as T                      # noqa: E402
from exp.d3detect import site_pool, blobs                     # noqa: E402
from exp import d3dfactor as F, d3zoo as Z                    # noqa: E402

OUT = os.environ.get("ULF_OUT", f"{ROOT}/out")
H2 = f"{ROOT}/datasets/raw/h2lo/repo"
COMMIT = "d3b3e4c2ef962df5872e4788a27acfdfb73b0338"
EXT = f"{ROOT}/datasets/scratch/external/ext_on_hfc"         # public OpenNeuro volumes (prep_external)
FOLD0_TRAIN = {f"ON-HYPE{i:02d}" for i in range(14)}
HALF, ZH, MARGIN = 26, 12, 6                                  # crop half-width, half-depth; z-chunk margin
N_BOOT = 10000


# ------------------------------------------------------------------ H2LO
def load_h2lo():
    spec = importlib.util.spec_from_file_location("h2lo_model", f"{H2}/h2lo/model.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ck = torch.load(f"{H2}/weights/h2lo_T2w_fold0.pth", map_location="cuda", weights_only=True)
    a = ck["args"]
    net = m.H2LO(a["feature_dim"], a["encoder_base_channels"], a["trunk_type"], a["decoder_depth"],
                 a["decoder_width"], a["omega"]).cuda()
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    return net, m


def norm_coords(idx, n):
    """Voxel index -> H2LO's [-1, 1] coordinate along an axis of length n (linspace(-1, 1, n))."""
    return -1.0 + 2.0 * idx / (n - 1)


@torch.no_grad()
def trunk_dot(net, feat, i0, j0, k0, shape, chunk=65536):
    """sum_k feat_k * t_k(x) over a block. feat (P, h, w, d) at volume offsets (i0, j0, k0)."""
    P, h, w, d = feat.shape
    H, W, Dd = shape
    ci = norm_coords(torch.arange(i0, i0 + h, device="cuda", dtype=torch.float32), H)
    cj = norm_coords(torch.arange(j0, j0 + w, device="cuda", dtype=torch.float32), W)
    ck = norm_coords(torch.arange(k0, k0 + d, device="cuda", dtype=torch.float32), Dd)
    coords = torch.stack(torch.meshgrid(ci, cj, ck, indexing="ij"), -1).view(-1, 3)
    fv = feat.reshape(P, -1).t()
    out = torch.empty(coords.shape[0], device="cuda")
    for s in range(0, coords.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.float16):
            t = net.trunk(coords[s:s + chunk])
        out[s:s + chunk] = (fv[s:s + chunk].float() * t.float()).sum(-1)
    return out.view(h, w, d)


@torch.no_grad()
def h2lo_full(net, vol, zc=8):
    """H2LO on a whole (H, W, D) volume in [0, 1], encoder in z-chunks with a margin >= its
    receptive-field radius, so the result equals the un-chunked forward."""
    H, W, Dd = vol.shape
    x = torch.from_numpy(vol).cuda()[None, None]
    out = torch.empty((H, W, Dd), device="cuda")
    for z in range(0, Dd, zc):
        a, b = max(0, z - MARGIN), min(Dd, z + zc + MARGIN)
        with torch.autocast("cuda", dtype=torch.float16):
            f = net.encoder(x[..., a:b])[0]
        n = min(zc, Dd - z)
        out[..., z:z + n] = trunk_dot(net, f[..., z - a:z - a + n], 0, 0, z, (H, W, Dd))
        del f
    return out + net.bias


@torch.no_grad()
def h2lo_delta(net, vol_t, shape, centres, patches):
    """Exact H2LO response to local input changes. vol_t: (H, W, D) cuda tensor in [0, 1];
    centres: [(i, j, k)] volume voxels; patches: [(2*HALF, 2*HALF) tensors] added at slice k,
    centred on (i, j). -> list of (i0, j0, k0, delta (2HALF, 2HALF, 2ZH+1))."""
    H, W, Dd = shape
    crops, offs = [], []
    for (i, j, k), p in zip(centres, patches):
        i0 = int(np.clip(i - HALF, 0, H - 2 * HALF)); j0 = int(np.clip(j - HALF, 0, W - 2 * HALF))
        k0 = int(np.clip(k - ZH, 0, Dd - (2 * ZH + 1)))
        c0 = vol_t[i0:i0 + 2 * HALF, j0:j0 + 2 * HALF, k0:k0 + 2 * ZH + 1]
        c1 = c0.clone()
        pi, pj = i - HALF - i0, j - HALF - j0                # patch offset inside the crop (0 unless clipped)
        assert pi == 0 and pj == 0, "lesion too close to the volume edge"
        if p.dim() == 3:                                     # 3D lesion, cut to the crop; outputs at k-1..k+1
            assert k - k0 == ZH, "3D lesion too close to the volume's z edge"   # stay exact (RF radius 5 < ZH - 1)
            c1 += p
        else:
            c1[:, :, k - k0] += p
        crops += [c0, c1]; offs.append((i0, j0, k0))
    x = torch.stack(crops)[:, None]
    res = []
    with torch.autocast("cuda", dtype=torch.float16):
        f = net.encoder(x)
    for n, (i0, j0, k0) in enumerate(offs):
        df = (f[2 * n + 1] - f[2 * n]).float()
        res.append((i0, j0, k0, trunk_dot(net, df, i0, j0, k0, shape)))
    return res


# ------------------------------------------------------------------ subject geometry
class Subject:
    """A test subject's 3T T2 volume and the map between its slab rows and the volume."""

    def __init__(self, man, hf_rows, ulf_rows):
        import nibabel as nib
        self.sid = man["subject_id"]
        d = f"{EXT}/{self.sid}"
        meta = json.load(open(f"{d}/meta.json"))
        z0, _, cy, cx = meta["frame"]
        self.oy, self.ox = int(round(cy - 127.5)), int(round(cx - 127.5))
        V = np.asarray(nib.load(f"{d}/hf_T2.nii.gz").dataobj, dtype=np.float32)
        V[~np.isfinite(V)] = 0; V = np.clip(V, 0, None)
        self.vmax = float(V.max()); self.vol = V / self.vmax          # H2LO's max normalisation
        self.shape = V.shape
        self.row0 = man["slice_start"]
        self.z = np.array([z0 + zi for zi in man["z_indices"]])
        # slab row = a * framed(V) + b (brain, unclipped voxels)
        ys, vs = [], []
        for r in range(0, man["slice_count"], 4):
            y = hf_rows[self.row0 + r, 0].astype(np.float32); v = self.framed(V, self.z[r])
            m = (y > 0.02) & (y < 0.98) & (v > 0); ys.append(y[m]); vs.append(v[m])
        ys, vs = np.concatenate(ys), np.concatenate(vs)
        self.a, self.b = np.polyfit(vs, ys, 1); self.r_ab = float(np.corrcoef(vs, ys)[0, 1])
        self.ulf = ulf_rows

    def framed(self, vol, z):
        """Framed 256 x 256 slice z of a (H, W, D) array (numpy or torch)."""
        H, W = vol.shape[:2]
        is_t = torch.is_tensor(vol)
        out = torch.zeros((256, 256), device=vol.device) if is_t else np.zeros((256, 256), np.float32)
        y0, x0 = self.oy, self.ox
        sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(H, y0 + 256), min(W, x0 + 256)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = vol[sy0:sy1, sx0:sx1, z]
        return out

    def to_volume_ij(self, cy, cx):
        return cy + self.oy, cx + self.ox


# ------------------------------------------------------------------ checks
@torch.no_grad()
def check():
    """(1) the chunked/cropped H2LO equals the repository's predict_full_volume; (2) slab mapping."""
    net, m = load_h2lo()
    coh = D.PairedCohort("test", "T2")
    man = D.manifest("test", "T2")
    s = Subject(man[0], coh.hf, coh.ulf)
    sub = np.ascontiguousarray(s.vol[:, :, 100:150])                # a sub-volume small enough for the reference
    ref = m.predict_full_volume(net, torch.from_numpy(sub).cuda()[None, None], use_amp=True)
    mine = h2lo_full(net, sub)
    print(f"chunked vs predict_full_volume: max|d| {float((ref - mine).abs().max()):.2e}, "
          f"max|ref| {float(ref.abs().max()):.3f}")
    vt = torch.from_numpy(sub).cuda()
    pat = torch.zeros((2 * HALF, 2 * HALF), device="cuda"); pat[HALF - 3:HALF + 3, HALF - 3:HALF + 3] = 0.1
    (i0, j0, k0, dl), = h2lo_delta(net, vt, sub.shape, [(100, 128, 25)], [pat])
    sub2 = sub.copy(); sub2[100 - HALF:100 + HALF, 128 - HALF:128 + HALF, 25] += pat.cpu().numpy()
    ref2 = m.predict_full_volume(net, torch.from_numpy(sub2).cuda()[None, None], use_amp=True) - ref
    full = torch.zeros_like(ref2); full[i0:i0 + 2 * HALF, j0:j0 + 2 * HALF, k0:k0 + 2 * ZH + 1] = dl
    print(f"cropped delta vs full-volume delta: max|d| {float((ref2 - full).abs().max()):.2e}, "
          f"max|delta| {float(ref2.abs().max()):.3f}, outside-crop max {float(ref2.abs().max() - full.abs().max()):+.2e}")
    print(f"slab map {s.sid}: a {s.a:.5f} b {s.b:+.4f} r {s.r_ab:.5f}")


# ------------------------------------------------------------------ run
@torch.no_grad()
def run(n_rows=96, sites=4, seed=11):
    T.seed_all(0)
    net, _ = load_h2lo()
    ops = {o: F.load_op(p) for o, p in F.OPS.items()}
    reg = [r for r in Z.registry() if r[2] == 0 and r[4] != "diff" and os.path.exists(f"{OUT}/{r[3]}.pt")]
    nets = [(r[1], Z.load(r[4], r[3])) for r in reg]
    coh = D.PairedCohort("test", "T2")
    man = D.manifest("test", "T2")
    subs = [Subject(mm, coh.hf, coh.ulf) for mm in man]
    row_sub = {}
    for s in subs:
        for i, z in enumerate(s.z): row_sub[s.row0 + i] = (s, z)
    # clean H2LO per subject, framed, and its map to the acquired 64 mT slab scale
    clean, fid = {}, {}
    for s in subs:
        full = h2lo_full(net, s.vol)
        fr = torch.stack([s.framed(full, z) for z in s.z])                    # (n, 256, 256)
        xr = torch.from_numpy(coh.ulf[s.row0:s.row0 + len(s.z), 1].astype(np.float32)).cuda()
        y = torch.from_numpy(coh.hf[s.row0:s.row0 + len(s.z), 0].astype(np.float32)).cuda()
        m = (y > 0) & (xr > 0)
        al, be = np.polyfit(fr[m].cpu().numpy(), xr[m].cpu().numpy(), 1)
        xs = ((al * fr + be).clamp(0, 1) * (y > 0)).float()
        dh = ops["HFC"](y[:, None])[:, 1]
        fid[s.sid] = dict(alpha=float(al), beta=float(be), a=float(s.a), b=float(s.b), r_ab=s.r_ab,
                          in_h2lo_train=s.sid in FOLD0_TRAIN,
                          ncc_h2lo_vs_acq=float(np.corrcoef(xs[m].cpu().numpy(), xr[m].cpu().numpy())[0, 1]),
                          ncc_hfc_vs_acq=float(np.corrcoef(dh[m].cpu().numpy(), xr[m].cpu().numpy())[0, 1]))
        clean[s.sid] = (full, xs, float(al), float(be))
        print(f"  {s.sid}: a {s.a:.5f} (r {s.r_ab:.4f}); alpha {al:.3f} beta {be:+.3f}; NCC vs acquired "
              f"H2LO {fid[s.sid]['ncc_h2lo_vs_acq']:.3f}, D_HFC {fid[s.sid]['ncc_hfc_vs_acq']:.3f}", flush=True)
    OPN = ["HFC", "HFE", "H2LO", "H2LO 3-slice", "H2LO sphere", "H2LO sphere 3-slice"]
    rec = {(o, nm): dict(sid=[], r=[], dy=[], dd=[], dp=[], dpa=[]) for o in OPN for nm, _ in nets}
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(coh.hf), min(n_rows, len(coh.hf)), replace=False))
    for i in range(0, len(rows), 8):
        rr = rows[i:i + 8]
        y = torch.from_numpy(coh.hf[rr].astype(np.float32)).cuda()
        ok = site_pool(y)
        jobs = []
        for s in range(len(y)):
            cand = torch.nonzero(ok[s])
            if len(cand) < 8: continue
            for _ in range(sites):
                cy, cx = cand[rng.integers(len(cand))].tolist()
                jobs.append((s, cy, cx, int(rng.choice(F.RADII)), float(rng.choice(F.CS))))
        if not jobs: continue
        si = torch.tensor([j[0] for j in jobs], device="cuda")
        b = blobs(torch.tensor([j[1] for j in jobs], device="cuda"), torch.tensor([j[2] for j in jobs], device="cuda"),
                  torch.tensor([j[3] for j in jobs], device="cuda")) * (y[si] > 0).float()
        c = torch.tensor([j[4] for j in jobs], device="cuda")[:, None, None, None]
        y0 = y[si]; yl = (y0 + c * b).clamp(0, 1)
        core, ring = b > 0.6, (b > 0.05) & (b < 0.25)
        good = [k for k in range(len(jobs)) if core[k].sum() >= 4 and ring[k].sum() >= 8]
        con = lambda d, k: float(d[k][core[k]].mean() - d[k][ring[k]].mean())
        # acquired 64 mT slab rows and the near mask of d3detreal.lesion_input
        xr = torch.from_numpy(coh.ulf[rr[si.cpu().numpy()]].astype(np.float32)).cuda()
        rad = torch.tensor([j[3] for j in jobs], device="cuda")
        near = (blobs(torch.tensor([j[1] for j in jobs], device="cuda"), torch.tensor([j[2] for j in jobs], device="cuda"),
                      torch.clamp(rad.float() * 4, min=24)) > 0.05).float()
        inputs = {}
        for o, f in ops.items():                                           # our operators, as d3dfactor
            x0, xl = f(y0), f(yl)
            inputs[o] = (x0, xl, xr + (xl - x0) * near)
        # H2LO: exact local response per lesion
        dl = (yl - y0)[:, 0]
        centres, patches, meta = [], [], []
        for k, (s_, cy, cx, r_, c_) in enumerate(jobs):
            sub, z = row_sub[int(rr[s_])]
            vi, vj = sub.to_volume_ij(cy, cx)
            p = torch.zeros((2 * HALF, 2 * HALF), device="cuda")
            ya, xa = cy - HALF, cx - HALF
            p_src = dl[k, max(0, ya):ya + 2 * HALF, max(0, xa):xa + 2 * HALF]
            p[max(0, -ya):max(0, -ya) + p_src.shape[0], max(0, -xa):max(0, -xa) + p_src.shape[1]] = p_src
            centres.append((vi, vj, int(z))); patches.append(p / sub.a / sub.vmax); meta.append((sub, z, vi, vj))
        # spherical variant: the same Gaussian extended through z with the same radius (c * b_r in 3D,
        # 3T brain only, no clipping), for an operator that models the 5 mm through-plane blur
        ii = torch.arange(2 * HALF, device="cuda", dtype=torch.float32) - HALF
        zz = torch.arange(2 * ZH + 1, device="cuda", dtype=torch.float32) - ZH
        d2 = ii[:, None, None] ** 2 + ii[None, :, None] ** 2 + zz[None, None, :] ** 2
        patches3 = []
        for k, (sub, z, vi, vj) in enumerate(meta):
            r_, c_ = jobs[k][3], jobs[k][4]
            brain = torch.from_numpy(sub.vol[vi - HALF:vi + HALF, vj - HALF:vj + HALF, z - ZH:z + ZH + 1] > 0).cuda()
            patches3.append(c_ * torch.exp(-2.0 * d2 / r_ ** 2) * brain / sub.a / sub.vmax)
        by_sub = {}
        for k, (sub, z, vi, vj) in enumerate(meta): by_sub.setdefault(sub.sid, []).append(k)

        def responses(pl):
            resp = [None] * len(jobs)
            for sid, ks in by_sub.items():
                sub = meta[ks[0]][0]
                vt = torch.from_numpy(sub.vol).cuda()
                for k0 in range(0, len(ks), 8):
                    kk = ks[k0:k0 + 8]
                    out = h2lo_delta(net, vt, sub.shape, [centres[k] for k in kk], [pl[k] for k in kk])
                    for k, o_ in zip(kk, out): resp[k] = o_
                del vt
            return resp

        def h2lo_inputs(resp):
            x0h, xlh, x0h3, xlh3 = [], [], [], []
            for k, (sub, z, vi, vj) in enumerate(meta):
                full, xs, al, be = clean[sub.sid]
                i0, j0, k0_, dlt = resp[k]
                d3 = torch.zeros((256, 256, 3), device="cuda")
                for t, dz in enumerate((-1, 0, 1)):                        # framed response at z-1, z, z+1
                    vol_slice = torch.zeros(sub.shape[:2], device="cuda")
                    vol_slice[i0:i0 + 2 * HALF, j0:j0 + 2 * HALF] = dlt[:, :, z + dz - k0_]
                    d3[..., t] = sub.framed(vol_slice[..., None].expand(-1, -1, 1), 0) * al
                zi = int(np.where(sub.z == z)[0][0])
                ym = (y0[k, 0] > 0).float()
                xc = xs[zi]                                                # clean centre slice
                xl_c = (xc + d3[..., 1]).clamp(0, 1) * ym
                x0h.append(xc.repeat(3, 1, 1)); xlh.append(xl_c.repeat(3, 1, 1))
                nb = [xs[max(zi - 1, 0)], xc, xs[min(zi + 1, len(sub.z) - 1)]]
                x0h3.append(torch.stack(nb)); xlh3.append(torch.stack([(nb[t] + d3[..., t]).clamp(0, 1) * ym for t in range(3)]))
            x0h, xlh, x0h3, xlh3 = (torch.stack(v) for v in (x0h, xlh, x0h3, xlh3))
            return (x0h, xlh, xr + (xlh - x0h) * near), (x0h3, xlh3, xr + (xlh3 - x0h3) * near)

        inputs["H2LO"], inputs["H2LO 3-slice"] = h2lo_inputs(responses(patches))
        inputs["H2LO sphere"], inputs["H2LO sphere 3-slice"] = h2lo_inputs(responses(patches3))
        for o in OPN:
            x0, xl, xa = inputs[o]
            dd = (xl - x0)[:, 1:2]
            for nm, g in nets:
                p0, pl, pa, pr = [], [], [], []
                for k0 in range(0, len(jobs), 12):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p0.append(g(x0[k0:k0 + 12]).float().clamp(0, 1)); pl.append(g(xl[k0:k0 + 12]).float().clamp(0, 1))
                        pa.append(g(xa[k0:k0 + 12]).float().clamp(0, 1)); pr.append(g(xr[k0:k0 + 12]).float().clamp(0, 1))
                dp = torch.cat(pl) - torch.cat(p0); dpa = torch.cat(pa) - torch.cat(pr)
                R = rec[(o, nm)]
                for k in good:
                    R["sid"].append(meta[k][0].sid); R["r"].append(jobs[k][3]); R["dy"].append(con(yl - y0, k))
                    R["dd"].append(con(dd, k)); R["dp"].append(con(dp, k)); R["dpa"].append(con(dpa, k))
        print(f"  rows {i + len(rr)}/{len(rows)}", flush=True)
    return rec, fid, [nm for nm, _ in nets]


def stats(R, ids=None):
    sid = np.array(R["sid"]); m = np.ones(len(sid), bool) if ids is None else np.isin(sid, list(ids))
    dy, dd, dp, dpa = (np.array(R[k])[m] for k in ("dy", "dd", "dp", "dpa"))
    return dict(n=int(m.sum()), R=float(np.mean(dp / dy)), R_D=float(dd.mean() / dy.mean()),
                R_GD=float(dp.mean() / dd.mean()), erased=float(np.mean(dp / dy < 0.25)),
                R_acq=float(np.mean(dpa / dy)), erased_acq=float(np.mean(dpa / dy < 0.25)))


def boot_rho(rec, names, o1, o2, n_boot=N_BOOT, seed=0):
    """Spearman of the enhancer ranking on mean R under two operators; subject-cluster bootstrap."""
    sids = sorted(set(rec[(o1, names[0])]["sid"]))
    rng = np.random.default_rng(seed)
    def mean_R(o, w):
        out = []
        for nm in names:
            R = rec[(o, nm)]; sid = np.array(R["sid"]); r = np.array(R["dp"]) / np.array(R["dy"])
            ww = np.array([w[s] for s in sid], float)
            out.append(float((ww * r).sum() / ww.sum()))
        return out
    w1 = {s: 1.0 for s in sids}
    rho = float(spearmanr(mean_R(o1, w1), mean_R(o2, w1)).statistic)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(sids, len(sids)); w = {s: float((pick == s).sum()) for s in sids}
        if sum(v > 0 for v in w.values()) < 2: continue
        bs.append(spearmanr(mean_R(o1, w), mean_R(o2, w)).statistic)
    return dict(rho=rho, ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))])


def main():
    rec, fid, names = run()
    ops = ["HFC", "HFE", "H2LO", "H2LO 3-slice", "H2LO sphere", "H2LO sphere 3-slice"]
    per = {o: {nm: stats(rec[(o, nm)]) for nm in names} for o in ops}
    groups = {"in_h2lo_train": FOLD0_TRAIN, "not_in_h2lo_train": {"ON-HYPE16", "ON-HYPE18"}}
    per_group = {g: {o: {nm: stats(rec[(o, nm)], ids) for nm in names} for o in ops} for g, ids in groups.items()}
    rd = {o: per[o][names[0]]["R_D"] for o in ops}
    rd_group = {g: {o: per_group[g][o][names[0]]["R_D"] for o in ops} for g in groups}
    rho = {f"{a} vs {b}": boot_rho(rec, names, a, b) for i, a in enumerate(ops) for b in ops[i + 1:]}
    ref = json.load(open(f"{OUT}/d3dfactor_T2.json"))["test"]["per"]
    repro = {f"{o}|{nm}": abs(per[o][nm]["R"] - ref[f"{o}|{nm}"]["R"]) for o in ("HFC", "HFE") for nm in names
             if f"{o}|{nm}" in ref}
    res = dict(
        meta=dict(h2lo_repo="https://github.com/Z7Gao/H2Lo-release", h2lo_commit=COMMIT,
                  weights="weights/h2lo_T2w_fold0.pth (epoch 420)", licence="MIT",
                  h2lo_fold0_train="HYPE00-13 (val 14-15, test 16-22)",
                  overlap_note="our test subjects ON-HYPE03, 06, 08 were in H2LO's training set; 16, 18 were not",
                  split="test", lesions="d3dfactor.run: seed 11, 96 rows, 4 sites, r in {2,3,4,6} (diam 4-12 mm), c in {0.12, 0.25}",
                  r_vox=list(F.RADII), diam_mm=[2 * r for r in F.RADII],
                  intensity_maps="slab = a*V + b (3T, per subject); acquired 64 mT slab ~ alpha*H2LO + beta (per subject); response scaled by alpha",
                  approximations=["H2LO input is N4-corrected (its training data were not)",
                                  "grid = subject's 3T-T1 1 mm grid, a few mm from the native T2 grid",
                                  "'H2LO': the framework's 2D lesion occupies one 1 mm slice and H2LO (3D, trained against 5 mm-slice data) spreads it through z; 'H2LO sphere': the same Gaussian extended through z (radius r), the like-for-like lesion for a 3D operator; our 2D D sees only the centre slice, so its numbers are the same for both",
                                  "only the OpenNeuro test split (H2LO's coordinate trunk is tied to ds006557)"],
                  enhancer_input="'H2LO': centre slice repeated to 3 channels, as d3dfactor.load_op; 'H2LO 3-slice': true adjacent slices"),
        fidelity=fid, R_D=rd, R_D_by_group=rd_group, per=per, per_group=per_group, rank_rho=rho,
        reproduction=dict(max_abs_dR_vs_d3dfactor=max(repro.values()) if repro else None, n=len(repro)))
    json.dump(res, open(f"{OUT}/d3h2lo_T2.json", "w"), indent=1)
    print("\nR_D:", {o: round(v, 3) for o, v in rd.items()})
    print("R_D by group:", {g: {o: round(v, 3) for o, v in d.items()} for g, d in rd_group.items()})
    print(f"reproduction of d3dfactor (HFC/HFE): max |dR| {res['reproduction']['max_abs_dR_vs_d3dfactor']:.4f}")
    print(f"\n  {'enhancer':28s}" + "".join(f"{o:>14s}" for o in ops))
    for nm in names:
        print(f"  {nm:28s}" + "".join(f"{per[o][nm]['R']:14.3f}" for o in ops))
    for k, v in rho.items():
        print(f"  ranking rho {k}: {v['rho']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    {"check": check, "run": main}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
