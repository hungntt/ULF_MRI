"""Build the public-data arrays: no unpaired 64 mT pool, no registration-gated data.

Everything is written to datasets/npy_public (point ULF_DATA there); datasets/npy is only read.

  split   OpenNeuro ds006557 (HFC) becomes the development cohort: 15 train + 3 dev-val
          (manifest_train) and 5 locked test subjects (manifest_test), cut from the existing
          ext_on_hfc arrays. The second Swoop unit (HFE) is kept for the 5 test subjects only
          (ext_on_hfe). Zenodo 20281403 (ext_v5) is linked unchanged as the external cohort.
  run3t   MRI-PairedField 3T T1/T2 of UNC (10), Penn (23) and Hippocampal (20). These volumes
          are already skull-stripped, N4-corrected and T1/T2-aligned on a 0.5 mm grid; here
          they are resampled to 1 mm and framed and scaled like the paired cohorts (3T T1
          sets the frame). 3T only: there is no 64 mT array for these splits.
  ixi     IXI, Hammersmith (Philips 3T) subjects only, from the official tarballs
          (CC BY-SA 3.0): the unpaired 3T corpus, replacing the FOMO-50k copy of IXI.
  qc      counts and a montage per new split.

    python exp/prep_public.py split
    python exp/prep_public.py run3t [--workers 6]
    python exp/prep_public.py ixi   [--workers 6]
    python exp/prep_public.py qc
"""
import argparse, csv, glob, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "."
RAW = f"{ROOT}/datasets/raw"
SRC = f"{ROOT}/datasets/npy"
NPY = f"{ROOT}/datasets/npy_public"
WORK = f"{ROOT}/datasets/scratch/public"
FS = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
os.environ.setdefault("FREESURFER_HOME", FS)
os.environ.setdefault("SYNTHSTRIP_BIN", f"{FS}/bin/mri_synthstrip")

N_TEST, N_VAL = 5, 3
SOURCES_3T = {"pub3t_unc": "unc", "pub3t_penn": "penn", "pub3t_hip": "figshare_26075713"}


def _man(split, c, root=SRC):
    return json.load(open(f"{root}/manifest_{split}_{c}.json"))


def choose_split():
    """Age-stratified: sort the 23 subjects by age, cut into N_TEST bins and draw one test
    subject per bin, then N_VAL bins of the rest for dev-val (seed 0)."""
    rows = list(csv.DictReader(open(f"{RAW}/ds006557/participants.tsv"), delimiter="\t"))
    rows = sorted(rows, key=lambda r: float(r["age"]))
    rng = np.random.default_rng(0)
    def draw(pool, k):
        bins = np.array_split(np.arange(len(pool)), k)
        return [pool[int(rng.choice(b))] for b in bins]
    test = draw(rows, N_TEST)
    rest = [r for r in rows if r not in test]
    val = draw(rest, N_VAL)
    key = lambda r: f"ON-{r['participant_id']}"
    return [key(r) for r in test], [key(r) for r in val], {key(r): r for r in rows}


def subset(split_in, sids, split_out, c, has_ulf=True):
    """Copy the rows of `sids` from datasets/npy/{split_in} into npy_public/{split_out}."""
    m = _man(split_in, c)
    hf = np.load(f"{SRC}/paired_{split_in}_hf_{c}.npy")
    ulf = np.load(f"{SRC}/paired_{split_in}_ulf_{c}.npy") if has_ulf else None
    H, U, man, off = [], [], [], 0
    for s in sorted(m["subjects"], key=lambda s: s["subject_id"]):
        if s["subject_id"] not in sids:
            continue
        r = slice(s["slice_start"], s["slice_start"] + s["slice_count"])
        H.append(np.asarray(hf[r]))
        if has_ulf:
            U.append(np.asarray(ulf[r]))
        man.append(dict(s, slice_start=off, split=split_out))
        off += s["slice_count"]
    assert len(man) == len(sids), (split_in, c, len(man), len(sids))
    np.save(f"{NPY}/paired_{split_out}_hf_{c}.npy", np.concatenate(H))
    if has_ulf:
        np.save(f"{NPY}/paired_{split_out}_ulf_{c}.npy", np.concatenate(U))
    json.dump(dict(contrast=c, split=split_out, total_slices=off, num_subjects=len(man),
                   subjects=man), open(f"{NPY}/manifest_{split_out}_{c}.json", "w"), indent=1)
    print(f"  {split_out:11s} {c}: {len(man)} subjects, {off} slices")


def split():
    os.makedirs(NPY, exist_ok=True)
    test, val, demo = choose_split()
    allids = sorted(demo)
    train = [s for s in allids if s not in test]          # includes the dev-val subjects
    json.dump(val, open(f"{NPY}/dev_val_ids.json", "w"), indent=1)
    info = {g: [dict(id=s, age=float(demo[s]["age"]), sex=demo[s]["sex"]) for s in ids]
            for g, ids in (("test", test), ("dev_val", val),
                           ("dev_train", [s for s in train if s not in val]))}
    json.dump(info, open(f"{NPY}/split_public.json", "w"), indent=1)
    for g, v in info.items():
        ages = [x["age"] for x in v]
        print(f"  {g:9s} n={len(v):2d}  age {min(ages):.0f}-{max(ages):.0f} (median "
              f"{np.median(ages):.0f})  F={sum(x['sex'] == 'F' for x in v)}  "
              f"{' '.join(x['id'] for x in v)}")
    for c in ("T1", "T2"):
        subset("ext_on_hfc", set(train), "train", c)
        subset("ext_on_hfc", set(test), "test", c)
        subset("ext_on_hfe", set(test), "ext_on_hfe", c)    # second 64 mT unit, test only
        for kind in ("ulf", "hf"):
            dst = f"{NPY}/paired_ext_v5_{kind}_{c}.npy"
            if not os.path.exists(dst):
                os.symlink(f"{SRC}/paired_ext_v5_{kind}_{c}.npy", dst)
        dst = f"{NPY}/manifest_ext_v5_{c}.json"
        if not os.path.exists(dst):
            os.symlink(f"{SRC}/manifest_ext_v5_{c}.json", dst)
    print("  ext_v5 (Zenodo) linked unchanged")


# ------------------------------------------------------------------ 3T-only cohorts
def jobs_3t():
    base = f"{RAW}/mri_pairedfield"
    rows = list(csv.DictReader(open(f"{base}/pairs.tsv"), delimiter="\t"))
    out = []
    for split, code in SOURCES_3T.items():
        cases = {}
        for r in rows:
            if r["source_dataset"] == code and r["field_strength"] == "3T" and r["modality"] in ("T1W", "T2W"):
                cases.setdefault(r["pair_id"], {})[r["modality"]] = f"{base}/{r['relative_path']}"
        for pid, p in sorted(cases.items()):
            if len(p) == 2:
                out.append((split, pid, p))
    return out


def process_3t(job):
    split, key, p = job
    import nibabel as nib
    from ulfkit import mri_preprocess as mp
    from ulfkit.slices import extract_paired_slices
    od = f"{WORK}/{split}/{key}"
    if os.path.exists(f"{od}/arrays.npz"):
        return key, "cached"
    os.makedirs(od, exist_ok=True)
    try:
        vols, masks = {}, {}
        for c, m in (("T1", "T1W"), ("T2", "T2W")):
            img = mp.reorient_to_ras(nib.load(p[m]))
            img = mp.resample_to_isotropic(img, 1.0, order=1)
            v = np.clip(np.nan_to_num(np.asarray(img.dataobj, np.float32)), 0, None)
            vols[c], masks[c] = v, v > 1e-3 * np.percentile(v[v > 0], 99)
        n1, _, frame = mp.frame_and_normalize(vols["T1"], masks["T1"])
        normed = {"T1": n1, "T2": mp.frame_and_normalize(vols["T2"], masks["T2"], frame=frame)[0]}
        res, meta = {}, {"split": split, "key": key, "paths": p,
                         "frame": [frame.z_start, frame.z_end, *frame.centroid_xy]}
        for c in ("T1", "T2"):
            v = normed[c] * (normed[c] > 0)
            _, hf, _, z = extract_paired_slices(v, v, slice_selection="middle",
                                                middle_fraction=1.0, filter_empty=False)
            res[f"hf_{c}"] = hf.astype(np.float16)
            meta[f"z_{c}"] = [int(x) for x in z]
        np.savez(f"{od}/arrays.npz", **res)
        json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
        return key, f"ok {res['hf_T1'].shape[0]} slices"
    except Exception as e:  # noqa: BLE001 - report and continue with the other subjects
        return key, f"FAILED {type(e).__name__}: {e}"


def run3t(workers=6):
    jobs = jobs_3t()
    print(f"{len(jobs)} 3T-only subjects", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for key, msg in ex.map(process_3t, jobs):
            print(f"  {key}: {msg}", flush=True)
    for split in SOURCES_3T:
        dirs = sorted(glob.glob(f"{WORK}/{split}/*/arrays.npz"))
        for c in ("T1", "T2"):
            H, man, off = [], [], 0
            for f in dirs:
                a = np.load(f); m = json.load(open(f.replace("arrays.npz", "meta.json")))
                H.append(a[f"hf_{c}"]); n = a[f"hf_{c}"].shape[0]
                man.append(dict(subject_id=m["key"], split=split, contrast=c, slice_start=off,
                                slice_count=n, volume_depth=int(m["frame"][1] - m["frame"][0]),
                                z_indices=m[f"z_{c}"], raw=m["paths"][f"{c}W"]))
                off += n
            np.save(f"{NPY}/paired_{split}_hf_{c}.npy", np.concatenate(H))
            json.dump(dict(contrast=c, split=split, total_slices=off, num_subjects=len(man),
                           subjects=man), open(f"{NPY}/manifest_{split}_{c}.json", "w"), indent=1)
            print(f"  {split} {c}: {len(man)} subjects, {off} slices")


# ------------------------------------------------------------------ IXI corpus
def process_ixi(path):
    from ulfkit import mri_preprocess as mp
    key = os.path.basename(path).replace(".nii.gz", "")
    out = f"{WORK}/ixi/{key}.npy"
    if os.path.exists(out):
        return key, "cached"
    try:
        sl, rep = mp.harmonize_fomo_volume(path)      # RAS, SynthStrip, N4, 1 mm, frame, scale
        np.save(out, sl.astype(np.float16))
        return key, f"ok {sl.shape[0]} slices"
    except Exception as e:  # noqa: BLE001
        return key, f"FAILED {type(e).__name__}: {e}"


def ixi(workers=6):
    src = f"{WORK}/ixi_nii"
    os.makedirs(src, exist_ok=True); os.makedirs(f"{WORK}/ixi", exist_ok=True)
    for c in ("T1", "T2"):
        if not glob.glob(f"{src}/*-HH-*-{c}.nii.gz"):
            subprocess.run(["tar", "-xf", f"{RAW}/ixi/IXI-{c}.tar", "-C", src,
                            "--wildcards", "*-HH-*"], check=True)
    paths = sorted(glob.glob(f"{src}/*-HH-*-T[12].nii.gz"))
    print(f"{len(paths)} Hammersmith volumes", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for key, msg in ex.map(process_ixi, paths):
            if not msg.startswith(("ok", "cached")):
                print(f"  {key}: {msg}", flush=True)
    for c in ("T1", "T2"):
        X, man, off = [], [], 0
        for f in sorted(glob.glob(f"{WORK}/ixi/*-HH-*-{c}.npy")):
            a = np.load(f); X.append(a)
            man.append(dict(subject_id=os.path.basename(f).split("-")[0], slice_start=off,
                            slice_count=len(a)))
            off += len(a)
        np.save(f"{NPY}/ixi_hf_{c}.npy", np.concatenate(X))
        json.dump(dict(contrast=c, source="IXI Hammersmith (Philips 3T), CC BY-SA 3.0",
                       total_slices=off, num_subjects=len(man), subjects=man),
                  open(f"{NPY}/manifest_ixi_{c}.json", "w"), indent=1)
        print(f"  ixi {c}: {len(man)} subjects, {off} slices")


def qc():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    splits = ["train", "test", "ext_on_hfe", "ext_v5", *SOURCES_3T]
    f, ax = plt.subplots(len(splits) + 1, 4, figsize=(12, 3 * (len(splits) + 1)))
    for i, s in enumerate(splits):
        for j, c in enumerate(("T1", "T2")):
            m = _man(s, c, NPY); h = np.load(f"{NPY}/paired_{s}_hf_{c}.npy")
            print(f"  {s:11s} {c}: {m['num_subjects']:3d} subjects, {m['total_slices']:5d} slices")
            sub = m["subjects"][0]; mid = sub["slice_start"] + sub["slice_count"] // 2
            ax[i, j].imshow(h[mid, 0].astype(float), cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_title(f"{s} {sub['subject_id']} 3T {c}", fontsize=8)
            up = f"{NPY}/paired_{s}_ulf_{c}.npy"
            if os.path.exists(up):
                ax[i, j + 2].imshow(np.load(up)[mid, 1].astype(float), cmap="gray", vmin=0, vmax=1)
                ax[i, j + 2].set_title(f"{s} 64mT {c}", fontsize=8)
    for j, c in enumerate(("T1", "T2")):
        p = f"{NPY}/ixi_hf_{c}.npy"
        if os.path.exists(p):
            m = json.load(open(f"{NPY}/manifest_ixi_{c}.json")); x = np.load(p)
            print(f"  ixi         {c}: {m['num_subjects']:3d} subjects, {m['total_slices']:5d} slices")
            s0 = m["subjects"][0]
            ax[-1, j].imshow(x[s0["slice_start"] + s0["slice_count"] // 2, 0].astype(float), cmap="gray", vmin=0, vmax=1)
            ax[-1, j].set_title(f"IXI {s0['subject_id']} {c}", fontsize=8)
    for a in ax.ravel(): a.axis("off")
    plt.savefig(f"{WORK}/qc_public.png", dpi=50, bbox_inches="tight")
    print(f"  montage: {WORK}/qc_public.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "run3t", "ixi", "qc"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(NPY, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    {"split": split, "qc": qc}.get(a.stage, lambda: None)()
    if a.stage == "run3t": run3t(a.workers)
    if a.stage == "ixi": ixi(a.workers)
