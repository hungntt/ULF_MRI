"""A handcrafted baseline for the retention map.

The map (out_public/d3detect_T2*.pt, four training seeds) predicts erasure of a lesion by the absolute-output U-Net
with AUROC 0.827 on the 1605 test lesions; radius alone reaches 0.727. The baseline uses features a reader could
compute from the lesion-free 3T image at the site, without any learned image model:
radius, contrast, SynthSeg tissue class at the centre, distance to CSF (3D, mm), lesion-free 3T mean in the core
and in the ring, local standard deviation and mean gradient magnitude over the footprint (b > 0.05).
Logistic regression (standardised) and gradient boosting are fitted on the map's own 5754 training labels
(15 training subjects) and scored on the same 1605 test lesions, overall and within each radius.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public ULF_SCRATCH=datasets/scratch_public python exp/coauthor_baseline.py
"""
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
DATA = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy_public")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch_public")
OUT_JSON = f"{OUT}/coauthor_baseline_T2.json"
THR = 0.25
CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)
TISSUES = ("WM", "cortical GM", "deep GM", "CSF", "cerebellum", "brainstem", "other")
SEG_DIRS = (f"{SCR}/physcheck", f"{SCR}/coauthor_seg")


def labels(split):
    f = {"train": "detect_labels_T2.npy", "test": "detect_test_T2.npy"}[split]
    return np.load(f"{DATA}/{f}", allow_pickle=True).astype(float)


def csf_distance(seg):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~np.isin(seg, CSF_LABELS))


def tissue_one_hot(classes):
    return np.array([[float((c if c in TISSUES else "other") == t) for t in TISSUES] for c in classes])


def footprint_features(y, rows, cy, cx, r):
    """Lesion-free intensity features over each site's Gaussian footprint (core b > 0.6, ring 0.05-0.25)."""
    H, W = y.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    out = {k: [] for k in ("core_mean", "ring_mean", "std", "grad")}
    grads = {}
    for row, a, b_, rr in zip(rows, cy, cx, r):
        row = int(row)
        if row not in grads:
            gy, gx = np.gradient(y[row].astype(np.float64))
            grads[row] = np.hypot(gy, gx)
        b = np.exp(-2.0 * ((yy - a) ** 2 + (xx - b_) ** 2) / float(rr) ** 2) * (y[row] > 0)
        core, ring, foot = b > 0.6, (b > 0.05) & (b < 0.25), b > 0.05
        out["core_mean"].append(float(y[row][core].mean())); out["ring_mean"].append(float(y[row][ring].mean()))
        out["std"].append(float(y[row][foot].std())); out["grad"].append(float(grads[row][foot].mean()))
    return {k: np.array(v) for k, v in out.items()}


def load_seg_any(sid):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    for d in SEG_DIRS:
        if os.path.exists(f"{d}/seg/{sid}.nii.gz"):
            ref, s = nib.load(f"{d}/in/{sid}.nii.gz"), nib.load(f"{d}/seg/{sid}.nii.gz")
            if s.shape != ref.shape or not np.allclose(s.affine, ref.affine):
                s = resample_from_to(s, ref, order=0)
            return np.transpose(np.asarray(s.dataobj).astype(np.int16), (2, 0, 1))
    raise FileNotFoundError(f"no SynthSeg for {sid}")


def features(split):
    """-> (X, names, erased, radius, subject id) for the map's labels of a split."""
    from exp.lesion_table import row_sid, tissue_class
    lab = labels(split)
    rows, cy, cx, r, c, ret = (lab[:, k] for k in range(6))
    rows, cy, cx = rows.astype(int), cy.astype(int), cx.astype(int)
    hf = np.load(f"{DATA}/paired_{split}_hf_T2.npy")[:, 0].astype(np.float32)
    sid, loc = row_sid(split)
    tis, dcsf = [], []
    cache = {}
    for rw, a, b in zip(rows, cy, cx):
        s = sid[rw]
        if s not in cache:
            seg = load_seg_any(s)
            cache = {s: (seg, csf_distance(seg))}
        seg, dist = cache[s]
        tis.append(tissue_class(seg[loc[rw], a, b])); dcsf.append(float(dist[loc[rw], a, b]))
    fp = footprint_features(hf, rows, cy, cx, r)
    oh = tissue_one_hot(tis)
    X = np.column_stack([r, c, np.array(dcsf), fp["core_mean"], fp["ring_mean"], fp["std"], fp["grad"], oh])
    names = ["radius", "contrast", "dist_csf_mm", "core_mean", "ring_mean", "local_std", "grad"] + \
            [f"tissue={t}" for t in TISSUES]
    return X, names, ret < THR, r, np.array([sid[rw] for rw in rows])


def map_scores():
    seeds = [json.load(open(f"{OUT}/d3detect_T2{t}.json")) for t in ("", "_s1", "_s2", "_s3")]
    return [np.array([-x["detect"] for x in s]) for s in seeds]


def fit_and_score():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from exp.coauthor_stats import boot_subjects
    Xtr, names, ytr, _, _ = features("train")
    Xte, _, yte, rte, ste = features("test")
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
              "boosting": HistGradientBoostingClassifier(random_state=0)}
    P = {k: m.fit(Xtr, ytr).predict_proba(Xte)[:, 1] for k, m in models.items()}
    P["radius"] = -rte
    maps = map_scores()
    auc = lambda s, m=slice(None), w=None: float(roc_auc_score(yte[m], s[m], sample_weight=w))
    res = dict(n_train=int(len(ytr)), n_test=int(len(yte)), erased_test=float(yte.mean()), features=names,
               auroc={k: auc(v) for k, v in P.items()},
               within_radius={k: {str(int(2 * rr)): auc(v, rte == rr) for rr in (2, 3, 4, 6)} for k, v in P.items()})
    res["auroc"]["map"] = float(np.mean([auc(s) for s in maps]))
    res["within_radius"]["map"] = {str(int(2 * rr)): float(np.mean([auc(s, rte == rr) for s in maps])) for rr in (2, 3, 4, 6)}
    best = max(("logistic", "boosting"), key=lambda k: res["auroc"][k])
    meta = dict(sid=ste, subjects=sorted(set(ste)))

    def diff(w):
        k = w > 0
        return float(np.mean([roc_auc_score(yte[k], s[k], sample_weight=w[k]) for s in maps])
                     - roc_auc_score(yte[k], P[best][k], sample_weight=w[k]))
    res["best"] = best
    res["map_minus_best"] = boot_subjects(diff, meta, B=2000)
    res["boosting_importance_note"] = "permutation importance on the test set"
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(models["boosting"], Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0)
    res["boosting_importance"] = dict(sorted(zip(names, pi.importances_mean.tolist()), key=lambda kv: -kv[1]))
    m, b, rad = res["auroc"]["map"], res["auroc"][best], res["auroc"]["radius"]
    within_ok = all(res["within_radius"][best][s] < res["within_radius"]["map"][s] for s in ("4", "6", "8", "12"))
    res["E12a"] = {"pass": bool(m - b >= 0.03 and b > rad and within_ok),
                   "value": f"best handcrafted ({best}) {b:.3f} against map {m:.3f} and radius {rad:.3f}; within radius "
                            + ", ".join(f"{s} mm {res['within_radius'][best][s]:.3f} vs {res['within_radius']['map'][s]:.3f}"
                                        for s in ("4", "6", "8", "12"))}
    return res


if __name__ == "__main__":
    res = fit_and_score()
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=1))
