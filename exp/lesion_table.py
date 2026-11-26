"""Unique lesion instances.

Counts lesion instances by cohort, subject, size, contrast, polarity and tissue site for every
lesion set the paper uses, and the physical-check instances per thickness bin, which decide
whether k can be estimated separately for 6 and 8 mm lesions.

    ULF_DATA=datasets/npy_public ULF_OUT=out_public python exp/lesion_table.py
        -> out_public/coauthor_lesions.md, out_public/coauthor_lesions.json
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = "."
sys.path.insert(0, ROOT)
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
DATA = os.environ.get("ULF_DATA", f"{ROOT}/datasets/npy_public")
SCR = os.environ.get("ULF_SCRATCH", f"{ROOT}/datasets/scratch_public")
SEG = f"{SCR}/physcheck"                       # SynthSeg of the 3T T1, on the array grid (exp/d1_physcheck.py)
KEEP_MIN = 0.03                                 # measurable contrast, as d3reallesion / d3atlas
STAGE_SPLITS = ("test", "ext_on_hfe", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip")

_LABELS = {"WM": (2, 41), "cortical GM": (3, 42), "CSF": (4, 5, 14, 15, 24, 43, 44),
           "deep GM": (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60),
           "cerebellum": (7, 8, 46, 47), "brainstem": (16,), "background": (0,)}
_LAB2CLS = {lab: cls for cls, labs in _LABELS.items() for lab in labs}


def tissue_class(label):
    return _LAB2CLS.get(int(label), "other")


def subjects(split, contrast="T2"):
    """-> [(sid, slice_start, slice_count)] in array order."""
    m = json.load(open(f"{DATA}/manifest_{split}_{contrast}.json"))["subjects"]
    return [(s["subject_id"], s["slice_start"], s["slice_count"]) for s in m]


def row_sid(split):
    """-> (array row -> subject id, array row -> slice index within the subject)."""
    sid, loc = {}, {}
    for s, a, n in subjects(split):
        for i in range(n):
            sid[a + i], loc[a + i] = s, i
    return sid, loc


_SEG = {}


def seg_volume(sid):
    """SynthSeg labels (Z, H, W) on the subject's array grid, or None when it was not run."""
    if sid not in _SEG:
        f = f"{SEG}/seg/{sid}.nii.gz"
        if not os.path.exists(f):
            _SEG[sid] = None
        else:
            from exp.d1_physcheck import load_seg
            _SEG[sid] = load_seg(sid)
    return _SEG[sid]


def tissue_at(split, rows, cy, cx):
    """Tissue class at each lesion centre; 'no SynthSeg' for cohorts without a segmentation."""
    sid, loc = row_sid(split)
    out = []
    for r, y, x in zip(rows, cy, cx):
        v = seg_volume(sid[int(r)])
        out.append("no SynthSeg" if v is None else tissue_class(v[loc[int(r)], int(y), int(x)]))
    return out


def _count(xs):
    return dict(sorted(collections.Counter(xs).items(), key=lambda kv: str(kv[0])))


def stage_table(split, tissue=False):
    """The d3stage lesions (matched across all 36 models; seed, selection and k-scaling checks)."""
    z = np.load(f"{OUT}/d3stage_rows_{split}_T2.npz")
    meta = json.load(open(f"{OUT}/d3stage_rows_{split}_T2.meta.json"))
    g = z["meta_good"].astype(bool)
    sids = [meta["subjects"][i] for i in z["sid_idx"][g]]
    size = (2 * z["meta_r"][g]).tolist()
    c = z["meta_c"][g]
    t = dict(n=int(g.sum()), n_subjects=len(set(sids)), per_subject=_count(sids),
             by_size_mm=_count(size), by_contrast=_count(np.abs(c).round(2).tolist()),
             by_polarity=_count(["bright" if v > 0 else "dark" for v in c]),
             by_size_contrast=_count([f"{s} mm, c {abs(v):.2f}" for s, v in zip(size, c)]),
             unique_sites=len(set(zip(z["meta_row"][g].tolist(), z["meta_cy"][g].tolist(), z["meta_cx"][g].tolist()))))
    t["min_cell_per_subject"] = min(collections.Counter(zip(sids, size)).values())
    if tissue:
        t["by_tissue"] = _count(tissue_at(split, z["meta_row"][g], z["meta_cy"][g], z["meta_cx"][g]))
    return t


def detect_table():
    """The 1605 test lesions of the retention map (out_public/d3detect_T2.json, absolute-output U-Net)."""
    rec = json.load(open(f"{OUT}/d3detect_T2.json"))
    sid, _ = row_sid("test")
    return dict(n=len(rec), per_subject=_count([sid[x["row"]] for x in rec]),
                by_size_mm=_count([2 * x["r"] for x in rec]), by_contrast=_count([x["c"] for x in rec]),
                by_polarity=_count(["bright" if x["c"] > 0 else "dark" for x in rec]),
                slices=len({x["row"] for x in rec}),
                erased_per_subject={s: float(np.mean([x["retention"] < 0.25 for x in rec if sid[x["row"]] == s]))
                                    for s in sorted({sid[x["row"]] for x in rec})})


def lesiontype_table():
    """Cross-contrast sign patterns (d3lesiontype rows): lesions per pattern and cohort, one route counted."""
    out = {}
    for pat in ("t1b_t2b", "t1b_t2d", "t1d_t2b", "t1d_t2d"):
        for sp in ("test", "ext_v5", "pub3t_unc", "pub3t_penn", "pub3t_hip"):
            f = f"{OUT}/d3lesiontype_rows_{pat}{'' if sp == 'test' else '_' + sp}.json"
            if not os.path.exists(f):
                continue
            rows = json.load(open(f))
            r0 = rows[0]["route"]
            one = [x for x in rows if x["route"] == r0]
            out[f"{pat}|{sp}"] = dict(n=len(one), routes=sorted({x["route"] for x in rows}),
                                     n_subjects=len({x["sid"] for x in one}),
                                     by_size_mm=_count([2 * x["r"] for x in one]))
    return out


def realms_table():
    """Transplanted MS lesions (out_public/d3reallesion_rows_s0.json)."""
    rows = json.load(open(f"{OUT}/d3reallesion_rows_s0.json"))["rows"]
    fin = lambda l: all(np.isfinite(v) for o in l["out"].values() for k, v in o.items()
                        if k in ("real", "gauss", "all2d", "all3d"))
    ok = [l for l in rows if abs(l["tru_t2"]) >= KEEP_MIN and abs(l["tru_g_t2"]) >= KEEP_MIN and fin(l)]

    def sign(v):
        return "bright" if v >= KEEP_MIN else "dark" if v <= -KEEP_MIN else "iso"
    bins = lambda d: "<4" if d < 4 else "4-6" if d < 6 else "6-8" if d < 8 else "8-12" if d < 12 else ">=12"
    return dict(n_kept=len(rows), n_measurable=len(ok), n_patients=len({l["patient"] for l in ok}),
                n_targets=len({l["target"] for l in ok}), by_target=_count([l["target"] for l in ok]),
                by_size_mm=_count([bins(l["diam_mm"]) for l in ok]), by_location=_count([l["location"] for l in ok]),
                by_pattern=_count([f"T1-{sign(l['tru_t1'])}/T2-{sign(l['tru_t2'])}" for l in ok]))


def atlas_table():
    """ATLAS stroke lesions placed in the targets (out_public/d3atlas_rows_2d.json)."""
    rows = json.load(open(f"{OUT}/d3atlas_rows_2d.json"))["rows"]
    kept = [r for r in rows if r["kept"]]
    ok = [r for r in kept if abs(r["tru"]) >= KEEP_MIN]
    return dict(n_placed=len(kept), n_measurable=len(ok), n_sessions=len({r["atlas_sid"] for r in ok}),
                n_sites=len({r["site"] for r in ok}), by_stratum=_count([r["stratum"] for r in ok]),
                by_target=_count([r["target"] for r in ok]),
                by_polarity=_count(["bright" if r["tru"] > 0 else "dark" for r in ok]))


def k_bins():
    """Physical-check thickness bin used for each lesion size (mm) under both pre-specified rules."""
    from exp.d1_physcheck import LESION_BIN_FWHM, LESION_BIN_DIAM
    return dict(fwhm={2 * r: b for r, b in LESION_BIN_FWHM.items()},
                diameter={2 * r: b for r, b in LESION_BIN_DIAM.items()})


def physcheck_table(reg="reg", cmin=0.05):
    """Sulcal-CSF instances per 3D thickness bin, T2: the rows that gave k (dev-val, test and Zenodo subjects)."""
    from exp.d1_physcheck import BIN_NAMES, CLASSES
    d = np.load(f"{SEG}/rows.npz")
    rows, cols = d["rows"], [str(c) for c in d["cols"]]
    j = cols.index
    out = {}
    for comp in ("HFC|D_HFC", "HFE|D_HFE", "V5|D_HFC", "V5|D_HFE"):
        u, o = comp.split("|")
        cx, cd = rows[:, j(f"x_{u}_{reg}")], rows[:, j(f"{o}_{reg}")]
        base = ((rows[:, 1] == 0) & (rows[:, 2] == CLASSES.index("sulcal_csf")) & (rows[:, 3] == 0)
                & (np.abs(rows[:, 9]) >= cmin) & np.isfinite(cx) & np.isfinite(cd))
        out[comp] = {bn: dict(n=int((base & (rows[:, 4] == b)).sum()),
                              n_subj=int(len(np.unique(rows[base & (rows[:, 4] == b), 0]))))
                     for b, bn in enumerate(BIN_NAMES)}
    return out


def _md_counts(d):
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def markdown(T):
    L = ["# Lesion instances per analysis", "",
         "Generated by `exp/lesion_table.py` from the analyses' own outputs. One instance is one inserted lesion at",
         "one site; every model of an analysis sees the same instances. Sizes are diameters 2r.", "",
         "## Matched lesions of the stage-wise analysis (all 36 models; seed, selection and k-scaling checks)", "",
         "| cohort | lesions | subjects | per subject | by size (mm) | by contrast | polarity | smallest subject x size cell | tissue at centre |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sp, t in T["stage"].items():
        L.append(f"| {sp} | {t['n']} | {t['n_subjects']} | {min(t['per_subject'].values())}-{max(t['per_subject'].values())} | "
                 f"{_md_counts(t['by_size_mm'])} | {_md_counts(t['by_contrast'])} | {_md_counts(t['by_polarity'])} | "
                 f"{t['min_cell_per_subject']} | {_md_counts(t.get('by_tissue', {}))} |")
    d = T["detect"]
    L += ["", "## Retention-map test lesions (`d3detect_T2.json`, absolute-output U-Net, seed 0)", "",
          f"{d['n']} lesions on {d['slices']} slices of 5 subjects (per subject: {_md_counts(d['per_subject'])}).",
          f"By size: {_md_counts(d['by_size_mm'])}. By contrast: {_md_counts(d['by_contrast'])}. All bright.",
          "Erased per subject (R < 0.25): " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in d["erased_per_subject"].items()) + ".",
          "", "## Cross-contrast sign patterns (`d3lesiontype_rows_*`, per route)", "",
          "| pattern, cohort | lesions | subjects | by size (mm) |", "|---|---|---|---|"]
    for k, t in T["lesiontype"].items():
        L.append(f"| {k} | {t['n']} | {t['n_subjects']} | {_md_counts(t['by_size_mm'])} |")
    m = T["realms"]
    L += ["", "## Real MS lesions (`d3reallesion_rows_s0.json`)", "",
          f"{m['n_kept']} kept placements, {m['n_measurable']} with measurable T2 contrast (|c| >= 0.03) from "
          f"{m['n_patients']} patients in {m['n_targets']} targets.",
          f"By size (equivalent diameter, mm): {_md_counts(m['by_size_mm'])}.",
          f"By location: {_md_counts(m['by_location'])}.",
          f"By sign pattern: {_md_counts(m['by_pattern'])}."]
    a = T["atlas"]
    L += ["", "## ATLAS stroke lesions (`d3atlas_rows_2d.json`)", "",
          f"{a['n_placed']} placements met the placement rules, {a['n_measurable']} carried measurable contrast, "
          f"from {a['n_sessions']} sessions at {a['n_sites']} sites.",
          f"By size stratum: {_md_counts(a['by_stratum'])}. By polarity: {_md_counts(a['by_polarity'])}."]
    kb = T["k_bins"]
    L += ["", "## Physical-check instances per thickness bin (sulcal CSF, T2, 3D, per-tile alignment; the instances that gave k)", "",
          "| comparison | " + " | ".join(f"{b} mm" for b in next(iter(T["physcheck"].values()))) + " |",
          "|---|" + "---|" * len(next(iter(T["physcheck"].values())))]
    for comp, bins in T["physcheck"].items():
        L.append(f"| {comp} | " + " | ".join(f"{v['n']} ({v['n_subj']} subj.)" for v in bins.values()) + " |")
    L += ["", f"Lesion size to k bin, primary FWHM rule: {_md_counts(kb['fwhm'])}; diameter rule: {_md_counts(kb['diameter'])}.",
          "Under both pre-specified rules 6 and 8 mm lesions take the same k, so k-scaled erasure cannot separate them.",
          "Retention and erasure without k-scaling can be reported for 6 and 8 mm separately: the matched test",
          f"lesions have at least {T['stage']['test']['min_cell_per_subject']} lesions per subject and size."]
    return "\n".join(L) + "\n"


def build():
    return dict(stage={sp: stage_table(sp, tissue=sp in ("test", "ext_on_hfe", "ext_v5")) for sp in STAGE_SPLITS},
                detect=detect_table(), lesiontype=lesiontype_table(), realms=realms_table(), atlas=atlas_table(),
                k_bins=k_bins(), physcheck=physcheck_table())


def main():
    T = build()
    json.dump(T, open(f"{OUT}/coauthor_lesions.json", "w"), indent=1, default=str)
    open(f"{OUT}/coauthor_lesions.md", "w").write(markdown(T))
    print(markdown(T))


if __name__ == "__main__":
    main()
