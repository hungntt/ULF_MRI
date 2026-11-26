"""Ties at the erasure threshold.

Erasure is R < 0.25 (paper, Section 2). It checks whether moving to R <= 0.25 changes any number.
This script counts retention values exactly at (and within 1e-9 of) 0.10, 0.25 and 0.50 in every per-lesion
output, gives the erased fraction under both conventions, and records the sha256 of every script that applies
a retention threshold.

    python exp/audit_ties.py            -> out_public/audit_ties.json
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = "."
OUT = os.environ.get("ULF_OUT", f"{ROOT}/out_public")
THRESHOLDS = (0.10, 0.25, 0.50)
NEAR = 1e-9
KEY = re.compile(r"^(retention|ret|R)(_.*)?$")
THR_USE = re.compile(r"<=?\s*(0?\.25\b|0?\.1\b|0?\.5\b|THR\b|thr\b|a\.thr\b)")


def tie_counts(v, thr):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return dict(n=int(v.size), exact=int(np.sum(v == thr)), near=int(np.sum(np.abs(v - thr) < NEAR)),
                erased_lt=float(np.mean(v < thr)) if v.size else float("nan"),
                erased_le=float(np.mean(v <= thr)) if v.size else float("nan"))


def json_retentions(o):
    """Every number stored under a retention-like key (retention, ret, R, R_*), at any depth."""
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if KEY.match(str(k)) and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            elif KEY.match(str(k)) and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                out += [float(x) for x in v]
            else:
                out += json_retentions(v)
    elif isinstance(o, list):
        for v in o:
            out += json_retentions(v)
    return out


def npz_retentions(z):
    """(name, R) pairs: responses over the inserted contrast (d3stage style) or rec_*/tru_* pairs (d3robust)."""
    keys = list(z.keys()) if isinstance(z, dict) else list(z.files)
    if "meta_dy" in keys:
        dy = np.asarray(z["meta_dy"], dtype=np.float64)
        for k in keys:
            if k.split("|")[0] in ("dp", "dpa", "dd"):
                yield k, np.asarray(z[k], dtype=np.float64) / dy
    for k in keys:
        if k.startswith("rec_") and f"tru_{k[4:]}" in keys:
            yield k, np.asarray(z[k], dtype=np.float64) / np.asarray(z[f"tru_{k[4:]}"], dtype=np.float64)


def audit_file(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        vals = np.concatenate([r for _, r in npz_retentions(z)] or [np.zeros(0)])
    else:
        vals = np.asarray(json_retentions(json.load(open(path))), dtype=np.float64)
    return dict(path=os.path.relpath(path, ROOT), n=int(np.isfinite(vals).sum()),
                thr={f"{t:.2f}": tie_counts(vals, t) for t in THRESHOLDS})


def row_files():
    pats = ["d3detect_T2*.json", "*rows*.json", "*rows*.npz"]
    fs = sorted({f for p in pats for f in glob.glob(f"{OUT}/{p}")})
    return [f for f in fs if not f.endswith(".meta.json")]


def threshold_scripts():
    out = []
    for f in sorted(glob.glob(f"{ROOT}/exp/*.py") + glob.glob(f"{ROOT}/paper/figures/*.py")):
        if os.path.basename(f) == "audit_ties.py":
            continue
        lines = [i + 1 for i, s in enumerate(open(f, errors="ignore")) if THR_USE.search(s)]
        if lines:
            uses_le = any("<=" in open(f, errors="ignore").read().splitlines()[i - 1]
                          and re.search(r"<=\s*(0?\.25|THR|thr|a\.thr)\b", open(f, errors="ignore").read().splitlines()[i - 1])
                          for i in lines)
            out.append(dict(path=os.path.relpath(f, ROOT), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(),
                            lines=lines, uses_le_at_threshold=bool(uses_le)))
    return out


def main():
    files = [audit_file(f) for f in row_files()]
    tot = {t: dict(n=sum(f["thr"][t]["n"] for f in files), exact=sum(f["thr"][t]["exact"] for f in files),
                   near=sum(f["thr"][t]["near"] for f in files)) for t in (f"{x:.2f}" for x in THRESHOLDS)}
    changed = [f["path"] for f in files if any(abs(f["thr"][t]["erased_lt"] - f["thr"][t]["erased_le"]) > 0
                                                for t in f["thr"] if f["thr"][t]["n"])]
    res = dict(meta=dict(task="ties at the erasure threshold", convention="R < 0.25 (kept)",
                         near_tol=NEAR, thresholds=list(THRESHOLDS)),
               totals=tot, files_where_le_changes_a_rate=changed, files=files, scripts=threshold_scripts())
    json.dump(res, open(f"{OUT}/audit_ties.json", "w"), indent=1)
    print(f"{len(files)} per-lesion files, {tot['0.25']['n']} retention values")
    for t, v in tot.items():
        print(f"  thr {t}: exact ties {v['exact']}, within {NEAR:g} {v['near']}")
    print(f"  files where <= changes a rate: {changed or 'none'}")
    print(f"  scripts applying a threshold: {len(res['scripts'])}; using <= at a threshold: "
          f"{[s['path'] for s in res['scripts'] if s['uses_le_at_threshold']] or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
