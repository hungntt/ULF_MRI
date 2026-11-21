"""How well does the synthetic-input retention agree with the acquired-input retention, per subject?

The closed loop measures retention through the learned operator D. The acquired-input arm replaces
the operator's rendering of the BACKGROUND by the real 64 mT scan and adds only D's local response
to the lesion (d3detreal.lesion_input). Agreement between the two arms therefore shows that the
enhancer is insensitive to a real versus synthetic background, not that D is physically right; this
script labels it "background insensitivity". It replaces the site-level Spearman, which pools
dependent lesions and says nothing about bias, with agreement statistics at the subject level.

Per cohort and operator, for each of the 36 zoo models, each subject has R_sim (mean over its
lesions of dp/dy in the synthetic arm) and R_acq (the same in the acquired arm), from the lesion
rows of d3stage.py. Reported: Lin's concordance correlation coefficient (CCC), and Bland-Altman bias
and 95% limits of agreement of R_acq - R_sim, over the pooled (model, subject) pairs, and over the 12
enhancer means. Intervals are cluster bootstraps with 10,000 replicates: subjects are resampled with
replacement and, within each drawn subject, lesion sites are resampled with replacement (the same
draw for every model, so pairing is kept). Resampling sites too adds noise to the arm means, which
biases the CCC downward, so the point estimate can sit near the top of its percentile interval;
`ci_subjects_only` resamples subjects alone and is reported beside it. The no-enhancer row is excluded (its two arms differ only
by the real background of the input residual). Cells: matched unit (HFC operator on the HFC test
scans, HFE operator on the HFE scans), cross unit, and the Zenodo cohort.

    ULF_OUT=out_public python exp/stats_agree.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")

OUT = os.environ.get("ULF_OUT", "./out")
FIG = os.environ.get("ULF_FIG", "./fig")
NOENH = "no enhancer"
B = 10000
CELLS = [("HFC", "test", "matched unit"), ("HFE", "ext_on_hfe", "matched unit"),
         ("HFE", "test", "cross unit"), ("HFC", "ext_on_hfe", "cross unit"),
         ("HFC", "ext_v5", "other site"), ("HFE", "ext_v5", "other site")]


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    return float(2 * ((x - mx) * (y - my)).mean() / (vx + vy + (mx - my) ** 2))


def ba(x, y):
    d = y - x
    return float(d.mean()), float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))


def pct(v):
    v = np.array(v); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def stats(Rs, Ra, enh_ix, sub_les, ns, rng, boot=B, nested=True):
    """Rs, Ra: (models, lesions). enh_ix: list of model-index arrays per enhancer.
    sub_les: list of lesion-index arrays per subject. -> point estimates and bootstrap draws."""
    def one(subs, les):
        # les: per drawn subject, lesion indices (already resampled)
        X = np.stack([Rs[:, l].mean(1) for l in les], 1)             # (models, draws)
        Y = np.stack([Ra[:, l].mean(1) for l in les], 1)
        pool = (ccc(X.ravel(), Y.ravel()),) + ba(X.ravel(), Y.ravel())
        Xe = np.array([X[i].mean() for i in enh_ix]); Ye = np.array([Y[i].mean() for i in enh_ix])
        # enhancer-level: mean over drawn subjects too
        return pool + (ccc(Xe, Ye),) + ba(Xe, Ye)
    pt = one(None, sub_les)
    reps = []
    for _ in range(boot):
        draw = rng.integers(0, ns, ns)
        les = [sub_les[s][rng.integers(0, len(sub_les[s]), len(sub_les[s]))] if nested else sub_les[s] for s in draw]
        reps.append(one(draw, les))
    return pt, np.array(reps)


def main():
    from exp import d3stage as S
    res = dict(meta=dict(label="background insensitivity (synthetic vs acquired background), not agreement with 64 mT",
                         bootstrap_reps=B, resampling="subjects, then lesion sites within subject",
                         units="R_acq - R_sim, retention units; Lin CCC", noenh_excluded=True),
               cells={})
    rng = np.random.default_rng(0)
    for op, split, kind in CELLS:
        if not os.path.exists(f"{OUT}/d3stage_rows_{split}_T2.npz"):
            continue
        z, m = S.load_rows(split)
        if not m["with_acq"]:
            continue
        dy = z["meta_dy"]; keep = z["meta_good"].astype(bool) & (np.abs(dy) > 0)
        models = [nm for nm in m["models"] if nm != NOENH]
        Rs = np.stack([z[f"dp|{op}|{nm}"][keep] / dy[keep] for nm in models]).astype(np.float64)
        Ra = np.stack([z[f"dpa|{op}|{nm}"][keep] / dy[keep] for nm in models]).astype(np.float64)
        sid = z["sid_idx"][keep]; ns = len(m["subjects"])
        sub_les = [np.where(sid == k)[0] for k in range(ns)]
        sub_les = [l for l in sub_les if len(l) >= 5]
        enh = [m["models"][nm][0] for nm in models]
        enh_ix = [np.array([i for i, e in enumerate(enh) if e == u]) for u in sorted(set(enh))]
        pt, reps = stats(Rs, Ra, enh_ix, sub_les, len(sub_les), rng)
        _, reps_s = stats(Rs, Ra, enh_ix, sub_les, len(sub_les), rng, nested=False)
        names = ["ccc", "bias", "loa_lo", "loa_hi"]
        cell = dict(operator=op, split=split, kind=kind, n_subjects=len(sub_les), n_models=len(models),
                    n_lesions=int(keep.sum()))
        for lvl, off in (("model_subject_pairs", 0), ("enhancer_means", 4)):
            cell[lvl] = {n: dict(est=float(pt[off + i]), ci=pct(reps[:, off + i]), ci_subjects_only=pct(reps_s[:, off + i])) for i, n in enumerate(names)}
        # no-enhancer reference row: the two arms differ only by the real background
        Rs0 = z[f"dp|{op}|{NOENH}"][keep] / dy[keep]; Ra0 = z[f"dpa|{op}|{NOENH}"][keep] / dy[keep]
        cell["no_enhancer_mean_R_sim_acq"] = [float(Rs0.mean()), float(Ra0.mean())]
        res["cells"][f"{op}|{split}"] = cell
        a, b, lo, hi = (cell["model_subject_pairs"][n] for n in names)
        e = cell["enhancer_means"]
        print(f"{op}|{split:10s} {kind:12s} subj={len(sub_les):2d}  pairs: CCC {a['est']:.3f} [{a['ci'][0]:.3f},{a['ci'][1]:.3f}] "
              f"[subj-only {a['ci_subjects_only'][0]:.3f},{a['ci_subjects_only'][1]:.3f}] bias {b['est']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}] LoA [{lo['est']:+.3f},{hi['est']:+.3f}] | "
              f"enhancer means: CCC {e['ccc']['est']:.3f} [{e['ccc']['ci'][0]:.3f},{e['ccc']['ci'][1]:.3f}] "
              f"bias {e['bias']['est']:+.3f}", flush=True)
        if (op, split) == ("HFC", "test"):
            keep_pairs = (Rs, Ra, sub_les, enh)
    # context: the site-level Spearman this replaces (one enhancer, HFC operator)
    p = f"{OUT}/d3external_summary.json"
    if os.path.exists(p):
        old = json.load(open(p))
        res["site_level_spearman_reference"] = {k: v["bright"]["closed_loop"] for k, v in old.items()
                                                if "closed_loop" in v.get("bright", {})}
    json.dump(res, open(f"{OUT}/stats_agree_T2.json", "w"), indent=1)
    try:
        figure(res, keep_pairs)
    except NameError:
        pass


def figure(res, pairs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Rs, Ra, sub_les, enh = pairs
    X = np.stack([Rs[:, l].mean(1) for l in sub_les], 1).ravel()
    Y = np.stack([Ra[:, l].mean(1) for l in sub_les], 1).ravel()
    c = res["cells"]["HFC|test"]["model_subject_pairs"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax[0].scatter(X, Y, s=8, alpha=0.6, color="#0072B2"); lim = [0, max(X.max(), Y.max()) * 1.05]
    ax[0].plot(lim, lim, "k:", lw=0.7); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("R, synthetic background (subject mean)", fontsize=8)
    ax[0].set_ylabel("R, acquired background (subject mean)", fontsize=8)
    ax[0].set_title(f"HFC, test: CCC {c['ccc']['est']:.2f} [{c['ccc']['ci'][0]:.2f}, {c['ccc']['ci'][1]:.2f}]", fontsize=8)
    mean, diff = (X + Y) / 2, Y - X
    ax[1].scatter(mean, diff, s=8, alpha=0.6, color="#0072B2")
    for v, ls in ((c["bias"]["est"], "-"), (c["loa_lo"]["est"], "--"), (c["loa_hi"]["est"], "--")):
        ax[1].axhline(v, color="k", lw=0.7, ls=ls)
    ax[1].set_xlabel("mean of the two arms", fontsize=8); ax[1].set_ylabel("acquired - synthetic", fontsize=8)
    ax[1].set_title(f"Bland-Altman: bias {c['bias']['est']:+.3f}", fontsize=8)
    for a in ax: a.tick_params(labelsize=7)
    fig.suptitle("background insensitivity of retention (model x subject pairs)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/agree_background_insensitivity.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
