"""Tables of the MIDL paper (public-data retrain), written from out_public/, and the numbers the
text quotes (pub_numbers.json), so every value in the paper can be traced to a result file.

    python -I -B paper/figures/make_tabs_pub.py

Writes, next to this script:
    tab_sites_pub.tex      cross-cohort results (main text)
    tab_polarity_pub.tex   wrong-sign rate by T1/T2 sign pattern, OpenNeuro test set (main text)
    tab_zoo_sites_pub.tex  every enhancer in every cohort (appendix)
    tab_coupling_pub.tex   fitted T1 / T2 weights of the synthesised T1, per cohort (appendix)
    tab_general_pub.tex    the map on unseen lesion families and enhancers, and within size (appendix)
    pub_numbers.json       the values quoted in the paper
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = os.environ.get("ULF_OUT", "./out_public")
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRED = [("test", "", "OpenNeuro"), ("ext_on_hfe", "_ext_on_hfe", "OpenNeuro"), ("ext_v5", "_ext_v5", "Zenodo")]
THREET = [("pub3t_unc", "_pub3t_unc", "UNC"), ("pub3t_penn", "_pub3t_penn", "Penn"), ("pub3t_hip", "_pub3t_hip", "Hippocampal")]
ALL = PAIRED + THREET
NAME = {"U-Net L1": "U-Net $L_1$", "U-Net L1+SSIM": "U-Net $L_1$+SSIM", "U-Net absolute": "U-Net, absolute output",
        "cGAN / pix2pix": "pix2pix", "LoHiResGAN": "LoHiResGAN", "corpus-calibrated": "Corpus, calibrated",
        "U-Net L1, long schedule": "U-Net $L_1$, long schedule", "corpus, uncalibrated": "Corpus, uncalibrated",
        "corpus, physics operator": "Corpus, physics operator", "Gaussian-NLL head": "Gaussian head",
        "quantile (CQR) head": "Quantile (CQR) head", "wavelet diffusion": "Wavelet diffusion"}
NUM = {}


def J(name):
    return json.load(open(f"{OUT}/{name}.json"))


def m(v):                                      # Helvetica-safe minus in text mode tables
    return f"{v:+.2f}".replace("-", "$-$")


def enh_table(sfx, synthseg=True):
    z = J(f"d3zoo{sfx}_T2")
    ss = J(f"d3zoo_synthseg{sfx}_T2")["runs"] if synthseg else {}
    E = {}
    for nm, v in z.items():
        E.setdefault(v["enhancer"], []).append(v | ({"dice": ss[nm]["dice"]} if nm in ss else {}))
    out = {}
    for e, rs in E.items():
        out[e] = dict(n=len(rs), ret=np.mean([np.mean(r["ret"]) for r in rs]),
                      er=100 * np.mean([np.mean(r["erased"]) for r in rs]),
                      er4=100 * np.mean([r["erased4"] for r in rs]),
                      ssim=np.mean([np.mean(r["ssim"]) for r in rs]) if "ssim" in rs[0] else None,
                      dice=np.mean([np.mean(r["dice"]) for r in rs]) if "dice" in rs[0] else None)
    inp = np.mean(ss["64 mT input (no enhancement)"]["dice"]) if synthseg else None
    return out, inp


def wrong(split_sfx):
    d = J(f"d3lesiontype{split_sfx}")
    w = lambda q, r: 100 * np.mean(d[q]["routes"][r]["inverted"])
    C = ["C_both", "C_bothtoT1_ctl", "C_bothtoT1_aug", "C_bothtoT1_augw"]
    same, usual = ("t1b_t2b", "t1d_t2d"), ("t1d_t2b", "t1b_t2d")
    return dict(A=[w(q, "A_T1in") for q in same + usual],
                B_broken=[w(q, "B_T2in") for q in same], B_usual=[w(q, "B_T2in") for q in usual],
                C_broken=[w(q, c) for q in same for c in C], C_usual=[w(q, c) for q in usual for c in C],
                table={q: dict(A=w(q, "A_T1in"), B=w(q, "B_T2in"), C=[w(q, c) for c in C]) for q in same + usual})


def rng(v):
    lo, hi = min(v), max(v)
    return f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}--{hi:.0f}"


# --------------------------------------------------------------- sites ----
def tab_sites():
    ext = J("d3external_summary")
    cols = []
    for split, sfx, lab in ALL:
        c = dict(lab=lab)
        b = ext[split]["bright"]
        c["map"], c["n"] = b["sim"]["map"][0], b["sim"]["n"]
        c["radius"], c["qa"], c["ens"] = b["sim"]["radius"], b["sim"]["qa"], b["sim"]["ens"]
        c["erased"] = 100 * b["sim"]["erased"]
        c["closed"] = b["closed_loop"]["spearman"] if "closed_loop" in b else None
        c["closed_diff"] = abs(b["closed_loop"]["ret_sim"] - b["closed_loop"]["ret_real"]) if "closed_loop" in b else None
        c["dark_map"] = ext[split]["dark"]["sim"]["map"][0]
        c["realin"] = ext[split]["dark"]["real"]["realin_map"] if "real" in ext[split]["dark"] else None
        lt = wrong("" if split in ("test", "ext_on_hfe") else sfx)
        c.update(B_broken=lt["B_broken"], C_broken=lt["C_broken"], C_usual=lt["C_usual"], B_usual=lt["B_usual"], A=lt["A"])
        if split in ("test", "ext_on_hfe", "ext_v5"):
            st = J(f"d3zoo_stats{sfx}_T2")
            E, inp = enh_table(sfx)
            pick = st["selection"]["synthseg"]["pick"]; best = st["selection"]["ret"]["pick"]
            c.update(ss_rho=st["synthseg~ret"]["enhancer"]["rho"], ss_rank=st["selection"]["synthseg"]["ret_rank"],
                     ssim_rho=st["ssim~ret"]["enhancer"]["rho"], ssim_rank=st["selection"]["ssim"]["ret_rank"],
                     psnr_rho=st["psnr~ret"]["enhancer"]["rho"], er_pick=E[pick]["er"], er_best=E[best]["er"],
                     er4_pick=E[pick]["er4"], er4_best=E[best]["er4"], pick=pick, best=best,
                     ssim_pick=st["selection"]["ssim"]["pick"],
                     inp=inp, dlo=min(e["dice"] for e in E.values()), dhi=max(e["dice"] for e in E.values()),
                     below=sum(e["dice"] <= inp for e in E.values()), n_enh=len(E))
        cols.append(c)
    # retention ranking vs OpenNeuro unit 1
    base, _ = enh_table("")
    names = sorted(base)
    from scipy.stats import spearmanr
    for c, (split, sfx, _) in zip(cols, ALL):
        E, _ = enh_table(sfx, synthseg=split in ("test", "ext_on_hfe", "ext_v5"))
        c["rank_rho"] = float(spearmanr([base[k]["ret"] for k in names], [E[k]["ret"] for k in names]).statistic)
    NUM["sites"] = cols

    same = lambda i: i == 1                      # OpenNeuro unit 2 shares unit 1's 3T scans
    def row(label, fn):
        cells = []
        for i, c in enumerate(cols):
            try:
                v = fn(c, i)
            except (KeyError, TypeError):
                v = "--"
            cells.append(v)
        return "\\hspace{0.6em}" + label + " & " + " & ".join(cells) + " \\\\"
    L = [r"% Generated by figures/make_tabs_pub.py from out_public/ (d3external_summary, d3lesiontype*, d3zoo*).",
         r"\begin{table}[t]", r"\floatconts{tab:sites}",
         r"{\caption{\textbf{Cross-cohort evaluation of focal signal preservation.} All models were trained on 15 "
         r"OpenNeuro subjects. Unit 2 shares the 3T scans of unit 1, so values computed from the 3T scans alone are "
         r"identical (=). The 3T-only cohorts have no 64mT scans, so their lesions pass through $D$ only. "
         r"Closed-loop $\rho$: Spearman correlation between retention with $D$'s rendering and with the acquired "
         r"64mT input. Wrong sign: lesions whose synthesised 3T T1 has the wrong sign (route C: four models). "
         r"Coupling broken: lesion bright in both T1 and T2 or dark in both. Pick: retention rank (of 12) of the "
         r"enhancer the score selects. Erased: lesions erased by the SynthSeg pick and by the retention-leading "
         r"enhancer.}}",
         r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{Paired 64mT and 3T}} & \multicolumn{3}{c}{\textbf{3T only}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Cohort} & \textbf{OpenNeuro} & \textbf{OpenNeuro} & \textbf{Zenodo} & \textbf{UNC} & \textbf{Penn} & \textbf{Hippoc.} \\",
         r"64mT unit & Unit 1 & Unit 2 & & -- & -- & -- \\",
         r"Test subjects & 5 & 5 & 11 & 10 & 23 & 20 \\", r"\midrule",
         r"\multicolumn{7}{@{}l}{\textbf{Simulation and lesion-retention map}} \\",
         row("Closed-loop $\\rho$", lambda c, i: f"{c['closed']:.2f}" if c["closed"] is not None else "--"),
         row("Map AUROC", lambda c, i: "=" if same(i) else f"{c['map']:.2f}"),
         row("Lesion radius AUROC", lambda c, i: "=" if same(i) else f"{c['radius']:.2f}"),
         row("QA, ensemble AUROC", lambda c, i: "=" if same(i) else f"{c['qa']:.2f}, {c['ens']:.2f}"),
         row("Lesion sites", lambda c, i: "=" if same(i) else f"{c['n']}"),
         r"\midrule", r"\multicolumn{7}{@{}l}{\textbf{Wrong sign (\%)}} \\",
         row("Route B, coupling broken", lambda c, i: "=" if same(i) else rng(c["B_broken"])),
         row("Route C, coupling broken", lambda c, i: "=" if same(i) else rng(c["C_broken"])),
         row("Route C, usual coupling", lambda c, i: "=" if same(i) else rng(c["C_usual"])),
         r"\midrule", r"\multicolumn{7}{@{}l}{\textbf{Global scores vs retention, 12 enhancers}} \\",
         row("SynthSeg Dice $\\rho$", lambda c, i: m(c["ss_rho"])),
         row("SynthSeg pick, rank", lambda c, i: f"{c['ss_rank']}"),
         row("Erased, pick / best (\\%)", lambda c, i: f"{c['er_pick']:.0f} / {c['er_best']:.0f}"),
         row("SSIM $\\rho$", lambda c, i: m(c["ssim_rho"])),
         row("SSIM pick, rank", lambda c, i: f"{c['ssim_rank']}"),
         row("PSNR $\\rho$", lambda c, i: m(c["psnr_rho"])),
         row("Ranking vs unit 1, $\\rho$", lambda c, i: "--" if i == 0 else "=" if i == 1 else f"{c['rank_rho']:.2f}"),
         r"\midrule", r"\multicolumn{7}{@{}l}{\textbf{SynthSeg Dice, 64mT input vs enhancers}} \\",
         row("Input", lambda c, i: f"{c['inp']:.3f}"),
         row("Enhancers", lambda c, i: f"{c['dlo']:.3f}--{c['dhi']:.3f}"),
         row("Not above input", lambda c, i: f"{c['below']} of {c['n_enh']}"),
         r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(f"{HERE}/tab_sites_pub.tex", "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------ polarity ----
def tab_polarity():
    t = wrong("")["table"]
    NUM["polarity_test"] = t
    lab = [("t1d_t2b", "Dark / bright", "White-matter lesion, oedema, infarct"),
           ("t1b_t2d", "Bright / dark", "Melanin, early-subacute blood"),
           ("t1b_t2b", "Bright / bright", "Late-subacute blood, fat"),
           ("t1d_t2d", "Dark / dark", "Acute blood, haemosiderin")]
    L = [r"% Generated by figures/make_tabs_pub.py from out_public/d3lesiontype.json (OpenNeuro test set).",
         r"\begin{table}[t]", r"\floatconts{tab:polarity}",
         r"{\caption{\textbf{Polarity inversion (\%) on the OpenNeuro test set, 200 lesions per cell.} A: 64mT T1 "
         r"input. B: 64mT T2 input. C: both inputs, mean [range] over four independently trained models. Examples "
         r"follow standard radiological descriptions.}}",
         r"{\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}llrrl@{}}", r"\toprule",
         r"\textbf{T1 / T2} & \textbf{Examples} & \textbf{A} & \textbf{B} & \textbf{C} \\", r"\midrule"]
    for q, a, b in lab:
        r = t[q]; broken = q in ("t1b_t2b", "t1d_t2d")
        B = f"\\textbf{{{r['B']:.0f}}}" if broken else f"{r['B']:.0f}"
        Cm = f"\\textbf{{{np.mean(r['C']):.0f}}}" if broken else f"{np.mean(r['C']):.0f}"
        L.append(f"{a} & {b} & {r['A']:.0f} & {B} & {Cm} [{min(r['C']):.0f}--{max(r['C']):.0f}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(f"{HERE}/tab_polarity_pub.tex", "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------- zoo, sites ----
def tab_zoo_sites():
    T = {split: enh_table(sfx, synthseg=split in ("test", "ext_v5"))[0] for split, sfx, _ in ALL if split != "ext_on_hfe"}
    inp = {split: enh_table(sfx)[1] for split, sfx in (("test", ""), ("ext_v5", "_ext_v5"))}
    order = sorted(T["test"], key=lambda e: -T["test"][e]["ret"])
    best = {(s, k): (max if k != "er" else min)(T[s][e][k] for e in order) for s in T for k in ("ret", "er", "dice") if T[s][order[0]][k] is not None}
    def cell(s, e, k, fmt):
        v = T[s][e][k]
        txt = fmt.format(v)
        return f"\\textbf{{{txt}}}" if abs(v - best[(s, k)]) < 1e-12 else txt
    L = [r"% Generated by figures/make_tabs_pub.py from out_public/d3zoo*_T2.json and d3zoo_synthseg*_T2.json.",
         r"\begin{table}[!htbp]", r"\floatconts{tab:zoo_sites}",
         r"{\caption{All 12 enhancers in every cohort, sorted by retention on the OpenNeuro test set. $\bar R$: mean "
         r"retention. Er.: lesions erased (\%). Dice: SynthSeg Dice on the acquired 64mT input. Seeds: trained "
         r"models averaged. The 3T-only cohorts have no 64mT input, so only $\bar R$ is reported. Bold: best value "
         r"in each column.}}",
         r"{\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccccccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{OpenNeuro} & \multicolumn{3}{c}{Zenodo} & UNC & Penn & Hippoc. \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}\cmidrule(lr){11-11}",
         r"Enhancer & Seeds & $\bar R$ & Er. & Dice & $\bar R$ & Er. & Dice & $\bar R$ & $\bar R$ & $\bar R$ \\", r"\midrule"]
    for e in order:
        cells = [NAME[e], f"{T['test'][e]['n']}"]
        for s in ("test", "ext_v5"):
            cells += [cell(s, e, "ret", "{:.3f}"), cell(s, e, "er", "{:.0f}"), cell(s, e, "dice", "{:.3f}")]
        for s in ("pub3t_unc", "pub3t_penn", "pub3t_hip"):
            cells.append(cell(s, e, "ret", "{:.3f}"))
        L.append(" & ".join(cells) + " \\\\")
    L += [r"\midrule", f"64mT input & -- & -- & -- & {inp['test']:.3f} & -- & -- & {inp['ext_v5']:.3f} & -- & -- & -- \\\\",
          r"\bottomrule", r"\end{tabular*}}", r"\end{table}"]
    open(f"{HERE}/tab_zoo_sites_pub.tex", "w").write("\n".join(L) + "\n")
    NUM["zoo_test"] = {e: dict(T["test"][e]) for e in order}


# ------------------------------------------------------------ coupling ----
def tab_coupling():
    c = J("d3coupling")
    NUM["coupling"] = c
    ns = [len(json.load(open(f"{OUT}/d3coupling_rows_{sp}.json"))) for sp in c["splits"]]
    aa = [c["splits"][sp]["fit"]["A"]["a"] for sp in c["splits"]]
    LES, AR = f"{min(ns)}--{max(ns)}", f"{min(aa):.2f}$--${max(aa):.2f}"
    lab = {"test": "OpenNeuro", "ext_v5": "Zenodo", "pub3t_unc": "UNC", "pub3t_penn": "Penn", "pub3t_hip": "Hippocampal"}
    ci = lambda q, k: f"{m(q[k])} [{m(q[k + '_ci'][0])}, {m(q[k + '_ci'][1])}]"
    L = [r"% Generated by figures/make_tabs_pub.py from out_public/d3coupling.json (exp/d3coupling.py).",
         r"\begin{table}[!htbp]", r"\floatconts{tab:coupling}",
         r"{\caption{Synthesised-T1 response fitted as $a\,c_1 + b\,c_2$ over the inserted T1 and T2 contrasts "
         r"$c_1$ and $c_2$, with 95\% subject-bootstrap intervals, from " + LES + r" lesion responses per cohort. "
         r"Route A has $a=" + AR + r"$ and $b=0.00$, and route B has $a=0.00$, in every cohort. Wrong sign: route C, "
         r"T2-bright lesions with $f=c_1/|c_2|$ from $0.25$ to $1$ (same sign) or $-1$ to $-0.25$ (opposite sign).}}",
         r"{\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccrc@{}}", r"\toprule",
         r" & B & \multicolumn{2}{c}{C} & & \multicolumn{1}{c}{Wrong sign, C (\%)} \\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){6-6}",
         r"Cohort & $b$ & $a$ & $b$ & $|b|/a$ & same / opposite \\", r"\midrule"]
    NUM["coupling_summary"] = {}
    for sp, nm in lab.items():
        s = c["splits"][sp]; F = s["fit"]
        same = [100 * s["wrong"][f"+1|{f:+.2f}"]["C"] for f in (0.25, 0.5, 0.75, 1.0)]
        opp = [100 * s["wrong"][f"+1|{f:+.2f}"]["C"] for f in (-1.0, -0.75, -0.5, -0.25)]
        n = len(json.load(open(f"{OUT}/d3coupling_rows_{sp}.json")))
        L.append(f"{nm} & {ci(F['B'], 'b')} & {ci(F['C'], 'a')} & {ci(F['C'], 'b')} & "
                 f"{abs(F['C']['b']) / F['C']['a']:.1f} & {rng(same)} / {rng(opp)} \\\\")
        NUM["coupling_summary"][sp] = dict(dom=abs(F["C"]["b"]) / F["C"]["a"], same=same, opp=opp, n=n)
    L += [r"\bottomrule", r"\end{tabular*}}", r"\end{table}"]
    open(f"{HERE}/tab_coupling_pub.tex", "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------- general ----
def tab_general():
    o = J("d3ood_T2"); x = J("d3xarch_T2")
    seeds = {t: J(f"d3detect_T2{t}") for t in ("", "_s1", "_s2", "_s3")}
    # high map score / large radius = retained, so they enter negated; high QA / ensemble s.d. = predicted error
    sign = {"detect": -1, "r": -1, "qa": 1, "ens": 1}
    auc = lambda rows, k: roc_auc_score([r["retention"] < 0.25 for r in rows], [sign[k] * r[k] for r in rows])
    main = [auc(v, "detect") for v in seeds.values()]
    within = {r: [auc([q for q in v if q["r"] == r], "detect") for v in seeds.values()] for r in (2, 3, 4, 6)}
    rows0 = seeds[""]
    NUM["map_test"] = dict(auroc=[float(np.mean(main)), float(np.std(main, ddof=1))], seeds=main,
                           radius=auc(rows0, "r"), qa=auc(rows0, "qa"), ens=auc(rows0, "ens"),
                           erased=100 * float(np.mean([r["retention"] < 0.25 for r in rows0])), n=len(rows0),
                           within={2 * r: [float(np.mean(v)), float(np.std(v, ddof=1))] for r, v in within.items()},
                           size_ret={2 * r: float(np.mean([q["retention"] for q in rows0 if q["r"] == r])) for r in (2, 3, 4, 6)})
    NUM["ood"] = o; NUM["xarch"] = x
    fam = [("iso   (training family)", "Isotropic (training family)"), ("elong (3:1 aspect)", "Elongated, aspect 3:1"),
           ("faint (c=0.06, unseen)", "Faint, $c=0.06$ (below range)"), ("ring  (annulus)", "Ring (annulus)"),
           ("big   (r=8, unseen)", "Large, 16\\,mm (above range)"), ("hypo  (negative)", "Dark (sign flipped)")]
    L = [r"% Generated by figures/make_tabs_pub.py from out_public/d3ood_T2.json, d3xarch_T2.json, d3detect_T2*.json.",
         r"\begin{table}[!htbp]", r"\floatconts{tab:general}",
         r"{\caption{The retention map beyond its training setting, OpenNeuro test set. Left: lesion families absent "
         r"from training (absolute-output U-Net). Middle: five enhancers the map was not trained on. Right: AUROC "
         r"within each lesion size, mean (s.d.) over four training seeds. The map was trained on isotropic bright "
         r"Gaussians of 4--12\,mm with contrasts 0.12 and 0.25.}}",
         r"{\scriptsize\setlength{\tabcolsep}{3pt}\begin{minipage}[t]{0.46\linewidth}\begin{tabular*}{\linewidth}[t]{@{\extracolsep{\fill}}lrrr@{}}", r"\toprule",
         r"Lesion family & $n$ & Er. (\%) & AUROC \\", r"\midrule"]
    for k, nm in fam:
        L.append(f"{nm} & {o[k]['n']} & {100 * o[k]['erased']:.0f} & {o[k]['auroc']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}\end{minipage}\hfill\begin{minipage}[t]{0.25\linewidth}\begin{tabular*}{\linewidth}[t]{@{\extracolsep{\fill}}lr@{}}", r"\toprule",
          r"Enhancer & AUROC \\", r"\midrule"]
    xn = {"U-Net L1": "U-Net $L_1$", "U-Net L1+SSIM": "U-Net $L_1$+SSIM", "cGAN / pix2pix": "pix2pix",
          "LoHiResGAN": "LoHiResGAN", "D1 calibrated corpus": "Corpus, calibrated"}
    for k, nm in xn.items():
        L.append(f"{nm} & {x[k]['auroc']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}\end{minipage}\hfill\begin{minipage}[t]{0.25\linewidth}\begin{tabular*}{\linewidth}[t]{@{\extracolsep{\fill}}lr@{}}", r"\toprule",
          r"Size & AUROC \\", r"\midrule"]
    for r, v in within.items():
        L.append(f"{2 * r}\\,mm & {np.mean(v):.3f} ({np.std(v, ddof=1):.3f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular*}\end{minipage}}", r"\end{table}"]
    open(f"{HERE}/tab_general_pub.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    tab_sites(); tab_polarity(); tab_zoo_sites(); tab_coupling(); tab_general()
    # a few more quoted values
    NUM["dfactor"] = J("d3dfactor_T2")
    NUM["synthsr"] = {"1mm": J("d3synthsr"), "native": J("d3synthsr_native")}
    NUM["polinfo_n"] = len(J("d3polinfo"))
    NUM["realret"] = J("d3realret_pub_loose")
    json.dump(NUM, open(f"{HERE}/pub_numbers.json", "w"), indent=1, default=float)
    print("  wrote tab_sites_pub, tab_polarity_pub, tab_zoo_sites_pub, tab_coupling_pub, tab_general_pub, pub_numbers.json")
    mt = NUM["map_test"]
    print(f"  map test AUROC {mt['auroc'][0]:.3f} +- {mt['auroc'][1]:.3f}; radius {mt['radius']:.3f}; QA {mt['qa']:.3f}; "
          f"ens {mt['ens']:.3f}; erased {mt['erased']:.0f}% of {mt['n']}; within {mt['within']}; size ret {mt['size_ret']}")
    for c in NUM["sites"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()
                    if k not in ("A", "B_broken", "C_broken", "C_usual", "B_usual")})
