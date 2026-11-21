"""Data figures of the MIDL paper (public-data retrain), drawn from out_public/ at the MIDL
text width (15.24 cm), so nothing is scaled and fonts print at 5-7 pt.

    python -I -B paper/figures/make_plots_pub.py

Writes, next to this script:
    fig2_where_pub_tikz.tex     (a) retention at identical sites under six enhancers,
                                (b) AUROC for erasure per cohort: map, radius, QA, ensemble
    fig3_polarity_pub_tikz.tex  (a) one lesion, two T1 polarities (tiles from
                                fig_polarity_tiles_pub.py), (b) fitted T1 and T2 weights of
                                the synthesised T1 per cohort, (c) wrong-sign rate vs the
                                T1/T2 contrast ratio
    fig4_select_pub_tikz.tex    (a) SSIM and (b) SynthSeg Dice vs retention, 12 enhancers,
                                (c) lesions erased by the enhancer each score selects

Sources (all in ULF_OUT, default ./out_public):
    d3xarch_rows_T2.json, d3external_summary.json, d3external_rows_*.json, d3coupling.json,
    d3zoo{,_ext_on_hfe,_ext_v5}_T2.json, d3zoo_synthseg*_T2.json, d3zoo_stats*_T2.json,
    figures/tiles_pub/pol_meta.json.
Data panels carry no frame or heading band; the tile panel of Fig. 3 keeps its band.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tikzplot import preamble, close, Axes          # shared Nature-style scaffolding

OUT = os.environ.get("ULF_OUT", "./out_public")
W = 15.24                                             # MIDL \textwidth = 433.62 pt
COHORTS = [("test", "OpenNeuro"), ("ext_v5", "Zenodo"), ("pub3t_unc", "UNC"),
           ("pub3t_penn", "Penn (patients)"), ("pub3t_hip", "Hippocampal")]


def frame(ax, pfx, xt, yt, xf="{:g}", yf="{:g}"):
    """Axes.frame with Helvetica minus signs on negative tick labels."""
    return ax.frame(pfx, xt, yt, xf, yf).replace("{-", "{\\textminus ")


def J(name):
    return json.load(open(f"{OUT}/{name}.json"))


def legend(pfx, items, y, x0=0.0):
    """One line, no box: (kind, colour, label); kind = 'dot' | 'sq' | 'line' | 'ring'."""
    s = [r"\begin{scope}[key/.style={anchor=west, inner sep=0pt, text height=1.6ex, text depth=0.4ex}]"]
    x = x0
    for kind, col, txt, w in items:
        if kind == "sq":
            s.append(f"\\fill[{col}, rounded corners=0.8pt] ({x:.2f},{y - 0.085:.3f}) rectangle ++(0.35,0.17);")
            tx = x + 0.45
        elif kind == "line":
            s.append(f"\\draw[{col}, line width=1pt] ({x:.2f},{y:.3f}) -- ++(0.42,0);")
            tx = x + 0.52
        elif kind == "ring":
            s.append(f"\\node[dot, fill=white, draw={col}, line width=0.8pt] at ({x + 0.08:.2f},{y:.3f}) {{}};")
            tx = x + 0.22
        else:
            s.append(f"\\node[dot, fill={col}] at ({x + 0.08:.2f},{y:.3f}) {{}};")
            tx = x + 0.22
        s.append(f"\\node[key, font=\\sffamily\\{pfx}S] at ({tx:.2f},{y:.3f}) {{{txt}}};")
        x += w
    s.append(r"\end{scope}")
    return s


# ------------------------------------------------------------------ fig 2 ---
def fig2_where():
    pfx = "wh"
    TOP, TITLE = 4.35, 4.62
    s = [preamble(pfx, W, -0.62, TITLE + 0.26)]
    # ---- a: retention at 200 identical sites, six enhancers
    rows = J("d3xarch_rows_T2")
    archs = sorted({x["arch"] for x in rows})
    by = {a: [x for x in rows if x["arch"] == a] for a in archs}
    n = len(by[archs[0]])
    M = np.array([[by[a][i]["retention"] for a in archs] for i in range(n)])
    g = M.mean(); ss_tot = ((M - g) ** 2).sum()
    v_site = M.shape[1] * ((M.mean(1) - g) ** 2).sum() / ss_tot
    v_arch = M.shape[0] * ((M.mean(0) - g) ** 2).sum() / ss_tot
    order = np.argsort(M.mean(1))
    s.append(rf"\{pfx}head{{0}}{{{TITLE}}}{{a}}{{Where beats which}}")
    ax = Axes(0.95, 7.05, 0.78, TOP, 0.5, n + 0.5, -0.5, 1.25)
    s.append(frame(ax, pfx, [1, 100, 200], [0.0, 0.5, 1.0], "{:g}", "{:.1f}"))
    s.append(rf"\node[note] at ({(ax.x0 + ax.x1) / 2:.3f},0.30) {{Lesion site, ranked by mean retention ({n} sites)}};")
    s.append(rf"\node[note, rotate=90] at (0.18,{(ax.y0 + ax.y1) / 2:.3f}) {{Retention $\mathcal{{R}}$}};")
    s.append(f"\\draw[line, {pfx}Ink, dash pattern=on 1.4pt off 1.1pt] ({ax.x0:.3f},{ax.Y(0.25):.3f}) -- ({ax.x1:.3f},{ax.Y(0.25):.3f});")
    for k, i in enumerate(order):
        X = ax.X(k + 1)
        s.append(f"\\draw[line width=0.35pt, draw={pfx}Gray] ({X:.3f},{ax.Y(M[i].min()):.3f}) -- ({X:.3f},{ax.Y(M[i].max()):.3f});")
    pts = " ".join(f"({ax.X(k + 1):.3f},{ax.Y(M[i].mean()):.3f}) circle (0.022)" for k, i in enumerate(order))
    s.append(f"\\fill[{pfx}A] {pts};")
    s.append(rf"\node[noteL, anchor=north west] at ({ax.x0 + 0.12:.3f},{ax.y1:.3f}) {{Share of retention variance\\"
             rf"between sites: \textbf{{{100 * v_site:.0f}\%}}\\between enhancers: \textbf{{{100 * v_arch:.1f}\%}}\\"
             rf"dashed: erasure threshold, $\mathcal{{R}}$ = 0.25}};")
    # ---- b: AUROC for erasure per cohort
    s.append(rf"\{pfx}head{{7.75}}{{{TITLE}}}{{b}}{{Predicting erasure in every cohort}}")
    ext = J("d3external_summary")
    bx = Axes(10.15, 15.10, 0.78, TOP, 0.30, 1.00, 0, 1)
    s.append(frame(bx, pfx, [0.3, 0.5, 0.7, 0.9], [], "{:.1f}", "{:g}"))
    s.append(rf"\node[note] at ({(bx.x0 + bx.x1) / 2:.3f},0.30) {{AUROC for lesion erasure}};")
    s.append(f"\\draw[line, {pfx}Ink, dash pattern=on 1.4pt off 1.1pt] ({bx.X(0.5):.3f},{bx.y0:.3f}) -- ({bx.X(0.5):.3f},{TOP - 0.05:.3f});")
    s.append(f"\\node[note, anchor=south] at ({bx.X(0.5):.3f},{TOP - 0.02:.3f}) {{chance}};")
    step = (TOP - 0.30 - bx.y0) / len(COHORTS)
    for i, (sp, lab) in enumerate(COHORTS):
        y = TOP - 0.42 - i * step
        b = ext[sp]["bright"]["sim"]
        s.append(f"\\node[noteR, anchor=east] at ({bx.x0 - 0.12:.3f},{y:.3f}) {{{lab}}};")
        s.append(f"\\draw[line, {pfx}Rule] ({bx.x0:.3f},{y:.3f}) -- ({bx.x1:.3f},{y:.3f});")
        lo, hi = b["map_ci"]
        s.append(f"\\draw[line width=0.8pt, draw={pfx}A] ({bx.X(lo):.3f},{y:.3f}) -- ({bx.X(hi):.3f},{y:.3f});")
        for v, col in ((b["qa"], f"{pfx}Bad"), (b["ens"], f"{pfx}B"), (b["radius"], f"{pfx}Gray"), (b["map"][0], f"{pfx}A")):
            s.append(f"\\node[dot, fill={col}] at ({bx.X(v):.3f},{y:.3f}) {{}};")
        s.append(f"\\node[noteL, anchor=west] at ({bx.X(max(hi, b['radius'])) + 0.10:.3f},{y:.3f}) {{{b['map'][0]:.2f}}};")
    s += legend(pfx, [("dot", f"{pfx}A", "Retention map (95\\% CI, subjects)", 4.05),
                      ("dot", f"{pfx}Gray", "Lesion radius alone", 2.75),
                      ("dot", f"{pfx}Bad", "Intensity-error QA", 2.55),
                      ("dot", f"{pfx}B", "Deep-ensemble s.d.", 2.6),
                      ("line", f"{pfx}Gray", "Range over six enhancers (a)", 3.0)], -0.40, 0.0)
    s.append(close())
    return "\n".join(s), dict(v_site=v_site, v_arch=v_arch, n_sites=n)


# ------------------------------------------------------------------ fig 3 ---
TILE_PRE = r"""\definecolor{%(p)sBand}{RGB}{241,240,232}%%
\def\%(p)sdir{figures/tiles_pub}%%
\newcommand{\%(p)simg}[4]{%%
  \IfFileExists{\%(p)sdir/#4}{%%
    \begin{scope}%%
      \clip[rounded corners=1.2pt] ($(#2)-(#3/2,#3/2)$) rectangle ++(#3,#3);%%
      \node[inner sep=0pt] at (#2) {\includegraphics[width=#3cm,height=#3cm]{\%(p)sdir/#4}};%%
    \end{scope}%%
    \draw[%(p)sRule, line width=0.3pt, rounded corners=1.2pt] ($(#2)-(#3/2,#3/2)$) rectangle ++(#3,#3);}%%
  {\draw[%(p)sRule, dash pattern=on 1.5pt off 1pt] ($(#2)-(#3/2,#3/2)$) rectangle ++(#3,#3);%%
   \node[note] at (#2) {missing\\\texttt{\detokenize{#4}}};}}%%
"""


def fig3_polarity():
    pfx = "pz"
    TOP, TITLE = 4.25, 4.52
    BH = 0.52                                   # tile panel heading band
    pre = preamble(pfx, W, -0.62, TITLE + 0.26)
    pre = pre.replace(r"\begin{tikzpicture}", TILE_PRE % {"p": pfx} + r"\begin{tikzpicture}", 1)
    s = [pre]
    meta = json.load(open(f"{HERE}/tiles_pub/pol_meta.json"))
    # ---- a: tiles
    s.append(rf"\{pfx}head{{0}}{{{TITLE}}}{{a}}{{One lesion, two T1 polarities}}")
    s.append(f"\\fill[{pfx}Band] (0,{TOP - BH:.3f}) rectangle (6.95,{TOP:.3f});")
    T, xs = 1.18, [2.06 + 1.33 * k for k in range(4)]
    for x, t in zip(xs, ["3T T1\\\\(truth)", "64mT T1\\\\input", "64mT T2\\\\input", "Synthesised\\\\3T T1"]):
        s.append(f"\\node[lbl, font=\\sffamily\\bfseries\\{pfx}S] at ({x:.3f},{TOP - BH / 2:.3f}) {{{t}}};")
    for (tag, ytile, l1, l2) in (("canon", 2.98, "T1 dark,\\\\T2 bright", "oedema,\\\\infarct"),
                                 ("viol", 1.50, "T1 bright,\\\\T2 bright", "late-subacute\\\\blood, fat")):
        r = meta[tag]["retention"]
        s.append(f"\\node[noteL, anchor=west, font=\\sffamily\\bfseries\\{pfx}S] at (0.05,{ytile + 0.38:.3f}) {{{l1}}};")
        s.append(f"\\node[noteL, anchor=west] at (0.05,{ytile - 0.06:.3f}) {{{l2}}};")
        rr = f"{r:+.2f}".replace("-", "\\textminus ").replace("+", "+")
        s.append(f"\\node[noteL, anchor=west] at (0.05,{ytile - 0.45:.3f}) {{$\\mathcal{{R}}$ = {rr}}};")
        for x, f in zip(xs, ["dy1", "dx1", "dx2", "dp"]):
            s.append(rf"\{pfx}img{{}}{{{x:.3f},{ytile:.3f}}}{{{T}}}{{pol_{tag}_{f}.png}}")
    s.append(f"\\shade[left color={pfx}A, right color=white] (1.31,0.30) rectangle (1.91,0.46);")
    s.append(f"\\shade[left color=white, right color={pfx}Bad] (1.91,0.30) rectangle (2.51,0.46);")
    s.append(f"\\draw[{pfx}Rule, line width=0.3pt] (1.31,0.30) rectangle (2.51,0.46);")
    s.append(f"\\node[noteL, anchor=west] at (2.62,0.38) {{lesion darker / brighter than its surround}};")
    s.append(f"\\node[noteL, anchor=north west] at (1.31,0.20) {{Scale $\\pm$0.25 for 3T, $\\pm$0.08 for 64mT and synthesised}};")
    # ---- b: fitted weights per cohort
    c = J("d3coupling")["splits"]
    s.append(rf"\{pfx}head{{7.25}}{{{TITLE}}}{{b}}{{What the synthesised T1 follows}}")
    bx = Axes(9.10, 11.45, 0.78, TOP - 0.10, -0.30, 0.10, 0, 1)
    s.append(frame(bx, pfx, [-0.3, -0.2, -0.1, 0.0, 0.1], [], "{:.1f}", "{:g}"))
    s.append(f"\\draw[line, {pfx}Ink, dash pattern=on 1.4pt off 1.1pt] ({bx.X(0):.3f},{bx.y0:.3f}) -- ({bx.X(0):.3f},{bx.y1:.3f});")
    s.append(rf"\node[note] at ({(bx.x0 + bx.x1) / 2:.3f},0.30) {{Weight on the inserted contrast}};")
    step = (bx.y1 - 0.20 - bx.y0) / len(COHORTS)
    for i, (sp, lab) in enumerate(COHORTS):
        y = bx.y1 - 0.30 - i * step
        fC, fB = c[sp]["fit"]["C"], c[sp]["fit"]["B"]
        s.append(f"\\node[noteR, anchor=east] at ({bx.x0 - 0.12:.3f},{y:.3f}) {{{lab.split(' ')[0]}}};")
        s.append(f"\\draw[line, {pfx}Rule] ({bx.x0:.3f},{y:.3f}) -- ({bx.x1:.3f},{y:.3f});")
        for (v, ci, col, fill) in ((fB["b"], fB["b_ci"], f"{pfx}Bad", "white"),
                                   (fC["b"], fC["b_ci"], f"{pfx}Bad", f"{pfx}Bad"),
                                   (fC["a"], fC["a_ci"], f"{pfx}A", f"{pfx}A")):
            s.append(f"\\draw[line width=0.6pt, draw={col}] ({bx.X(ci[0]):.3f},{y:.3f}) -- ({bx.X(ci[1]):.3f},{y:.3f});")
            s.append(f"\\node[dot, fill={fill}, draw={col}, line width=0.6pt] at ({bx.X(v):.3f},{y:.3f}) {{}};")
    # ---- c: wrong-sign rate vs f (T2-bright lesions), routes A, B, C
    s.append(rf"\{pfx}head{{11.95}}{{{TITLE}}}{{c}}{{Wrong sign is a step}}")
    cx = Axes(12.70, 15.10, 0.78, TOP - 0.10, -1.08, 1.08, 0, 100)
    s.append(frame(cx, pfx, [-1, 0, 1], [0, 50, 100], "{:g}", "{:g}"))
    s.append(rf"\node[note] at ({(cx.x0 + cx.x1) / 2:.3f},0.30) {{T1 / T2 lesion contrast ratio $f$}};")
    s.append(rf"\node[note, rotate=90] at (12.12,{(cx.y0 + cx.y1) / 2:.3f}) {{Wrong sign (\%)}};")
    s.append(rf"\node[noteL, anchor=north west] at ({cx.X(-1.0):.3f},{cx.Y(72):.3f}) {{T2-bright\\lesions}};")
    fr = [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]
    for route, col, wd in (("A", f"{pfx}Gray", "0.8pt"), ("B", f"{pfx}Bad", "0.8pt"), ("C", f"{pfx}A", "1.0pt")):
        for sp, _ in COHORTS:                      # thin per-cohort traces
            ys = [100 * c[sp]["wrong"][f"+1|{f:+.2f}"][route] for f in fr]
            for seg in (range(0, 4), range(4, 8)):
                pts = " -- ".join(f"({cx.X(fr[k]):.3f},{cx.Y(ys[k]):.3f})" for k in seg)
                s.append(f"\\draw[line width=0.25pt, draw={col}!45] {pts};")
        mean = [np.mean([100 * c[sp]["wrong"][f"+1|{f:+.2f}"][route] for sp, _ in COHORTS]) for f in fr]
        for seg in (range(0, 4), range(4, 8)):
            pts = " -- ".join(f"({cx.X(fr[k]):.3f},{cx.Y(mean[k]):.3f})" for k in seg)
            s.append(f"\\draw[line width={wd}, draw={col}] {pts};")
    s += legend(pfx, [("dot", f"{pfx}A", "Route C (T1 and T2 in): T1 weight", 4.45),
                      ("dot", f"{pfx}Bad", "Route C: T2 weight", 2.55),
                      ("ring", f"{pfx}Bad", "Route B (T2 in): T2 weight", 3.4),
                      ("line", f"{pfx}Gray", "Route A (T1 in)", 2.2)], -0.40, 0.0)
    s.append(close())
    return "\n".join(s)


# ------------------------------------------------------------------ fig 4 ---
HILITE = {"cGAN / pix2pix": "A", "U-Net L1+SSIM": "Bad", "corpus-calibrated": "OK",
          "corpus, uncalibrated": "OK", "corpus, physics operator": "OK"}
PAIRED = [("", "OpenNeuro\\\\unit 1"), ("_ext_on_hfe", "OpenNeuro\\\\unit 2"), ("_ext_v5", "Zenodo")]


def enh_points(sfx, key):
    z = J(f"d3zoo{sfx}_T2")
    if key == "synthseg":
        ss = J(f"d3zoo_synthseg{sfx}_T2")["runs"]
        for nm in z:
            z[nm]["synthseg"] = ss[nm]["dice"]
    E = {}
    for v in z.values():
        E.setdefault(v["enhancer"], []).append((np.mean(v[key]), np.mean(v["ret"]), np.mean(v["erased"])))
    return {e: dict(x=np.mean([a for a, _, _ in p]), r=np.mean([b for _, b, _ in p]),
                    er=np.mean([c for _, _, c in p]), n=len(p),
                    sx=np.std([a for a, _, _ in p], ddof=1) if len(p) > 1 else 0.0,
                    sr=np.std([b for _, b, _ in p], ddof=1) if len(p) > 1 else 0.0) for e, p in E.items()}


def fig4_select():
    pfx = "sl"
    TOP, TITLE = 3.95, 4.22
    s = [preamble(pfx, W, -0.62, TITLE + 0.26)]
    st = J("d3zoo_stats_T2")
    for (x0, x1), key, xlab, title, letter in (((0.0, 4.85), "ssim", "SSIM (brain-masked)", "Image quality vs retention", "a"),
                                               ((5.15, 10.0), "synthseg", "SynthSeg Dice vs 3T", "Segmentation vs retention", "b")):
        P = enh_points("", key)
        rlo = min(p["r"] - p["sr"] for p in P.values()) - 0.01
        rhi = max(p["r"] + p["sr"] for p in P.values()) + 0.045   # headroom for the rho label
        lo = min(p["x"] - p["sx"] for p in P.values()); hi = max(p["x"] + p["sx"] for p in P.values())
        pad = 0.06 * (hi - lo)
        ax = Axes(x0 + 0.95, x1 - 0.10, 0.92, TOP, lo - pad, hi + pad, rlo, rhi)
        xt = [t for t in np.round(np.arange(0.40, 0.90, 0.04), 2) if lo - pad <= t <= hi + pad]
        yt = [t for t in (0.10, 0.15, 0.20, 0.25) if rlo <= t <= rhi]
        s.append(rf"\{pfx}head{{{x0}}}{{{TITLE}}}{{{letter}}}{{{title}}}")
        s.append(ax.frame(pfx, list(xt), yt, "{:.2f}", "{:.2f}"))
        s.append(rf"\node[note] at ({(ax.x0 + ax.x1) / 2:.3f},0.42) {{{xlab}}};")
        s.append(rf"\node[note, rotate=90] at ({x0 + 0.20:.3f},{(ax.y0 + ax.y1) / 2:.3f}) {{Retention $\mathcal{{R}}$}};")
        for e, p in sorted(P.items(), key=lambda kv: kv[0] in HILITE):
            X, Y = ax.X(p["x"]), ax.Y(p["r"])
            col = f"{pfx}{HILITE[e]}" if e in HILITE else f"{pfx}Gray"
            if p["n"] > 1:
                s.append(f"\\draw[ax] ({X:.3f},{ax.Y(p['r'] - p['sr']):.3f}) -- ({X:.3f},{ax.Y(p['r'] + p['sr']):.3f});")
                s.append(f"\\draw[ax] ({ax.X(p['x'] - p['sx']):.3f},{Y:.3f}) -- ({ax.X(p['x'] + p['sx']):.3f},{Y:.3f});")
                s.append(f"\\node[dot, fill={col}] at ({X:.3f},{Y:.3f}) {{}};")
            else:
                s.append(f"\\node[dot, fill=white, draw={col}, line width=0.8pt] at ({X:.3f},{Y:.3f}) {{}};")
        q = st[f"{key}~ret"]["enhancer"]
        fm = lambda v: f"{v:+.2f}".replace("-", "\\textminus ")
        lx, anc = (ax.x1, "north east") if letter == "a" else (ax.x0 + 0.10, "north west")
        s.append(rf"\node[noteL, anchor={anc}] at ({lx:.3f},{ax.y1:.3f}) "
                 rf"{{Spearman $\rho$ = {fm(q['rho'])}\\95\% CI {fm(q['lo'])} to {fm(q['hi'])}}};")
    # ---- c: erased by the enhancer each score selects
    s.append(rf"\{pfx}head{{10.30}}{{{TITLE}}}{{c}}{{Lesions erased by the pick}}")
    bx = Axes(11.15, 15.10, 0.92, TOP, -0.5, 2.5, 0, 100)
    s.append(bx.frame(pfx, [], [0, 50, 100], "{:g}", "{:g}"))
    s.append(rf"\node[note, rotate=90] at (10.55,{(bx.y0 + bx.y1) / 2:.3f}) {{Erased (\%)}};")
    bars = [("synthseg", f"{pfx}OK"), ("ssim", f"{pfx}Gray"), ("ret", f"{pfx}A")]
    bw = 0.30
    for i, (sfx, lab) in enumerate(PAIRED):
        stt = J(f"d3zoo_stats{sfx}_T2")
        P = enh_points(sfx, "ssim")
        s.append(f"\\node[note, below=1pt, align=center] at ({bx.X(i):.3f},{bx.y0 - 0.03:.3f}) {{{lab}}};")
        for j, (k, col) in enumerate(bars):
            v = 100 * P[stt["selection"][k]["pick"]]["er"]
            xc = bx.X(i) + (j - 1) * (bw + 0.04)
            s.append(f"\\fill[{col}] ({xc - bw / 2:.3f},{bx.y0:.3f}) rectangle ({xc + bw / 2:.3f},{bx.Y(v):.3f});")
            s.append(f"\\node[note, above=0.6pt, font=\\sffamily\\fontsize{{5}}{{6}}\\selectfont] at ({xc:.3f},{bx.Y(v):.3f}) {{{v:.0f}}};")
    s += legend(pfx, [("dot", f"{pfx}A", "pix2pix (best retention)", 3.35),
                      ("dot", f"{pfx}OK", "Corpus-pretrained (SynthSeg's pick)", 4.45),
                      ("dot", f"{pfx}Bad", "U-Net $L_1$+SSIM", 2.45),
                      ("dot", f"{pfx}Gray", "Other / SSIM's pick (c)", 3.0),
                      ("ring", f"{pfx}Gray", "single run", 1.6)], -0.40, 0.0)
    s.append(close())
    return "\n".join(s)


if __name__ == "__main__":
    body, info = fig2_where()
    open(f"{HERE}/fig2_where_pub_tikz.tex", "w").write(
        "% Generated by figures/make_plots_pub.py from out_public/ (see its docstring).\n" + body)
    print(f"  wrote fig2_where_pub_tikz.tex  (variance: sites {info['v_site']:.3f}, "
          f"enhancers {info['v_arch']:.3f}, {info['n_sites']} sites)")
    open(f"{HERE}/fig3_polarity_pub_tikz.tex", "w").write(
        "% Generated by figures/make_plots_pub.py; tiles from figures/fig_polarity_tiles_pub.py.\n" + fig3_polarity())
    print("  wrote fig3_polarity_pub_tikz.tex")
    open(f"{HERE}/fig4_select_pub_tikz.tex", "w").write(
        "% Generated by figures/make_plots_pub.py from out_public/ (see its docstring).\n" + fig4_select())
    print("  wrote fig4_select_pub_tikz.tex")
