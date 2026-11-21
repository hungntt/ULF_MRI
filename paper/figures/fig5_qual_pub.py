"""Qualitative figure for the MIDL paper (fig5_qual_pub_tikz.tex): the same lesion through three enhancers.

Columns: 3T reference, acquired 64 mT input, then pix2pix (best retention), the corpus-calibrated
U-Net (SynthSeg Dice's pick) and U-Net L1+SSIM (lowest retention), first-seed models from
out_public/. Rows: the enhanced whole slice, a zoom on the inserted lesion, and the lesion difference.
Every enhancer receives the ACQUIRED 64 mT slab of a locked OpenNeuro test subject plus D's local
response to the lesion (exp/d3detreal.lesion_input, the acquired-input arm of the agreement check),
so the slices are real outputs and only the lesion is synthetic.

The site is TYPICAL OF THE TEST SET, not a dramatic one: among the 40 interior sites of the slice
with the most uniform zoom window, the one whose retentions under the three enhancers are closest to
those enhancers' median retention for the same lesion (6 mm, clear) over the 200 test-set sites of
out_public/d3xarch_rows_T2.json. (Medians over this slice alone tie pix2pix and the corpus model,
which the test set does not.) SSIM and SynthSeg Dice in the labels are the shown models' means over
the 5 test subjects (out_public/d3zoo_T2.json, d3zoo_synthseg_T2.json).

    ULF_DATA=./datasets/npy_public ULF_OUT=./out_public \\
    python -I -B -u paper/figures/fig5_qual_pub.py
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ".")
sys.path.insert(0, HERE)
from ulfkit import data as D, train as T
from exp.d3detect import operator, blobs, site_pool
from exp.d3detreal import lesion_input
from exp import d3zoo as Z
from fig_tiles_pub import diverging, gray, brain_box, save, TILES

OUT = os.environ.get("ULF_OUT", "./out_public")
ENH = [("pix2pix", "cGAN / pix2pix s0", "base_pix2pix_T2", "unet_res", "best retention"),
       ("Corpus-calibrated", "corpus-calibrated s0", "d1_learned_cal_T2", "unet_res", "SynthSeg's pick"),
       ("U-Net $L_1$+SSIM", "U-Net L1+SSIM s0", "base_unet_ssim_T2", "unet_res", "lowest retention")]
R_PX, C, H = 3, 0.25, 16                 # 6 mm diameter, clear contrast, zoom half-width (px)
LIM_3T, LIM_IN, LIM_OUT = 0.25, 0.08, 0.10


@torch.no_grad()
def make_tiles():
    T.seed_all(0)
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    test = D.PairedCohort("test", "T2")
    sid = test.subjects[1]
    idx = test.subject_rows(sid); k = idx[len(idx) // 2]
    y = torch.from_numpy(test.hf[k:k + 1].astype(np.float32)).cuda()
    x = torch.from_numpy(test.ulf[k:k + 1].astype(np.float32)).cuda()
    m = (y[0, 0] > 0).cpu().numpy()
    fwd = operator("T2", "learned")
    nets = [Z.load(kind, ck) for _, _, ck, kind, _ in ENH]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p0 = [n(x).float().clamp(0, 1) for n in nets]

    def run(cy, cx):
        t = lambda v: torch.tensor([v], device="cuda")
        b = blobs(t(cy), t(cx), t(R_PX)) * (y > 0).float()
        yl = (y + C * b).clamp(0, 1)
        xl = lesion_input(x, y, yl, fwd, gen, t(cy), t(cx), t(R_PX))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl = [n(xl).float().clamp(0, 1) for n in nets]
        co, ri = b[0, 0] > 0.6, (b[0, 0] > 0.05) & (b[0, 0] < 0.25)
        cr = lambda d: (d[co].mean() - d[ri].mean()).item()
        dy = (yl - y)[0, 0]
        return [cr((q - q0)[0, 0]) / cr(dy) for q, q0 in zip(pl, p0)], yl, xl, pl

    yn = y[0, 0].cpu().numpy()
    clean = []
    for cy, cx in torch.nonzero(site_pool(y)[0]).tolist():
        if not (H < cy < 256 - H and H < cx < 256 - H): continue
        clean.append((float(yn[cy-H:cy+H, cx-H:cx+H].std()) + 2 * (1 - float(m[cy-H:cy+H, cx-H:cx+H].mean())), cy, cx))
    clean = sorted(clean)[:40]
    R = np.array([run(cy, cx)[0] for _, cy, cx in clean])            # (40 sites, 3 enhancers)
    rows = json.load(open(f"{OUT}/d3xarch_rows_T2.json"))
    arch = {"base_pix2pix_T2": "cGAN / pix2pix", "d1_learned_cal_T2": "D1 calibrated corpus",
            "base_unet_ssim_T2": "U-Net L1+SSIM"}
    med = np.array([np.median([q["retention"] for q in rows if q["arch"] == arch[ck] and q["r"] == R_PX
                               and q["c"] > 0.2]) for _, _, ck, _, _ in ENH])
    i = int(np.argmin(np.abs(R - med).sum(1)))
    _, cy, cx = clean[i]
    ret, yl, xl, pl = run(cy, cx)
    BB = brain_box(m)
    zb = (cy - H, cy + H, cx - H, cx + H)
    g = lambda a, lo=0.0, hi=0.8: gray(a.cpu().numpy(), lo, hi, bg=~m)
    gz = lambda a: gray(a.cpu().numpy(), 0.04, 0.62)
    save(g(yl[0, 0]), "q_3t_slice", BB); save(gz(yl[0, 0]), "q_3t_zoom", zb, up=12)
    save(diverging((yl - y)[0, 0].cpu().numpy(), LIM_3T), "q_3t_diff", zb, up=12)
    save(g(xl[0, 1]), "q_in_slice", BB); save(gz(xl[0, 1]), "q_in_zoom", zb, up=12)
    save(diverging((xl - x)[0, 1].cpu().numpy(), LIM_IN), "q_in_diff", zb, up=12)
    for j, q in enumerate(pl):
        save(g(q[0, 0]), f"q_e{j}_slice", BB); save(gz(q[0, 0]), f"q_e{j}_zoom", zb, up=12)
        save(diverging((q - p0[j])[0, 0].cpu().numpy(), LIM_OUT), f"q_e{j}_diff", zb, up=12)
    y0, y1, x0, x1 = BB
    meta = dict(sid=sid, row=int(k), site=[cy, cx], r_px=R_PX, c=C, retention=ret, medians=med.tolist(),
                selection="closest of the 40 cleanest sites to the test-set medians",
                zoom_in_slice=[(cx - H - x0) / (x1 - x0), (cx + H - x0) / (x1 - x0),
                               (cy - H - y0) / (y1 - y0), (cy + H - y0) / (y1 - y0)])
    z = json.load(open(f"{OUT}/d3zoo_T2.json")); ss = json.load(open(f"{OUT}/d3zoo_synthseg_T2.json"))["runs"]
    meta["scores"] = [dict(ssim=float(np.mean(z[run_][("ssim")])), dice=float(np.mean(ss[run_]["dice"])))
                      for _, run_, _, _, _ in ENH]
    json.dump(meta, open(f"{TILES}/q_meta.json", "w"), indent=1)
    print(f"  {sid} row {k}, site ({cy},{cx}); retention {', '.join(f'{r:+.2f}' for r in ret)}; "
          f"test-set medians {', '.join(f'{v:+.2f}' for v in med)}")
    return meta


def write_tikz(meta):
    W, TOP, BH = 15.24, 7.0, 0.66
    xs = [1.45 + 2.758 * (k + 0.5) for k in range(5)]
    S1, S2 = 2.45, 1.55
    r1 = TOP - BH - 0.10 - S1 / 2
    r2 = r1 - S1 / 2 - 0.14 - S2 / 2
    r3 = r2 - S2 / 2 - 0.12 - S2 / 2
    heads = [("3T reference", "", "q_3t"), ("64mT input", "acquired", "q_in")] + \
            [(n, role, f"q_e{j}") for j, (n, _, _, _, role) in enumerate(ENH)]
    zx0, zx1, zy0, zy1 = meta["zoom_in_slice"]
    L = [r"% Generated by figures/fig5_qual_pub.py (tiles in figures/tiles_pub/q_*.png, values in q_meta.json).",
         r"% Same lesion through three enhancers; every slice is a real output from the acquired 64mT scan.",
         r"\begingroup%", r"\renewcommand{\sffamily}{\fontfamily{phv}\selectfont}%",
         r"\renewcommand{\bfseries}{\fontseries{b}\selectfont}%", r"\def\qdir{figures/tiles_pub}%",
         r"\newcommand{\qL}{\fontsize{8}{9.6}\selectfont}%", r"\newcommand{\qM}{\fontsize{7}{8.4}\selectfont}%",
         r"\newcommand{\qS}{\fontsize{6}{7.0}\selectfont}%",
         r"\definecolor{qInk}{RGB}{0,0,0}\definecolor{qRule}{RGB}{185,185,185}\definecolor{qBand}{RGB}{241,240,232}%",
         r"\definecolor{qA}{RGB}{0,114,178}\definecolor{qBad}{RGB}{213,94,0}%   must match fig_tiles_pub.py",
         r"\newcommand{\qimg}[3]{%",
         r"  \IfFileExists{\qdir/#3}{%",
         r"    \begin{scope}\clip[rounded corners=1.2pt] ($(#1)-(#2/2,#2/2)$) rectangle ++(#2,#2);%",
         r"      \node[inner sep=0pt] at (#1) {\includegraphics[width=#2cm,height=#2cm]{\qdir/#3}};\end{scope}%",
         r"    \draw[qRule, line width=0.3pt, rounded corners=1.2pt] ($(#1)-(#2/2,#2/2)$) rectangle ++(#2,#2);}%",
         r"  {\draw[qRule, dash pattern=on 1.5pt off 1pt] ($(#1)-(#2/2,#2/2)$) rectangle ++(#2,#2);%",
         r"   \node[font=\sffamily\qS, align=center] at (#1) {missing\\\texttt{\detokenize{#3}}};}}%",
         r"\begin{tikzpicture}[x=1cm, y=1cm, font=\sffamily\qM, text=qInk,",
         r"  note/.style={inner sep=0pt, align=center, font=\sffamily\qS},",
         r"  noteL/.style={inner sep=0pt, align=left, font=\sffamily\qS}]",
         f"\\path[use as bounding box] (0,-0.62) rectangle ({W},{TOP + 0.02});",
         f"\\fill[qBand] (1.45,{TOP - BH:.3f}) rectangle ({W},{TOP});"]
    for x, (name, role, _) in zip(xs, heads):
        sub = f"\\\\[-0.5pt]{{\\fontseries{{m}}\\selectfont {role}}}" if role else ""
        L.append(f"\\node[note, font=\\sffamily\\bfseries\\qS] at ({x:.3f},{TOP - BH / 2:.3f}) {{{name}{sub}}};")
    for y, lab in ((r1, "Whole\\\\slice"), (r2, "Lesion,\\\\zoom"), (r3, "Lesion\\\\difference")):
        L.append(f"\\node[noteL, anchor=west, font=\\sffamily\\bfseries\\qS] at (0.02,{y:.3f}) {{{lab}}};")
    for x, (_, _, stem) in zip(xs, heads):
        L.append(f"\\qimg{{{x:.3f},{r1:.3f}}}{{{S1}}}{{{stem}_slice.png}}")
        bx0, by1 = x - S1 / 2 + zx0 * S1, r1 + S1 / 2 - zy0 * S1      # image rows grow downwards
        L.append(f"\\draw[white, line width=0.6pt] ({bx0:.3f},{by1:.3f}) rectangle ({x - S1 / 2 + zx1 * S1:.3f},{r1 + S1 / 2 - zy1 * S1:.3f});")
        L.append(f"\\qimg{{{x:.3f},{r2:.3f}}}{{{S2}}}{{{stem}_zoom.png}}")
        L.append(f"\\qimg{{{x:.3f},{r3:.3f}}}{{{S2}}}{{{stem}_diff.png}}")
    ylab = r3 - S2 / 2 - 0.16
    for j, (x, sc, r) in enumerate(zip(xs[2:], meta["scores"], meta["retention"])):
        rr = f"{r:.2f}".replace("-", "\\textminus ")
        L.append(f"\\node[note, anchor=north] at ({x:.3f},{ylab:.3f}) {{SSIM {sc['ssim']:.3f}, Dice {sc['dice']:.3f}\\\\"
                 f"$\\mathcal{{R}}$ = {rr} at this lesion}};")
    L += [r"\shade[left color=qA, right color=white] (0.02,-0.52) rectangle (0.62,-0.36);",
          r"\shade[left color=white, right color=qBad] (0.62,-0.52) rectangle (1.22,-0.36);",
          r"\draw[qRule, line width=0.3pt] (0.02,-0.52) rectangle (1.22,-0.36);",
          r"\node[noteL, anchor=west] at (1.34,-0.44) {lesioned minus lesion-free, scale $\pm$0.25 (3T), $\pm$0.08 (64mT), "
          r"$\pm$0.10 (enhanced). White box: zoom region.};",
          r"\end{tikzpicture}%", r"\endgroup"]
    open(f"{HERE}/fig5_qual_pub_tikz.tex", "w").write("\n".join(L) + "\n")
    print("  wrote fig5_qual_pub_tikz.tex")


if __name__ == "__main__":
    write_tikz(make_tiles())
