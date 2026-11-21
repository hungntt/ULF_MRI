"""Image tiles for the MIDL polarity figure (fig3_polarity_pub_tikz.tex), public-data models.

The same 8 mm lesion is inserted at the same site of a locked OpenNeuro test subject, bright
in the 3T T2 in both rows. Row 1 makes it dark in the 3T T1 (as for oedema or infarct), row 2
bright (as for late-subacute blood or fat). Both 3T contrasts pass through the learned 64 mT
operators, and route C (64 mT T1 and T2 -> 3T T1, out_public/d2_bothtoT1.pt) synthesises the
3T T1. Tiles are lesioned-minus-lesion-free differences on symmetric diverging scales.

The site is the most TYPICAL clean site, not the cleanest: among the 40 interior sites with the
most uniform zoom window, the one whose two route-C retentions are closest to the medians over
all comparable test lesions in out_public/d3coupling_rows_test.json (T2-bright, 8 mm, |c| = 0.25
(core-minus-ring T2 contrast > 0.12),
f = -1 and +1). The cleanest site alone happened to invert the opposite way from the majority.

    ULF_DATA=./datasets/npy_public ULF_OUT=./out_public \\
    python -I -B -u paper/figures/fig_polarity_tiles_pub.py
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ".")
sys.path.insert(0, HERE)
from ulfkit import data as D, train as T
from exp.d3detect import operator, blobs, site_pool
from exp.d3pipe import d2_model, pack
from fig_tiles_pub import diverging, save, TILES

R_PX, C = 4, 0.25                  # 8 mm diameter, clear contrast
LIM_3T, LIM_IN = 0.25, 0.08        # 3T T1 scale; 64 mT inputs and the synthesised T1
H = 18                             # zoom half-width, px


@torch.no_grad()
def main():
    T.seed_all(0)
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    mc = D.MultiContrastCohort("test")
    sid = sorted(set(mc.sid))[0]
    pos = np.where(mc.sid == sid)[0]
    p = pos[len(pos) // 2: len(pos) // 2 + 1]
    _, hf = mc.batch(p)
    y1 = torch.from_numpy(hf[:, 0:1].astype(np.float32)).cuda()
    y2 = torch.from_numpy(hf[:, 1:2].astype(np.float32)).cuda()
    f1, f2 = operator("T1", "learned"), operator("T2", "learned")
    eC = d2_model("bothtoT1")
    x1_0, x2_0 = f1(y1, gen), f2(y2, gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p0 = eC(pack(x1_0, x2_0)).float().clamp(0, 1)
    ok = (site_pool(y1) & site_pool(y2))[0]
    m = (y1[0, 0] > 0).cpu().numpy()
    a, b_ = y1[0, 0].cpu().numpy(), y2[0, 0].cpu().numpy()
    clean = []
    for cy, cx in torch.nonzero(ok).tolist():
        if not (H < cy < 256 - H and H < cx < 256 - H): continue
        s = a[cy-H:cy+H, cx-H:cx+H].std() + b_[cy-H:cy+H, cx-H:cx+H].std() + 2 * (1 - m[cy-H:cy+H, cx-H:cx+H].mean())
        clean.append((s, cy, cx))
    clean = sorted(clean)[:40]
    rows = json.load(open(os.environ.get("ULF_OUT", "./out_public")
                          + "/d3coupling_rows_test.json"))
    med = {f: float(np.median([x["C"] / x["c1"] for x in rows if x["s2"] == 1 and x["r"] == R_PX
                               and x["f"] == f and abs(x["c2"]) > 0.12 and abs(x["c1"]) > 1e-4]))
           for f in (-1.0, 1.0)}
    assert all(np.isfinite(v) for v in med.values()), med   # c2 is core-minus-ring, ~0.17 for |c| = 0.25

    def ret_c(cy, cx, s1):
        bb = blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
                   torch.tensor([R_PX], device="cuda")) * (y1 > 0).float()
        y1l, y2l = (y1 + s1 * C * bb).clamp(0, 1), (y2 + C * bb).clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl = eC(pack(f1(y1l, gen), f2(y2l, gen))).float().clamp(0, 1)
        co, ri = bb[0, 0] > 0.6, (bb[0, 0] > 0.05) & (bb[0, 0] < 0.25)
        d1, dp = (y1l - y1)[0, 0], (pl - p0)[0, 0]
        return ((dp[co].mean() - dp[ri].mean()) / (d1[co].mean() - d1[ri].mean())).item()
    dist = sorted((abs(ret_c(cy, cx, -1.0) - med[-1.0]) + abs(ret_c(cy, cx, 1.0) - med[1.0]), cy, cx)
                  for _, cy, cx in clean)
    _, cy, cx = dist[0]
    print(f"  median route-C retention, test lesions: T1-dark {med[-1.0]:+.3f}, T1-bright {med[1.0]:+.3f}")
    box = (cy - H, cy + H, cx - H, cx + H)
    b = blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
              torch.tensor([R_PX], device="cuda")) * (y1 > 0).float()
    core, ring = b[0, 0] > 0.6, (b[0, 0] > 0.05) & (b[0, 0] < 0.25)
    cr = lambda d: (d[core].mean() - d[ring].mean()).item()
    meta = {"sid": sid, "site": [cy, cx], "r_px": R_PX, "c": C, "median_retention": med,
            "selection": "most typical of the 40 cleanest sites"}
    for tag, s1 in (("canon", -1.0), ("viol", +1.0)):
        y1l, y2l = (y1 + s1 * C * b).clamp(0, 1), (y2 + C * b).clamp(0, 1)
        x1l, x2l = f1(y1l, gen), f2(y2l, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl = eC(pack(x1l, x2l)).float().clamp(0, 1)
        dy1, dx1, dx2, dp = (y1l - y1)[0, 0], (x1l - x1_0)[0, 1], (x2l - x2_0)[0, 1], (pl - p0)[0, 0]
        ret = cr(dp) / cr(dy1)
        meta[tag] = dict(retention=ret, core_ring=dict(y1=cr(dy1), x1=cr(dx1), x2=cr(dx2), p=cr(dp)))
        save(diverging(dy1.cpu().numpy(), LIM_3T), f"pol_{tag}_dy1", box, up=12)
        save(diverging(dx1.cpu().numpy(), LIM_IN), f"pol_{tag}_dx1", box, up=12)
        save(diverging(dx2.cpu().numpy(), LIM_IN), f"pol_{tag}_dx2", box, up=12)
        save(diverging(dp.cpu().numpy(), LIM_IN), f"pol_{tag}_dp", box, up=12)   # synthesised: same scale as 64 mT
        print(f"  {tag}: retention {ret:+.3f}  core-ring 3T T1 {cr(dy1):+.3f}  64mT T1 {cr(dx1):+.4f}"
              f"  64mT T2 {cr(dx2):+.4f}  synth {cr(dp):+.3f}")
    json.dump(meta, open(f"{TILES}/pol_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
