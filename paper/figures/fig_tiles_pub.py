"""Image tiles for the MIDL overview figure (fig1_overview_pub_tikz.tex), public-data models.

Every tile is a measured result. The 3T reference comes from a locked OpenNeuro test subject,
the synthesised images from the enhancer the map audits (out_public/base_unet_noresid_T2.pt,
"U-Net absolute"), the retention map from out_public/d3detect_T2.pt queried at a fixed lesion
size and contrast, and the uncertainty map from the four deep-ensemble members
(out_public/d3qa_fold{0-3}_T2.pt), all on the same slice and on D's rendering of it.

    ULF_DATA=./datasets/npy_public ULF_OUT=./out_public \\
    python -I -u paper/figures/fig_tiles_pub.py

Palette: the diverging Okabe-Ito blue / vermillion and the viridis and magma ramps must stay
in sync with the \\definecolor and shading declarations in fig1_overview_pub_tikz.tex.
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from ulfkit import data as D, models as Mo, train as T
from exp.d3qa import enhancer
from exp.d3detect import operator, blobs, site_pool

OUT = os.environ.get("ULF_OUT", "./out_public")
TILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles_pub")
CONTRAST = "T2"
UP = 4

VERMILLION = np.array([213, 94, 0], dtype=np.float32)   # \definecolor{ovPos}{RGB}{213,94,0}
OKABE_BLUE = np.array([0, 114, 178], dtype=np.float32)  # \definecolor{ovNeg}{RGB}{0,114,178}


def diverging(x, lim):
    t = np.clip(x / lim, -1, 1)[..., None]
    pos, neg = np.where(t > 0, t, 0.0), np.where(t < 0, -t, 0.0)
    return (255.0 + pos * (VERMILLION - 255.0) + neg * (OKABE_BLUE - 255.0)).astype(np.uint8)


def ramp(x, name):
    import matplotlib
    return (matplotlib.colormaps[name](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def gray(x, lo=0.0, hi=0.8, bg=None):
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    rgb = (np.stack([x] * 3, -1) * 255).astype(np.uint8)
    if bg is not None:
        rgb[bg] = 255
    return rgb


def brain_box(mask, margin=5):
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + margin
    h = min(h, cy, cx, mask.shape[0] - cy, mask.shape[1] - cx)
    return (cy - h, cy + h, cx - h, cx + h)


def save(rgb, name, box=None, up=UP):
    if box is not None:
        y0, y1, x0, x1 = box
        rgb = rgb[y0:y1, x0:x1]
    im = Image.fromarray(rgb)
    im = im.resize((im.width * up, im.height * up), Image.LANCZOS)
    os.makedirs(TILES, exist_ok=True)
    im.save(f"{TILES}/{name}.png")
    print(f"  {name}.png  {im.width}x{im.height}")


@torch.no_grad()
def main():
    T.seed_all(0)
    test = D.PairedCohort("test", CONTRAST)
    enh = enhancer(); enh.load_state_dict(torch.load(f"{OUT}/base_unet_noresid_{CONTRAST}.pt")); enh.eval()
    det = Mo.UNet(6, 1, base=32, residual_from=None).cuda()
    det.load_state_dict(torch.load(f"{OUT}/d3detect_{CONTRAST}.pt")); det.eval()
    ens = []
    for f in range(4):
        n = enhancer(); n.load_state_dict(torch.load(f"{OUT}/d3qa_fold{f}_{CONTRAST}.pt")); n.eval()
        ens.append(n)
    fwd = operator(CONTRAST, "learned")
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)

    sid = test.subjects[0]
    idx = test.subject_rows(sid)
    k = idx[len(idx) // 2]
    print(f"subject {sid}, slice row {k}")
    y = torch.from_numpy(test.hf[k:k + 1].astype(np.float32)).cuda()
    xr = torch.from_numpy(test.ulf[k:k + 1].astype(np.float32)).cuda()
    m = (y[0, 0] > 0).cpu().numpy()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pr = enh(xr).float().clamp(0, 1)
    BB = brain_box(m)
    save(gray(y[0, 0].cpu().numpy(), bg=~m), "ov_3t", BB)
    save(gray(xr[0, 1].cpu().numpy(), bg=~m), "ov_ulf", BB)
    save(gray(pr[0, 0].cpu().numpy(), bg=~m), "ov_syn", BB)

    x0 = fwd(y, gen)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p0 = enh(x0).float().clamp(0, 1)
    R_PX, C = 3, 0.25                               # 6 mm diameter, clear contrast

    def retention_at(cy, cx):
        b = blobs(torch.tensor([cy], device="cuda"), torch.tensor([cx], device="cuda"),
                  torch.tensor([R_PX], device="cuda")) * (y > 0).float()
        yl = (y + C * b).clamp(0, 1)
        xl = fwd(yl, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl = enh(xl).float().clamp(0, 1)
        dy, dp = (yl - y)[0, 0], (pl - p0)[0, 0]
        co, ri = b[0, 0] > 0.6, (b[0, 0] > 0.05) & (b[0, 0] < 0.25)
        return ((dp[co].mean() - dp[ri].mean()) / (dy[co].mean() - dy[ri].mean())).item(), yl, xl, pl

    # identical lesions at two clean interior sites with opposite fates
    H = 20
    cand = torch.nonzero(site_pool(y)[0])
    yn = y[0, 0].cpu().numpy()
    clean = []
    for cy, cx in cand.tolist():
        if not (H < cy < 256 - H and H < cx < 256 - H): continue
        win, wm = yn[cy - H:cy + H, cx - H:cx + H], m[cy - H:cy + H, cx - H:cx + H]
        clean.append((float(win.std()) + 2.0 * (1.0 - float(wm.mean())), cy, cx))
    clean.sort()
    scored = sorted((retention_at(cy, cx)[0], cy, cx) for _, cy, cx in clean[:120][::2])
    lost, kept = scored[1], scored[-2]              # avoid the extreme outliers
    meta = {"sid": sid, "row": int(k), "r_px": R_PX, "c": C, "n_candidates": len(scored)}
    for tag, (r, cy, cx) in [("lost", lost), ("kept", kept)]:
        ret, yl, xl, pl = retention_at(cy, cx)
        box = (cy - H, cy + H, cx - H, cx + H)
        g = lambda t: gray(t.cpu().numpy(), 0.04, 0.62)
        save(g(yl[0, 0]), f"ov_{tag}_3t", box, up=10)
        save(g(xl[0, 1]), f"ov_{tag}_in", box, up=10)
        save(g(pl[0, 0]), f"ov_{tag}_out", box, up=10)
        save(diverging((yl - y)[0, 0].cpu().numpy(), 0.25), f"ov_{tag}_dy", box, up=10)
        save(diverging((pl - p0)[0, 0].cpu().numpy(), 0.25), f"ov_{tag}_dp", box, up=10)
        meta[tag] = dict(retention=ret, site=[cy, cx])
        print(f"  {tag}: retention {ret:.3f} at ({cy},{cx})")

    # retention map (faint 6 mm lesion) and ensemble uncertainty, same slice, same input
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = det(torch.cat([x0, p0, torch.full_like(p0, R_PX / 6.0), torch.full_like(p0, 0.12 / 0.25)], 1)
                ).float()[0, 0].cpu().numpy()
        sd = torch.stack([e(x0).float().clamp(0, 1) for e in ens]).std(0)[0, 0].cpu().numpy()
    VMAX_R = 0.3                                    # map range on this slice: about -0.06 to 0.28
    tile = ramp(np.clip(q, 0, VMAX_R) / VMAX_R, "viridis"); tile[~m] = 255
    save(tile, "ov_map6", BB)
    VMAX_U = float(np.percentile(sd[m], 99))
    tile = ramp(np.clip(sd, 0, VMAX_U) / VMAX_U, "magma"); tile[~m] = 255
    save(tile, "ov_unc", BB)
    rho = float(np.corrcoef(q[m], sd[m])[0, 1])
    meta.update(map_vmax=VMAX_R, unc_vmax=VMAX_U, map_vs_unc_corr=rho,
                map_range=[float(q[m].min()), float(q[m].max())])
    print(f"  map range {q[m].min():.3f}-{q[m].max():.3f} (shown 0-{VMAX_R}); ensemble s.d. "
          f"shown 0-{VMAX_U:.4f}; corr(map, s.d.) over brain {rho:+.2f}")
    json.dump(meta, open(f"{TILES}/ov_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
