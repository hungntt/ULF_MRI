"""
Unified MRI volume harmonization for ULF and 3T domains.

Pipeline:
  RAS reorient → SynthStrip (skull-strip + brain mask) → N4ITK bias correction
  (masked) → 1 mm isotropic resample → pad/crop to fixed in-plane size about
  mask centroid → brain-mask percentile clip + min-max intensity scaling.

No cross-domain registration for unpaired ULF↔3T. For paired data, share the
HF-derived frame (centroid + z-window) so LF/HF stay spatially matched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

TARGET_SPACING_MM = 1.0
TARGET_INPLANE = (256, 256)
DEFAULT_PERCENTILES = (0.5, 99.5)
DEFAULT_MIDDLE_FRACTION = 0.45
SYNTHSTRIP_ENV = "SYNTHSTRIP_BIN"
DEFAULT_SYNTHSTRIP_BIN = "mri_synthstrip"
# Repo Docker wrapper (Windows-friendly); used when FreeSurfer is not on PATH.
REPO_SYNTHSTRIP_DOCKER = (
    Path(__file__).resolve().parents[1] / "tools" / "synthstrip-docker.py"
)


@dataclass(frozen=True)
class HarmonizeReport:
    """Diagnostics for one volume after harmonization."""

    path: str
    orig_shape: tuple[int, ...]
    orig_zooms: tuple[float, ...]
    orig_ornt: tuple[str, ...]
    ras_shape: tuple[int, ...]
    resampled_shape: tuple[int, ...]
    mask_z_bbox: tuple[int, int]
    selected_z: tuple[int, int]
    n_slices: int
    centroid_xy: tuple[float, float]
    applied_n4: bool = True
    applied_synthstrip: bool = True


@dataclass(frozen=True)
class FrameSpec:
    """Shared in-plane crop + z-window for paired LF/HF framing."""

    centroid_xy: tuple[float, float]
    z_start: int
    z_end: int
    mask_z_bbox: tuple[int, int]
    target_inplane: tuple[int, int] = TARGET_INPLANE


def load_nifti(path: Union[str, Path]) -> nib.Nifti1Image:
    return nib.load(str(path))


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Force closest RAS canonical orientation."""
    return nib.as_closest_canonical(img)


def _is_python_wrapper(path: Path) -> bool:
    return path.suffix.lower() == ".py"


def resolve_synthstrip_bin(bin_path: str | Path | None = None) -> str:
    """
    Resolve SynthStrip executable or Python wrapper script.

    Order: explicit ``bin_path`` → ``SYNTHSTRIP_BIN`` env → ``mri_synthstrip`` on
    PATH → repo ``tools/synthstrip-docker.py`` if present.
    """
    candidates: list[str] = []
    if bin_path is not None:
        candidates.append(str(bin_path))
    env = os.environ.get(SYNTHSTRIP_ENV)
    if env:
        candidates.append(env)
    candidates.append(DEFAULT_SYNTHSTRIP_BIN)
    if REPO_SYNTHSTRIP_DOCKER.is_file():
        candidates.append(str(REPO_SYNTHSTRIP_DOCKER))

    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            if _is_python_wrapper(p) or os.access(p, os.X_OK):
                return str(p.resolve())
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError(
        "SynthStrip executable not found. Install FreeSurfer ≥7.3 "
        f"(mri_synthstrip), or use tools/synthstrip-docker.py, then pass "
        f"--synthstrip_bin / set {SYNTHSTRIP_ENV}. Tried: {candidates}"
    )


def synthstrip_command(exe: str | Path) -> list[str]:
    """Build argv prefix so ``.py`` wrappers run under the current interpreter."""
    path = Path(exe)
    if _is_python_wrapper(path):
        return [sys.executable, str(path.resolve())]
    return [str(exe)]


def synthstrip(
    img: nib.Nifti1Image,
    *,
    bin_path: str | Path | None = None,
    work_dir: Union[str, Path, None] = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Skull-strip via SynthStrip CLI (``mri_synthstrip -i … -o … -m …``).

    Returns ``(stripped_image, binary_mask)`` in the input geometry.
    """
    exe = resolve_synthstrip_bin(bin_path)
    own_tmpdir = None
    if work_dir is None:
        own_tmpdir = tempfile.TemporaryDirectory(prefix="ulf_synthstrip_")
        work = Path(own_tmpdir.name)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        in_path = work / "input.nii.gz"
        out_path = work / "stripped.nii.gz"
        mask_path = work / "mask.nii.gz"
        nib.save(img, str(in_path))
        cmd = [
            *synthstrip_command(exe),
            "-i",
            str(in_path),
            "-o",
            str(out_path),
            "-m",
            str(mask_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to launch SynthStrip ({exe}): {exc}") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"SynthStrip failed (exit {proc.returncode}): {err or 'no output'}"
            )
        if not out_path.is_file() or not mask_path.is_file():
            raise RuntimeError(
                "SynthStrip completed but did not write stripped image and mask"
            )
        stripped = nib.load(str(out_path))
        mask = nib.load(str(mask_path))
        # Re-load into memory so temp files can be deleted.
        stripped = nib.Nifti1Image(
            np.ascontiguousarray(stripped.get_fdata(dtype=np.float32)),
            stripped.affine,
            stripped.header,
        )
        mask_data = np.ascontiguousarray(mask.get_fdata(dtype=np.float32))
        mask = nib.Nifti1Image((mask_data > 0.5).astype(np.uint8), mask.affine, mask.header)
        return stripped, mask
    finally:
        if own_tmpdir is not None:
            own_tmpdir.cleanup()


def _nib_to_sitk(img: nib.Nifti1Image):
    """Array-only SimpleITK view (spacing from header; identity direction)."""
    import SimpleITK as sitk

    data = np.ascontiguousarray(np.asarray(img.dataobj, dtype=np.float32))
    if data.ndim > 3:
        data = data[..., 0]
    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    sitk_img.SetSpacing(zooms)
    return sitk.Cast(sitk_img, sitk.sitkFloat32)


def _sitk_to_nib(sitk_img, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    """Write SimpleITK intensities back into the reference nibabel geometry."""
    import SimpleITK as sitk

    arr_zyx = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    data = np.transpose(arr_zyx, (2, 1, 0))
    out = nib.Nifti1Image(data, reference.affine, reference.header)
    return out


def n4_bias_correct(
    img: nib.Nifti1Image,
    *,
    mask: nib.Nifti1Image | None = None,
    shrink_factor: int = 4,
) -> nib.Nifti1Image:
    """
    N4 bias-field correction via SimpleITK (N4ITK).

    Estimates the field on a shrinked grid (default 4×), then applies it at
    full resolution. When ``mask`` is provided (e.g. SynthStrip brain mask),
    it is used instead of Otsu thresholding.
    """
    import SimpleITK as sitk

    sitk_img = _nib_to_sitk(img)
    work = sitk_img
    if shrink_factor > 1:
        work = sitk.Shrink(sitk_img, [int(shrink_factor)] * sitk_img.GetDimension())

    if mask is not None:
        sitk_mask_full = _nib_to_sitk(mask)
        sitk_mask_full = sitk.BinaryThreshold(
            sitk_mask_full, lowerThreshold=0.5, upperThreshold=1e12, insideValue=1, outsideValue=0
        )
        if shrink_factor > 1:
            sitk_mask = sitk.Resample(
                sitk_mask_full,
                work,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk_mask_full.GetPixelID(),
            )
        else:
            sitk_mask = sitk_mask_full
    else:
        sitk_mask = sitk.OtsuThreshold(work, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Lightweight schedule: enough to remove slow bias, fast enough for batch prep.
    corrector.SetMaximumNumberOfIterations([30, 20, 10])
    corrector.SetConvergenceThreshold(1e-3)
    corrector.Execute(work, sitk_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
    corrected = sitk.Divide(sitk_img, sitk.Exp(log_bias))
    return _sitk_to_nib(corrected, img)


def resample_to_isotropic(
    img: nib.Nifti1Image,
    spacing_mm: float = TARGET_SPACING_MM,
    *,
    order: int = 1,
) -> nib.Nifti1Image:
    """Resample to isotropic ``spacing_mm`` using nibabel (linear by default)."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    target_zooms = np.array([spacing_mm, spacing_mm, spacing_mm], dtype=np.float64)
    target_shape = np.maximum(1, np.round(shape * zooms / target_zooms).astype(int))

    # Build target affine with same origin/orientation, new zooms.
    aff = np.asarray(img.affine, dtype=np.float64).copy()
    rotation = aff[:3, :3] / zooms
    aff[:3, :3] = rotation * target_zooms

    target = nib.Nifti1Image(
        np.zeros(tuple(int(s) for s in target_shape), dtype=np.float32),
        aff,
    )
    return resample_from_to(img, target, order=order)


def soft_foreground_mask(vol: np.ndarray) -> np.ndarray:
    """
    Soft tissue/foreground mask for geometry fallback (not skull-strip).

    Uses a low percentile of positive voxels so air/padding is excluded from
    z-bbox / centroid without tightly carving brain tissue.
    """
    data = np.asarray(vol, dtype=np.float32)
    positive = data[data > 0]
    if positive.size < 16:
        return data > 0
    thr = float(np.percentile(positive, 5))
    thr = max(thr, float(np.percentile(positive, 1)) * 0.5)
    mask = data >= thr
    # Drop near-empty axial slices (noise) while keeping the main tissue slab.
    if mask.any():
        per_z = mask.sum(axis=(0, 1))
        max_z = float(per_z.max()) if per_z.size else 0.0
        if max_z > 0:
            keep = per_z >= 0.005 * max_z
            mask = mask.copy()
            mask[:, :, ~keep] = False
    return mask


def mask_z_bbox(mask: np.ndarray) -> tuple[int, int]:
    """Inclusive ``[z_min, z_max]`` of True voxels along axis 2."""
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if zs.size == 0:
        raise ValueError("Foreground mask is empty; cannot compute z-bbox")
    return int(zs[0]), int(zs[-1])


def middle_window_in_bbox(
    z_min: int,
    z_max: int,
    middle_fraction: float,
) -> tuple[int, int]:
    """
    Apply a central ``middle_fraction`` window inside inclusive ``[z_min, z_max]``.

    Returns half-open ``[start, end)`` absolute indices.
    """
    depth = z_max - z_min + 1
    if depth <= 0:
        raise ValueError(f"Invalid z-bbox [{z_min}, {z_max}]")
    if not 0.0 < middle_fraction <= 1.0:
        raise ValueError(f"middle_fraction must be in (0, 1], got {middle_fraction}")
    n_keep = max(1, int(round(depth * middle_fraction)))
    start_rel = (depth - n_keep) // 2
    end_rel = start_rel + n_keep
    return z_min + start_rel, z_min + end_rel


def mask_centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    """Centroid of True voxels in (x, y); axes 0 and 1."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        h, w = mask.shape[0], mask.shape[1]
        return (h - 1) / 2.0, (w - 1) / 2.0
    cy = float(coords[:, 0].mean())
    cx = float(coords[:, 1].mean())
    return cy, cx


def pad_crop_inplane_about_center(
    vol: np.ndarray,
    center_yx: tuple[float, float],
    target_hw: tuple[int, int] = TARGET_INPLANE,
) -> np.ndarray:
    """Pad/crop spatial axes (0, 1) to ``target_hw``, centered on ``center_yx``."""
    th, tw = target_hw
    h, w = vol.shape[0], vol.shape[1]
    cy, cx = center_yx
    y0 = int(round(cy - (th - 1) / 2.0))
    x0 = int(round(cx - (tw - 1) / 2.0))
    y1 = y0 + th
    x1 = x0 + tw

    out_shape = (th, tw, *vol.shape[2:])
    out = np.zeros(out_shape, dtype=vol.dtype)

    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(h, y1)
    src_x1 = min(w, x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1, ...] = vol[src_y0:src_y1, src_x0:src_x1, ...]
    return out


def normalize_foreground(
    vol: np.ndarray,
    mask: np.ndarray,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> np.ndarray:
    """
    Percentile clip + min-max to [0, 1] using foreground voxels for stats only.

    Scaling is applied to the full volume (no hard zeroing of non-mask voxels),
    so this is intensity harmonization — not skull-stripping.
    """
    data = np.asarray(vol, dtype=np.float32)
    fg = mask.astype(bool)
    if not fg.any():
        raise ValueError("Cannot normalize: empty foreground mask")
    vals = data[fg]
    lo, hi = np.percentile(vals, list(percentiles))
    if hi <= lo:
        out = np.zeros_like(data, dtype=np.float32)
        out[fg] = 1.0
        return out
    clipped = np.clip(data, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _as_float32_volume(img: nib.Nifti1Image) -> np.ndarray:
    vol = np.ascontiguousarray(np.asarray(img.dataobj, dtype=np.float32))
    if vol.ndim > 3:
        vol = vol[..., 0]
    return vol


def _as_bool_mask(img: nib.Nifti1Image) -> np.ndarray:
    data = np.ascontiguousarray(np.asarray(img.dataobj, dtype=np.float32))
    if data.ndim > 3:
        data = data[..., 0]
    return data > 0.5


def prepare_volume_geometry(
    path: Union[str, Path],
    *,
    target_spacing_mm: float = TARGET_SPACING_MM,
    apply_n4: bool = True,
    apply_synthstrip: bool = True,
    synthstrip_bin: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    RAS → SynthStrip → N4 → 1 mm isotropic.

    Returns ``(vol, mask, meta)`` before in-plane framing / intensity norm.
    """
    path = Path(path)
    img0 = load_nifti(path)
    meta = {
        "path": str(path),
        "orig_shape": tuple(int(s) for s in img0.shape[:3]),
        "orig_zooms": tuple(float(z) for z in img0.header.get_zooms()[:3]),
        "orig_ornt": tuple(str(c) for c in nib.orientations.aff2axcodes(img0.affine)),
        "applied_n4": bool(apply_n4),
        "applied_synthstrip": bool(apply_synthstrip),
    }

    img = reorient_to_ras(img0)
    meta["ras_shape"] = tuple(int(s) for s in img.shape[:3])

    brain_mask_img: nib.Nifti1Image | None = None
    if apply_synthstrip:
        img, brain_mask_img = synthstrip(img, bin_path=synthstrip_bin)

    if apply_n4:
        img = n4_bias_correct(img, mask=brain_mask_img)

    img = resample_to_isotropic(img, spacing_mm=target_spacing_mm, order=1)
    vol = _as_float32_volume(img)

    if brain_mask_img is not None:
        mask_img = resample_to_isotropic(brain_mask_img, spacing_mm=target_spacing_mm, order=0)
        mask = _as_bool_mask(mask_img)
        if mask.shape != vol.shape:
            raise ValueError(
                f"Resampled mask shape {mask.shape} != volume shape {vol.shape} for {path}"
            )
        if not mask.any():
            mask = soft_foreground_mask(vol)
    else:
        mask = soft_foreground_mask(vol)

    meta["resampled_shape"] = tuple(int(s) for s in vol.shape)
    return vol, mask, meta


def frame_and_normalize(
    vol: np.ndarray,
    mask: np.ndarray,
    *,
    frame: FrameSpec | None = None,
    target_inplane: tuple[int, int] = TARGET_INPLANE,
    middle_fraction: float = DEFAULT_MIDDLE_FRACTION,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    select_z_band: bool = True,
) -> tuple[np.ndarray, np.ndarray, FrameSpec]:
    """
    Pad/crop about mask centroid and intensity-normalize with brain-mask stats.

    If ``frame`` is given (paired LF following HF), reuse that centroid and
    z-window instead of deriving them from ``mask``.
    When ``select_z_band`` is True, return only the middle z-window volume;
    otherwise return the full framed depth (caller selects slices).
    """
    if frame is None:
        z_min, z_max = mask_z_bbox(mask)
        z_start, z_end = middle_window_in_bbox(z_min, z_max, middle_fraction)
        cy, cx = mask_centroid_xy(mask)
        frame = FrameSpec(
            centroid_xy=(cy, cx),
            z_start=z_start,
            z_end=z_end,
            mask_z_bbox=(z_min, z_max),
            target_inplane=target_inplane,
        )
    else:
        target_inplane = frame.target_inplane

    vol_f = pad_crop_inplane_about_center(vol, frame.centroid_xy, target_inplane)
    mask_f = (
        pad_crop_inplane_about_center(mask.astype(np.float32), frame.centroid_xy, target_inplane)
        > 0.5
    )

    if select_z_band:
        z0, z1 = frame.z_start, frame.z_end
        # Clamp if this volume is shallower than the shared frame (rare for paired).
        z1 = min(z1, vol_f.shape[2])
        z0 = min(max(0, z0), z1)
        vol_out = vol_f[:, :, z0:z1]
        mask_out = mask_f[:, :, z0:z1]
    else:
        vol_out = vol_f
        mask_out = mask_f

    if not mask_out.any():
        mask_out = soft_foreground_mask(vol_out)

    normed = normalize_foreground(vol_out, mask_out, percentiles=percentiles)
    return normed, mask_out, frame


def harmonize_volume(
    path: Union[str, Path],
    *,
    target_spacing_mm: float = TARGET_SPACING_MM,
    target_inplane: tuple[int, int] = TARGET_INPLANE,
    middle_fraction: float = DEFAULT_MIDDLE_FRACTION,
    apply_n4: bool = True,
    apply_synthstrip: bool = True,
    synthstrip_bin: str | Path | None = None,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    frame: FrameSpec | None = None,
    select_z_band: bool = True,
) -> tuple[np.ndarray, np.ndarray, HarmonizeReport, FrameSpec]:
    """
    Full harmonization → framed float32 volume in [0, 1] plus brain mask.

    Returns ``(vol, mask, report, frame)``. When ``select_z_band`` is True,
    ``vol`` has shape ``(H, W, n_z)`` for the middle window only.
    Pass ``frame`` from an HF pass to keep paired LF spatially matched.
    """
    vol, mask, meta = prepare_volume_geometry(
        path,
        target_spacing_mm=target_spacing_mm,
        apply_n4=apply_n4,
        apply_synthstrip=apply_synthstrip,
        synthstrip_bin=synthstrip_bin,
    )
    normed, mask_out, used_frame = frame_and_normalize(
        vol,
        mask,
        frame=frame,
        target_inplane=target_inplane,
        middle_fraction=middle_fraction,
        percentiles=percentiles,
        select_z_band=select_z_band,
    )
    report = HarmonizeReport(
        path=meta["path"],
        orig_shape=meta["orig_shape"],
        orig_zooms=meta["orig_zooms"],
        orig_ornt=meta["orig_ornt"],
        ras_shape=meta["ras_shape"],
        resampled_shape=meta["resampled_shape"],
        mask_z_bbox=used_frame.mask_z_bbox,
        selected_z=(used_frame.z_start, used_frame.z_end),
        n_slices=int(normed.shape[2]),
        centroid_xy=used_frame.centroid_xy,
        applied_n4=meta["applied_n4"],
        applied_synthstrip=meta["applied_synthstrip"],
    )
    return normed, mask_out, report, used_frame


def extract_hf_slices_from_harmonized(vol: np.ndarray) -> np.ndarray:
    """Stack axial slices from harmonized ``(H, W, D)`` → ``(N, 1, H, W)``."""
    if vol.ndim != 3:
        raise ValueError(f"Expected (H, W, D) volume, got {vol.shape}")
    return np.transpose(vol, (2, 0, 1))[:, np.newaxis, :, :].astype(np.float32)


def harmonize_fomo_volume(
    path: Union[str, Path],
    *,
    target_spacing_mm: float = TARGET_SPACING_MM,
    target_inplane: tuple[int, int] = TARGET_INPLANE,
    middle_fraction: float = DEFAULT_MIDDLE_FRACTION,
    apply_n4: bool = True,
    apply_synthstrip: bool = True,
    synthstrip_bin: str | Path | None = None,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> tuple[np.ndarray, HarmonizeReport]:
    """
    Harmonize one volume to HF slices ``(N, 1, H, W)``.

    Thin wrapper around :func:`harmonize_volume` for unpaired 3T writers.
    """
    vol, _mask, report, _frame = harmonize_volume(
        path,
        target_spacing_mm=target_spacing_mm,
        target_inplane=target_inplane,
        middle_fraction=middle_fraction,
        apply_n4=apply_n4,
        apply_synthstrip=apply_synthstrip,
        synthstrip_bin=synthstrip_bin,
        percentiles=percentiles,
        select_z_band=True,
    )
    slices = extract_hf_slices_from_harmonized(vol)
    report = HarmonizeReport(
        path=report.path,
        orig_shape=report.orig_shape,
        orig_zooms=report.orig_zooms,
        orig_ornt=report.orig_ornt,
        ras_shape=report.ras_shape,
        resampled_shape=report.resampled_shape,
        mask_z_bbox=report.mask_z_bbox,
        selected_z=report.selected_z,
        n_slices=int(slices.shape[0]),
        centroid_xy=report.centroid_xy,
        applied_n4=report.applied_n4,
        applied_synthstrip=report.applied_synthstrip,
    )
    return slices, report


def format_harmonize_log(report: HarmonizeReport, tag: str = "fomo50k") -> str:
    zooms = ",".join(f"{z:.3g}" for z in report.orig_zooms)
    ornt = "".join(report.orig_ornt)
    steps = "RAS"
    if report.applied_synthstrip:
        steps += "->SynthStrip"
    if report.applied_n4:
        steps += "->N4"
    steps += "->1mm"
    return (
        f"[{tag}] {Path(report.path).name}: "
        f"orig={report.orig_shape} zooms=({zooms}) ornt={ornt} -> "
        f"{steps} {report.resampled_shape} "
        f"mask_z=[{report.mask_z_bbox[0]},{report.mask_z_bbox[1]}] "
        f"sel_z=[{report.selected_z[0]},{report.selected_z[1]}) "
        f"-> {report.n_slices} slices"
    )
