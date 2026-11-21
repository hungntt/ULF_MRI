"""Axial slab extraction used by the preprocessing scripts."""
import numpy as np


def extract_paired_slices(lf_vol, hf_vol, slab_size=3, *, slice_selection="middle",
                          middle_fraction=1.0, filter_empty=False):
    """Pair each 3T axial slice with the low-field slab centred on it.

    Keeps the central `middle_fraction` of the stack, clamped so that every slab fits.
    Returns (ulf [N, slab, H, W], hf [N, 1, H, W], n_candidates, z_indices).
    """
    if lf_vol.shape != hf_vol.shape:
        raise ValueError(f"LF/HF shape mismatch: {lf_vol.shape} vs {hf_vol.shape}")
    if slice_selection != "middle" or filter_empty:
        raise NotImplementedError("only slice_selection='middle', filter_empty=False is used")
    half, depth = slab_size // 2, lf_vol.shape[2]
    n_keep = max(1, int(round(depth * middle_fraction)))
    start = max((depth - n_keep) // 2, half)
    end = min((depth - n_keep) // 2 + n_keep, depth - half)
    if end <= start:
        raise ValueError(f"no slab-safe axial indices for depth={depth}")
    z = list(range(start, end))
    ulf = np.stack([np.transpose(lf_vol[:, :, k - half:k + half + 1], (2, 0, 1)) for k in z])
    hf = np.stack([hf_vol[:, :, k] for k in z])[:, None]
    return ulf.astype(np.float32), hf.astype(np.float32), end - start, z
