"""
Preprocess utilities for 3D packing (container plane features & downsampling).

Channels per cell (i, j):
  0: h_ij    (height)
  1: e_l     (distance to left boundary, along +x)
  2: e_w     (distance to top boundary, along +y)
  3: e_-l    (distance to right boundary, along -x)
  4: e_-w    (distance to bottom boundary, along -y)
  5: f_l     (forward run distance along +x until first higher cell; to boundary if none)
  6: f_w     (forward run distance along +y until first higher cell; to boundary if none)

Notes
-----
- Heightmap H has shape (H, W) = (rows, cols) = (y, x).
- Distances are in grid cells (integer-like, >= 0).
- "Higher" means strictly greater height (>) than the current cell.
- Downsampling rule: for each (patch x patch) tile, select the cell with
  maximal (e_l * e_w). Tie-break follows np.argmax (row-major first hit).
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

Array = np.ndarray


def _validate_heightmap(H: Array) -> Tuple[int, int]:
    if not isinstance(H, np.ndarray):
        raise TypeError("heightmap must be a numpy.ndarray")
    if H.ndim != 2:
        raise ValueError(f"heightmap must be 2D, got shape {H.shape}")
    h, w = H.shape
    if h <= 0 or w <= 0:
        raise ValueError("heightmap must have positive dimensions")
    return h, w


def compute_plane_features(heightmap: Array) -> Array:
    """Compute 7-channel container state from a (H, W) heightmap.

    Returns
    -------
    state7 : np.ndarray of shape (H, W, 7), dtype float32
    """
    H, W = _validate_heightmap(heightmap)
    h = heightmap.astype(np.float32)

    # --- Boundary distances (broadcast to (H, W)) ---
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)

    e_l  = np.broadcast_to(xs[None, :], (H, W))            # 0..W-1 across columns
    e_w  = np.broadcast_to(ys[:, None], (H, W))            # 0..H-1 down rows
    e_ml = np.broadcast_to((W - 1 - xs)[None, :], (H, W))  # W-1..0 across columns
    e_mw = np.broadcast_to((H - 1 - ys)[:, None], (H, W))  # H-1..0 down rows

    # --- Forward-run distances until first higher cell (> base); else to boundary ---
    f_l = np.zeros_like(h, dtype=np.float32)
    for i in range(H):
        row = h[i]
        for j in range(W):
            base = row[j]
            dist = W - 1 - j
            for k in range(j + 1, W):
                if row[k] > base:
                    dist = k - j
                    break
            f_l[i, j] = float(dist)

    f_w = np.zeros_like(h, dtype=np.float32)
    for j in range(W):
        col = h[:, j]
        for i in range(H):
            base = col[i]
            dist = H - 1 - i
            for k in range(i + 1, H):
                if col[k] > base:
                    dist = k - i
                    break
            f_w[i, j] = float(dist)

    state7 = np.stack([h, e_l, e_w, e_ml, e_mw, f_l, f_w], axis=-1).astype(np.float32)
    return state7


def downsample_patches(state7: Array, patch: int = 10, rule: str = "el*ew_max") -> Array:
    """Downsample (H, W, 7) into (H/patch, W/patch, 7) by tile selection.

    Parameters
    ----------
    state7 : (H, W, 7) float32
    patch  : tile size (default 10)
    rule   : selection rule; currently supports "el*ew_max"

    Returns
    -------
    ds7 : (H//patch, W//patch, 7)
    """
    if state7.ndim != 3 or state7.shape[-1] != 7:
        raise ValueError(f"state7 must be (H,W,7), got {state7.shape}")
    H, W, C = state7.shape
    if H % patch != 0 or W % patch != 0:
        raise ValueError(f"H and W must be divisible by patch={patch}, got {(H, W)}")

    out = np.zeros((H // patch, W // patch, C), dtype=state7.dtype)

    for pi in range(0, H, patch):
        for pj in range(0, W, patch):
            tile = state7[pi:pi + patch, pj:pj + patch, :]  # (patch, patch, 7)
            if rule == "el*ew_max":
                el = tile[..., 1]
                ew = tile[..., 2]
                score = el * ew
                idx = int(np.argmax(score))  # row-major argmax (deterministic)
            else:
                idx = 0  # top-left fallback
            ii, jj = divmod(idx, patch)
            out[pi // patch, pj // patch] = tile[ii, jj]

    return out


def flatten_for_encoder(ds7: Array) -> Array:
    """Flatten (h, w, 7) to (h*w, 7) row-major order for Transformer encoder."""
    if ds7.ndim != 3 or ds7.shape[-1] != 7:
        raise ValueError(f"ds7 must be (h,w,7), got {ds7.shape}")
    h, w, c = ds7.shape
    return ds7.reshape(h * w, c)
