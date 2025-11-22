"""
GPU-accelerated preprocessing for 3D packing container states.

This module provides PyTorch implementations of plane feature computation
that can run entirely on GPU, avoiding CPU-GPU transfers.
"""
import torch
from typing import Tuple
import numpy as np


def compute_plane_features(heightmap: torch.Tensor) -> torch.Tensor:
    """Compute 7-channel container state from a (H, W) heightmap on GPU.
    
    This is the MAIN function you need. All logic is self-contained.
    
    Parameters
    ----------
    heightmap : torch.Tensor of shape (H, W)
        Height values at each position, on GPU
        
    Returns
    -------
    state7 : torch.Tensor of shape (H, W, 7), dtype float32
        Seven feature channels:
        0: h_ij    (height)
        1: e_l     (distance to right boundary)
        2: e_w     (distance to bottom boundary)
        3: e_-l    (distance to left boundary)
        4: e_-w    (distance to top boundary)
        5: f_l     (forward run distance along +x)
        6: f_w     (forward run distance along +y)
        
    Example
    -------
    >>> heightmap = torch.randint(0, 50, (100, 100)).cuda().float()
    >>> state7 = compute_plane_features_gpu(heightmap)
    >>> print(state7.shape)  # torch.Size([100, 100, 7])
    """
    device = heightmap.device
    H, W = heightmap.shape
    
    # Convert to float32 if needed
    h = heightmap.float()
    
    # ========================================
    # 1. Boundary distances (간단!)
    # ========================================
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    
    e_l = (W - 1 - xs)[None, :].expand(H, W).clone()  # W-1..0 across columns
    e_w = (H - 1 - ys)[:, None].expand(H, W).clone()  # H-1..0 down rows
    e_ml = xs[None, :].expand(H, W).clone()           # 0..W-1 across columns
    e_mw = ys[:, None].expand(H, W).clone()           # 0..H-1 down rows

    
    # ========================================
    # 2. Forward distances (복잡하지만 여기서 처리)
    # ========================================
    
    # f_l: distance to first higher cell in +x direction
    j_grid = torch.arange(W, device=device)[None, :].expand(H, W)
    f_l = (W - 1 - j_grid).float()  # Initialize to boundary
    
    for offset in range(1, W):
        shifted_h = torch.roll(h, shifts=-offset, dims=1)
        valid = j_grid < (W - offset)
        is_higher = shifted_h > h
        update_mask = is_higher & valid & (f_l > offset)
        f_l = torch.where(update_mask, torch.tensor(offset, dtype=torch.float32, device=device), f_l)
    
    # f_w: distance to first higher cell in +y direction
    i_grid = torch.arange(H, device=device)[:, None].expand(H, W)
    f_w = (H - 1 - i_grid).float()  # Initialize to boundary
    
    for offset in range(1, H):
        shifted_h = torch.roll(h, shifts=-offset, dims=0)
        valid = i_grid < (H - offset)
        is_higher = shifted_h > h
        update_mask = is_higher & valid & (f_w > offset)
        f_w = torch.where(update_mask, torch.tensor(offset, dtype=torch.float32, device=device), f_w)
    
    # ========================================
    # 3. Stack all channels
    # ========================================
    state7 = torch.stack([h, e_l, e_w, e_ml, e_mw, f_l, f_w], dim=-1)
    
    return state7


# Note: Helper functions removed for simplicity.
# All logic is now in compute_plane_features_gpu() above.


def downsample_patches(state7: torch.Tensor, patch: int = 10, 
                           rule: str = "el*ew_max") -> torch.Tensor:
    """GPU version of downsample_patches.
    
    Parameters
    ----------
    state7 : torch.Tensor of shape (H, W, 7)
        Container state with 7 channels
    patch : int
        Tile size (default 10)
    rule : str
        Selection rule, currently supports "el*ew_max"
        
    Returns
    -------
    ds7 : torch.Tensor of shape (H//patch, W//patch, 7)
        Downsampled state
    """
    H, W, C = state7.shape
    device = state7.device
    
    if H % patch != 0 or W % patch != 0:
        raise ValueError(f"H and W must be divisible by patch={patch}, got {(H, W)}")
    
    # Reshape into patches: (H//patch, patch, W//patch, patch, 7)
    state_reshaped = state7.view(H // patch, patch, W // patch, patch, C)
    # Rearrange to: (H//patch, W//patch, patch, patch, 7)
    state_reshaped = state_reshaped.permute(0, 2, 1, 3, 4)
    # Flatten patch dimensions: (H//patch, W//patch, patch*patch, 7)
    state_reshaped = state_reshaped.reshape(H // patch, W // patch, patch * patch, C)
    
    if rule == "el*ew_max":
        # Extract e_l (channel 1) and e_w (channel 2)
        el = state_reshaped[..., 1]  # (H//patch, W//patch, patch*patch)
        ew = state_reshaped[..., 2]  # (H//patch, W//patch, patch*patch)
        
        # Compute score
        score = el * ew  # (H//patch, W//patch, patch*patch)
        
        # Find argmax for each tile
        idx = torch.argmax(score, dim=-1)  # (H//patch, W//patch)
        
        # Gather selected features
        # Expand idx for gathering: (H//patch, W//patch, 1, 1) -> (H//patch, W//patch, 1, 7)
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, C)
        
        # Gather: (H//patch, W//patch, 1, 7)
        out = torch.gather(state_reshaped, dim=2, index=idx_expanded)
        
        # Squeeze: (H//patch, W//patch, 7)
        out = out.squeeze(2)
    else:
        # Fallback: select first (top-left)
        out = state_reshaped[..., 0, :]
    
    return out


def flatten_for_encoder(ds7: torch.Tensor) -> torch.Tensor:
    """GPU version of flatten_for_encoder.
    
    Parameters
    ----------
    ds7 : torch.Tensor of shape (h, w, 7)
        Downsampled state
        
    Returns
    -------
    flat : torch.Tensor of shape (h*w, 7)
        Flattened state in row-major order
    """
    h, w, c = ds7.shape
    return ds7.reshape(h * w, c)

# Batch processing versions
def preprocess_batch_gpu(heightmaps, patch=10, device='cuda'):
    """Complete preprocessing pipeline on GPU for batch of environments."""
    
    # NumPy → Torch
    if isinstance(heightmaps, np.ndarray):
        heightmaps = torch.from_numpy(heightmaps).float().to(device)
    
    B = heightmaps.shape[0]
    results = []
    
    for i in range(B):
        s7 = compute_plane_features(heightmaps[i])      # 직접 호출!
        ds7 = downsample_patches(s7, patch=patch)
        flat = flatten_for_encoder(ds7)
        results.append(flat)
    
    return torch.stack(results, dim=0)