from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Positional Encoding (fixed 2D sin-cos for 10x10 grid → 100 tokens)
# ------------------------------------------------------------

def _build_2d_sincos_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """Return (h*w, dim) fixed 2D sin-cos positional embedding.
    dim must be even and preferably divisible by 4, but we handle general even dim.
    """
    assert dim % 2 == 0, "positional dim must be even"
    # Split channels for x/y equally
    dim_half = dim // 2
    # For each axis, build 1D sincos of length h or w
    def _sincos_1d(n: int, d: int) -> torch.Tensor:
        # frequencies
        omega = torch.arange(d // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (d / 2)))
        pos = torch.arange(n, dtype=torch.float32)
        out = torch.einsum('n,d->nd', pos, omega)  # (n, d//2)
        emb = torch.cat([out.sin(), out.cos()], dim=1)  # (n, d)
        if d % 2 == 1:
            # pad if odd (shouldn't happen for our usage)
            emb = F.pad(emb, (0, 1))
        return emb  # (n, d)

    emb_y = _sincos_1d(h, dim_half)  # (h, dim//2)
    emb_x = _sincos_1d(w, dim_half)  # (w, dim//2)
    # combine to 2D by concatenating per grid cell
    pos = []
    for iy in range(h):
        row = []
        for ix in range(w):
            row.append(torch.cat([emb_y[iy], emb_x[ix]], dim=0))  # (dim)
        pos.append(torch.stack(row, dim=0))  # (w, dim)
    pos = torch.stack(pos, dim=0)   # (h, w, dim)
    pos = pos.view(h * w, dim)      # (h*w, dim)
    return pos


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=4*d_model, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):  # x: (B, L, d)
        return self.enc(x)


# ------------------------------------------------------------
# Encoders
# ------------------------------------------------------------

@dataclass
class EncoderConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    grid_hw: tuple[int, int] = (10, 10)  # for container (downsampled 10x10)


class BoxEncoder(nn.Module):
    """Box Encoder
    Input:  boxes as (B, N, 3) with (l, w, h)
    Procedure (per paper): embed each of l, w, h to d_model → average → TransformerEncoder
    Output: (B, N, d_model)
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d = cfg.d_model
        # scalar → d_model (shared for l,w,h)
        self.scalar_mlp = MLP(1, d, d)
        self.enc = TransformerEncoder(d_model=d, nhead=cfg.nhead, num_layers=cfg.num_layers)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        # boxes: (B, N, 3)
        assert boxes.dim() == 3 and boxes.size(-1) == 3
        B, N, _ = boxes.shape
        x = boxes.unsqueeze(-1)            # (B, N, 3, 1)
        x = self.scalar_mlp(x)             # (B, N, 3, d)
        x = x.mean(dim=2)                  # (B, N, d)  (average over {l,w,h})
        x = self.enc(x)                    # (B, N, d)
        return x


class ContainerEncoder(nn.Module):
    """Container Encoder
    Input:  container features (B, 100, 7) from downsampled 10x10x7 → flatten 100x7
    Add fixed 2D positional encoding (10x10) → TransformerEncoder
    Output: (B, 100, d_model)
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.d = cfg.d_model
        self.grid_hw = cfg.grid_hw
        H, W = self.grid_hw
        assert H * W == 100, "expected 10x10 downsampled grid → 100 tokens"

        self.proj = MLP(7, self.d, self.d)
        pos = _build_2d_sincos_pos_embed(H, W, self.d)      # (100, d)
        self.register_buffer('pos', pos.unsqueeze(0), persistent=False)  # (1, 100, d)
        self.enc = TransformerEncoder(d_model=self.d, nhead=cfg.nhead, num_layers=cfg.num_layers)

    def forward(self, cont_feat: torch.Tensor) -> torch.Tensor:
        # cont_feat: (B, 100, 7)
        assert cont_feat.dim() == 3 and cont_feat.size(1) == 100 and cont_feat.size(-1) == 7
        x = self.proj(cont_feat)           # (B, 100, d)
        x = x + self.pos                   # add fixed positional encoding
        x = self.enc(x)                    # (B, 100, d)
        return x


class PolicyBackbone(nn.Module):
    """Convenience wrapper holding both encoders for the policy network.
    Usage:
        bb = PolicyBackbone(EncoderConfig())
        box_enc, cont_enc = bb(boxes, cont_feat)
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.box_encoder = BoxEncoder(cfg)
        self.container_encoder = ContainerEncoder(cfg)

    def forward(self, boxes: torch.Tensor, cont_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.box_encoder(boxes), self.container_encoder(cont_feat)
