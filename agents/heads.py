from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# 재사용: backbone의 MLP를 그대로 씁니다.
try:
    from agents.backbone import MLP
except Exception:
    # 테스트 독립성을 위한 fallback MLP
    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, out_dim)
            )
        def forward(self, x):
            return self.net(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4*d_model
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
    def forward(self, q, kv):  # q: (B, Lq, d), kv: (B, Lk, d)
        return self.dec(q, kv)


class PositionDecoder(nn.Module):
    """Query=Container Encoding, Key/Value=Box Encoding → pos logits over 100.
    Returns log_probs, logits, and decoder output (context for pos-embedding).
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.dec = TransformerDecoder(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, cont_enc: torch.Tensor, box_enc: torch.Tensor):
        # cont_enc: (B,100,d), box_enc: (B,N,d)
        h = self.dec(cont_enc, box_enc)          # (B,100,d)
        logits = self.fc(h).squeeze(-1)          # (B,100)
        return logits, h                   # h는 pos-embedding 생성을 위한 컨텍스트


class PositionEmbeddingBuilder(nn.Module):
    """Build position embedding from selected (x,y) index.
    pos_emb = MLP_d( cont_ctx[idx] + Proj7(raw_state[idx]) )  → (B,1,d)
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.raw_proj = MLP(7, d_model, d_model)
        self.fuse = MLP(d_model, d_model, d_model)
    def forward(self, cont_ctx: torch.Tensor, raw_state: torch.Tensor, idx: torch.Tensor):
        # cont_ctx: (B,100,d), raw_state: (B,100,7), idx: (B,) long
        B, L, d = cont_ctx.shape
        assert L == 100 and raw_state.shape == (B, L, 7)
        if idx.dtype != torch.long:
            idx = idx.long()
        # gather selected token per batch
        idx_d = idx.view(B, 1, 1).expand(-1, 1, d)
        pos_vec = cont_ctx.gather(1, idx_d)        # (B,1,d)
        idx_7 = idx.view(B, 1, 1).expand(-1, 1, 7)
        pos_raw = raw_state.gather(1, idx_7).squeeze(1)  # (B,7)
        fused = pos_vec.squeeze(1) + self.raw_proj(pos_raw)  # (B,d)
        pos_emb = self.fuse(fused).unsqueeze(1)    # (B,1,d)
        return pos_emb


class SelectionDecoder(nn.Module):
    """Query=Box Encoding, Key/Value=Position Embedding → box logits over N."""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.dec = TransformerDecoder(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, box_enc: torch.Tensor, pos_emb: torch.Tensor):
        # box_enc: (B,N,d), pos_emb: (B,1,d)
        h = self.dec(box_enc, pos_emb)            # (B,N,d)
        logits = self.fc(h).squeeze(-1)           # (B,N)
        return logits, h


class OrientationDecoder(nn.Module):
    """Query=Orientation Embedding(6), Key/Value=Position Embedding → 6-way logits."""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.dec = TransformerDecoder(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, orient_emb: torch.Tensor, pos_emb: torch.Tensor):
        # orient_emb: (B,6,d), pos_emb: (B,1,d)
        h = self.dec(orient_emb, pos_emb)         # (B,6,d)
        logits = self.fc(h).squeeze(-1)           # (B,6)
        return logits, h


class OrientationEmbedder(nn.Module):
    """Make 6 orientation embeddings from selected box dims (l,w,h).
    Each scalar is embedded then averaged → (B,6,d).
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.scalar_mlp = MLP(1, d_model, d_model)
    def forward(self, box_lwh: torch.Tensor) -> torch.Tensor:
        # box_lwh: (B,3) with (l,w,h)
        B, three = box_lwh.shape
        assert three == 3
        l, w, h = box_lwh[:, 0:1], box_lwh[:, 1:2], box_lwh[:, 2:3]
        # 6 permutations
        perms = [
            torch.cat([l,w,h], dim=1), torch.cat([l,h,w], dim=1),
            torch.cat([w,l,h], dim=1), torch.cat([w,h,l], dim=1),
            torch.cat([h,l,w], dim=1), torch.cat([h,w,l], dim=1),
        ]  # each (B,3)
        outs = []
        for p in perms:
            x = p.unsqueeze(-1)               # (B,3,1)
            x = self.scalar_mlp(x)            # (B,3,d)
            x = x.mean(dim=1)                 # (B,d)
            outs.append(x)
        out = torch.stack(outs, dim=1)        # (B,6,d)
        return out
