from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse encoders & small MLP from backbone
from agents.backbone import BoxEncoder, ContainerEncoder, EncoderConfig, MLP


class _TransformerDecoder(nn.Module):
    """Light wrapper around nn.TransformerDecoder (batch_first=True)."""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4*d_model
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
    def forward(self, q, kv):  # q: (B, Lq, d), kv: (B, Lk, d)
        return self.dec(q, kv)


class ValueNet(nn.Module):
    """Value network (Critic).

    논문과 동일하게 Box/Container 인코더를 사용하고, 디코더 1개를 통해
    컨테이너 토큰(100)이 박스 토큰(N)을 참조(cross-attn)하도록 한 뒤,
    평균 풀링 + MLP로 스칼라 V(s)를 출력합니다.

    Inputs
    ------
    boxes:     (B, N, 3)   # (l, w, h)
    cont_feat: (B, 100, 7) # downsampled 10x10x7 → flatten

    Output
    ------
    V: (B, 1) scalar value per state
    """
    def __init__(self, enc_cfg: EncoderConfig, d_model: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        # encoders (critic 전용, actor와 파라미터 분리)
        self.box_encoder = BoxEncoder(enc_cfg)
        self.container_encoder = ContainerEncoder(enc_cfg)
        # one cross-attention decoder (Query=container, KV=box)
        self.dec = _TransformerDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        # head to scalar
        self.head = MLP(d_model, d_model, 1)

    def forward(self, boxes: torch.Tensor, cont_feat: torch.Tensor) -> torch.Tensor:
        # Encode
        be = self.box_encoder(boxes)         # (B, N, d)
        ce = self.container_encoder(cont_feat)  # (B, 100, d)
        # Cross-attend: container queries box
        h = self.dec(ce, be)                 # (B, 100, d)
        # Pool & project to scalar
        pooled = h.mean(dim=1)               # (B, d)
        V = self.head(pooled)                # (B, 1)
        return V
