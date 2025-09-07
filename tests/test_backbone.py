import torch
import pytest
from agents.backbone import PolicyBackbone, EncoderConfig, BoxEncoder, ContainerEncoder


def test_box_encoder_shapes():
    cfg = EncoderConfig(d_model=128, nhead=4, num_layers=2)
    enc = BoxEncoder(cfg)
    B, N = 2, 20
    boxes = torch.randint(10, 51, (B, N, 3)).float()
    out = enc(boxes)
    assert out.shape == (B, N, cfg.d_model)


def test_container_encoder_shapes():
    cfg = EncoderConfig(d_model=128, nhead=4, num_layers=2)
    enc = ContainerEncoder(cfg)
    B = 2
    cont = torch.randn(B, 100, 7)
    out = enc(cont)
    assert out.shape == (B, 100, cfg.d_model)


def test_policy_backbone_forward():
    cfg = EncoderConfig(d_model=128, nhead=4, num_layers=2)
    bb = PolicyBackbone(cfg)
    B, N = 2, 20
    boxes = torch.randint(10, 51, (B, N, 3)).float()
    cont = torch.randn(B, 100, 7)
    be, ce = bb(boxes, cont)
    assert be.shape == (B, N, cfg.d_model)
    assert ce.shape == (B, 100, cfg.d_model)
