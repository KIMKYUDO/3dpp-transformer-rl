import torch
from agents.heads import (
    PositionDecoder, SelectionDecoder, OrientationDecoder,
    PositionEmbeddingBuilder
)


def test_position_decoder_shapes_and_probs():
    B, N, d = 2, 20, 128
    cont_enc = torch.randn(B, 100, d)
    box_enc  = torch.randn(B, N, d)
    posdec = PositionDecoder(d_model=d, nhead=8, num_layers=2)
    logp, logits, ctx = posdec(cont_enc, box_enc)
    assert logits.shape == (B, 100)
    assert ctx.shape == (B, 100, d)
    probs = logp.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones(B), atol=1e-6)


def test_position_embedding_builder_shape():
    B, d = 2, 128
    ctx = torch.randn(B, 100, d)
    raw = torch.randn(B, 100, 7)
    idx = torch.tensor([3, 55], dtype=torch.long)
    builder = PositionEmbeddingBuilder(d_model=d)
    pos_emb = builder(ctx, raw, idx)
    assert pos_emb.shape == (B, 1, d)


def test_selection_decoder_shapes_and_probs():
    B, N, d = 2, 20, 128
    box_enc = torch.randn(B, N, d)
    pos_emb = torch.randn(B, 1, d)
    seldec = SelectionDecoder(d_model=d, nhead=8, num_layers=2)
    logp, logits, h = seldec(box_enc, pos_emb)
    assert logits.shape == (B, N)
    probs = logp.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones(B), atol=1e-6)
    assert h.shape == (B, N, d)


def test_orientation_decoder_shapes_and_probs():
    B, d = 2, 128
    orient_emb = torch.randn(B, 6, d)
    pos_emb = torch.randn(B, 1, d)
    ordec = OrientationDecoder(d_model=d, nhead=8, num_layers=2)
    logp, logits, h = ordec(orient_emb, pos_emb)
    assert logits.shape == (B, 6)
    probs = logp.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones(B), atol=1e-6)
    assert h.shape == (B, 6, d)
