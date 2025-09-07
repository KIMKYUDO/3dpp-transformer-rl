import torch
from agents.value_head import ValueNet
from agents.backbone import EncoderConfig


def test_value_net_forward_shape_and_grad():
    cfg = EncoderConfig(d_model=128, nhead=4, num_layers=2)
    net = ValueNet(cfg, d_model=128, nhead=8, num_layers=2)
    B, N = 3, 20
    boxes = torch.randint(10, 51, (B, N, 3)).float().requires_grad_()
    cont  = torch.randn(B, 100, 7).requires_grad_()

    out = net(boxes, cont)  # (B,1)
    assert out.shape == (B, 1)

    # Simple scalar loss to test backward
    loss = out.mean()
    loss.backward()

    # Check gradients flowed to inputs (sanity)
    assert boxes.grad is not None
    assert cont.grad is not None
