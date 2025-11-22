from __future__ import annotations
from typing import Dict
import torch
from torch import nn


def apply_parameter_noise(module: nn.Module, std: float) -> Dict[str, torch.Tensor] | None:
    if std <= 0.0:
        return None

    noise: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            eps = torch.randn_like(p) * std
            p.add_(eps)
            noise[name] = eps
    return noise


def remove_parameter_noise(module: nn.Module, noise: Dict[str, torch.Tensor] | None) -> None:
    if not noise:
        return

    with torch.no_grad():
        for name, p in module.named_parameters():
            eps = noise.get(name)
            if eps is None:
                continue
            p.sub_(eps)
