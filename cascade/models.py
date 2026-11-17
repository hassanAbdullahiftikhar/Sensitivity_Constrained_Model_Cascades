"""Stage-1 (MobileNetV3-Small) and Stage-2 (ResNet-18) classifiers, ImageNet-pretrained."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

SMALL, LARGE = "mobilenet_v3_small", "resnet18"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == SMALL:
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == LARGE:
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_flops(model: nn.Module, size: int) -> int:
    """Forward FLOPs for one image (multiply-add counted as 2 FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.eval().cpu()
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, size, size))
    return int(fc.get_total_flops())
