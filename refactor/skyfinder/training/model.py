"""Model factories: `build_model` and the FDS-wrapped variant `FDSModel`.

`build_model` returns a fresh ResNet-50 or ViT-B/16 with a 1-output regression head.
`FDSModel` wraps a vanilla model with an FDS calibration module between backbone
and head — see the `FDS` class in `skyfinder.training.fds`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights, resnet50,
                                vit_b_16)

from .fds import FDS
from .lds import MIN_TEMP


def build_model(name: str, freeze_backbone: bool = False) -> nn.Module:
    """Build a fresh model with our 1-output regression head.

    `freeze_backbone` (D4 linear probe): freezes all pretrained params, then
    swaps in a new head. The new `nn.Linear` is constructed AFTER the freeze,
    so it stays trainable by default — no name-matching needed downstream.
    """
    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad_(False)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.parameters():
                p.requires_grad_(False)
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
        return m
    raise ValueError(f"unknown model: {name}")


class FDSModel(nn.Module):
    """Wraps a vanilla ResNet-50 or ViT-B/16 with an FDS calibration module
    between its backbone and the regression head.

    Forward signature:
        net(x)              # eval mode; returns predictions (B,)
        net(x, labels=y)    # train mode; returns (preds, raw_features) for FDS update
    """

    def __init__(self, vanilla_model: nn.Module, fds: FDS,
                 bin_width: float = 1.0, min_temp: float = MIN_TEMP):
        super().__init__()
        self.fds = fds
        self.bin_width = bin_width
        self.min_temp = min_temp
        self.current_epoch = 0

        # Split off the final FC for both architectures.
        if hasattr(vanilla_model, "fc") and isinstance(vanilla_model.fc, nn.Linear):
            self.head = vanilla_model.fc
            vanilla_model.fc = nn.Identity()
        elif hasattr(vanilla_model, "heads"):
            self.head = vanilla_model.heads.head
            vanilla_model.heads.head = nn.Identity()
        else:
            raise ValueError(f"Don't know how to FDS-wrap model {type(vanilla_model)}")
        self.backbone = vanilla_model

    def _bucketize(self, temps: torch.Tensor) -> torch.Tensor:
        return ((temps - self.min_temp) / self.bin_width).long().clamp(0, self.fds.bucket_num - 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        smoothed = feats
        if self.training and labels is not None and self.current_epoch >= self.fds.start_smooth:
            smoothed = self.fds.smooth(feats, self._bucketize(labels), self.current_epoch)
        pred = self.head(smoothed).squeeze(-1)
        if self.training:
            return pred, feats   # raw feats for the FDS running-stats update
        return pred
