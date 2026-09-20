"""Plain in-RAM VGG-style CNN, structurally identical to models.cnn.build_vgg,
used as a ground-truth baseline to validate the disk-offloaded version's
correctness and to compare peak RAM usage / convergence behavior.
"""
from __future__ import annotations

from typing import List, Union

import torch.nn as nn

from models.cnn import VGG_CFGS


def build_vgg_baseline(cfg_name: str, in_channels: int, num_classes: int,
                        image_size: int = 32) -> nn.Module:
    cfg: List[Union[int, str]] = VGG_CFGS[cfg_name]
    layers: List[nn.Module] = []
    c_in = in_channels
    spatial = image_size
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            spatial //= 2
        else:
            layers.append(nn.Conv2d(c_in, v, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            c_in = v
    features = nn.Sequential(*layers)

    flat_dim = c_in * spatial * spatial
    classifier = nn.Sequential(
        nn.Linear(flat_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )

    return _VGGBaseline(features, classifier)


class _VGGBaseline(nn.Module):
    def __init__(self, features: nn.Module, classifier: nn.Module):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
