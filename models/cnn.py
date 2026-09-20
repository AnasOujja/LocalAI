"""A VGG-style CNN built entirely from disk-offloaded conv/linear layers.

BatchNorm/ReLU/Pool layers stay as ordinary `nn.Module`s: they either have
no parameters or negligibly small ones, so keeping them resident in RAM
costs nothing, while the convolution and linear layers -- which hold the
vast majority of a deep CNN's parameters -- are streamed from disk.
"""
from __future__ import annotations

from typing import List, Union

import torch.nn as nn

from disk_offload import DiskTensorStore, OffloadedConv2d, OffloadedLinear, OffloadedSequential

VGG_CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


def build_vgg(cfg_name: str, in_channels: int, num_classes: int,
              store: DiskTensorStore, image_size: int = 32) -> nn.Module:
    cfg: List[Union[int, str]] = VGG_CFGS[cfg_name]
    layers: List[nn.Module] = []
    c_in = in_channels
    conv_idx = 0
    spatial = image_size
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            spatial //= 2
        else:
            layers.append(OffloadedConv2d(
                c_in, v, kernel_size=3, padding=1, store=store,
                name=f"features.conv{conv_idx}",
            ))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            c_in = v
            conv_idx += 1
    features = OffloadedSequential(layers, store=store)

    flat_dim = c_in * spatial * spatial
    classifier = OffloadedSequential([
        OffloadedLinear(flat_dim, 512, bias=True, store=store, name="classifier.fc0"),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        OffloadedLinear(512, num_classes, bias=True, store=store, name="classifier.fc1"),
    ], store=store)

    return _VGG(features, classifier)


class _VGG(nn.Module):
    def __init__(self, features: nn.Module, classifier: nn.Module):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
