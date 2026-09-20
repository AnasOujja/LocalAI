"""A deliberately oversized disk-backed MLP for capacity validation.

Each hidden layer is 8192 by 8192 float32 parameters, about 256 MiB. The
builder chooses enough layers to exceed a requested fraction of physical RAM,
while keeping every individual layer small enough to stream one at a time.
"""
from __future__ import annotations

import math
from typing import List

import torch.nn as nn

from disk_offload import DiskTensorStore, OffloadedLinear, OffloadedSequential


DEFAULT_WIDTH = 8192
DEFAULT_LAYER_BYTES = DEFAULT_WIDTH * DEFAULT_WIDTH * 4


def build_over_ram_mlp(
    store: DiskTensorStore,
    total_ram_bytes: int,
    target_ram_multiple: float = 1.15,
    width: int = DEFAULT_WIDTH,
    input_dim: int = 1024,
    output_dim: int = 10,
) -> tuple[nn.Module, int, int]:
    """Build a model whose disk-backed parameters exceed physical RAM.

    Returns `(model, hidden_layer_count, parameter_bytes)`.
    """
    hidden_layer_count = max(
        1,
        math.ceil((total_ram_bytes * target_ram_multiple) / (width * width * 4)),
    )
    layers: List[nn.Module] = []
    layers.append(
        OffloadedLinear(
            input_dim,
            width,
            bias=False,
            store=store,
            name="stress.input",
        )
    )
    layers.append(nn.ReLU())
    for index in range(hidden_layer_count):
        input_width = width
        layers.append(
            OffloadedLinear(
                input_width,
                width,
                bias=False,
                store=store,
                name=f"stress.hidden{index}",
            )
        )
        layers.append(nn.ReLU())

    layers.append(
        OffloadedLinear(
            width,
            output_dim,
            bias=True,
            store=store,
            name="stress.output",
        )
    )
    model = OffloadedSequential(layers, store=store)
    parameter_bytes = (
        input_dim * width * 4
        + hidden_layer_count * width * width * 4
        + width * output_dim * 4
        + output_dim * 4
    )
    return model, hidden_layer_count, parameter_bytes
