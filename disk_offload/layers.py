"""nn.Module wrappers that hold no resident parameter tensors -- only disk
keys and shape metadata. Real weight/bias tensors are streamed in from a
`DiskTensorStore` for the duration of a single forward/backward call.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .ops import OffloadedConv2dFn, OffloadedLinearFn
from .storage import DiskTensorStore

_Pair = Union[int, Tuple[int, int]]


class OffloadedLinear(nn.Module):
    """Disk-backed drop-in replacement for `nn.Linear`."""

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 store: DiskTensorStore, name: str):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.store = store
        self.weight_key = f"{name}.weight"
        self.bias_key = f"{name}.bias" if bias else None
        # Forces autograd to always create a graph node / call backward for
        # this layer, even when its actual input doesn't require grad.
        self._grad_anchor = nn.Parameter(torch.zeros(1))

        if not store.exists(self.weight_key):
            weight = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            store.save(self.weight_key, weight)
        if self.bias_key and not store.exists(self.bias_key):
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            b = torch.empty(out_features).uniform_(-bound, bound)
            store.save(self.bias_key, b)

    def param_keys(self) -> List[str]:
        keys = [self.weight_key]
        if self.bias_key:
            keys.append(self.bias_key)
        return keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return OffloadedLinearFn.apply(x, self._grad_anchor, self.weight_key, self.bias_key, self.store)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, offloaded=True"


class OffloadedConv2d(nn.Module):
    """Disk-backed drop-in replacement for `nn.Conv2d`."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _Pair,
                 store: DiskTensorStore, name: str, stride: _Pair = 1,
                 padding: _Pair = 0, dilation: _Pair = 1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        ks = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.store = store
        self.weight_key = f"{name}.weight"
        self.bias_key = f"{name}.bias" if bias else None
        # Forces autograd to always create a graph node / call backward for
        # this layer, even when its actual input doesn't require grad.
        self._grad_anchor = nn.Parameter(torch.zeros(1))

        if not store.exists(self.weight_key):
            weight = torch.empty(out_channels, in_channels // groups, *ks)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            store.save(self.weight_key, weight)
        if self.bias_key and not store.exists(self.bias_key):
            fan_in = (in_channels // groups) * ks[0] * ks[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            b = torch.empty(out_channels).uniform_(-bound, bound)
            store.save(self.bias_key, b)

    def param_keys(self) -> List[str]:
        keys = [self.weight_key]
        if self.bias_key:
            keys.append(self.bias_key)
        return keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return OffloadedConv2dFn.apply(
            x, self._grad_anchor, self.weight_key, self.bias_key, self.store,
            self.stride, self.padding, self.dilation, self.groups,
        )


def _pair(v: _Pair) -> Tuple[int, int]:
    return v if isinstance(v, tuple) else (v, v)


class OffloadedSequential(nn.Module):
    """Like `nn.Sequential`, but prefetches the *next* offloaded layer's
    weights from disk while the *current* layer is still computing,
    overlapping disk I/O latency with CPU compute."""

    def __init__(self, layers: Iterable[nn.Module], store: DiskTensorStore):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.store = store
        self._offload_indices = [
            i for i, l in enumerate(self.layers) if hasattr(l, "param_keys")
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            nxt = next((j for j in self._offload_indices if j > i), None)
            if nxt is not None:
                for key in self.layers[nxt].param_keys():
                    self.store.prefetch(key)
            x = layer(x)
        return x

    def param_keys(self) -> List[str]:
        keys: List[str] = []
        for i in self._offload_indices:
            keys.extend(self.layers[i].param_keys())
        return keys


def collect_param_keys(module: nn.Module) -> List[str]:
    """Walk a module tree and gather disk keys of every offloaded layer."""
    keys: List[str] = []
    for m in module.modules():
        if hasattr(m, "param_keys") and not isinstance(m, OffloadedSequential):
            keys.extend(m.param_keys())
    return keys
