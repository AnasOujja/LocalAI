"""Disk-backed tensor storage with background prefetching and a memory-aware
residency budget.

Parameters, gradients and optimizer states normally live on disk as raw
binary files, materialized in RAM only for the brief window they are
actually needed. But not every layer needs to pay that disk-I/O tax: if a
layer's weights would fit comfortably in *currently available* RAM, there's
no reason to stream it from disk on every pass. `auto_configure_residency`
inspects available system memory once, decides how many parameters can be
pinned permanently in RAM, and promotes them -- so only the layers that
actually don't fit keep streaming from disk. Callers never have to decide
this layer by layer.

A small background thread pool lets the caller prefetch the *next*
streamed layer's weights while the *current* layer is still computing,
overlapping disk I/O with CPU compute instead of paying for it serially.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, Optional, Set

import numpy as np
import psutil
import torch


class DiskTensorStore:
    """Stores tensors as raw binary blobs on disk, keyed by string name."""

    def __init__(self, root: str, num_prefetch_workers: int = 2):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "grads"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "state"), exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

        # Keys promoted to full RAM residency: no disk I/O at all for their
        # parameter, gradient or optimizer state once here.
        self._resident_keys: Set[str] = set()
        self._resident_params: Dict[str, torch.Tensor] = {}
        self._resident_grads: Dict[str, torch.Tensor] = {}
        self._resident_state: Dict[str, Dict[str, torch.Tensor]] = {}

    # ---- path helpers -------------------------------------------------
    def _paths(self, key: str, subdir: str = ""):
        base = os.path.join(self.root, subdir, key.replace("/", "__"))
        return base + ".bin", base + ".json"

    @staticmethod
    def _replace_with_retry(src: str, dst: str, attempts: int = 8, base_delay: float = 0.01):
        """os.replace() over a file that a background prefetch thread still
        has open for reading can raise a transient PermissionError on
        Windows. Retry with a short backoff instead of crashing training."""
        for attempt in range(attempts):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))

    # ---- raw read/write -------------------------------------------------
    def _write(self, key: str, tensor: torch.Tensor, subdir: str = ""):
        bin_path, meta_path = self._paths(key, subdir)
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        arr = tensor.detach().to("cpu").contiguous().numpy()
        tmp_bin = bin_path + ".tmp"
        arr.tofile(tmp_bin)
        self._replace_with_retry(tmp_bin, bin_path)
        meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _read(self, key: str, subdir: str = "") -> torch.Tensor:
        bin_path, meta_path = self._paths(key, subdir)
        with open(meta_path) as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        arr = np.fromfile(bin_path, dtype=dtype)
        arr = arr.reshape(shape)
        return torch.from_numpy(arr.copy())

    def _exists(self, key: str, subdir: str = "") -> bool:
        bin_path, meta_path = self._paths(key, subdir)
        return os.path.exists(bin_path) and os.path.exists(meta_path)

    # ---- parameters -----------------------------------------------------
    def save(self, key: str, tensor: torch.Tensor):
        if key in self._resident_keys:
            self._resident_params[key] = tensor.detach().to("cpu").contiguous().clone()
            return
        self._write(key, tensor)

    def load(self, key: str) -> torch.Tensor:
        if key in self._resident_keys:
            return self._resident_params[key]
        with self._lock:
            fut = self._futures.pop(key, None)
        if fut is not None:
            return fut.result()
        return self._read(key)

    def prefetch(self, key: str):
        """Kick off a background read for `key` if not already in flight."""
        if key in self._resident_keys:
            return  # already in RAM, nothing to prefetch
        with self._lock:
            if key in self._futures:
                return
            self._futures[key] = self._executor.submit(self._read, key)

    def exists(self, key: str) -> bool:
        return key in self._resident_keys or self._exists(key)

    # ---- gradients --------------------------------------------------------
    def save_grad(self, key: str, grad: torch.Tensor, accumulate: bool = False):
        if key in self._resident_keys:
            if accumulate and key in self._resident_grads:
                grad = self._resident_grads[key] + grad
            self._resident_grads[key] = grad.detach().clone()
            return
        if accumulate and self._exists(key, "grads"):
            existing = self._read(key, "grads")
            grad = existing + grad
        self._write(key, grad, "grads")

    def load_grad(self, key: str, missing_ok: bool = True) -> Optional[torch.Tensor]:
        if key in self._resident_keys:
            grad = self._resident_grads.get(key)
            if grad is None and not missing_ok:
                raise FileNotFoundError(f"No gradient stored for {key!r}")
            return grad
        if not self._exists(key, "grads"):
            if missing_ok:
                return None
            raise FileNotFoundError(f"No gradient stored for {key!r}")
        return self._read(key, "grads")

    def clear_grad(self, key: str):
        if key in self._resident_keys:
            self._resident_grads.pop(key, None)
            return
        bin_path, meta_path = self._paths(key, "grads")
        for p in (bin_path, meta_path):
            if os.path.exists(p):
                os.remove(p)

    # ---- optimizer state ---------------------------------------------------
    def save_state(self, key: str, name: str, tensor: torch.Tensor):
        if key in self._resident_keys:
            self._resident_state.setdefault(key, {})[name] = tensor.detach().clone()
            return
        self._write(f"{key}.{name}", tensor, "state")

    def load_state(self, key: str, name: str, default: torch.Tensor) -> torch.Tensor:
        if key in self._resident_keys:
            return self._resident_state.get(key, {}).get(name, default.clone())
        if not self._exists(f"{key}.{name}", "state"):
            return default.clone()
        return self._read(f"{key}.{name}", "state")

    # ---- memory-aware residency --------------------------------------------
    def get_nbytes(self, key: str) -> int:
        """Size in bytes of a (disk-backed) parameter, read from its on-disk
        shape/dtype metadata without loading the actual tensor data."""
        if key in self._resident_keys:
            return self._resident_params[key].numel() * self._resident_params[key].element_size()
        _, meta_path = self._paths(key)
        with open(meta_path) as f:
            meta = json.load(f)
        nbytes = np.dtype(meta["dtype"]).itemsize
        for dim in meta["shape"]:
            nbytes *= dim
        return nbytes

    def is_resident(self, key: str) -> bool:
        return key in self._resident_keys

    def make_resident(self, key: str):
        """Promote an already disk-backed parameter to live permanently in
        RAM: no further disk I/O for its weight/bias, gradient, or optimizer
        state."""
        if key in self._resident_keys:
            return
        self._resident_params[key] = self._read(key)
        self._resident_keys.add(key)

    def auto_configure_residency(self, keys: Iterable[str], memory_fraction: float = 0.7) -> Set[str]:
        """Decide, based on *currently available* system RAM, how many of
        `keys` can be kept permanently resident instead of streamed from
        disk on every forward/backward -- so the caller doesn't have to make
        that call per layer.

        Greedily fits the largest parameters first into a budget of
        `memory_fraction` of available RAM (leaving the rest for the OS,
        activations, and everything else the process needs), promotes the
        ones that fit, and leaves the remainder disk-offloaded as before.
        """
        keys = list(keys)
        sizes = {k: self.get_nbytes(k) for k in keys}
        budget = int(psutil.virtual_memory().available * memory_fraction)

        resident: Set[str] = set()
        remaining = budget
        for key, nbytes in sorted(sizes.items(), key=lambda kv: -kv[1]):
            if nbytes <= remaining:
                resident.add(key)
                remaining -= nbytes

        for key in resident:
            self.make_resident(key)
        return resident

    def flush_resident_to_disk(self):
        """Write current values of all resident parameters back to disk, so
        the cache directory reflects a full, up-to-date checkpoint even for
        keys that never touched disk again after being promoted."""
        for key, tensor in self._resident_params.items():
            self._write(key, tensor)

    def shutdown(self):
        self._executor.shutdown(wait=True)
