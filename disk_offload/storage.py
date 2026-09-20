"""Disk-backed tensor storage with background prefetching and RAM-sized groups.

Parameters, gradients and optimizer states normally live on disk as raw
binary files, materialized in RAM only for the brief window they are
actually needed. `configure_layer_batches` inspects available system memory
once, partitions whole layers into contiguous RAM-sized groups, and loads one
group at a time during forward and backward. Groups are temporary. They are
not pinned permanently, so layers still go to RAM and come back out even
when a complete group fits.

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
from typing import Dict, Iterable, Optional

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

        self._batch_for_key: dict[str, int] = {}
        self._batches: list[list[str]] = []
        self._active_batch_id: Optional[int] = None
        self._active_batch_params: Dict[str, torch.Tensor] = {}
        self._backward_remaining: dict[int, int] = {}

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
        self._write(key, tensor)

    def load(self, key: str) -> torch.Tensor:
        if key in self._active_batch_params:
            return self._active_batch_params[key]
        with self._lock:
            fut = self._futures.pop(key, None)
        if fut is not None:
            return fut.result()
        return self._read(key)

    def prefetch(self, key: str):
        """Kick off a background read for `key` if not already in flight."""
        with self._lock:
            if key in self._futures:
                return
            self._futures[key] = self._executor.submit(self._read, key)

    def exists(self, key: str) -> bool:
        return self._exists(key)

    # ---- gradients --------------------------------------------------------
    def save_grad(self, key: str, grad: torch.Tensor, accumulate: bool = False):
        if accumulate and self._exists(key, "grads"):
            existing = self._read(key, "grads")
            grad = existing + grad
        self._write(key, grad, "grads")

    def load_grad(self, key: str, missing_ok: bool = True) -> Optional[torch.Tensor]:
        if not self._exists(key, "grads"):
            if missing_ok:
                return None
            raise FileNotFoundError(f"No gradient stored for {key!r}")
        return self._read(key, "grads")

    def clear_grad(self, key: str):
        bin_path, meta_path = self._paths(key, "grads")
        for p in (bin_path, meta_path):
            if os.path.exists(p):
                os.remove(p)

    # ---- optimizer state ---------------------------------------------------
    def save_state(self, key: str, name: str, tensor: torch.Tensor):
        self._write(f"{key}.{name}", tensor, "state")

    def load_state(self, key: str, name: str, default: torch.Tensor) -> torch.Tensor:
        if not self._exists(f"{key}.{name}", "state"):
            return default.clone()
        return self._read(f"{key}.{name}", "state")

    # ---- memory-aware sizing -----------------------------------------------
    def get_nbytes(self, key: str) -> int:
        """Size in bytes of a (disk-backed) parameter, read from its on-disk
        shape/dtype metadata without loading the actual tensor data."""
        _, meta_path = self._paths(key)
        with open(meta_path) as f:
            meta = json.load(f)
        nbytes = np.dtype(meta["dtype"]).itemsize
        for dim in meta["shape"]:
            nbytes *= dim
        return nbytes

    # ---- temporary layer batches ------------------------------------------
    def configure_layer_batches(
        self,
        layer_groups: Iterable[Iterable[str]],
        memory_fraction: float = 0.7,
    ) -> list[list[str]]:
        """Partition complete layers into temporary groups that fit a RAM
        budget based on currently available memory.

        The groups remain contiguous in model order. A single layer larger
        than the budget still gets its own group because it must remain
        computable.
        """
        budget = int(psutil.virtual_memory().available * memory_fraction)
        batches: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for layer_group in layer_groups:
            keys = list(layer_group)
            layer_bytes = sum(self.get_nbytes(key) for key in keys)
            if current and current_bytes + layer_bytes > budget:
                batches.append(current)
                current = []
                current_bytes = 0
            current.extend(keys)
            current_bytes += layer_bytes
            if budget == 0 or layer_bytes > budget:
                batches.append(current)
                current = []
                current_bytes = 0
        if current:
            batches.append(current)

        self._batches = batches
        self._batch_for_key = {
            key: batch_id
            for batch_id, batch in enumerate(batches)
            for key in batch
        }
        self._backward_remaining = {
            batch_id: sum(key.endswith(".weight") for key in batch)
            for batch_id, batch in enumerate(batches)
        }
        return batches

    def batch_id(self, key: str) -> Optional[int]:
        return self._batch_for_key.get(key)

    def activate_batch_for(self, key: str):
        """Load the complete temporary group containing `key` into RAM."""
        batch_id = self._batch_for_key.get(key)
        if batch_id is None or batch_id == self._active_batch_id:
            return
        self.release_active_batch()
        self._active_batch_id = batch_id
        self._active_batch_params = {
            batch_key: self.load(batch_key)
            for batch_key in self._batches[batch_id]
        }

    def release_active_batch(self):
        self._active_batch_params.clear()
        self._active_batch_id = None

    def begin_backward_for(self, key: str):
        self.activate_batch_for(key)

    def finish_backward_for(self, key: str):
        batch_id = self._batch_for_key.get(key)
        if batch_id is None:
            return
        self._backward_remaining[batch_id] -= 1
        if self._backward_remaining[batch_id] <= 0:
            self.release_active_batch()

    def shutdown(self):
        self._executor.shutdown(wait=True)
