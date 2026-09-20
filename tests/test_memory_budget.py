"""Validate the memory-aware residency budget: parameters that fit in
currently-available RAM get promoted to live permanently in memory (no more
disk I/O for them at all), while the rest keep streaming from disk -- and
the caller never has to make that decision layer by layer."""
import os
import shutil
import tempfile
import types

import torch

from disk_offload import DiskTensorStore


def _make_store():
    tmpdir = tempfile.mkdtemp(prefix="offload_budget_test_")
    return DiskTensorStore(tmpdir), tmpdir


def test_auto_configure_residency_respects_budget(monkeypatch):
    store, tmpdir = _make_store()
    try:
        # Three 1000-byte params (250 float32 elements each).
        keys = [f"p{i}" for i in range(3)]
        for key in keys:
            store.save(key, torch.zeros(250))

        fake_vmem = types.SimpleNamespace(available=2200)
        monkeypatch.setattr("disk_offload.storage.psutil.virtual_memory", lambda: fake_vmem)

        resident = store.auto_configure_residency(keys, memory_fraction=1.0)

        assert len(resident) == 2  # only 2 of the 3 1000-byte params fit in a 2200-byte budget
        for key in keys:
            assert store.is_resident(key) == (key in resident)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_auto_configure_residency_zero_budget_streams_everything(monkeypatch):
    store, tmpdir = _make_store()
    try:
        keys = [f"p{i}" for i in range(3)]
        for key in keys:
            store.save(key, torch.zeros(250))

        fake_vmem = types.SimpleNamespace(available=0)
        monkeypatch.setattr("disk_offload.storage.psutil.virtual_memory", lambda: fake_vmem)

        resident = store.auto_configure_residency(keys, memory_fraction=1.0)
        assert resident == set()
        assert not any(store.is_resident(key) for key in keys)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resident_param_survives_disk_file_removal():
    store, tmpdir = _make_store()
    try:
        store.save("w", torch.ones(4))
        store.make_resident("w")

        bin_path, meta_path = store._paths("w")
        os.remove(bin_path)
        os.remove(meta_path)

        assert torch.allclose(store.load("w"), torch.ones(4))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_flush_resident_to_disk_writes_latest_value():
    store, tmpdir = _make_store()
    try:
        store.save("w", torch.ones(4))
        store.make_resident("w")
        store.save("w", torch.full((4,), 5.0))  # updates the in-RAM cache only

        store.flush_resident_to_disk()

        on_disk = store._read("w")
        assert torch.allclose(on_disk, torch.full((4,), 5.0))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
