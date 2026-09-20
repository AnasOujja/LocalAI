"""Validate the temporary RAM-sized layer batch planner."""
import shutil
import tempfile
import types

import torch

from disk_offload import DiskTensorStore


def _make_store():
    tmpdir = tempfile.mkdtemp(prefix="offload_budget_test_")
    return DiskTensorStore(tmpdir), tmpdir


def test_configure_layer_batches_groups_whole_layers(monkeypatch):
    store, tmpdir = _make_store()
    try:
        keys = [f"layer{i}.weight" for i in range(3)]
        for key in keys:
            store.save(key, torch.zeros(250))

        fake_vmem = types.SimpleNamespace(available=2200)
        monkeypatch.setattr("disk_offload.storage.psutil.virtual_memory", lambda: fake_vmem)

        batches = store.configure_layer_batches([[keys[0]], [keys[1]], [keys[2]]], memory_fraction=1.0)

        assert batches == [[keys[0], keys[1]], [keys[2]]]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_temporary_batch_loads_and_releases_as_a_unit(monkeypatch):
    store, tmpdir = _make_store()
    try:
        keys = [f"layer{i}.weight" for i in range(2)]
        for index, key in enumerate(keys):
            store.save(key, torch.full((4,), float(index)))

        fake_vmem = types.SimpleNamespace(available=1000)
        monkeypatch.setattr("disk_offload.storage.psutil.virtual_memory", lambda: fake_vmem)
        store.configure_layer_batches([[keys[0]], [keys[1]]], memory_fraction=1.0)

        store.activate_batch_for(keys[0])
        assert torch.allclose(store.load(keys[0]), torch.zeros(4))
        assert store.batch_id(keys[0]) == store._active_batch_id
        store.release_active_batch()
        assert store._active_batch_id is None
        assert not store._active_batch_params
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
