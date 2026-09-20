"""Numerically validate that disk-offloaded layers produce the same forward
outputs and gradients as their standard in-RAM `nn.Linear`/`nn.Conv2d`
counterparts, given identical weights."""
import os
import shutil
import tempfile

import torch
import torch.nn as nn

from disk_offload import DiskTensorStore, OffloadedConv2d, OffloadedLinear


def _make_store():
    tmpdir = tempfile.mkdtemp(prefix="offload_test_")
    return DiskTensorStore(tmpdir), tmpdir


def test_offloaded_linear_matches_reference():
    torch.manual_seed(0)
    store, tmpdir = _make_store()
    try:
        offloaded = OffloadedLinear(16, 8, bias=True, store=store, name="lin")
        reference = nn.Linear(16, 8, bias=True)
        with torch.no_grad():
            reference.weight.copy_(store.load(offloaded.weight_key))
            reference.bias.copy_(store.load(offloaded.bias_key))

        x1 = torch.randn(4, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out1 = offloaded(x1)
        out2 = reference(x2)
        assert torch.allclose(out1, out2, atol=1e-6)

        out1.sum().backward()
        out2.sum().backward()

        grad_w = store.load_grad(offloaded.weight_key)
        grad_b = store.load_grad(offloaded.bias_key)
        assert torch.allclose(grad_w, reference.weight.grad, atol=1e-5)
        assert torch.allclose(grad_b, reference.bias.grad, atol=1e-5)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_offloaded_conv2d_matches_reference():
    torch.manual_seed(0)
    store, tmpdir = _make_store()
    try:
        offloaded = OffloadedConv2d(3, 6, kernel_size=3, store=store, name="conv", padding=1)
        reference = nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            reference.weight.copy_(store.load(offloaded.weight_key))
            reference.bias.copy_(store.load(offloaded.bias_key))

        x1 = torch.randn(2, 3, 8, 8, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out1 = offloaded(x1)
        out2 = reference(x2)
        assert torch.allclose(out1, out2, atol=1e-5)

        out1.sum().backward()
        out2.sum().backward()

        grad_w = store.load_grad(offloaded.weight_key)
        grad_b = store.load_grad(offloaded.bias_key)
        assert torch.allclose(grad_w, reference.weight.grad, atol=1e-4)
        assert torch.allclose(grad_b, reference.bias.grad, atol=1e-4)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-4)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_optimizer_step_updates_disk_param():
    from disk_offload import DiskOffloadedAdam

    torch.manual_seed(0)
    store, tmpdir = _make_store()
    try:
        offloaded = OffloadedLinear(4, 2, bias=True, store=store, name="lin")
        opt = DiskOffloadedAdam(store, offloaded.param_keys(), lr=0.1)

        before = store.load(offloaded.weight_key).clone()
        x = torch.randn(3, 4)
        out = offloaded(x)
        out.sum().backward()
        opt.step()
        after = store.load(offloaded.weight_key)

        assert not torch.allclose(before, after)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_layer_recreates_missing_bias_even_if_weight_exists():
    """Regression test: weight and bias initialization must be independent,
    so a stale/partial cache (weight present, bias missing) doesn't crash
    layer construction."""
    store, tmpdir = _make_store()
    try:
        OffloadedLinear(4, 2, bias=True, store=store, name="lin")
        store.clear_grad("lin.bias")  # no-op, just documents grads are separate
        bin_path, meta_path = store._paths("lin.bias")
        os.remove(bin_path)
        os.remove(meta_path)

        # Reconstructing with the same name must recreate only the bias.
        recreated = OffloadedLinear(4, 2, bias=True, store=store, name="lin")
        assert store.exists(recreated.bias_key)
        x = torch.randn(2, 4)
        out = recreated(x)
        out.sum().backward()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

