"""End-to-end validation: a disk-offloaded VGG and a structurally identical
in-RAM baseline, given the same initial weights and the same input batches,
should produce (near-)identical loss trajectories, and loss should trend
downward as training proceeds. Also sanity-checks the peak-RSS tracker.
"""
import shutil
import tempfile

import torch
import torch.nn as nn

from disk_offload import DiskOffloadedAdam, DiskTensorStore, PeakMemoryTracker, collect_param_keys
from disk_offload.layers import OffloadedConv2d, OffloadedLinear
from models.baseline_cnn import build_vgg_baseline
from models.cnn import build_vgg


def _sync_baseline_from_store(offloaded_model, baseline_model, store):
    off_layers = [m for m in offloaded_model.modules() if isinstance(m, (OffloadedConv2d, OffloadedLinear))]
    base_layers = [m for m in baseline_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(off_layers) == len(base_layers)
    with torch.no_grad():
        for off, base in zip(off_layers, base_layers):
            base.weight.copy_(store.load(off.weight_key))
            if off.bias_key:
                base.bias.copy_(store.load(off.bias_key))


def test_offloaded_training_matches_baseline_and_converges():
    torch.manual_seed(0)
    tmpdir = tempfile.mkdtemp(prefix="offload_train_test_")
    try:
        store = DiskTensorStore(tmpdir)
        offloaded_model = build_vgg("vgg11", in_channels=3, num_classes=4, store=store, image_size=32)
        baseline_model = build_vgg_baseline("vgg11", in_channels=3, num_classes=4, image_size=32)
        _sync_baseline_from_store(offloaded_model, baseline_model, store)

        off_optimizer = DiskOffloadedAdam(store, collect_param_keys(offloaded_model), lr=1e-3)
        # Only conv/linear weights are optimized on the offloaded side; keep
        # the baseline's optimizer scope identical (conv/linear only), BN
        # params use their deterministic default init on both sides.
        base_conv_linear_params = [
            p for m in baseline_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
            for p in m.parameters()
        ]
        base_optimizer = torch.optim.Adam(base_conv_linear_params, lr=1e-3)

        criterion = nn.CrossEntropyLoss()
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 4, (2,))

        off_losses = []
        base_losses = []
        for step in range(20):
            torch.manual_seed(100 + step)
            off_optimizer.zero_grad()
            out_off = offloaded_model(x)
            loss_off = criterion(out_off, y)
            loss_off.backward()
            off_optimizer.step()
            off_losses.append(loss_off.item())

            torch.manual_seed(100 + step)
            base_optimizer.zero_grad()
            out_base = baseline_model(x)
            loss_base = criterion(out_base, y)
            loss_base.backward()
            base_optimizer.step()
            base_losses.append(loss_base.item())

        for lo, lb in zip(off_losses, base_losses):
            assert abs(lo - lb) < 1e-3, (off_losses, base_losses)

        assert off_losses[-1] < off_losses[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_peak_memory_tracker_reports_positive_value():
    with PeakMemoryTracker(interval=0.01) as tracker:
        _ = [torch.randn(1000, 1000) for _ in range(5)]
    assert tracker.peak_mb > 0


def test_mixed_residency_matches_baseline():
    """Promoting some (but not all) layers to RAM residency shouldn't
    change the math -- it only changes where the data physically lives."""
    torch.manual_seed(0)
    tmpdir = tempfile.mkdtemp(prefix="offload_residency_test_")
    try:
        store = DiskTensorStore(tmpdir)
        offloaded_model = build_vgg("vgg11", in_channels=3, num_classes=4, store=store, image_size=32)
        baseline_model = build_vgg_baseline("vgg11", in_channels=3, num_classes=4, image_size=32)
        _sync_baseline_from_store(offloaded_model, baseline_model, store)

        offloaded_keys = collect_param_keys(offloaded_model)
        for key in offloaded_keys[::2]:  # promote every other parameter to full residency
            store.make_resident(key)
        assert any(store.is_resident(k) for k in offloaded_keys)
        assert not all(store.is_resident(k) for k in offloaded_keys)

        off_optimizer = DiskOffloadedAdam(store, offloaded_keys, lr=1e-3)
        base_optimizer = torch.optim.Adam(
            [p for m in baseline_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)) for p in m.parameters()],
            lr=1e-3,
        )

        criterion = nn.CrossEntropyLoss()
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 4, (2,))

        for step in range(5):
            torch.manual_seed(200 + step)
            off_optimizer.zero_grad()
            loss_off = criterion(offloaded_model(x), y)
            loss_off.backward()
            off_optimizer.step()

            torch.manual_seed(200 + step)
            base_optimizer.zero_grad()
            loss_base = criterion(baseline_model(x), y)
            loss_base.backward()
            base_optimizer.step()

            assert abs(loss_off.item() - loss_base.item()) < 1e-3
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
