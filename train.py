"""Train a VGG-style CNN on MNIST or CIFAR-10, optionally streaming its
conv/linear weights from disk instead of keeping the whole model resident
in RAM. Compare `--offload` on vs off to see the RAM trade-off directly.

Examples:
    python train.py --dataset mnist   --offload --epochs 2
    python train.py --dataset cifar10 --offload --arch vgg16 --epochs 5
    python train.py --dataset mnist   --no-offload   # in-RAM baseline
"""
from __future__ import annotations

import argparse
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from disk_offload import (
    DiskOffloadedAdam,
    DiskOffloadedSGD,
    DiskTensorStore,
    PeakMemoryTracker,
    collect_param_groups,
    collect_param_keys,
)
from models.baseline_cnn import build_vgg_baseline
from models.cnn import build_vgg


def get_dataset(name: str, data_root: str):
    if name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        return train_ds, test_ds, 1, 10
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
        return train_ds, test_ds, 3, 10
    raise ValueError(f"Unknown dataset {name!r}")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run(args)
    print(f"offload={args.offload} arch={args.arch} dataset={args.dataset}")
    print(f"total_time={result['total_time_s']:.1f}s peak_rss={result['peak_rss_mb']:.1f}MB")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--arch", choices=["vgg11", "vgg13", "vgg16"], default="vgg11")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--limit", type=int, default=None, help="subset size for quick runs")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--cache-dir", default="./offload_cache")
    parser.add_argument("--offload", dest="offload", action="store_true", default=True)
    parser.add_argument("--no-offload", dest="offload", action="store_false")
    parser.add_argument("--fresh-cache", action="store_true", help="wipe offload cache before starting")
    parser.add_argument("--memory-fraction", type=float, default=0.7,
                         help="fraction of currently available system RAM used for temporary layer batches")
    parser.add_argument("--offload-width-multiplier", type=float, default=1.0,
                         help="width multiplier for the offloaded model only")
    return parser


def run(args) -> dict:
    """Run one training experiment and return structured metrics + the
    trained model, so callers (e.g. benchmarks/compare.py) can reuse it
    for further evaluation (inference timing, etc.) without retraining."""
    device = torch.device("cpu")
    train_ds, test_ds, in_channels, num_classes = get_dataset(args.dataset, args.data_root)
    image_size = 32

    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    resident_optimizer = None
    if args.offload:
        if args.fresh_cache:
            shutil.rmtree(args.cache_dir, ignore_errors=True)
        store = DiskTensorStore(args.cache_dir)
        model = build_vgg(
            args.arch,
            in_channels,
            num_classes,
            store,
            image_size=image_size,
            width_multiplier=args.offload_width_multiplier,
        ).to(device)
        offloaded_keys = collect_param_keys(model)
        layer_groups = collect_param_groups(model)
        batches = store.configure_layer_batches(layer_groups, memory_fraction=args.memory_fraction)
        largest_batch_bytes = max(
            (sum(store.get_nbytes(key) for key in batch) for batch in batches),
            default=0,
        )
        print(f"[memory budget] {len(batches)} temporary layer batches; "
              f"largest batch {largest_batch_bytes / (1024 ** 2):.1f} MB; "
              f"{len(offloaded_keys)} parameter tensors streamed")

        opt_cls = DiskOffloadedAdam if args.optimizer == "adam" else DiskOffloadedSGD
        optimizer = opt_cls(store, offloaded_keys, lr=args.lr)
        # BatchNorm affine params are tiny and stay as regular nn.Parameters.
        resident_params = [p for p in model.parameters() if p.requires_grad]
        if resident_params:
            resident_optimizer = torch.optim.Adam(resident_params, lr=args.lr) if args.optimizer == "adam" \
                else torch.optim.SGD(resident_params, lr=args.lr)
    else:
        model = build_vgg_baseline(args.arch, in_channels, num_classes, image_size=image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" \
            else torch.optim.SGD(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    history = []

    with PeakMemoryTracker() as tracker:
        start = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                if resident_optimizer is not None:
                    resident_optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                optimizer.step()
                if resident_optimizer is not None:
                    resident_optimizer.step()

                running_loss += loss.item()
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            acc = evaluate(model, test_loader, device)
            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / (step + 1)
            print(f"[epoch {epoch}] avg_loss={avg_loss:.4f} test_acc={acc:.4f} epoch_time={epoch_time:.1f}s")
            history.append({"epoch": epoch, "train_loss": avg_loss, "test_acc": acc, "epoch_time_s": epoch_time})
        elapsed = time.time() - start

    return {
        "history": history,
        "total_time_s": elapsed,
        "peak_rss_mb": tracker.peak_mb,
        "model": model,
        "test_loader": test_loader,
        "device": device,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": (
            sum(store.get_nbytes(key) for key in offloaded_keys)
            if args.offload else sum(p.numel() * p.element_size() for p in model.parameters())
        ),
    }


if __name__ == "__main__":
    main()

