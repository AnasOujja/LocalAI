"""Standalone baseline trainer: an ordinary, fully in-RAM VGG-style CNN with
no disk offloading whatsoever. Used as the reference point that
`disk_offload` is benchmarked against (see ../benchmarks/compare.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline_cnn import build_vgg_baseline  # noqa: E402


class PeakMemoryTracker:
    """Minimal RSS peak sampler, duplicated here so this folder has no
    dependency on the disk_offload package."""

    def __init__(self, interval: float = 0.05):
        import threading
        self._process = psutil.Process()
        self._interval = interval
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._peak_bytes = max(self._peak_bytes, self._process.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(self._interval)

    def __enter__(self):
        self._peak_bytes = self._process.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 ** 2)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--arch", choices=["vgg11", "vgg13", "vgg16"], default="vgg11")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--limit", type=int, default=None, help="subset size for quick runs")
    default_data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    parser.add_argument("--data-root", default=default_data_root)
    return parser


def run(args) -> dict:
    """Run one plain, fully-resident training experiment and return
    structured metrics + the trained model."""
    device = torch.device("cpu")
    train_ds, test_ds, in_channels, num_classes = get_dataset(args.dataset, args.data_root)
    image_size = 32

    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

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
    }


def main():
    args = build_arg_parser().parse_args()
    result = run(args)
    print(f"offload=False arch={args.arch} dataset={args.dataset}")
    print(f"total_time={result['total_time_s']:.1f}s peak_rss={result['peak_rss_mb']:.1f}MB")


if __name__ == "__main__":
    main()
