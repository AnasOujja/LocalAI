"""Run a capacity-focused experiment with a model larger than physical RAM.

This intentionally stops after a real MNIST forward pass. A useful training
epoch for an 8+ GiB model is not a responsible benchmark on a machine with
7.79 GiB total RAM, but the capacity test proves that the model can be
created, stored, and evaluated one streamed layer at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

import psutil
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disk_offload import DiskTensorStore, PeakMemoryTracker, collect_param_keys
from models.over_ram import build_over_ram_mlp


def gb(value: int | float) -> float:
    return value / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--cache-dir", default="./over_ram_cache")
    parser.add_argument("--target-ram-multiple", type=float, default=1.15)
    parser.add_argument("--memory-fraction", type=float, default=0.0)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--output", default="./benchmarks/results/over_ram_results.json")
    args = parser.parse_args()

    memory = psutil.virtual_memory()
    cpu_count = os.cpu_count()
    disk_free = shutil.disk_usage(os.path.abspath(args.cache_dir)).free
    if args.fresh_cache:
        shutil.rmtree(args.cache_dir, ignore_errors=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    start_init = time.time()
    store = DiskTensorStore(args.cache_dir)
    model, hidden_layers, parameter_bytes = build_over_ram_mlp(
        store,
        total_ram_bytes=memory.total,
        target_ram_multiple=args.target_ram_multiple,
    )
    parameter_keys = collect_param_keys(model)
    resident_keys = store.auto_configure_residency(parameter_keys, memory_fraction=args.memory_fraction)
    init_seconds = time.time() - start_init

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    inputs, label = dataset[0]
    inputs = inputs.flatten().unsqueeze(0)

    model.eval()
    start_forward = time.time()
    with PeakMemoryTracker() as tracker, torch.no_grad():
        logits = model(inputs)
    forward_seconds = time.time() - start_forward
    prediction = int(logits.argmax(dim=1).item())

    cache_bytes = sum(
        os.path.getsize(os.path.join(root, filename))
        for root, _, files in os.walk(args.cache_dir)
        for filename in files
        if filename.endswith(".bin")
    )
    result = {
        "hardware": {
            "cpu": platform.processor(),
            "logical_processors": cpu_count,
            "physical_cores": psutil.cpu_count(logical=False),
            "physical_ram_gb": gb(memory.total),
            "available_ram_at_start_gb": gb(memory.available),
            "free_disk_at_start_gb": gb(disk_free),
            "torch_version": torch.__version__,
        },
        "model": {
            "name": "OverRAMMLP",
            "width": 8192,
            "hidden_layers": hidden_layers,
            "parameter_tensors": len(parameter_keys),
            "parameter_size_gb": gb(parameter_bytes),
            "parameter_to_physical_ram_ratio": parameter_bytes / memory.total,
            "resident_tensors": len(resident_keys),
        },
        "capacity_test": {
            "cache_size_gb": gb(cache_bytes),
            "initialization_seconds": init_seconds,
            "forward_seconds": forward_seconds,
            "peak_rss_gb": tracker.peak_mb / 1024,
            "mnist_label": int(label),
            "prediction": prediction,
            "memory_fraction": args.memory_fraction,
        },
    }
    with open(args.output, "w") as output_file:
        json.dump(result, output_file, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved capacity results to {args.output}")


if __name__ == "__main__":
    main()