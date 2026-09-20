"""Head-to-head benchmark: disk-offloaded training (root train.py) vs.
plain in-RAM training (normal_training/train.py) on the same architecture,
dataset and hyperparameters.

Three variants are compared:
  - "offload (auto memory budget)": default --memory-fraction, i.e. the
    memory-aware residency behavior described in the README -- layers that
    fit in currently-available RAM are kept fully resident automatically.
  - "offload (forced streaming)": --memory-fraction 0.0, i.e. every layer
    streams from disk on every pass, regardless of available RAM. This is
    the worst case / what you get on a machine where the model genuinely
    doesn't fit in RAM.
  - "in-RAM baseline": normal_training, no offloading code involved at all.

Measures, for each: per-epoch train loss / test accuracy, total training
wall time, peak resident memory (RSS), and inference latency/throughput.

Writes benchmarks/results/results.json plus a handful of comparison plots.

Usage:
    python benchmarks/compare.py --dataset mnist --arch vgg11 --epochs 3 --limit 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train as offload_train  # noqa: E402
from normal_training import train as baseline_train  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def benchmark_inference(model, test_loader, device, num_batches: int = 20, warmup: int = 3):
    model.eval()
    batches = []
    for i, (x, y) in enumerate(test_loader):
        if i >= num_batches + warmup:
            break
        batches.append(x.to(device))
    if not batches:
        return {"avg_latency_ms": 0.0, "throughput_samples_per_s": 0.0}

    with torch.no_grad():
        for x in batches[:warmup]:
            model(x)

        timed = batches[warmup:] or batches
        start = time.time()
        n_samples = 0
        for x in timed:
            model(x)
            n_samples += x.size(0)
        elapsed = time.time() - start

    avg_latency_ms = (elapsed / len(timed)) * 1000
    throughput = n_samples / elapsed if elapsed > 0 else 0.0
    return {"avg_latency_ms": avg_latency_ms, "throughput_samples_per_s": throughput}


def run_variant(module, common: dict, extra: dict) -> dict:
    parser = module.build_arg_parser()
    args = parser.parse_args([])
    for k, v in {**common, **extra}.items():
        setattr(args, k, v)
    result = module.run(args)
    result["inference"] = benchmark_inference(result["model"], result["test_loader"], result["device"])
    return result


def to_serializable(result: dict) -> dict:
    return {
        "history": result["history"],
        "total_time_s": result["total_time_s"],
        "peak_rss_mb": result["peak_rss_mb"],
        "inference": result["inference"],
    }


def make_plots(results: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    labels = list(results.keys())
    colors = COLORS[: len(labels)]
    markers = ["o", "s", "^"]

    # Loss / accuracy curves
    for metric, fname, title, ylabel in [
        ("train_loss", "loss_curve.png", "Training loss", "Train loss"),
        ("test_acc", "accuracy_curve.png", "Test accuracy", "Test accuracy"),
    ]:
        plt.figure(figsize=(6.5, 4))
        for label, marker in zip(labels, markers):
            hist = results[label]["history"]
            plt.plot([h["epoch"] for h in hist], [h[metric] for h in hist], marker=marker, label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    # Bar charts
    for metric_path, fname, title, ylabel in [
        (("peak_rss_mb",), "peak_memory_bar.png", "Peak memory usage during training", "Peak RSS (MB)"),
        (("total_time_s",), "training_time_bar.png", "Training wall time", "Total training time (s)"),
        (("inference", "avg_latency_ms"), "inference_latency_bar.png", "Inference latency",
         "Avg inference latency (ms/batch)"),
        (("inference", "throughput_samples_per_s"), "inference_throughput_bar.png", "Inference throughput",
         "Throughput (samples/sec)"),
    ]:
        values = []
        for label in labels:
            v = results[label]
            for key in metric_path:
                v = v[key]
            values.append(v)
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=colors)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=10, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()


def print_summary(results: dict):
    labels = list(results.keys())
    print("\n=== Summary ===")
    header = f"{'metric':<32}" + "".join(f"{label:>22}" for label in labels)
    print(header)

    def row(name, getter, fmt="{:.4f}"):
        print(f"{name:<32}" + "".join(fmt.format(getter(results[label])).rjust(22) for label in labels))

    row("final train loss", lambda r: r["history"][-1]["train_loss"])
    row("final test accuracy", lambda r: r["history"][-1]["test_acc"])
    row("total training time (s)", lambda r: r["total_time_s"], "{:.1f}")
    row("peak RSS (MB)", lambda r: r["peak_rss_mb"], "{:.1f}")
    row("inference latency (ms/batch)", lambda r: r["inference"]["avg_latency_ms"], "{:.2f}")
    row("inference throughput (samples/s)", lambda r: r["inference"]["throughput_samples_per_s"], "{:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--arch", choices=["vgg11", "vgg13", "vgg16"], default="vgg11")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--limit", type=int, default=None, help="subset size for quick runs")
    parser.add_argument("--memory-fraction", type=float, default=0.7,
                         help="memory-fraction used for the 'offload (auto memory budget)' variant")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    common = dict(dataset=args.dataset, arch=args.arch, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr, optimizer=args.optimizer, limit=args.limit)

    results = {}

    print(">>> Running disk-offloaded training (auto memory budget)...")
    results["offload (auto memory budget)"] = run_variant(
        offload_train, common,
        extra={"offload": True, "fresh_cache": True, "memory_fraction": args.memory_fraction,
               "cache_dir": "./offload_cache_auto"},
    )

    print(">>> Running disk-offloaded training (forced streaming, memory-fraction=0)...")
    results["offload (forced streaming)"] = run_variant(
        offload_train, common,
        extra={"offload": True, "fresh_cache": True, "memory_fraction": 0.0,
               "cache_dir": "./offload_cache_streamed"},
    )

    print(">>> Running in-RAM baseline training...")
    results["in-RAM baseline"] = run_variant(baseline_train, common, extra={})

    print_summary(results)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({
            "config": common,
            **{label: to_serializable(res) for label, res in results.items()},
        }, f, indent=2)

    make_plots(results, args.out_dir)
    print(f"\nSaved results + plots to {args.out_dir}")


if __name__ == "__main__":
    main()

