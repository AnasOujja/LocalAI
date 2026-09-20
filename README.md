# Disk-Offloaded Deep Learning Trainer

Train deep neural networks whose total parameter size exceeds available RAM
by streaming each layer's weights from disk into RAM only for the brief
moment they're actually used, instead of holding the whole model resident.

The repo includes a plain in-RAM baseline ([normal_training/](normal_training))
and a benchmark suite ([benchmarks/](benchmarks)) that trains both variants
under identical conditions and compares training accuracy, training speed,
peak memory, and inference performance -- see [Benchmarks](#benchmarks) below.

## How it works

A normal `nn.Linear`/`nn.Conv2d` keeps its weight tensor alive in the
autograd graph from the moment `forward()` runs until the matching
`backward()` call consumes it. For an N-layer network that means, by the
time the last layer's forward pass finishes, **all N layers' weights are
resident in RAM simultaneously** -- exactly what you don't want when the
model is bigger than your RAM.

This project avoids that by never letting the autograd graph capture the
weight tensor at all:

- Every offloaded layer ([disk_offload/layers.py](disk_offload/layers.py):
  `OffloadedLinear`, `OffloadedConv2d`) stores only a string *key*, not a
  tensor. The real weight/bias data lives on disk
  ([disk_offload/storage.py](disk_offload/storage.py): `DiskTensorStore`).
- A custom `torch.autograd.Function`
  ([disk_offload/ops.py](disk_offload/ops.py)) loads the weight from disk
  right before `F.linear`/`F.conv2d` runs, and `del`s it immediately after
  computing the output -- it never touches the autograd tape.
- Gradients w.r.t. the weight are computed **manually** in the custom
  `backward()` (re-loading the weight from disk once more) and written
  straight to a "grad" file on disk, again without ever registering as a
  leaf tensor requiring grad.
- A disk-backed optimizer ([disk_offload/optim.py](disk_offload/optim.py):
  `DiskOffloadedSGD` / `DiskOffloadedAdam`) reads each parameter + its
  gradient + its optimizer state (momentum / Adam moments) from disk,
  computes the update on CPU, and writes the result back -- one layer at a
  time.
- A small **grad-anchor** trick (`nn.Parameter(torch.zeros(1))` per layer,
  fed into the `Function`) forces autograd to always build a graph node
  and call `backward()`, even for the very first layer of a network whose
  input (raw pixels) doesn't itself require grad. Without it that layer's
  gradient would silently never be computed.
- Because each layer's weight is only ever resident for its own
  forward/backward, **peak RAM usage becomes roughly O(1 layer) + O(activations)
  instead of O(N layers)**, so model size can scale with disk capacity.
- To hide disk latency, [disk_offload/layers.py](disk_offload/layers.py)'s
  `OffloadedSequential` prefetches the *next* layer's weights on a
  background thread ([disk_offload/storage.py](disk_offload/storage.py))
  while the *current* layer is still computing, overlapping I/O with CPU
  compute.
- BatchNorm/ReLU/Pool layers are left as ordinary resident `nn.Module`s --
  they're parameter-free or negligibly small, so there's no benefit to
  offloading them.

### Memory-aware residency (automatic, no per-layer decisions)

Streaming *every* layer from disk is wasteful once the model is small
enough to partly (or fully) fit in RAM -- so the store doesn't do that.
Instead, [`DiskTensorStore.auto_configure_residency`](disk_offload/storage.py)
is called once, after the model is built:

1. It checks *currently available* system RAM (`psutil.virtual_memory().available`).
2. It takes a configurable fraction of that as a budget (`--memory-fraction`,
   default `0.7`, leaving headroom for activations, the OS, and everything
   else the process needs).
3. It greedily fits as many parameter tensors (largest first) into that
   budget as possible and **promotes** them to full RAM residency: from then
   on they behave exactly like normal `nn.Parameter`s -- no disk read/write
   at all, for their weights, gradients, *or* optimizer state.
4. Whatever doesn't fit keeps streaming from disk exactly as before.

This means you never have to decide, layer by layer, what should be
offloaded -- the whole model is handed to `auto_configure_residency` once,
and it figures out the split based on what's actually available on the
machine it's running on. On a machine with enough RAM, this makes
disk-offloaded training converge to the same performance as the in-RAM
baseline automatically (see the `--memory-fraction 0.0` vs default
comparison in [Benchmarks](#benchmarks)). A `flush_resident_to_disk()` call
at the end of training writes the final values of resident parameters back
to disk, so the cache directory is always a complete, up-to-date checkpoint.

```
disk_offload/
  storage.py      DiskTensorStore: raw binary tensor files + background prefetch + memory-aware residency
  ops.py          Custom autograd Functions (Linear, Conv2d)
  layers.py       OffloadedLinear, OffloadedConv2d, OffloadedSequential
  optim.py        DiskOffloadedSGD, DiskOffloadedAdam
  memory_utils.py PeakMemoryTracker (RSS sampling)
models/
  cnn.py          Disk-offloaded VGG-style CNN (vgg11/13/16)
  baseline_cnn.py In-RAM reference CNN, structurally identical, for comparison
train.py          CLI: train on MNIST or CIFAR-10, offloaded or in-RAM
normal_training/  Standalone plain in-RAM trainer (no offloading at all)
benchmarks/
  compare.py      Runs both variants under identical conditions, produces plots + results.json
  over_ram.py     Builds and evaluates a model whose parameters exceed physical RAM
  results/        Generated plots + results.json (see Benchmarks section)
tests/
  test_correctness.py  Offloaded layers vs nn.Linear/nn.Conv2d: forward + grad match
  test_training.py     Full-model offloaded vs baseline: loss trajectories match, loss converges
```

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
# Disk-offloaded training (default) on MNIST
python train.py --dataset mnist --arch vgg11 --epochs 5

# Disk-offloaded training on CIFAR-10 with a deeper network
python train.py --dataset cifar10 --arch vgg16 --epochs 10

# In-RAM baseline for comparison (same architecture, no offloading)
python train.py --dataset mnist --no-offload --epochs 5

# Quick smoke test on a small subset
python train.py --dataset mnist --limit 256 --epochs 1 --fresh-cache

# Force (almost) everything to stream from disk, ignoring available RAM
python train.py --dataset mnist --memory-fraction 0.0
```

Each run prints per-epoch loss/accuracy plus total wall time and peak RSS
memory, so you can directly compare `--offload` vs `--no-offload`. It also
prints a `[memory budget]` line showing how many of the model's parameter
tensors were kept resident in RAM vs. left streaming from disk, e.g.:

```
[memory budget] 20/20 parameter tensors kept resident in RAM (36.2 MB); 0 still streamed from disk
```

`--memory-fraction` (default `0.7`) controls how much of *currently
available* RAM the residency budget is allowed to use -- lower it to leave
more headroom for other processes, or set it to `0.0` to force the old
fully-streamed behavior (useful for testing/benchmarking the disk path in
isolation).

Disk cache location defaults to `./offload_cache` (override with
`--cache-dir`); pass `--fresh-cache` to wipe it before a run.

## Validation

```powershell
python -m pytest tests -v
```

- `test_correctness.py` checks that `OffloadedLinear`/`OffloadedConv2d`
  produce bit-for-bit-equivalent (within float tolerance) forward outputs
  and gradients as `nn.Linear`/`nn.Conv2d` given identical weights.
- `test_training.py` builds a disk-offloaded VGG and a structurally
  identical in-RAM baseline from the same initial weights, trains both for
  several steps on identical batches, and checks their loss trajectories
  match closely and both converge -- validating the full forward+backward+
  optimizer pipeline, not just single layers. It also checks that promoting
  only *some* layers to RAM residency (mixed mode) still matches the
  baseline exactly.
- `test_memory_budget.py` checks that `auto_configure_residency` respects
  the given RAM budget, that resident parameters survive their on-disk file
  being deleted (proving they're no longer read from disk), and that
  `flush_resident_to_disk` writes the latest in-RAM value back out.

## Test Hardware

The measurements in this README and in [REPORT.md](REPORT.md) were captured
on the following machine:

| Property | Measured value |
|---|---:|
| CPU | Intel Core i5-1135G7 |
| Physical cores | 4 |
| Logical processors | 8 |
| Physical RAM | 7.79 GB |
| Available RAM when measured | 0.30 GB to 0.42 GB |
| Free system-drive space | 81.30 GB |
| Python | 3.10.0 for the original benchmark capture |
| PyTorch | 2.14.0+cpu |

Available RAM changes while Windows and other programs are running, so the
important planning number is physical RAM plus the live availability check.
The trainer uses that live value rather than relying only on this table.

## Benchmarks

Generated by [benchmarks/compare.py](benchmarks/compare.py), which trains
three variants back-to-back on the *same* architecture and hyperparameters:
the disk-offloaded trainer with its default memory-aware residency budget,
the disk-offloaded trainer forced to stream every layer from disk
(`--memory-fraction 0.0`, i.e. the old behavior / worst case), and the
[normal_training/](normal_training) in-RAM baseline. It then measures
training accuracy, wall time, peak RSS and inference performance for all
three. Reproduce with:

```powershell
python benchmarks/compare.py --dataset mnist --arch vgg11 --epochs 2 --limit 1000
```

Setup: VGG11 (20 parameter tensors, ~36 MB total), MNIST (1000-image
subset, resized to 32x32), batch size 64, Adam lr=1e-3, 2 epochs, CPU-only.
Raw numbers in [benchmarks/results/results.json](benchmarks/results/results.json).

| Metric                        | offload (auto memory budget) | offload (forced streaming) | in-RAM baseline |
|--------------------------------|---------------:|---------------:|----------------:|
| Final train loss               | 0.949           | 1.364           | 1.521            |
| Final test accuracy            | 65.0%           | 53.3%           | 22.7%            |
| Total training time (2 epochs) | 26.1 s          | 106.2 s         | 23.4 s           |
| Peak RSS during training       | 527.2 MB        | 497.1 MB        | 376.7 MB         |
| Inference latency (ms/batch)   | 152.1 ms        | 152.9 ms        | 133.4 ms         |
| Inference throughput           | 409 samples/s   | 407 samples/s   | 466 samples/s    |

**Reading the results:**
- **"offload (auto memory budget)" tracks the in-RAM baseline's training
  time closely** (26.1 s vs 23.4 s) -- because with default settings, all 20
  of VGG11's parameter tensors comfortably fit in currently-available RAM
  and get promoted to full residency (`[memory budget] 20/20 parameter
  tensors kept resident`), so this run does effectively zero disk I/O once
  training starts. This is the point of the memory-aware residency feature:
  on a machine with enough RAM, disk-offloaded training automatically
  behaves like normal in-RAM training, with no manual per-layer tuning.
- **"offload (forced streaming)" is ~4x slower** than the auto/baseline
  runs (106.2 s vs ~25 s) -- with `--memory-fraction 0.0`, every layer's
  weight is read from disk twice per step (forward + backward) and the
  optimizer reads+writes it again, for every single training step. This is
  the cost you pay only for the *portion* of a model that genuinely doesn't
  fit in RAM; on this small model it's 100% of the layers, which is why the
  gap is so large here.
- **Test accuracy differs across variants because each run uses an
  independent random weight initialization**, not because offloading
  changes the math -- this benchmark measures speed/memory/throughput, not
  numerical equivalence. Numerical equivalence (same initial weights, same
  batches, same loss trajectory) is separately and rigorously verified in
  [tests/test_correctness.py](tests/test_correctness.py) and
  [tests/test_training.py](tests/test_training.py), including a
  mixed-residency case.
- **Peak RAM** is dominated by fixed overhead (Python/PyTorch/OS, the
  background prefetch thread pool) at this tiny model size (~36 MB of
  weights); the auto and forced-streaming offload runs are close to each
  other and a bit above the baseline. The RAM benefit of streaming from
  disk only shows up once *model size* approaches or exceeds available
  RAM -- which this deliberately small demo network does not do, so the
  benchmark's job here is to validate correctness and characterize
  overhead/timing, not to show a memory win on a model that already fits
  comfortably in RAM.
- **Inference** is run in-RAM for all three (weights already loaded once
  via the disk-backed layers by the time inference runs); latency is close
  between the two offloaded variants and both are somewhat slower than the
  baseline due to the extra layer-wrapper indirection.

![Training loss](benchmarks/results/loss_curve.png)
![Test accuracy](benchmarks/results/accuracy_curve.png)
![Peak memory](benchmarks/results/peak_memory_bar.png)
![Training time](benchmarks/results/training_time_bar.png)
![Inference latency](benchmarks/results/inference_latency_bar.png)
![Inference throughput](benchmarks/results/inference_throughput_bar.png)

## Over-RAM Capacity Test

The small VGG examples above fit in memory. To test the actual purpose of
disk streaming, [benchmarks/over_ram.py](benchmarks/over_ram.py) builds an
`OverRAMMLP` sized from the machine's physical RAM. On this laptop it used
36 hidden layers of width 8192, producing 39 disk-backed parameter tensors
and 9.03 GB of parameters. That is 1.16 times the machine's 7.79 GB of
physical RAM.

The test uses `--memory-fraction 0.0`, creates the full cache on disk, then
runs one real forward pass on an MNIST test image. It deliberately does not
claim to complete a useful training epoch at this size. A training epoch
would require many disk reads and writes and would not be a fair use of this
machine for a first capacity validation.

| Measurement | Result |
|---|---:|
| Model parameter size | 9.03 GB |
| Physical RAM | 7.79 GB |
| Parameter-to-RAM ratio | 1.16x |
| Resident parameter tensors | 0 of 39 |
| Cache creation time | 101.0 s |
| Cache size | 9.03 GB |
| One-image forward time | 17.0 s |
| Peak RSS during forward | 0.857 GB |
| MNIST label | 7 |
| Predicted class | 0 |

The important result is capacity, not classification quality from one
untrained example. The complete model existed on disk and was evaluated
without keeping all 9.03 GB in RAM. Peak RSS stayed far below the model
size, while each individual layer was loaded only when needed.

The raw capture is in
[benchmarks/results/over_ram_results.json](benchmarks/results/over_ram_results.json).
Run it again with:

```powershell
python benchmarks/over_ram.py --cache-dir .\over_ram_cache_final
```


## Known limitations / next steps

- Only **parameters** are offloaded, not activations -- for very deep/wide
  networks with large batch sizes, activation memory (needed for backward)
  can still dominate. Adding `torch.utils.checkpoint`-style activation
  recomputation would extend the same idea to activations.
- Each offloaded layer's weight is read from disk **twice** per training
  step (forward + backward); prefetching hides most of this but a fast
  disk (SSD/NVMe) matters a lot for throughput.
- Currently CPU-only, matching the "use whole RAM + CPU compute" goal;
  extending `DiskTensorStore` to stream disk → pinned host memory → GPU
  would let the same technique offload GPU-VRAM-bound models too.

## Author

Anas Oujja

## License

[MIT](LICENSE)
