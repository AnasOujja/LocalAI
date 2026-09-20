# Normal (In-RAM) Training

Plain baseline trainer for the same VGG-style CNN architecture used by the
disk-offloaded trainer at the repo root, but with **no offloading at all**:
every parameter is a regular resident `nn.Parameter`, trained with a
standard `torch.optim` optimizer. This is the reference point the
disk-offloaded approach is benchmarked against. See
[../benchmarks](../benchmarks).

## Usage

```powershell
python normal_training/train.py --dataset mnist --arch vgg11 --epochs 5
python normal_training/train.py --dataset cifar10 --arch vgg16 --epochs 10
```
