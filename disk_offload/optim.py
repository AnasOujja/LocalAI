"""Optimizers whose parameter and state tensors live entirely on disk.

Offloaded layers (see layers.py) never register real `nn.Parameter`s, so a
regular `torch.optim.Optimizer` can't touch them. These optimizers instead
iterate over disk keys directly: for each key they load the current
parameter, its freshly written gradient (produced by the layer's custom
backward) and any running state (momentum / Adam moments), compute the
update on CPU, and write the updated parameter + state back to disk.
"""
from __future__ import annotations

from typing import Iterable, List

import torch

from .storage import DiskTensorStore


class DiskOffloadedSGD:
    def __init__(self, store: DiskTensorStore, param_keys: Iterable[str],
                 lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        self.store = store
        self.param_keys: List[str] = list(param_keys)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        for key in self.param_keys:
            grad = self.store.load_grad(key, missing_ok=True)
            if grad is None:
                continue
            param = self.store.load(key)
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            if self.momentum:
                buf = self.store.load_state(key, "momentum", default=torch.zeros_like(param))
                buf.mul_(self.momentum).add_(grad)
                grad = buf
                self.store.save_state(key, "momentum", buf)
            param -= self.lr * grad
            self.store.save(key, param)
            del param, grad

    def zero_grad(self):
        for key in self.param_keys:
            self.store.clear_grad(key)


class DiskOffloadedAdam:
    def __init__(self, store: DiskTensorStore, param_keys: Iterable[str],
                 lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.store = store
        self.param_keys: List[str] = list(param_keys)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for key in self.param_keys:
            grad = self.store.load_grad(key, missing_ok=True)
            if grad is None:
                continue
            param = self.store.load(key)
            if self.weight_decay:
                grad = grad + self.weight_decay * param

            m = self.store.load_state(key, "m", default=torch.zeros_like(param))
            v = self.store.load_state(key, "v", default=torch.zeros_like(param))

            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

            self.store.save(key, param)
            self.store.save_state(key, "m", m)
            self.store.save_state(key, "v", v)
            del param, m, v, grad

    def zero_grad(self):
        for key in self.param_keys:
            self.store.clear_grad(key)
