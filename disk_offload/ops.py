"""Custom autograd Functions that never let a layer's weight tensor sit in
the autograd graph. The weight is loaded from disk right before it's used
and dropped immediately afterwards, both in `forward` and in `backward`.

This is the key trick that keeps peak RAM bounded to O(1) layer's worth of
parameters (plus activations) instead of O(N) for an N-layer network: a
normal `nn.Linear`/`nn.Conv2d` autograd graph keeps every weight tensor
alive from forward until its matching backward runs, i.e. all N layers'
weights are resident simultaneously by the time the last layer's forward
finishes. Here, gradients w.r.t. weights are computed manually and written
straight to disk, so the weight tensor is never captured by the tape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

from .storage import DiskTensorStore


class OffloadedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_anchor, weight_key, bias_key, store: DiskTensorStore):
        # `grad_anchor` is a dummy requires_grad tensor: it forces autograd to
        # build a graph node (and later call backward) even when `input`
        # itself doesn't require grad -- e.g. the very first layer of a
        # network, whose input is raw data with requires_grad=False. Without
        # it, that layer's weight gradient would silently never be computed.
        weight = store.load(weight_key)
        bias = store.load(bias_key) if bias_key else None
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input)
        ctx.weight_key = weight_key
        ctx.bias_key = bias_key
        ctx.store = store
        del weight, bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        store: DiskTensorStore = ctx.store
        store.begin_backward_for(ctx.weight_key)
        weight = store.load(ctx.weight_key)

        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None

        flat_grad_out = grad_output.reshape(-1, grad_output.shape[-1])
        flat_input = input.reshape(-1, input.shape[-1])
        grad_weight = flat_grad_out.t().matmul(flat_input)
        store.save_grad(ctx.weight_key, grad_weight)

        if ctx.bias_key:
            grad_bias = flat_grad_out.sum(0)
            store.save_grad(ctx.bias_key, grad_bias)

        del weight
        store.finish_backward_for(ctx.weight_key)
        return grad_input, None, None, None, None


class OffloadedConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_anchor, weight_key, bias_key, store: DiskTensorStore,
                stride, padding, dilation, groups):
        # See OffloadedLinearFn for why `grad_anchor` is needed.
        weight = store.load(weight_key)
        bias = store.load(bias_key) if bias_key else None
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input)
        ctx.weight_shape = weight.shape
        ctx.weight_key = weight_key
        ctx.bias_key = bias_key
        ctx.store = store
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        del weight, bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        store: DiskTensorStore = ctx.store
        store.begin_backward_for(ctx.weight_key)
        weight = store.load(ctx.weight_key)

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = conv2d_input(
                input.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )

        grad_weight = conv2d_weight(
            input, ctx.weight_shape, grad_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
        )
        store.save_grad(ctx.weight_key, grad_weight)

        if ctx.bias_key:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
            store.save_grad(ctx.bias_key, grad_bias)

        del weight
        store.finish_backward_for(ctx.weight_key)
        return grad_input, None, None, None, None, None, None, None, None
