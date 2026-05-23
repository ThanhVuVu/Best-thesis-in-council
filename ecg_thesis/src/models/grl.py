from __future__ import annotations

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambd)
