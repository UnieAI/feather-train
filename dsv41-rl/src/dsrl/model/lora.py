"""LoRA on the frozen fp8 attention projections.

Targets are chosen so a trained adapter can be merged back into the served checkpoint: the
attention projections are fp8 block-32 tensors that sglang loads as-is, so merge = dequantize,
add B@A, requantize with the same block scheme.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dsrl.model.quant import frozen_linear

DEFAULT_TARGETS = ("wq_a", "wq_b", "wkv", "wo_b")


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, device=None):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32, device=device))
        self.B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.float() @ self.A.t()) @ self.B.t() * self.scaling).to(x.dtype)

    @torch.no_grad()
    def delta(self) -> torch.Tensor:
        """The dense update B @ A * scaling, [out, in] fp32, for merging."""
        return self.B @ self.A * self.scaling


def lora_linear(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Base projection with serving numerics, plus the adapter if one is attached."""
    y = frozen_linear(x, linear.weight)
    adapter = getattr(linear, "lora", None)
    if adapter is not None:
        y = y + adapter(x)
    return y


def attach(block: nn.Module, rank: int, alpha: float, targets=DEFAULT_TARGETS) -> list[nn.Parameter]:
    """Attach adapters to `block.attn.<target>`; returns the new trainable parameters."""
    params = []
    for name in targets:
        lin = getattr(block.attn, name)
        lin.lora = LoRA(lin.in_features, lin.out_features, rank, alpha, device=lin.weight.device)
        params += list(lin.lora.parameters())
    return params
