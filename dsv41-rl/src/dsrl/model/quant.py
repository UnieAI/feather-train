"""Frozen quantized weights with a backward pass.

The base model stays exactly as served: fp8 (block 32, ue8m0 scales) dense weights and fp4
(e2m1, one ue8m0 scale per 32 inputs) routed experts. Forward reuses the reference GEMM kernels
so activations see serving numerics. Base weights never get gradients; backward only needs
dX = dY @ W, computed against W dequantized to bf16 on the fly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dsrl.model.ref import load_reference

FP8_BLOCK = 32
FP4_BLOCK = 32
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """[out, in] e4m3 with [ceil(out/32), ceil(in/32)] block scales -> dense."""
    out_f, in_f = weight.shape
    s = scale.float().repeat_interleave(FP8_BLOCK, 0)[:out_f].repeat_interleave(FP8_BLOCK, 1)[:, :in_f]
    return (weight.float() * s).to(dtype)


def dequant_fp4(weight: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """[out, in/2] packed e2m1 (low nibble first) with [out, in/32] scales -> dense [out, in]."""
    packed = weight.view(torch.uint8)
    table = _FP4_TABLE.to(packed.device)
    vals = torch.stack([table[(packed & 0x0F).long()], table[(packed >> 4).long()]], dim=-1).flatten(-2)
    s = scale.float().repeat_interleave(FP4_BLOCK, 1)
    return (vals * s).to(dtype)


def dequant(weight: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    if weight.dtype == torch.float4_e2m1fn_x2:
        return dequant_fp4(weight, weight.scale, dtype)
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8(weight, weight.scale, dtype)
    return weight.to(dtype)


class _FrozenQLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor):
        ref, _ = load_reference()
        ctx.weight = weight  # a frozen parameter: holding the reference costs nothing
        # the reference GEMMs take bf16 activations and allocate their output in the default
        # dtype, which the reference process sets to bf16 globally
        with ref.set_dtype(torch.bfloat16):
            return ref.linear(x.to(torch.bfloat16).contiguous(), weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w = dequant(ctx.weight, grad_out.dtype)
        return grad_out @ w, None


def frozen_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y = x W^T with serving numerics, differentiable in x only."""
    if weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
        return _FrozenQLinear.apply(x, weight)
    return F.linear(x.to(weight.dtype), weight)


def _ste(x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Forward value `quantized`, gradient of identity."""
    return x + (quantized - x).detach()


def fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    """The reference's in-place fp8 act_quant (block 32, ue8m0), as a straight-through op."""
    _, kernel = load_reference()
    with torch.no_grad():
        q = kernel.act_quant(x.detach().contiguous().clone(), FP8_BLOCK, "ue8m0", torch.float8_e8m0fnu, True)
    return _ste(x, q)


def fake_quant_fp4(x: torch.Tensor, block: int, scale_dtype: torch.dtype) -> torch.Tensor:
    """The reference's in-place fp4 quantization, as a straight-through op."""
    _, kernel = load_reference()
    with torch.no_grad():
        q = kernel.fp4_act_quant(x.detach().contiguous().clone(), block, True, scale_dtype=scale_dtype)
    return _ste(x, q)
