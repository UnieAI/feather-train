"""Build reference modules for single layers and fill them from the HF checkpoint."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open

from dsrl.model.quant import dequant_fp8
from dsrl.model.ref import DEFAULT_CKPT, load_reference, set_globals


class CheckpointIndex:
    """Tensor name -> shard file for an HF checkpoint, opened lazily (mmap'd by safetensors)."""

    def __init__(self, ckpt: str = DEFAULT_CKPT):
        self.ckpt = ckpt
        self.weight_map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self._files: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        f = self._files.get(shard)
        if f is None:
            f = self._files[shard] = safe_open(os.path.join(self.ckpt, shard), framework="pt", device="cpu")
        return f.get_tensor(name)

    def names_with_prefix(self, prefix: str) -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]


def _assign(param: torch.Tensor, src: torch.Tensor, name: str) -> None:
    if param.dtype == torch.float4_e2m1fn_x2:
        assert src.dtype == torch.int8 and src.shape == param.shape, (name, src.dtype, src.shape, param.shape)
        param.data.view(torch.uint8).copy_(src.view(torch.uint8))
    else:
        assert src.shape == param.shape, (name, src.shape, param.shape)
        param.data.copy_(src)


def load_block(layer_id: int, args, index: CheckpointIndex, device="cuda", engram_layout=None):
    """A reference `Block` for `layer_id` on `device`, weights loaded, all parameters frozen."""
    ref, _ = load_reference()
    set_globals(args)
    prev_dev, prev_dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = ref.Block(layer_id, args, engram_layout)
    finally:
        torch.set_default_device(prev_dev)
        torch.set_default_dtype(prev_dtype)

    params = dict(block.named_parameters())
    prefix = f"layers.{layer_id}."
    pending = defaultdict(dict)
    for name in index.names_with_prefix(prefix):
        local = name[len(prefix) :]
        if local.startswith("attn.wo_a."):
            # the reference keeps wo_a dequantized to bf16 (it is applied with an einsum)
            pending["wo_a"][local.rsplit(".", 1)[1]] = index.get(name)
            continue
        if local.startswith("engram.") and engram_layout is None:
            continue
        if local not in params:
            raise KeyError(f"checkpoint tensor {name} has no parameter in the reference Block")
        _assign(params[local], index.get(name), name)
    if "wo_a" in pending:
        w = pending["wo_a"]
        _assign(params["attn.wo_a.weight"], dequant_fp8(w["weight"], w["scale"]), "attn.wo_a.weight")

    for p in block.parameters():
        p.requires_grad_(False)
    return block
