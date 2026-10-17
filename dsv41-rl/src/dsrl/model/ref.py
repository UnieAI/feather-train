"""Access to DeepSeek's reference implementation (`inference/` in the released checkpoint).

The reference is inference-only, but it defines the parameter layout we load weights into and
the quantized GEMM / quantization kernels we reuse in the forward pass. It is imported from the
checkpoint directory rather than vendored, so the trainer always matches the weights it serves.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import fields
from types import ModuleType

DEFAULT_CKPT = os.environ.get("DSRL_CKPT", "/models/deepseek-ai/DeepSeek-V4.1-Flash-1017")
# 1017 ships no inference/ directory; the code is shared with the base V4.1-Flash release.
DEFAULT_REF_DIR = os.environ.get("DSRL_REF_DIR", "/models/deepseek-ai/DeepSeek-V4.1-Flash/inference")

_ref: ModuleType | None = None
_kernel: ModuleType | None = None


def load_reference(ref_dir: str = DEFAULT_REF_DIR) -> tuple[ModuleType, ModuleType]:
    """Import the reference `model` and `kernel` modules once per process."""
    global _ref, _kernel
    if _ref is None:
        if ref_dir not in sys.path:
            sys.path.insert(0, ref_dir)
        _kernel = importlib.import_module("kernel")
        _ref = importlib.import_module("model")
    return _ref, _kernel


def model_args(max_seq_len: int, max_batch_size: int = 1, ref_dir: str = DEFAULT_REF_DIR, **overrides):
    """ModelArgs for the released shapes, sized for one training sequence."""
    ref, _ = load_reference(ref_dir)
    cfg = json.load(open(os.path.join(ref_dir, "config.json")))
    known = {f.name for f in fields(ref.ModelArgs)}
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in known}
    cfg.update(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **overrides)
    return ref.ModelArgs(**cfg)


def set_globals(args) -> None:
    """The reference keeps world size and storage dtype as module globals, set by Transformer's
    constructor. Layers built on their own (as the trainer does) need them set explicitly."""
    import torch

    ref, _ = load_reference()
    ref.world_size = 1
    ref.rank = 0
    ref.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
