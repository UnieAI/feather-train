"""Real-weight checks of the trainable forward against the reference prefill.

What can be compared exactly is compared exactly; what cannot is bounded:

- MoE, top-k / candidate selection, and block assembly (hyper-connections, Sinkhorn, norms)
  match the reference to rounding (< 1e-3).
- Attention differs at the 1e-2 level: its output and the window KV are fp8-quantized, and the
  tilelang kernel's online-softmax rounding pushes some elements across a different fp8 step.
  Through MoE routing that noise compounds at block level (layer 20: ~10% of tokens pick a
  different expert set), which is why block-level comparisons inject reference attention.
  Agreement that matters for RL — trainer log-probs vs the sglang sampler — is measured on the
  full model separately.

Layers: 2 (ratio-2 kv/index source), 3 (consumer of 2), 20 (ratio-1 source, candidate source).
Needs ~20 GB of GPU memory.
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not os.path.isdir("/models/deepseek-ai"), reason="needs the checkpoint"),
]

SEQ = int(os.environ.get("DSRL_TEST_SEQ", "2048"))


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def env():
    from dsrl.model.loader import CheckpointIndex
    from dsrl.model.ref import load_reference, model_args

    ref, _ = load_reference()
    args = model_args(max_seq_len=SEQ)
    index = CheckpointIndex()
    embed = index.get("embed.weight").cuda()
    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (1, SEQ), device="cuda")
    return ref, args, index, embed[ids]


def ref_mode(ref):
    """The reference assumes process-wide defaults: cuda device, bf16 dtype."""
    import contextlib

    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    stack.enter_context(ref.set_dtype(torch.bfloat16))
    stack.enter_context(torch.device("cuda"))
    return stack


@pytest.mark.parametrize("layer", [2, 20])
def test_sublayers(env, layer):
    from dsrl.model import block as B
    from dsrl.model.loader import load_block

    ref, args, index, h = env
    blk = load_block(layer, args, index)
    a = blk.attn_norm(h)
    with ref_mode(ref):
        ref_attn, ref_moe = blk.attn(a.clone(), 0), blk.ffn(a.clone())
        ref_topk = ref.shared_attn.topk_idxs.clone()
        ref_cand = None if ref.shared_attn.candidates is None else ref.shared_attn.candidates.clone()
    ctx = B.LayerContext()
    with torch.no_grad():
        my_attn, my_moe = B.attention(blk.attn, a.clone(), ctx), B.moe(blk.ffn, a.clone())
    print(f"layer {layer}: attention {rel(my_attn, ref_attn):.2e}, moe {rel(my_moe, ref_moe):.2e}")
    assert torch.equal(ctx.topk_idxs, ref_topk)
    if layer == args.candidate_source_layer:
        assert torch.equal(ctx.candidates, ref_cand)
    assert rel(my_moe, ref_moe) < 1e-3
    assert rel(my_attn, ref_attn) < 3e-2


@pytest.mark.parametrize("layer", [2, 20])
def test_block_assembly(env, layer, monkeypatch):
    from dsrl.model import block as B
    from dsrl.model.loader import load_block

    ref, args, index, h = env
    blk = load_block(layer, args, index)
    x = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pm = ref.make_identity_pre_mix(x, args.hc_mult)
    with ref_mode(ref):
        x_ref, m_ref = blk(x, 0, pm, None)

    def reference_attention(attn, inp, ctx):
        with ref_mode(ref):
            return attn(inp.clone(), 0).clone()

    monkeypatch.setattr(B, "attention", reference_attention)
    with torch.no_grad():
        x_ours, m_ours = B.block_forward(blk, x, pm, B.LayerContext())
    print(f"layer {layer}: block {rel(x_ours, x_ref):.2e}, pre_mix {rel(m_ours, m_ref):.2e}")
    assert rel(x_ours, x_ref) < 1e-3 and rel(m_ours, m_ref) < 1e-3


def test_cross_layer_state(env):
    """Layer 3 reads layer 2's compressed KV and top-k through LayerContext."""
    from dsrl.model import block as B
    from dsrl.model.loader import load_block

    ref, args, index, h = env
    b2, b3 = load_block(2, args, index), load_block(3, args, index)
    a = b2.attn_norm(h)
    with ref_mode(ref):
        b2.attn(a.clone(), 0)
        ref3 = b3.attn(a.clone(), 0)
    ctx = B.LayerContext()
    with torch.no_grad():
        B.attention(b2.attn, a.clone(), ctx)
        ours3 = B.attention(b3.attn, a.clone(), ctx)
    print(f"layer 3 attention via layer-2 state: {rel(ours3, ref3):.2e}")
    assert rel(ours3, ref3) < 3e-2


def test_backward_reaches_lora_and_input(env):
    from dsrl.model import lora
    from dsrl.model.block import LayerContext, block_forward
    from dsrl.model.loader import load_block

    ref, args, index, h = env
    blocks = [load_block(i, args, index) for i in (2, 3)]
    params = [p for b in blocks for p in lora.attach(b, rank=16, alpha=32)]
    for p in params:  # B starts at zero; give it a value so A receives gradient too
        if p.shape[1] == 16:
            torch.nn.init.normal_(p, std=1e-3)
    x = h[:, :1024].unsqueeze(2).repeat(1, 1, args.hc_mult, 1).detach().requires_grad_(True)
    m = ref.make_identity_pre_mix(x, args.hc_mult)
    ctx = LayerContext()
    y = x
    for b in blocks:
        y, m = block_forward(b, y, m, ctx)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
