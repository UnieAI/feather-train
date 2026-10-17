"""Component checks on synthetic tensors: each differentiable twin against the reference kernel
(forward) and against plain autograd on dequantized / dense math (backward)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def test_frozen_fp8_linear_matches_reference_and_dequant_grad():
    from dsrl.model.quant import dequant_fp8, frozen_linear
    from dsrl.model.ref import load_reference

    ref, kernel = load_reference()
    torch.manual_seed(0)
    w_dense = torch.randn(512, 1024, device="cuda") * 0.05
    # quantize per 32x32 block the way the checkpoint stores it
    blocks = w_dense.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-8)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w = (blocks / scale).flatten(2, 3).flatten(0, 1).to(torch.float8_e4m3fn)
    w.scale = scale.squeeze(1).squeeze(-1).to(torch.float8_e8m0fnu)

    x = torch.randn(4, 77, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = frozen_linear(x, w)
    with ref.set_dtype(torch.bfloat16):
        expected = ref.linear(x.detach(), w)
    assert rel(y, expected) == 0.0

    g = torch.randn_like(y)
    (y * g).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    (F.linear(x2, dequant_fp8(w, w.scale)) * g).sum().backward()
    assert rel(x.grad, x2.grad) < 1e-2


def test_sparse_attn_matches_reference_kernel():
    from dsrl.model import ops
    from dsrl.model.ref import load_reference

    _, kernel = load_reference()
    torch.manual_seed(0)
    b, s, h, d, n, k = 1, 300, 64, 512, 450, 96
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, device="cuda")
    idxs = torch.randint(0, n, (b, s, k), device="cuda", dtype=torch.int32)
    idxs[:, :, -10:] = -1
    idxs[:, 0] = -1  # a row with nothing to attend to must give zeros, not NaN
    scale = d**-0.5
    ours = ops.sparse_attn(q, kv, sink, idxs, scale, chunk=128)
    theirs = kernel.sparse_attn(q, kv, sink, idxs, scale)
    assert torch.isfinite(ours).all()
    assert ours[:, 0].abs().max() == 0
    assert rel(ours, theirs) < 2e-2


def test_sparse_attn_grad_matches_dense_masked_attention():
    from dsrl.model import ops

    torch.manual_seed(0)
    b, s, h, d, n, k = 1, 40, 4, 64, 60, 12
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float64, requires_grad=True)
    kv = torch.randn(b, n, d, device="cuda", dtype=torch.float64, requires_grad=True)
    sink = torch.randn(h, device="cuda", dtype=torch.float64)
    idxs = torch.stack([torch.randperm(n, device="cuda")[:k] for _ in range(s)]).int().unsqueeze(0)
    idxs[:, :, -2:] = -1
    scale = d**-0.5
    out = ops.sparse_attn(q, kv, sink, idxs, scale, chunk=16)
    g = torch.randn_like(out)
    dq, dkv = torch.autograd.grad((out * g).sum(), (q, kv))

    # dense twin: a [s, n] mask built from the same index lists
    mask = torch.zeros(s, n, dtype=torch.bool, device="cuda")
    for i in range(s):
        mask[i, idxs[0, i][idxs[0, i] >= 0].long()] = True
    q2, kv2 = q.detach().clone().requires_grad_(True), kv.detach().clone().requires_grad_(True)
    scores = torch.einsum("bshd,bnd->bshn", q2, kv2) * scale
    scores = scores.masked_fill(~mask[None, :, None, :], float("-inf"))
    full = torch.cat([scores, sink.view(1, 1, h, 1).expand(b, s, h, 1)], -1).softmax(-1)[..., :-1]
    out2 = torch.einsum("bshn,bnd->bshd", full, kv2)
    dq2, dkv2 = torch.autograd.grad((out2 * g).sum(), (q2, kv2))
    # the chunk math runs in fp32 by design, so agreement is to fp32 precision
    assert rel(out, out2) < 1e-5
    assert rel(dq, dq2) < 1e-5 and rel(dkv, dkv2) < 1e-5


def test_sinkhorn_matches_reference_kernel():
    from dsrl.model import ops
    from dsrl.model.ref import load_reference

    _, kernel = load_reference()
    torch.manual_seed(0)
    hc = 4
    mixes = torch.randn(2, 33, (2 + hc) * hc, device="cuda")
    scale, base = torch.rand(3, device="cuda"), torch.randn((2 + hc) * hc, device="cuda")
    ours = ops.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    theirs = kernel.hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    for a, b in zip(ours, theirs):
        assert rel(a, b) < 1e-5


def test_rope_matches_reference_inplace_rotation():
    from dsrl.model import ops
    from dsrl.model.ref import load_reference

    ref, _ = load_reference()
    freqs = ref.precompute_freqs_cis(64, 128, 65536, 160000, 16, 32, 1).cuda()[:50]
    x = torch.randn(2, 50, 8, 64, device="cuda", dtype=torch.bfloat16)
    for inverse in (False, True):
        assert rel(ops.rope(x, freqs, inverse), ref.apply_rotary_emb(x.clone(), freqs, inverse)) == 0.0
