"""Stateless, differentiable full-sequence forward for one reference `Block`.

Mirrors the reference prefill path (start_pos == 0) line for line, with these substitutions:
quantized GEMMs go through `frozen_linear`, in-place quantization becomes straight-through
fake quantization, `sparse_attn` and Sinkhorn use the torch twins in `ops`, and the KV caches
are replaced by `LayerContext`, which carries what source layers publish to later ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dsrl.model import ops
from dsrl.model.lora import lora_linear
from dsrl.model.quant import fake_quant_fp4, fake_quant_fp8, frozen_linear
from dsrl.model.ref import load_reference

INDEX_CHUNK = 2048  # queries per indexer-score chunk; [chunk, heads, positions] is materialized


@dataclass
class LayerContext:
    """Cross-layer state for one sequence, the training twin of `SharedAttentionRuntime`.

    compress_kv carries gradient (it is the compressor output of a kv-source layer, read by the
    layers after it); the index keys, top-k and candidates are selections and carry none."""

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def _rmsnorm(norm, x):
    return norm(x)  # the reference RMSNorm is already out-of-place and differentiable


def _compressor(comp, x):
    """Prefill compressor: pools `ratio` tokens into one latent; a trailing partial group is
    dropped (the reference parks it in decode state, which a full-sequence pass never reads)."""
    ratio = comp.compress_ratio
    if ratio == 1:
        return _rmsnorm(comp.norm, frozen_linear(x, comp.wkv.weight))
    dtype = x.dtype
    xf = x.float()
    kv, score = frozen_linear(xf, comp.wkv.weight), frozen_linear(xf, comp.wgate.weight)
    cutoff = x.size(1) - x.size(1) % ratio
    kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
    score = score[:, :cutoff].unflatten(1, (-1, ratio))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    return _rmsnorm(comp.norm, kv.to(dtype))


@torch.no_grad()
def _indexer(idx, x, qr, latent, ctx: LayerContext, offset: int, freqs_cis):
    """Top-k compressed positions per query. Frozen weights and a hard selection: no gradient."""
    ref_ratio, rd = idx.compress_ratio, idx.rope_head_dim
    bsz, seqlen, _ = x.shape
    if idx.owns_k and latent is not None:
        k = _rmsnorm(idx.k_norm, frozen_linear(latent, idx.wk.weight))
        k = ops.rope_tail(k, freqs_cis[: seqlen - seqlen % ref_ratio : ref_ratio], rd)
        ctx.index_k = fake_quant_fp4(k, 32, torch.float8_e8m0fnu)
    q = frozen_linear(qr, idx.wq_b.weight).unflatten(-1, (idx.n_local_heads, idx.index_head_dim))
    q = ops.rope_tail(q, freqs_cis[:seqlen], rd)
    q = fake_quant_fp4(q, 32, torch.float8_e8m0fnu)
    weights = frozen_linear(x, idx.weights_proj.weight) * (idx.softmax_scale * idx.n_heads**-0.5)

    n_comp = seqlen // ref_ratio
    topk = min(idx.index_topk, n_comp)
    index_k = ctx.index_k[:, :n_comp]
    out = []
    new_candidates = [] if idx.is_candidate_source else None
    for start in range(0, seqlen, INDEX_CHUNK):
        sl = slice(start, start + INDEX_CHUNK)
        score = torch.einsum("bshd,btd->bsht", q[:, sl], index_k)
        score = (score.relu_() * weights[:, sl].unsqueeze(-1)).sum(dim=2)
        pos = torch.arange(start, min(start + INDEX_CHUNK, seqlen), device=x.device)
        compress_lens = ((pos + 1) // ref_ratio).unsqueeze(-1)
        score.masked_fill_(torch.arange(n_comp, device=x.device) >= compress_lens, float("-inf"))
        if idx.is_candidate_source:
            ref, _ = load_reference()
            cand = ref.select_candidate_blocks(score, compress_lens, idx.candidate_topk_blocks, idx.candidate_block_size)
            new_candidates.append(cand)
        elif idx.uses_candidates:
            score = score.masked_fill(~ctx.candidates[:, sl], float("-inf"))
        sel = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        out.append(torch.where(sel < compress_lens, sel + offset, -1).int())
    if new_candidates is not None:
        ctx.candidates = torch.cat(new_candidates, dim=1)
    return torch.cat(out, dim=1)


def attention(attn, x, ctx: LayerContext):
    """Reference Attention.forward at start_pos == 0, differentiable."""
    bsz, seqlen, _ = x.shape
    rd = attn.rope_head_dim
    freqs_cis = attn.freqs_cis[:seqlen]

    qr = _rmsnorm(attn.q_norm, lora_linear(attn.wq_a, x))
    q = lora_linear(attn.wq_b, qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q = ops.rope_tail(q, freqs_cis, rd)

    kv = _rmsnorm(attn.kv_norm, lora_linear(attn.wkv, x))
    kv = fake_quant_fp8(ops.rope_tail(kv, freqs_cis, rd))
    idxs = ops.window_idxs(attn.window_size, bsz, seqlen, x.device)

    if attn.compress_ratio:
        ratio = attn.compress_ratio
        latent = None
        if attn.is_kv_source:
            latent = _compressor(attn.compressor, x)
        if attn.is_index_source:
            if seqlen // ratio == 0:
                ctx.topk_idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
            else:
                ctx.topk_idxs = _indexer(attn.indexer, x, qr, latent, ctx, seqlen, attn.freqs_cis)
        if latent is not None:
            pos_freqs = attn.freqs_cis[: seqlen - seqlen % ratio : ratio]
            latent = ops.rope_tail(latent, pos_freqs, rd)
            ctx.compress_kv = fake_quant_fp4(latent, 16, torch.float8_e4m3fn)
        kv = torch.cat([kv, ctx.compress_kv[:, : seqlen // ratio]], dim=1)
        idxs = torch.cat([idxs, ctx.topk_idxs], dim=-1)

    o = ops.sparse_attn(q, kv, attn.attn_sink, idxs, attn.softmax_scale)
    o = ops.rope_tail(o, freqs_cis, rd, inverse=True)
    o = fake_quant_fp8(o.view(bsz, seqlen, attn.n_local_groups, -1))
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return lora_linear(attn.wo_b, o.flatten(2))


def expert(e, x, weights=None):
    h = ops.swiglu(frozen_linear(x, e.w1.weight), frozen_linear(x, e.w3.weight), e.swiglu_limit)
    if weights is not None:
        h = weights * h
    return frozen_linear(h.to(x.dtype), e.w2.weight)


def moe(m, x):
    shape = x.shape
    x = x.reshape(-1, m.dim)
    weights, indices = m.gate(x)  # routing scores stay differentiable in x; indices do not
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=m.n_routed_experts).tolist()
    for i in range(m.n_routed_experts):
        if counts[i] == 0:
            continue
        tok, slot = torch.where(indices == i)
        y = y.index_add(0, tok, expert(m.experts[i], x[tok], weights[tok, slot, None]).float())
    y = y + expert(m.shared_experts, x).float()
    return y.type_as(x).view(shape)


def hc_mixes(block, x, fn, scale, base):
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + block.norm_eps)
    mixes = F.linear(xf, fn) * rsqrt
    return ops.hc_split_sinkhorn(mixes, scale, base, block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps)


def block_forward(block, x, pre_mix, ctx: LayerContext):
    """Reference Block.forward at start_pos == 0 (text only). x: [b, s, hc, d]."""
    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(block, x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    h = block.hc_pre(x, pre_mix)
    h = attention(block.attn, block.attn_norm(h), ctx)
    x = block.hc_post(h, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = hc_mixes(block, x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    h = block.hc_pre(x, attn_pre)
    h = moe(block.ffn, block.ffn_norm(h))
    x = block.hc_post(h, residual, ffn_post, ffn_comb)
    return x, ffn_pre
