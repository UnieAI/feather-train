"""Differentiable replacements for the reference's forward-only pieces."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Out-of-place `apply_rotary_emb`: adjacent pairs as complex numbers. x: [b, s, d] or
    [b, s, h, d]; freqs_cis: [s, d/2]."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = freqs_cis.conj() if inverse else freqs_cis
    f = f.view(1, x.size(1), xc.size(-1)) if x.ndim == 3 else f.view(1, x.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * f).flatten(-2).to(dtype)


def rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """Rotate only the last `rd` channels, as every attention path in V4.1 does."""
    return torch.cat([x[..., :-rd], rope(x[..., -rd:], freqs_cis, inverse)], dim=-1)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc: int, iters: int, eps: float):
    """Torch twin of the tilelang kernel. mixes: [..., (2+hc)*hc] fp32."""
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _sparse_attn_chunk(q, kv, sink, idxs, scale):
    """q: [b, c, h, d]; kv: [b, n, d]; idxs: [b, c, k] with -1 = empty slot."""
    b, c, h, d = q.shape
    valid = idxs >= 0
    flat = idxs.clamp_min(0).long()
    gathered = torch.gather(kv, 1, flat.flatten(1).unsqueeze(-1).expand(-1, -1, d)).view(b, c, -1, d)
    scores = torch.einsum("bchd,bckd->bchk", q.float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # the sink is an extra logit with no value: it only enlarges the denominator
    sink_logit = sink.float().view(1, 1, h, 1).expand(b, c, h, 1)
    full = torch.cat([scores, sink_logit], dim=-1)
    # like the serving kernel: exponentials are rounded to the activation dtype before the PV
    # product and normalized afterwards (fp32 accumulation either way)
    m = full.amax(-1, keepdim=True)
    e = torch.exp(full - m)
    denom = e.sum(-1, keepdim=True)
    p = e[..., :-1].to(q.dtype).float()
    return (torch.einsum("bchk,bckd->bchd", p, gathered.float()) / denom).to(q.dtype)


def sparse_attn(q, kv, sink, idxs, scale, chunk: int = 1024):
    """Training twin of the reference `sparse_attn`: every query attends to the kv rows its
    `idxs` name (window + selected compressed positions). Chunked over queries with recompute,
    so memory is O(chunk * topk * d) rather than O(seq * topk * d)."""
    outs = []
    for start in range(0, q.size(1), chunk):
        sl = slice(start, start + chunk)
        if torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad):
            outs.append(checkpoint(_sparse_attn_chunk, q[:, sl], kv, sink, idxs[:, sl], scale, use_reentrant=False))
        else:
            outs.append(_sparse_attn_chunk(q[:, sl], kv, sink, idxs[:, sl], scale))
    return torch.cat(outs, dim=1)


def window_idxs(window: int, bsz: int, seqlen: int, device) -> torch.Tensor:
    """Prefill sliding-window slots, identical to the reference's start_pos == 0 case."""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window), device=device)
    idxs = torch.where(idxs > end, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up
