"""Python front end of the fused MLA decode prologue (csrc/fused/mla_decode_prologue.cu) and its PyTorch reference.

    q_out = mla_decode_prologue(q_nope_abs, q_pe, kv_c, k_pe, cos_sin, block_table, cache_seqlens, s_q, kv_cache, ...)

does, in ONE kernel launch and in place (CUDA-graph safe, no allocation inside):
    q_out[t, h] = [ q_nope_abs[t, h] | rope(q_pe[t, h]) ]                              (T, H, 576)
    kv_cache[slot(t)] = [ rmsnorm?(kv_c[t]) | rope(k_pe[t]) ]       bf16/fp16, or FP8/INT8 + per-token scale
with slot(t) = block_table[b, pos // page] * page + pos % page, b = t // s_q, pos = cache_seqlens[b] - s_q + (t % s_q).
"""
import torch

from .kv_quant import KIND_ID, ROW_BYTES, dequantize_kv, kind_id, quantize_kv  # noqa: F401

try:
    import flash_mla_fused_cuda as _ext
except ImportError:  # extension not built: the torch reference stays usable
    _ext = None


def make_rope_cache(max_pos, rope_dim=64, base=10000.0, device="cuda", mscale=1.0):
    """[max_pos, rope_dim] fp32: cos[0..rope_dim/2) | sin[0..rope_dim/2); inv_freq_i = base^(-2i/rope_dim).
    YaRN-style scaling can be folded in by passing `mscale` (multiplies cos and sin) or building another cache."""
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None]
    return torch.cat([ang.cos() * mscale, ang.sin() * mscale], dim=-1).float().to(device).contiguous()


def apply_rope_ref(x, cos_sin, pos, interleaved=False):
    """x: [N, ..., 64] (any float dtype), pos: [N] -> rotated x in float32. neox: pairs (i, i+32); interleaved: (2i, 2i+1)."""
    cs = cos_sin[pos.long()]  # [N, 64]
    c, s = cs[:, :32], cs[:, 32:]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [32]
    c, s = c.view(shape), s.view(shape)
    xf = x.float()
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        o1, o2 = x1 * c - x2 * s, x2 * c + x1 * s
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    x1, x2 = xf[..., :32], xf[..., 32:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def rmsnorm_ref(x, w, eps):
    """DeepseekV2RMSNorm: w * (x * rsqrt(mean(x^2) + eps)).to(dtype)."""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return w * y


def token_positions(cache_seqlens, s_q, T):
    b = torch.arange(T, device=cache_seqlens.device) // s_q
    j = torch.arange(T, device=cache_seqlens.device) % s_q
    return cache_seqlens[b].long() - s_q + j, b


def _slots(block_table, cache_seqlens, s_q, T, page):
    pos, b = token_positions(cache_seqlens, s_q, T)
    return block_table[b, (pos // page)].long() * page + pos % page, pos


def mla_decode_prologue_ref(q_nope, q_pe, kv_c, k_pe, cos_sin, block_table, cache_seqlens, s_q, kv_cache, kv_scale=None,
                            kv_norm_weight=None, eps=1e-6, interleaved=False, kv_dtype="native", q_out=None):
    """Pure PyTorch version (several kernels). Same signature / in-place semantics as `mla_decode_prologue`."""
    T, H = q_nope.shape[0], q_nope.shape[1]
    kd = kind_id(kv_dtype)
    page = kv_cache.shape[1]
    slot, pos = _slots(block_table, cache_seqlens, s_q, T, page)
    dt = q_nope.dtype
    if q_out is None:
        q_out = torch.empty(T, H, 576, dtype=dt, device=q_nope.device)
    q_out[..., :512] = q_nope
    q_out[..., 512:] = apply_rope_ref(q_pe, cos_sin, pos, interleaved).to(dt)
    kvc = rmsnorm_ref(kv_c, kv_norm_weight, eps) if kv_norm_weight is not None else kv_c
    row = torch.cat([kvc, apply_rope_ref(k_pe, cos_sin, pos, interleaved).to(dt)], dim=-1)  # [T, 576]
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    if kd == 0:
        flat[slot] = row
    else:
        kvq, sc = quantize_kv(row, kd)
        flat[slot] = kvq
        kv_scale.view(-1)[slot] = sc
    return q_out


def mla_decode_prologue(q_nope, q_pe, kv_c, k_pe, cos_sin, block_table, cache_seqlens, s_q, kv_cache, kv_scale=None,
                        kv_norm_weight=None, eps=1e-6, interleaved=False, kv_dtype="native", q_out=None):
    """Fused prologue (see module docstring). Returns q_out [T, H, 576]. `kv_cache`: native [nblk, page, 1, 576] (q dtype)
    or quantized uint8 [nblk, page, 640] with `kv_scale` fp32 [nblk, page] (docs/kv_quant_format.md)."""
    if _ext is None:
        raise RuntimeError("flash_mla_fused_cuda is not built (tools/build.sh)")
    if q_out is None:
        q_out = torch.empty(q_nope.shape[0], q_nope.shape[1], 576, dtype=q_nope.dtype, device=q_nope.device)
    _ext.mla_decode_prologue(q_nope, q_pe, kv_c, k_pe, cos_sin, block_table, cache_seqlens, s_q, q_out, kv_cache, kv_scale,
                             kv_norm_weight, eps, interleaved, kind_id(kv_dtype))
    return q_out
