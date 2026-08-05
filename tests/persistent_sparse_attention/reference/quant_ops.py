"""Pure PyTorch references for attention/output FP8 quantization."""

import torch


FP8_MAX = 448.0
SCALE_EPSILON = 1.0e-10


def _maxnum_abs(values, dim):
    """CUDA maxNum semantics: ignore NaN unless every value is NaN."""
    magnitudes = values.abs()
    magnitudes = torch.where(
        torch.isnan(magnitudes), torch.zeros_like(magnitudes), magnitudes
    )
    return magnitudes.amax(dim=dim)


def _float_to_raw_e4m3(values):
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def _pack_scale_bytes(scale_bytes):
    words = scale_bytes.to(torch.int32)
    return (
        words[..., 0]
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    )


def attention_output_quant_reference(o, positions, cos_sin_cache):
    """Inverse RoPE then E4M3 quantize each 128-value head chunk.

    Returns the logical group-major FP8 view [T,16,4096], logical packed
    scale view [T,16,8], and the complete physical scale backing including
    aligned-token padding initialized to the accepted canary value.
    """
    if o.dtype != torch.bfloat16 or o.ndim != 3 or o.shape[1:] != (128, 512):
        raise ValueError("o must be bfloat16 [tokens, 128, 512]")
    if bool(torch.isinf(o).any()):
        raise ValueError(
            "K11 Inf handling uses CUDA float-to-int exponent semantics; "
            "validate Inf with the native source oracle"
        )
    tokens = o.shape[0]
    values = o.float().clone()
    table = cos_sin_cache.index_select(0, positions.to(torch.int64))
    rope = values[..., 448:].reshape(tokens, 128, 32, 2)
    cosine = table[:, None, :32]
    sine = table[:, None, 32:64]
    even = rope[..., 0].clone()
    odd = rope[..., 1].clone()
    rope[..., 0] = even * cosine + odd * sine
    rope[..., 1] = odd * cosine - even * sine

    chunks = values.reshape(tokens, 128, 4, 128)
    absmax = _maxnum_abs(chunks, -1).clamp_min(SCALE_EPSILON)
    exponent = torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX)))
    scale = torch.exp2(exponent)
    inverse = torch.exp2(-exponent)
    normalized = chunks * inverse[..., None]
    # K11 applies the lower maxNum clamp first; NaN therefore maps to -448.
    normalized = torch.where(
        torch.isnan(normalized),
        torch.full_like(normalized, -FP8_MAX),
        normalized.clamp(-FP8_MAX, FP8_MAX),
    )
    fp8_heads = _float_to_raw_e4m3(normalized).reshape(tokens, 128, 512)
    fp8_base = torch.empty(
        (16, tokens, 8 * 512), dtype=torch.uint8, device=o.device
    )
    fp8 = fp8_base.transpose(0, 1)
    fp8.copy_(fp8_heads.reshape(tokens, 16, 8 * 512))

    scale_bytes = (
        scale.view(torch.int32).bitwise_right_shift(23).bitwise_and(0xFF)
    ).to(torch.uint8)
    packed = _pack_scale_bytes(scale_bytes).reshape(tokens, 16, 8)
    aligned_tokens = (tokens + 3) & ~3
    scale_backing = torch.full(
        (16 * 8 * aligned_tokens,), 0x5A5A5A5A,
        dtype=torch.int32, device=o.device,
    )
    scale_view = scale_backing.as_strided(
        (16, tokens, 8), (8 * aligned_tokens, 1, aligned_tokens)
    ).transpose(0, 1)
    scale_view.copy_(packed)
    return fp8, scale_view, scale_backing


def _exact_output_scale(absmax):
    bounded = absmax.clamp_min(SCALE_EPSILON)
    raw_scale = (bounded * (1.0 / FP8_MAX)).to(torch.float32)
    raw_bits = raw_scale.view(torch.int32)
    exponent = raw_bits.bitwise_right_shift(23).bitwise_and(0xFF)
    mantissa = raw_bits.bitwise_and(0x007FFFFF)
    scale_byte = exponent + (mantissa != 0).to(torch.int32)
    inverse_bits = (254 - scale_byte).clamp_min(0).bitwise_left_shift(23)
    inverse = inverse_bits.view(torch.float32)
    inverse = torch.where(scale_byte < 255, inverse, torch.zeros_like(inverse))
    return inverse, scale_byte.to(torch.uint8)


def output_quant_reference(x):
    """BF16 to raw E4M3 plus physical [packs, rows] UE8M0 backing."""
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 512:
        raise ValueError("x must be bfloat16 [rows, hidden], hidden % 512 == 0")
    rows, hidden = x.shape
    packs = hidden // 512
    groups = x.float().reshape(rows, packs, 4, 128)
    absmax = _maxnum_abs(groups, -1)
    inverse, scale_bytes = _exact_output_scale(absmax)
    normalized = groups * inverse[..., None]
    # K12 applies the upper maxNum clamp first; NaN therefore maps to +448.
    normalized = torch.where(
        torch.isnan(normalized),
        torch.full_like(normalized, FP8_MAX),
        normalized.clamp(-FP8_MAX, FP8_MAX),
    )
    output = _float_to_raw_e4m3(normalized).reshape(rows, hidden)
    packed = _pack_scale_bytes(scale_bytes)
    scale_backing = packed.transpose(0, 1).contiguous()
    return output, scale_backing
