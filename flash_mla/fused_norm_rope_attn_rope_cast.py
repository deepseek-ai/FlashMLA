from typing import Optional, Tuple

import torch

import flash_mla.cuda as flash_mla_cuda


def prefill(
    enable_q_norm: bool,
    rms_norm_eps: float,

    token_positions: torch.Tensor,
    is_rope_neox_style: bool,
    rope_dim: int,
    cos_sin_cache: torch.Tensor,

    n_wv_group: int,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,

    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A fused kernel for Q Norm + Q RoPE + Core Attn (sparse attention) + O RoPE + O cast to FP8, for DeepSeek-V4 & DeepSeek-V4.1
    Only support sm100 / sm103 GPU architecture.

    Args (Norm):
        enable_q_norm: bool, whether to apply Q RMSNorm
        rms_norm_eps: float. EPS for RMSNorm

    Args (RoPE):
        token_positions: [s_q], int32
        is_rope_neox_style: must be False now
        rope_dim: int32, must be 64 now
        cos_sin_cache: [*, rope_dim], float32

    Args (Cast):
        n_wv_group: n_wv_group during o projection
        num_per_channels: quantization granularity, must be 32
        use_tma_aligned_col_major_sf: must be True
        round_sf: must be True
        use_packed_ue8m0: must be True

    Args (Core Attn):
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to < 0, or >= s_kv
        sm_scale: float, scaling factor for attention scores
        d_v: value dimension, default (and only) is 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros). This has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k tokens (even if provided in indices). This parameter is mainly used for variable-length topk attention scenarios, such as using sparse attention to simulate causal attention.
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        - out_fp8: [s_q, n_wv_group, wv_group_size * d_v], fp8_e4m3, quantized attention result
        - out_sf: [s_q, n_wv_group, wv_group_size * d_v / (32*4)], int32_t, scaling factor. This scaling factor is ALWAYS stored in the per-32 scaled format, even if num_per_channels is 128
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float
        If a q token does not attend to any k token, then max_logits is -inf, lse is +inf, out is all zeros.
    """
    results = flash_mla_cuda.fused_norm_rope_attn_rope_cast_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length,

        enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,

        n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    )
    out_fp8, out_sf, max_logits, lse = results
    return out_fp8, out_sf, max_logits, lse


def decode(
    enable_q_norm: bool,
    rms_norm_eps: float,

    token_positions: torch.Tensor,
    is_rope_neox_style: bool,
    rope_dim: int,
    cos_sin_cache: torch.Tensor,

    n_wv_group: int,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,

    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Decoding kernel: Q Norm + Q RoPE + Core Attn (decode, with paged FP8 KV cache) + O RoPE + O cast to FP8, for DeepSeek-V4 & DeepSeek-V4.1
    Only support sm100 / sm103 GPU architecture.

    The batch size is always 1, and the batch dimension is not included in any tensor shape below.

    Args (Norm):
        enable_q_norm: bool, whether to apply Q RMSNorm
        rms_norm_eps: float. EPS for RMSNorm

    Args (RoPE):
        token_positions: [s_q], int32
        is_rope_neox_style: must be False now
        rope_dim: int32, must be 64 now
        cos_sin_cache: [*, rope_dim], float32

    Args (Cast):
        n_wv_group: n_wv_group during o projection
        num_per_channels: quantization granularity, must be 32
        use_tma_aligned_col_major_sf: must be True
        round_sf: must be True
        use_packed_ue8m0: must be True

    Args (Core Attn):
        q: [s_q, h_q, d_qk], bfloat16. Already permuted by permute_q_b_proj
        k_cache: [num_blocks, page_block_size, h_kv, bytes_per_token], fp8_e4m3 (or int8/uint8). Paged quantized KV cache. The format is
            detected from bytes_per_token: 584 (V4), 528 (V4.1) or 288 (V4.1 with fp4 e2m1 + per-16 e4m3 scales). See tests/quant.py for the layouts
        indices_in_kvcache: [s_q, topk], int32. Page-relative KV token indices
        sm_scale: float, scaling factor for attention scores
        d_v: value dimension, default (and only) is 512
        attn_sink: optional, [h_q], float32. Per-head attention sink bias
        topk_length: optional, [s_q], int32. Actual valid topk count of the request
        extra_k_cache: optional, [extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token]. Secondary paged FP8 KV cache
        extra_indices_in_kvcache: optional, [s_q, extra_topk], int32. Indices into the extra KV cache
        extra_topk_length: optional, [s_q], int32. Actual valid extra topk count of the request

    Returns:
        - out_fp8: [s_q, n_wv_group, wv_group_size * d_v], fp8_e4m3, quantized attention result
        - out_sf: [s_q, n_wv_group, wv_group_size * d_v / (32*4)], int32, scaling factor
        - lse: [s_q, h_q], float
    """
    out_fp8, out_sf, lse = flash_mla_cuda.fused_norm_rope_attn_rope_cast_decode(
        q, k_cache, indices_in_kvcache, sm_scale, d_v,
        attn_sink, topk_length,
        extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
        enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache,
        n_wv_group, num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    )
    return out_fp8, out_sf, lse


def permute_q_b_proj(
    weight_and_sf: Tuple[torch.Tensor, torch.Tensor],
    h_q: int,
    d_q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the layout of the q_b_proj weight and its scale factors into the layout required by
    fused_norm_rope_attn_rope_cast_fwd / fused_norm_rope_attn_rope_cast_decode.

    Args:
        weight_and_sf: (q_b_proj, q_b_sf), where
        - q_b_proj: [h_q*d_q, q_lora_rank], fp8_e4m3, the weight in its original layout
        - q_b_sf: [h_q*d_q, q_lora_rank/gran/4], int32, the scale factors in their original layout (DeepGeMM format, i.e. contiguous on the first dim)
        h_q: int, the number of Q heads
        d_q: int, the Q head dimension

    Returns:
        - q_b_proj_permuted: [h_q*d_q, q_lora_rank], fp8_e4m3
        - scale_factors_permuted: [h_q*d_q, q_lora_rank/gran/4], int32
    """
    return flash_mla_cuda.permute_q_b_proj(weight_and_sf[0], weight_and_sf[1], h_q, d_q)


def permute_wv_proj(
    weight_and_sf: Tuple[torch.Tensor, torch.Tensor],
    wv_group_size: int,
    d_o: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the layout of the wv_proj weight and its scale factors into the layout required by
    fused_norm_rope_attn_rope_cast_fwd / fused_norm_rope_attn_rope_cast_decode.
    Note: regardless of the input quantization granularity, the output granularity is always 32.

    Args:
        weight_and_sf: (wv_proj, wv_sf), where
        - wv_proj: [n_wv_groups, d_proj_out, wv_group_size * d_o], fp8_e4m3, the weight in its original layout
        - wv_sf: [n_wv_groups, d_proj_out, wv_group_size * d_o/gran/4], int32, the scale factors in their original layout (DeepGeMM format, i.e. contiguous on the second dim)
        wv_group_size: int
        d_o: int, the O head dimension

    Returns:
        - wv_proj_permuted: the same shape as wv_proj, fp8_e4m3
        - scale_factors_permuted: [n_wv_groups, d_proj_out, wv_group_size * d_o/32/4], int32
    """
    return flash_mla_cuda.permute_wv_proj(weight_and_sf[0], weight_and_sf[1], wv_group_size, d_o)
