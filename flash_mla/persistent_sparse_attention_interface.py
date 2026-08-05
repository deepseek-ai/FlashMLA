from typing import Literal, Optional, Tuple

import torch


def _cuda_module():
    # Import lazily so the stock FlashMLA path remains usable when the private
    # extension was intentionally not built.
    from flash_mla import persistent_sparse_attention_cuda

    return persistent_sparse_attention_cuda


def flash_mla_persistent_sparse_attention_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    q_ready: Optional[torch.Tensor] = None,
    q_ready_chunk_size: int = 0,
    q_positions: Optional[torch.Tensor] = None,
    q_cos_sin_cache: Optional[torch.Tensor] = None,
    q_norm_eps: float = 1.0e-6,
    o_ready: Optional[torch.Tensor] = None,
    fused_o_fp8: Optional[torch.Tensor] = None,
    fused_o_scale: Optional[torch.Tensor] = None,
    fused_o_positions: Optional[torch.Tensor] = None,
    fused_o_skip_bf16: bool = False,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    psa: bool = False,
    csa_lane: Optional[Literal["generic", "prefix", "tail"]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the isolated SM100 head-128 D=512 sparse-prefill kernel."""
    if q_positions is not None and q_cos_sin_cache is None:
        raise ValueError("q_positions requires q_cos_sin_cache")
    if q_ready is not None and q_ready_chunk_size <= 0:
        raise ValueError(
            "q_ready_chunk_size must be positive when q_ready is provided"
        )
    if (extra_kv is None) != (extra_indices is None):
        raise ValueError("extra_kv and extra_indices must be provided together")
    if extra_topk_length is not None and extra_kv is None:
        raise ValueError("extra_topk_length requires extra_kv and extra_indices")
    if psa and num_sms is None:
        raise ValueError("psa=True requires num_sms")
    lane_codes = {None: 0, "generic": 0, "prefix": 1, "tail": 2}
    if csa_lane not in lane_codes:
        raise ValueError("csa_lane must be None, 'generic', 'prefix', or 'tail'")
    if csa_lane in ("prefix", "tail") and not psa:
        raise ValueError("CSA prefix/tail lanes require psa=True")

    values = (fused_o_fp8, fused_o_scale, fused_o_positions)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError(
            "fused_o_fp8, fused_o_scale, and fused_o_positions must be "
            "provided together"
        )
    if fused_o_fp8 is not None and q_cos_sin_cache is None:
        raise ValueError("the fused output epilogue requires q_cos_sin_cache")
    if fused_o_skip_bf16 and fused_o_fp8 is None:
        raise ValueError("fused_o_skip_bf16 requires fused output tensors")
    if psa and fused_o_fp8 is not None:
        raise ValueError("persistent topology and fused-O are mutually exclusive")

    return _cuda_module().sparse_prefill_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
        num_sms,
        q_ready,
        q_ready_chunk_size,
        q_positions,
        q_cos_sin_cache,
        q_norm_eps,
        o_ready,
        fused_o_fp8,
        fused_o_scale,
        fused_o_positions,
        fused_o_skip_bf16,
        extra_kv,
        extra_indices,
        extra_topk_length,
        psa,
        lane_codes[csa_lane],
    )


def flash_mla_sparse_decode_ready(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_sm_parts: int,
    row_ready: torch.Tensor,
    head_dim_v: int = 512,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    initial_ready_rows: int = 0,
    q_positions: Optional[torch.Tensor] = None,
    q_cos_sin_cache: Optional[torch.Tensor] = None,
    q_norm_eps: float = 1.0e-6,
    fused_o_fp8: Optional[torch.Tensor] = None,
    fused_o_scale: Optional[torch.Tensor] = None,
    fused_o_positions: Optional[torch.Tensor] = None,
    fused_o_skip_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the isolated row-ready SM100 sparse-decode specialization."""
    if isinstance(num_sm_parts, bool) or not isinstance(num_sm_parts, int):
        raise TypeError("num_sm_parts must be an int")
    if num_sm_parts <= 0:
        raise ValueError("num_sm_parts must be positive")
    if (q_positions is None) != (q_cos_sin_cache is None):
        raise ValueError(
            "q_positions and q_cos_sin_cache must be provided together"
        )
    if (extra_k_cache is None) != (extra_indices is None):
        raise ValueError(
            "extra_k_cache and extra_indices must be provided together"
        )
    if extra_topk_length is not None and extra_k_cache is None:
        raise ValueError(
            "extra_topk_length requires extra_k_cache and extra_indices"
        )

    fused_values = (fused_o_fp8, fused_o_scale, fused_o_positions)
    if any(value is not None for value in fused_values) and not all(
        value is not None for value in fused_values
    ):
        raise ValueError(
            "fused_o_fp8, fused_o_scale, and fused_o_positions must be "
            "provided together"
        )
    if fused_o_fp8 is not None and q_cos_sin_cache is None:
        raise ValueError("the fused output epilogue requires q_cos_sin_cache")
    if fused_o_skip_bf16 and fused_o_fp8 is None:
        raise ValueError("fused_o_skip_bf16 requires fused output tensors")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out, lse = _cuda_module().sparse_decode_ready_fwd(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
        head_dim_v,
        softmax_scale,
        out,
        num_sm_parts,
        row_ready,
        initial_ready_rows,
        q_positions,
        q_cos_sin_cache,
        q_norm_eps,
        fused_o_fp8,
        fused_o_scale,
        fused_o_positions,
        fused_o_skip_bf16,
    )
    return out, lse
