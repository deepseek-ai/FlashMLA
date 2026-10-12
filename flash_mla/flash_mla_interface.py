from typing import Optional, Tuple

import torch

import flash_mla_cuda


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.

    Computed fully on the device (no host sync), so it can be captured in a CUDA Graph; the shapes depend only on
    (batch_size, num_q_tokens_per_head_k, num_heads_k).
    """
    return flash_mla_cuda.get_mla_metadata(cache_seqlens, num_q_tokens_per_head_k, num_heads_k)


def get_mla_workspace_bytes(
    batch_size: int,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_sm_parts: int,
) -> int:
    """Bytes of `workspace` required by flash_mla_with_kvcache(impl="v2")."""
    return flash_mla_cuda.get_mla_workspace_bytes(batch_size, num_q_tokens_per_head_k, num_heads_k, num_sm_parts)


def alloc_mla_workspace(
    batch_size: int,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_sm_parts: int,
    device=None,
) -> torch.Tensor:
    """Zero-initialised persistent workspace (uint8) for flash_mla_with_kvcache(workspace=...)."""
    n = get_mla_workspace_bytes(batch_size, num_q_tokens_per_head_k, num_heads_k, num_sm_parts)
    return torch.zeros(n, dtype=torch.uint8, device=device if device is not None else "cuda")


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    impl: str = "v2",
    kv_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head dimension of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, returned by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, returned by get_mla_metadata.
        softmax_scale: float. The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.
        out: optional preallocated output (batch_size, seq_len_q, num_heads_q, head_dim_v), same dtype as q.
        lse: optional preallocated softmax_lse (batch_size, num_heads_q, seq_len_q), torch.float32.
        workspace: optional persistent scratch (see alloc_mla_workspace); with `out`, `lse` and `workspace` given the
            call allocates nothing and never synchronises with the host (CUDA-Graph safe).
        impl: "v2" (default; sm86-optimised kernels), "legacy" (the original sm80 kernel) or "legacy_ws"
            (the original warp-specialised sm80 kernel).
        kv_scale: per-token fp32 dequant scales of a quantized KV cache (docs/kv_quant_format.md).

    Returns:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla_cuda.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        impl,
        out,
        lse,
        workspace,
        kv_scale,
    )
    return out, softmax_lse
