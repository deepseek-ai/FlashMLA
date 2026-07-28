from typing import Optional, Sequence, Tuple
import dataclasses

import torch

import flash_mla.cuda as flash_mla_cuda

@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (1), dtype torch.int32.


def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def get_packed_kv_workspace_size(
    batch_size: int,
    query_len: int,
    topk: int,
) -> int:
    """Return the packed V3.2 sparse-decode workspace size in bytes."""
    if batch_size < 1 or query_len < 1:
        raise ValueError("batch_size and query_len must be positive")
    if topk < 1 or topk % 64:
        raise ValueError("topk must be positive and divisible by 64")
    return batch_size * query_len * topk * 656


def get_packed_mla_metadata(
    *,
    batch_size: int,
    query_len: int,
    num_heads: int,
    topk: int,
    device: torch.device | str,
    topk_length: Optional[int | Sequence[int]] = None,
    num_sms: Optional[int] = None,
) -> FlashMLASchedMeta:
    """Create the measured H=64 packed scheduler.

    ``topk_length`` is a host integer or sequence, not a CUDA tensor.
    Full-length H=128 intentionally returns empty metadata so packed decode
    uses the same scheduler heuristic as native sparse decode. H=64 uses the
    compact, request-local scheduler selected by the L20X experiments, except
    for the measured B*S_q=16 full-prefix regime where the native scheduler is
    faster. Partial H=128 requests receive a compact schedule so no row extends
    beyond the valid prefix.
    """
    if batch_size < 1 or query_len < 1:
        raise ValueError("batch_size and query_len must be positive")
    if num_heads not in (64, 128):
        raise ValueError("num_heads must be 64 or 128")
    if topk < 1 or topk % 64:
        raise ValueError("topk must be positive and divisible by 64")

    if topk_length is None:
        active_lengths = (topk,) * batch_size
    elif isinstance(topk_length, int):
        active_lengths = (topk_length,) * batch_size
    else:
        active_lengths = tuple(int(value) for value in topk_length)
    if len(active_lengths) != batch_size:
        raise ValueError("topk_length must have one value per batch")
    if any(value < 1 or value > topk for value in active_lengths):
        raise ValueError("topk lengths must be in [1, topk]")

    meta = FlashMLASchedMeta()
    if num_heads == 128 and all(
        value == topk for value in active_lengths
    ):
        return meta

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_sms < 1:
        raise ValueError("num_sms must be positive")

    if (
        num_heads == 64
        and num_sms == 132
        and batch_size * query_len == 16
        and topk == 2048
        and all(value == topk for value in active_lengths)
    ):
        return meta

    active_tiles = tuple((value + 63) // 64 for value in active_lengths)
    native_num_parts = max(
        num_sms // query_len // (num_heads // 64),
        1,
    )
    total_weighted_blocks = sum(tiles + 5 for tiles in active_tiles)
    real_tiles_per_split = max(
        2,
        (total_weighted_blocks + native_num_parts - 1)
        // native_num_parts,
    )
    while sum(
        (tiles + real_tiles_per_split - 1) // real_tiles_per_split
        for tiles in active_tiles
    ) > 160:
        real_tiles_per_split += 1

    rows: list[list[int]] = []
    cumulative_splits = [0]
    for batch_idx, tiles in enumerate(active_tiles):
        splits = (
            tiles + real_tiles_per_split - 1
        ) // real_tiles_per_split
        small_width, wide_split_count = divmod(tiles, splits)
        begin_block = 0
        is_split = int(splits > 1)
        for split_idx in range(splits):
            split_width = small_width + int(
                split_idx < wide_split_count
            )
            end_block = begin_block + split_width
            rows.append(
                [
                    batch_idx,
                    batch_idx,
                    begin_block,
                    end_block,
                    split_idx,
                    is_split,
                    is_split,
                    0,
                ]
            )
            begin_block = end_block
        cumulative_splits.append(cumulative_splits[-1] + splits)

    meta.tile_scheduler_metadata = torch.tensor(
        rows,
        dtype=torch.int32,
        device=device,
    )
    meta.num_splits = torch.tensor(
        cumulative_splits,
        dtype=torch.int32,
        device=device,
    )
    return meta


def pack_selected_kv(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack selected V3.2 FP8 KV records into FlashMLA's opaque layout.

    Valid indices must form a prefix in every selection row. Invalid suffix
    entries are zero-filled and ignored by packed attention. ``out`` may be
    any contiguous uint8 allocation with the size returned by
    ``get_packed_kv_workspace_size``; always use the returned opaque view.
    """
    return flash_mla_cuda.pack_selected_kv(
        k_cache,
        indices,
        topk_length,
        out,
    )


def flash_mla_with_packed_kvcache(
    q: torch.Tensor,
    packed_k_cache: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SM90 V3.2 sparse decode from a packed selected-KV workspace."""
    if not isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        raise TypeError(
            "tile_scheduler_metadata must be FlashMLASchedMeta"
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    sched_meta = tile_scheduler_metadata
    if not sched_meta.have_initialized:
        pages_per_row, remainder = divmod(
            packed_k_cache.shape[0],
            q.shape[0] * q.shape[1],
        )
        if remainder:
            raise ValueError(
                "packed page count must be divisible by B*S_q"
            )
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            64,
            1,
            False,
            True,
            pages_per_row * 64,
            None,
            None,
        )
    else:
        assert sched_meta.config is not None
        config = sched_meta.config
        pages_per_row, remainder = divmod(
            packed_k_cache.shape[0],
            q.shape[0] * q.shape[1],
        )
        if remainder:
            raise ValueError(
                "packed page count must be divisible by B*S_q"
            )
        expected = (
            config.b,
            config.s_q,
            config.h_q,
            config.topk,
        )
        actual = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            pages_per_row * 64,
        )
        if expected != actual:
            raise ValueError(
                "packed inputs are inconsistent with cached scheduler "
                f"metadata: expected {expected}, got {actual}"
            )

    out, lse, new_metadata, new_num_splits = (
        flash_mla_cuda.packed_sparse_decode_fwd(
            q,
            packed_k_cache,
            topk_length,
            attn_sink,
            sched_meta.tile_scheduler_metadata,
            sched_meta.num_splits,
            head_dim_v,
            softmax_scale,
        )
    )
    sched_meta.tile_scheduler_metadata = new_metadata
    sched_meta.num_splits = new_num_splits
    return out, lse


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
                Different modes (including fp8/bf16, and sparsity) has different KV cache layouts. See comments below for details.
                The KV cache must be contiguously valid for sparse attention on sm100. Here "contiguously valid" means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't IMA). In other words, the KV cache could be a slice of a larger array, but cannot be a list of disjoint memory blocks.
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32. Can be None when sparse attention is used.
        cache_seqlens: (batch_size), torch.int32. Can be None when sparse attention is used.
        head_dim_v: Head_dim of v. Must be 512
        sched_meta: FlashMLASchedMeta, return by get_mla_metadata. You may reuse the same sched_meta across different invocations, but only when the tensor shapes and the values of cache_seqlens, topk_length, and extra_topk_length remain the same.
        num_splits_placeholder: must be "None" (to be compatible with the old interface).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Whether to apply causal attention mask. Only valid for dense attention
        is_fp8_kvcache: bool.
        indices: (batch_size, seq_len_q, topk). KV indices when sparse attention is enabled.
                    Pay attention that indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * block_size + (the offset of token t among the page block),
                    where t is the k-th token of the j-th q-sequence in the i-th batch.
        attn_sink: Optional[torch.Tensor], (num_heads_q, ), torch.float32. If presented, the final output will be scaled by exp(lse) / (exp(lse) + exp(attn_sink)). Have no affect on the returned softmax_lse. +inf will cause the result to become 0.
        extra_k_cache and extra_indices_in_kvcache: If provided, will attend to these extra tokens in addition to those in k_cache and indices_in_kvcache. Their format requirements are the same as k_cache and indices_in_kvcache respectively.
        topk_length/extra_topk_length: (batch_size, ), torch.int32. If provided, only the leftmost topk_length indices will be processed. Useful when the actual topk for different queries are different so that we can save some computation, compared to masking.
    
    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2:
        head_dim should be 576 while head_dim_v should be 512.
        In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
            - The shape of the tensor `k_cache` is (num_blocks, page_block_size, num_heads_k, head_dim), and num_heads_k must be 1.
            - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
            - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"

    topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        # Sanity check. We only perform sanity check during the first invocation to save CPU time.
        if indices_in_kvcache is not None:
            assert not causal, "causal must be False when indices_in_kvcache is not None (i.e. sparse attention is enabled)"
            
        # Initialize the tile scheduler metadata during the first invocation.
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],

            causal,
            is_fp8_kvcache,
            topk,

            extra_k_page_block_size,
            extra_topk,
        )
    else:
        # Check whether the input arguments are consistent with sched_meta
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + helper_msg
        assert sched_meta.config.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + helper_msg
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + helper_msg
        assert sched_meta.config.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + helper_msg

    if topk is not None:
        # Sparse attention
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True when sparse attention is enabled"
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale
        )
    else:
        # Dense attention
        assert indices_in_kvcache is None and attn_sink is None and extra_k_cache is None and extra_indices_in_kvcache is None and topk_length is None and extra_topk_length is None, "indices_in_kvcache, attn_sink, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used."
        assert block_table is not None and cache_seqlens is not None, "block_table and cache_seqlens must be provided when dense attention is used."
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
            q, k_cache, head_dim_v,
            cache_seqlens, block_table,
            softmax_scale, causal,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits
        )
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits
    return (out, lse)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    return results


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # Make lse contiguous on seqlen dim
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    max_seqlen_qo_aligned = (max_seqlen_qo + 7) // 8 * 8
    bs = cu_seqlens_qo.shape[0] - 1
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO and scaled_lse
    if num_qo_heads != num_kv_heads:
        workspace_bytes += 2 * kv_total_len * num_qo_heads * (head_dim_qk + head_dim_vo)  # dKV_acc
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    flash_mla_cuda.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen,
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )
