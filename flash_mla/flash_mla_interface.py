from typing import Optional, Tuple
import dataclasses

import torch

import flash_mla.cuda as flash_mla_cuda


def _resolve_window_size(window_size) -> int:
    """Map a FlashAttention-style ``window_size`` to FlashMLA's kernel int.

    Accepts the FA convention ``(left, right)`` tuple (``(-1, -1)`` disables), a
    bare int (kernel convention: number of keys kept looking back; ``<=0``
    disables), or ``None``. Returns the kernel int W: the causal window keeps the
    last W keys, i.e. query q attends to keys k with ``q - W < k <= q``.

    The kernel implements a causal LEFT window only, so ``right`` must be 0
    (``(W-1, 0)`` for a causal window of width W, matching tests/test_fmha_sm100).
    """
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        return window_size
    left, right = window_size
    if left < 0 and right < 0:
        return -1
    assert right == 0, (
        "FlashMLA sliding window supports a causal left-window only "
        f"(right must be 0); got window_size={window_size}"
    )
    return left + 1

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
    extra_topk_length: Optional[torch.Tensor] = None,
    window_size=(-1, -1),   # FA-style (left, right); (-1,-1) disables. Dense decode only.
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
        # Dense attention (incl. SWA via window_size). attn_sink IS allowed here and is
        # applied as a post-rescale below (the dense decode kernel has no native sink);
        # the other sparse-only args must still be None.
        assert indices_in_kvcache is None and extra_k_cache is None and extra_indices_in_kvcache is None and topk_length is None and extra_topk_length is None, "indices_in_kvcache, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used."
        assert block_table is not None and cache_seqlens is not None, "block_table and cache_seqlens must be provided when dense attention is used."
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
            q, k_cache, head_dim_v,
            cache_seqlens, block_table,
            softmax_scale, causal, _resolve_window_size(window_size),
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits
        )
        if attn_sink is not None:
            # gpt-oss per-head sink for dense/SWA decode: out *= sigmoid(lse - sink)
            # (= exp(lse)/(exp(lse)+exp(sink))), lse unaffected (matches the sparse path's
            # native attn_sink convention). lse: (b, h_q, s_q); out: (b, s_q, h_q, d_v).
            scale = torch.sigmoid(lse.float() - attn_sink.float().view(1, -1, 1))
            out = (out.float() * scale.permute(0, 2, 1).unsqueeze(-1)).to(out.dtype)
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


def flash_mla_sparse_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do_grad: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    dq_acc_workspace: torch.Tensor,
    dkv_acc_workspace: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    fuse_reducesum: bool = True,
):
    """
    Sparse MLA attention prefill backward (sm100 only).

    The LSE handed in must be base-2 (matching FlashMLA fwd's output) —
    multiply external base-e LSEs by log2(e) before calling.

    dq_acc_workspace and dkv_acc_workspace are caller-allocated persistent
    fp32 buffers (mirrors the dense FMHA BWD workspace pattern). The kernel
    memsets and accumulates into these but does NOT manage their lifetime.
    Reuse a single buffer pair across all BWD calls (e.g. keep them at module
    level in the training framework). Sizes:
        dq_acc_workspace  : s_q * h_q * d_qk floats
        dkv_acc_workspace : s_kv * d_qk      floats

    Args:
        q:        [s_q, h_q, d_qk] bf16          (d_qk in {576, 512})
        kv:       [s_kv, 1, d_qk]  bf16          (h_kv must be 1)
        o:        [s_q, h_q, d_v]  bf16          (forward output, saved tensor)
        do_grad:  [s_q, h_q, d_v]  bf16          (upstream gradient)
        indices:  [s_q, 1, topk]   int32         (causal top-k; -1 = pad)
        lse:      [s_q, h_q]       float32       (base-2)
        dq_acc_workspace:  [>= s_q * h_q * d_qk]   float32 (caller-allocated)
        dkv_acc_workspace: [>= s_kv * d_qk]        float32 (caller-allocated)
        sm_scale: float                          (typically 1/sqrt(d_qk))
        d_v:      int                            (only 512 supported)
        fuse_reducesum: bool                     (also emit kl_target for the
                                                  un-gated KL indexer loss,
                                                  fused into the same launch)

    Returns:
        if fuse_reducesum:  (dq, dkv, kl_target)
        else:               (dq, dkv)
        where dq is [s_q, h_q, d_qk] bf16, dkv is [s_kv, 1, d_qk] bf16
              (dV occupies dkv[:, :, :d_v]), kl_target is [s_q, topk] float32.
    """
    return flash_mla_cuda.sparse_prefill_bwd(
        q, kv, o, do_grad, indices, lse,
        dq_acc_workspace, dkv_acc_workspace,
        sm_scale, d_v, fuse_reducesum
    )


def block_sparse_prefill_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do_grad: torch.Tensor,
    lse: torch.Tensor,
    q2k_blocks: torch.Tensor,
    sm_scale: float,
    window_size: int = -1,
    kv_block_size: int = 128,
    q_block_size: int = 64,
    attn_sink: Optional[torch.Tensor] = None,
):
    """Block-sparse KV-outer MLA prefill backward (sm100, MLA 192/128).

    The KV-outer counterpart to the per-token sparse bwd: a CTA owns a KV block
    and iterates only the Q blocks that selected it (via the k2q CSR built from
    q2k_blocks), accumulating dK/dV with no atomic scatter (deterministic). Up to
    ~58x faster than the dense MLA bwd at 256K when sparse (O(s*window) vs O(s^2)).

    Args:
        q,k:  [s_q, h, 192] bf16; v,o,do_grad: [s_q, h, 128] bf16 (B=1, causal)
        lse:  [h, s_q] float32 (base-e); passed to the kernel as a stride(0)==1
              [s_q, h] view internally.
        q2k_blocks: [num_q_blocks, topk] int32 -- per-Q-block selected KV-block
              ids (-1 pad); num_q_blocks = ceil(s_q / q_block_size).
        window_size: SWA width (<=0 disables); fuses with block-sparse.
        kv_block_size / q_block_size: must be 128 / 64 (== TileShapeK / TileShapeQ).

        attn_sink: optional per-head [h] fp32 gpt-oss sink. dQ/dK/dV are sink-correct via the
            sink-aware LSE produced by block_sparse_prefill_fwd(attn_sink=...); its own gradient is
            returned as d_sink. None disables.

    Returns: (dq, dk, dv, d_sink). d_sink is [h] fp32 when attn_sink is given, else None.
    """
    s_q, h, d_qk = q.shape
    d_v = v.shape[-1]
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    # d_sink ([h], fp32) is atomic-accumulated in the (reused dense) sum_OdO pass -> pre-zeroed.
    d_sink = torch.zeros(h, dtype=torch.float32, device=q.device) if attn_sink is not None else None
    ws_bytes = flash_mla_cuda.block_sparse_bwd_workspace_size(s_q, h, 1, d_qk)
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=q.device)
    lse_sh = lse.transpose(0, 1) if lse.shape[0] == h else lse  # -> [s_q, h], stride(0)==1
    flash_mla_cuda.block_sparse_prefill_bwd(
        workspace, do_grad, q, k, v, o, lse_sh, q2k_blocks, dq, dk, dv,
        sm_scale, window_size, kv_block_size, q_block_size, attn_sink, d_sink,
    )
    return dq, dk, dv, d_sink


# The gpt-oss attention sink is implemented entirely in-kernel (no torch/Triton): the
# forward folds it into the dense-MLA softmax denominator (sink-aware O + LSE, zero
# overhead) and the backward's d(sink) is folded into the bwd's existing sum_OdO pass
# (also zero overhead). It is a SWA-only feature exposed via the sink_bias argument of
# flash_attn_varlen_func.


def block_sparse_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_blocks: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
):
    """Block-sparse MLA prefill forward (sm100, MLA 192/128 non-absorbed, training).

    Forks the dense MLA fwd; each q-block (q_block_size=256 = TileShape Q) attends
    only its q2k-selected KV-blocks (kv_block_size=128, fixed by the kernel tile).
    O(s*selected) compute -> up to ~180x faster than dense (full-causal) at 256K.
    Emits O + LSE (base-e) in the dense fwd convention, so the returned lse pairs
    directly with block_sparse_prefill_bwd.

    Args:
        q,k: [s_q, h, 192] bf16; v: [s_k, h, 128] bf16 (B=1, causal, per-head K/V).
        q2k_blocks: [num_q_blocks, topk] int32 -- per-q-block selected KV-block ids
            (-1 pad); num_q_blocks = ceil(s_q / 256). Shared across heads.
        sm_scale: softmax scale (typically 192 ** -0.5).
        attn_sink: optional per-head [h] fp32 gpt-oss attention sink, folded into the softmax
            denominator in-kernel (O + LSE sink-aware). None disables (DSA/global layers).

    Returns: (o [s_q, h, 128] bf16, lse [h, s_q] float32, base-e). LSE is sink-aware when attn_sink
    is given (matches the dense MLA fwd convention), so it pairs directly with block_sparse_prefill_bwd.
    """
    s_q, h, d_qk = q.shape
    d_v = v.shape[-1]
    o = torch.empty(s_q, h, d_v, device=q.device, dtype=q.dtype)
    lse_hs = torch.empty(h, s_q, device=q.device, dtype=torch.float32)  # [h, s_q] contiguous
    lse_sh = lse_hs.transpose(0, 1)  # [s_q, h] view, stride(0)==1 (kernel output layout)
    flash_mla_cuda.block_sparse_prefill_fwd(q, k, v, o, lse_sh, q2k_blocks, sm_scale, attn_sink)
    return o, lse_hs


def expand_block_selection(q2k_blocks: torch.Tensor, factor: int) -> torch.Tensor:
    """Reconcile the fwd/bwd q-block granularity for an end-to-end train step.

    The fwd CTA covers q_block=256 rows (one q2k row per CTA); the bwd requires
    q_block=64 (== TileShapeQ). They are the SAME attention pattern because the fwd
    applies a 256-block's selection uniformly to all its rows and BOTH kernels apply
    a per-row causal mask -- so a 256-block's selection, replicated to its `factor`
    (=256/64=4) sub-blocks of 64, yields an identical (q,k) attended set. Future
    KV-blocks listed in the coarse selection are causally masked per row in both
    kernels (zero contribution), so the expansion is exact, not approximate.

    Usage:  o, lse = block_sparse_prefill_fwd(q, k, v, q2k256, scale)
            q2k64 = expand_block_selection(q2k256, 4)
            dq, dk, dv = block_sparse_prefill_bwd(q, k, v, o, do, lse, q2k64,
                                                  scale, -1, 128, 64)

    Args:
        q2k_blocks: [num_coarse_q_blocks, topk] int32 (the fwd's q_block=256 selection).
        factor: coarse/fine ratio = fwd_q_block // bwd_q_block (256 // 64 = 4).
    Returns: [num_coarse_q_blocks * factor, topk] int32 (the bwd's q_block=64 selection).
    """
    return q2k_blocks.repeat_interleave(factor, dim=0)


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
    window_size: int = -1,
    sink_bias: Optional[torch.Tensor] = None,
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
        window_size,
        # kernel reads sink as fp32 (const float*); bf16 training stores the sink param
        # in bf16, so cast here.
        (sink_bias.float() if sink_bias is not None else None),
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
    window_size: int = -1,
    sink_bias: Optional[torch.Tensor] = None,
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
    # gpt-oss attention sink: its logit gradient d_sink is accumulated INTO the bwd's
    # existing sum_OdO pass (zero extra cost). d_sink must be zero-initialized.
    d_sink = (torch.zeros(num_qo_heads, device=q.device, dtype=torch.float32)
              if sink_bias is not None else None)
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
        window_size,
        # kernel reads sink as fp32 (const float*); bf16 training stores the sink param
        # in bf16, so cast here. d_sink is fp32 (set above).
        (sink_bias.float() if sink_bias is not None else None),
        d_sink,
    )

    return dq, dk, dv, d_sink


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
        window_size: int = -1,
        sink_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # gpt-oss per-head attention sink (SWA training): folded into the softmax
        # denominator IN-KERNEL, so out + lse come back sink-aware (zero extra pass). The
        # sink-aware lse makes the bwd's P=exp(S-lse) correct -> dQ/dK/dV need no change;
        # only d(sink) is added below.
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen, window_size=window_size,
            sink_bias=sink_bias,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv,
                              sink_bias if sink_bias is not None else q.new_empty(0))
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        ctx.window_size = window_size
        ctx.has_sink = sink_bias is not None
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv, sink_bias = ctx.saved_tensors
        # d(sink) is folded into the bwd's sum_OdO pass (zero extra cost): out + lse are
        # sink-aware (in-kernel fwd), so dQ/dK/dV are already correct and d_sink comes back
        # from the same kernel. None when no sink was used.
        dq, dk, dv, d_sink = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen, window_size=ctx.window_size,
            sink_bias=(sink_bias if ctx.has_sink else None),
        )
        # Forward inputs: q,k,v,cu_qo,cu_kv,max_qo,max_kv,causal,softmax_scale,is_varlen,
        # window_size,sink_bias -> dq,dk,dv + 8 Nones + d_sink.
        # d_sink comes back fp32; match the sink param's dtype (bf16 in bf16 training) so
        # autograd accepts the grad for the bf16 sink Parameter.
        if d_sink is not None and ctx.has_sink:
            d_sink = d_sink.to(sink_bias.dtype)
        return dq, dk, dv, None, None, None, None, None, None, None, None, d_sink


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
    window_size=(-1, -1),   # FA-style (left, right); (-1,-1) disables. int also accepted.
    sink_bias: Optional[torch.Tensor] = None,   # gpt-oss per-head attention sink [h_q], fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen, _resolve_window_size(window_size),
        sink_bias,
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
    window_size=(-1, -1),   # FA-style (left, right); (-1,-1) disables. int also accepted.
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen, _resolve_window_size(window_size),
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
    window_size=(-1, -1),   # FA-style (left, right); (-1,-1) disables. int also accepted.
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen, _resolve_window_size(window_size),
    )
