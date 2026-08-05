"""Fused Indexer-Q RoPE/MXFP4 preparation and weight scaling."""

from __future__ import annotations

import torch


MXFP4_BLOCK_SIZE = 32


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def index_query_prepare(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    q_packed: torch.Tensor,
    q_scale: torch.Tensor,
    weights_out: torch.Tensor,
    *,
    num_sms: int = 56,
) -> None:
    """Apply interleaved tail RoPE, MXFP4-pack Q, and scale index weights.

    All outputs are caller-owned to keep the operation allocation-free and
    CUDA-Graph capture safe.  The accepted GB200 schedule uses 56 full CTAs;
    ``num_sms`` is explicit so callers can retarget the fixed-grid lane.
    """

    _check_tensor(positions, name="positions", dtype=torch.int64, ndim=1)
    _check_tensor(index_q, name="index_q", dtype=torch.bfloat16, ndim=3)
    if cos_sin_cache.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "cos_sin_cache must have dtype torch.float32 or torch.bfloat16"
        )
    _check_tensor(
        cos_sin_cache,
        name="cos_sin_cache",
        dtype=cos_sin_cache.dtype,
        ndim=2,
    )
    _check_tensor(
        index_weights,
        name="index_weights",
        dtype=torch.bfloat16,
        ndim=2,
    )
    _check_tensor(q_packed, name="q_packed", dtype=torch.uint8, ndim=3)
    _check_tensor(q_scale, name="q_scale", dtype=torch.uint8, ndim=3)
    _check_tensor(weights_out, name="weights_out", dtype=torch.float32, ndim=2)

    num_tokens, num_heads, head_dim = index_q.shape
    rope_dim = cos_sin_cache.shape[1]
    if head_dim % MXFP4_BLOCK_SIZE:
        raise ValueError("index_q head dimension must be divisible by 32")
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 16:
        raise ValueError("RoPE dimension must be positive, <= head_dim, and /16")
    if positions.shape != (num_tokens,):
        raise ValueError("positions length must equal the index_q token count")
    if index_weights.shape != (num_tokens, num_heads):
        raise ValueError("index_weights shape must be [tokens, heads]")
    if q_packed.shape != (num_tokens, num_heads, head_dim // 2):
        raise ValueError("q_packed shape must be [tokens, heads, head_dim/2]")
    if q_scale.shape != (
        num_tokens,
        num_heads,
        head_dim // MXFP4_BLOCK_SIZE,
    ):
        raise ValueError("q_scale shape must be [tokens, heads, head_dim/32]")
    if weights_out.shape != index_weights.shape:
        raise ValueError("weights_out shape must match index_weights")
    if not all(
        tensor.device == index_q.device
        for tensor in (
            positions,
            cos_sin_cache,
            index_weights,
            q_packed,
            q_scale,
            weights_out,
        )
    ):
        raise ValueError("all tensors must be on the same CUDA device")
    if int(num_sms) <= 0:
        raise ValueError("num_sms must be positive")
    if index_q.data_ptr() % 32:
        raise ValueError("index_q storage must be 32-byte aligned")
    if cos_sin_cache.data_ptr() % 32:
        raise ValueError("cos_sin_cache storage must be 32-byte aligned")
    if index_q.data_ptr() % 32:
        raise ValueError("index_q storage must be 32-byte aligned")

    try:
        from ._index_query_prepare_cute import launch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "index_query_prepare requires CUTLASS CuTeDSL and cuda-python"
        ) from error

    launch(
        positions,
        index_q,
        cos_sin_cache,
        index_weights,
        float(softmax_scale),
        float(head_scale),
        q_packed,
        q_scale,
        weights_out,
        int(num_sms),
    )


__all__ = ["MXFP4_BLOCK_SIZE", "index_query_prepare"]
