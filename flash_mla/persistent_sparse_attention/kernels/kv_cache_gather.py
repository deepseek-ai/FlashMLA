"""Paged FP8/BF16 KV-cache dequantization and gather."""

from __future__ import annotations

import torch


HEAD_DIM = 512
FP8_DIM = 448
SCALE_DIM = 8


def kv_cache_gather(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    *,
    num_sms: int = 4,
) -> None:
    """Gather the causal suffix of each request into a dense BF16 buffer.

    The cache page payload is split into a data plane followed by an eight-byte
    UE8M0 scale plane.  Each token contains 448 FP8 E4M3 values and a 64-value
    BF16 tail.  ``gather_lens=None`` gathers each complete sequence.
    """

    if out.dtype != torch.bfloat16 or out.ndim != 3:
        raise TypeError("out must be a 3-D BF16 tensor")
    if k_cache.dtype != torch.uint8 or k_cache.ndim != 3:
        raise TypeError("k_cache must be a 3-D uint8 tensor")
    if seq_lens.dtype != torch.int32 or seq_lens.ndim != 1:
        raise TypeError("seq_lens must be a 1-D int32 tensor")
    if gather_lens is not None and (
        gather_lens.dtype != torch.int32 or gather_lens.ndim != 1
    ):
        raise TypeError("gather_lens must be None or a 1-D int32 tensor")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise TypeError("block_table must be a 2-D int32 tensor")
    if not all(
        tensor.is_cuda
        for tensor in (out, k_cache, seq_lens, block_table)
    ) or (gather_lens is not None and not gather_lens.is_cuda):
        raise ValueError("all tensors must be CUDA tensors")
    if not all(
        tensor.device == out.device
        for tensor in (k_cache, seq_lens, block_table)
    ) or (gather_lens is not None and gather_lens.device != out.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if out.shape[0] != seq_lens.numel() or out.shape[2] != HEAD_DIM:
        raise ValueError("out shape must be [num_requests, capacity, 512]")
    if block_table.shape[0] != seq_lens.numel():
        raise ValueError("block_table request dimension must match seq_lens")
    if gather_lens is not None and gather_lens.shape != seq_lens.shape:
        raise ValueError("gather_lens shape must match seq_lens")
    if int(block_size) <= 0 or k_cache.shape[1] != int(block_size):
        raise ValueError("block_size must be positive and match k_cache.shape[1]")
    expected_bytes = FP8_DIM + (HEAD_DIM - FP8_DIM) * 2 + SCALE_DIM
    if k_cache.shape[2] != expected_bytes:
        raise ValueError(f"k_cache logical row width must be {expected_bytes}")
    if int(offset) < 0:
        raise ValueError("offset must be non-negative")
    if int(num_sms) <= 0:
        raise ValueError("num_sms must be positive")

    try:
        from ._kv_cache_gather_cute import launch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "kv_cache_gather requires CUTLASS CuTeDSL and cuda-python"
        ) from error

    launch(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        int(block_size),
        int(offset),
        int(num_sms),
    )


__all__ = ["FP8_DIM", "HEAD_DIM", "SCALE_DIM", "kv_cache_gather"]
