"""Pure PyTorch references for the persistent-attention leaf kernels."""

import torch


TOPK = 1024
SWA = 128
COMBINED = TOPK + SWA
INDEX_CACHE_BLOCK = 64
INDEX_VALUE_BYTES = 64
INDEX_SCALE_BYTES = 4


def combine_indices_reference(
    topk: torch.Tensor,
    compressed_rows: int,
):
    """Append the causal 128-token SWA interval to every learned TopK row."""
    if topk.ndim != 2 or topk.shape[1] != TOPK or topk.dtype != torch.int32:
        raise ValueError("topk must be int32 [tokens, 1024]")
    tokens = topk.shape[0]
    token = torch.arange(tokens, dtype=torch.int32, device=topk.device)[:, None]
    swa = torch.arange(SWA, dtype=torch.int32, device=topk.device)[None, :]
    combined = torch.cat((topk, int(compressed_rows) + token + swa), dim=1)
    lengths = torch.full(
        (tokens,), COMBINED, dtype=torch.int32, device=topk.device
    )
    return combined, lengths


def index_cache_gather_reference(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    output_capacity: int,
    *,
    dst_k: torch.Tensor | None = None,
    dst_scale: torch.Tensor | None = None,
):
    """Gather into caller-owned byte outputs, preserving invalid rows."""
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim < 2:
        raise ValueError("kv_cache must be a byte tensor with one row per page")
    if block_table.dtype != torch.int32 or cu_seq_lens.dtype != torch.int32:
        raise ValueError("block_table and cu_seq_lens must be int32")
    if dst_k is None:
        dst_k = torch.zeros(
            (output_capacity, INDEX_VALUE_BYTES), dtype=torch.uint8,
            device=kv_cache.device,
        )
    if dst_scale is None:
        dst_scale = torch.zeros(
            (output_capacity, INDEX_SCALE_BYTES), dtype=torch.uint8,
            device=kv_cache.device,
        )
    if dst_k.shape != (output_capacity, INDEX_VALUE_BYTES):
        raise ValueError("dst_k must be uint8 [output_capacity,64]")
    if dst_scale.shape != (output_capacity, INDEX_SCALE_BYTES):
        raise ValueError("dst_scale must be uint8 [output_capacity,4]")
    if dst_k.dtype != torch.uint8 or dst_scale.dtype != torch.uint8:
        raise ValueError("gather outputs must be uint8")
    pages = kv_cache.reshape(kv_cache.shape[0], -1)
    for batch in range(block_table.shape[0]):
        start = int(cu_seq_lens[batch])
        end = int(cu_seq_lens[batch + 1])
        for token in range(start, min(end, output_capacity)):
            local = token - start
            physical = int(block_table[batch, local // INDEX_CACHE_BLOCK])
            if physical < 0 or physical >= pages.shape[0]:
                continue
            in_block = local % INDEX_CACHE_BLOCK
            value_start = in_block * INDEX_VALUE_BYTES
            scale_start = (
                INDEX_CACHE_BLOCK * INDEX_VALUE_BYTES
                + in_block * INDEX_SCALE_BYTES
            )
            dst_k[token].copy_(
                pages[physical, value_start:value_start + INDEX_VALUE_BYTES]
            )
            dst_scale[token].copy_(
                pages[physical, scale_start:scale_start + INDEX_SCALE_BYTES]
            )
    return dst_k, dst_scale


def reset_ready_reference(*buffers: torch.Tensor) -> None:
    """In-place reference for the ready-protocol housekeeping launch."""
    if len(buffers) not in (3, 5):
        raise ValueError("reset_ready accepts exactly three or five buffers")
    for buffer in buffers:
        buffer.zero_()


def swa_cache_insert_reference(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int = 64,
) -> None:
    """Raw-layout reference for CSA's mixed FP8/BF16 SWA cache insert."""
    if kv.dtype != torch.bfloat16 or kv.ndim != 2 or kv.shape[1] != 512:
        raise ValueError("kv must be bfloat16 [tokens, 512]")
    if cache.dtype != torch.uint8 or cache.ndim != 2:
        raise ValueError("cache must be uint8 [pages, page_stride]")
    for token in range(kv.shape[0]):
        slot = int(slot_mapping[token])
        if slot < 0:
            continue
        page = slot // cache_block_size
        offset = slot % cache_block_size
        values = kv[token].float()
        rope = values[448:].reshape(32, 2)
        table = cos_sin_cache[int(position_ids[token])]
        cosine, sine = table[:32], table[32:64]
        rotated = torch.stack(
            (rope[:, 0] * cosine - rope[:, 1] * sine,
             rope[:, 0] * sine + rope[:, 1] * cosine), dim=1,
        ).reshape(64).to(torch.bfloat16)
        rounded = values[:448].to(torch.bfloat16).float().reshape(7, 64)
        absmax = rounded.abs().amax(dim=1).clamp_min(1.0e-4)
        exponent = torch.ceil(torch.log2(absmax / 448.0))
        scaled = (rounded * torch.exp2(-exponent[:, None])).clamp(-448, 448)
        packed = scaled.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
        page_row = cache[page]
        data_start = offset * 576
        page_row[data_start:data_start + 448].copy_(packed)
        page_row[data_start + 448:data_start + 576].copy_(
            rotated.view(torch.uint8)
        )
        scale_start = cache_block_size * 576 + offset * 8
        page_row[scale_start:scale_start + 7].copy_(
            (exponent + 127).clamp(0, 255).to(torch.uint8)
        )
        page_row[scale_start + 7] = 0
