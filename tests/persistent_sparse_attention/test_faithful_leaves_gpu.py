"""SM100 correctness gates for the persistent-attention faithful leaves."""

from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("FLASH_MLA_CSA_FAITHFUL_ACCEPTANCE") != "1",
    reason="set FLASH_MLA_CSA_FAITHFUL_ACCEPTANCE=1 on a 148-SM B200",
)


SEED = 20_260_731


def _extension():
    from flash_mla import persistent_sparse_attention_cuda as _C

    properties = torch.cuda.get_device_properties(0)
    assert properties.major == 10
    assert properties.multi_processor_count == 148
    return _C


def _index_inputs(tokens: int = 257):
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    angles = torch.randn(
        tokens, 32, device="cuda", dtype=torch.float32, generator=generator
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()
    query = torch.randn(
        tokens,
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.randn(
        tokens,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return positions, rope, query, weights


def _portable_index_prepare(positions, rope, query, weights):
    from flash_mla.persistent_sparse_attention.kernels.index_query_prepare import (
        index_query_prepare,
    )

    tokens = query.shape[0]
    packed = torch.empty(tokens, 64, 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(tokens, 64, 4, device="cuda", dtype=torch.uint8)
    scaled_weights = torch.empty(
        tokens, 64, device="cuda", dtype=torch.float32
    )
    index_query_prepare(
        positions,
        query,
        rope,
        weights,
        128**-0.5,
        64**-0.5,
        packed,
        scales,
        scaled_weights,
        num_sms=104,
    )
    return packed, scales, scaled_weights


def test_index_query_quant_matches_portable_cutedsl_exactly() -> None:
    extension = _extension()
    positions, rope, query, weights = _index_inputs()
    expected_packed, expected_scales, _ = _portable_index_prepare(
        positions, rope, query, weights
    )
    actual_packed = torch.empty_like(expected_packed)
    actual_scales = torch.empty_like(expected_scales)

    extension.index_query_quant(
        positions, query, rope, actual_packed, actual_scales, 104
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_packed, expected_packed)
    assert torch.equal(actual_scales, expected_scales)


def test_index_weight_scale_matches_portable_cutedsl_exactly() -> None:
    extension = _extension()
    positions, rope, query, weights = _index_inputs()
    _, _, expected = _portable_index_prepare(positions, rope, query, weights)
    actual = torch.empty_like(expected)

    extension.index_weight_scale(
        weights, actual, 128**-0.5, 64**-0.5, 44
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_index_cache_pack_matches_native_gather_exactly() -> None:
    extension = _extension()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    pages = 5
    rows = pages * 64
    source = torch.randint(
        0,
        256,
        (pages, 64, 132),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    ).view(pages, -1)
    block_table = torch.arange(
        pages, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    cu_lens = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    expected_values = torch.empty(
        rows, 64, device="cuda", dtype=torch.uint8
    )
    expected_scales = torch.empty(rows, device="cuda", dtype=torch.int32)
    actual_values = torch.empty_like(expected_values)
    actual_scales = torch.empty_like(expected_scales)

    extension.index_cache_gather(
        source,
        expected_values,
        expected_scales.view(torch.uint8).view(-1, 4),
        block_table,
        cu_lens,
        132,
        True,
    )
    extension.index_cache_pack(
        source, actual_values, actual_scales, rows, 0, 0, 132
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_values, expected_values)
    assert torch.equal(actual_scales, expected_scales)


def test_kv_cache_gather_matches_portable_cutedsl_exactly() -> None:
    from flash_mla.persistent_sparse_attention.kernels.kv_cache_gather import (
        kv_cache_gather,
    )

    extension = _extension()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    pages = 2
    block_size = 64
    sequence_length = 80
    cache = torch.zeros(
        pages, block_size, 584, device="cuda", dtype=torch.uint8
    )
    flat_cache = cache.view(pages, -1)

    # Production pages contain one contiguous 576-byte data plane per token,
    # followed by the page's contiguous eight-byte UE8M0 scale plane.
    data = flat_cache[:, : block_size * 576].view(pages, block_size, 576)
    scales = flat_cache[:, block_size * 576 :].view(pages, block_size, 8)
    fp8 = (
        torch.randn(
            pages,
            block_size,
            448,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    bf16_tail = (
        torch.randn(
            pages,
            block_size,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        .view(torch.uint8)
        .reshape(pages, block_size, 128)
    )
    data[:, :, :448].copy_(fp8)
    data[:, :, 448:].copy_(bf16_tail)
    scales.fill_(127)

    seq_lens = torch.tensor(
        [sequence_length], device="cuda", dtype=torch.int32
    )
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    expected = torch.empty(
        1, sequence_length, 512, device="cuda", dtype=torch.bfloat16
    )
    actual = torch.empty_like(expected)

    kv_cache_gather(
        expected,
        cache,
        seq_lens,
        None,
        block_table,
        block_size,
        0,
        num_sms=16,
    )
    extension.kv_cache_gather_persistent(
        actual,
        flat_cache,
        seq_lens,
        None,
        block_table,
        block_size,
        0,
        None,
        16,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
