"""SM100 acceptance gate for the fixed-topology ROOT fusion launch."""

from __future__ import annotations

import os
from math import prod

import pytest
import torch

from flash_mla.persistent_sparse_attention import (
    csa_root_fusion as root_fusion,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_MLA_CSA_ROOT_ACCEPTANCE") != "1",
    reason="set FLASH_MLA_CSA_ROOT_ACCEPTANCE=1 for the production ROOT gate",
)


def _guarded_contiguous(shape, dtype, sentinel):
    elements = prod(shape)
    backing = torch.empty(elements + 64, dtype=dtype, device="cuda")
    backing.fill_(sentinel)
    return backing, backing[32 : 32 + elements].view(shape)


def _guarded_mn_major(rows: int, columns: int, sentinel: int):
    backing = torch.full(
        (columns, rows + 64), sentinel, dtype=torch.int32, device="cuda"
    )
    return backing, backing[:, 32 : 32 + rows].t()


def _assert_outer_guards(backing: torch.Tensor, sentinel) -> None:
    flat = backing.flatten()
    assert bool((flat[:32] == sentinel).all())
    assert bool((flat[-32:] == sentinel).all())


def _assert_mn_guards(backing: torch.Tensor, sentinel: int) -> None:
    assert bool((backing[:, :32] == sentinel).all())
    assert bool((backing[:, -32:] == sentinel).all())


def test_root_fusion_production_zero_oracle_graph_and_canary() -> None:
    # B200 and GB200 select separate precompiled total grids and runtime lane
    # splits while preserving one resident CTA per assigned SM.
    physical_sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert physical_sms in {148, 152}
    grid_sms = physical_sms
    default_projection_sms = 108 if physical_sms == 152 else 104
    alternate_projection_sms = (104, 106) if physical_sms == 152 else (100,)
    tokens, hidden = 8192, 7168
    fp8 = torch.float8_e4m3fn

    hidden_input = torch.zeros(
        (tokens, hidden), dtype=torch.bfloat16, device="cuda"
    )
    hidden_fp8_backing, hidden_fp8 = _guarded_contiguous(
        (tokens, hidden), fp8, 1.0
    )
    hidden_scale_backing, hidden_scale = _guarded_mn_major(
        tokens, hidden // 512, -77
    )
    fused_wqa_wkv = torch.zeros(
        (2048, hidden), dtype=fp8, device="cuda"
    )
    fused_scale_backing, fused_wqa_wkv_scale = _guarded_mn_major(
        2048, hidden // 512, 0x7F7F7F7F
    )

    qkv_backing, qkv = _guarded_contiguous(
        (tokens, 2048), torch.bfloat16, 7.0
    )
    q_norm_backing, q_norm = _guarded_contiguous(
        (tokens, 1536), torch.bfloat16, 7.0
    )
    kv_norm_backing, kv_norm = _guarded_contiguous(
        (tokens, 512), torch.bfloat16, 7.0
    )
    q_norm_fp8_backing, q_norm_fp8 = _guarded_contiguous(
        (tokens, 1536), fp8, 1.0
    )
    q_norm_scale_backing, q_norm_scale = _guarded_mn_major(tokens, 3, -99)

    index_score_backing, index_kv_score = _guarded_contiguous(
        (tokens, 512), torch.float32, 13.0
    )
    index_weight_backing, index_weights = _guarded_contiguous(
        (tokens, 64), torch.bfloat16, 11.0
    )
    index_compressor_weight = torch.zeros(
        (512, hidden), dtype=torch.bfloat16, device="cuda"
    )
    index_weight_projection = torch.zeros(
        (64, hidden), dtype=torch.bfloat16, device="cuda"
    )

    projection_barrier = torch.zeros(1, dtype=torch.int32, device="cuda")
    index_barrier = torch.zeros(8, dtype=torch.int32, device="cuda")
    row_ready = torch.zeros(4, dtype=torch.int64, device="cuda")
    c4_workspace = torch.zeros(
        (52, 8, 128), dtype=torch.float32, device="cuda"
    )
    state_cache = torch.zeros(
        (1, 4, 512), dtype=torch.float32, device="cuda"
    )
    token_to_request = torch.zeros(tokens, dtype=torch.int32, device="cuda")
    positions = torch.arange(tokens, dtype=torch.int64, device="cuda")
    state_slot_mapping = torch.full(
        (tokens,), -1, dtype=torch.int64, device="cuda"
    )
    state_block_table = torch.full(
        (1, tokens // 4), -1, dtype=torch.int32, device="cuda"
    )
    rms_weight = torch.ones(128, dtype=torch.float32, device="cuda")
    cos_sin_cache = torch.zeros(
        (tokens, 64), dtype=torch.float32, device="cuda"
    )
    kv_cache = torch.full((1, 4352), 0xA5, dtype=torch.uint8, device="cuda")
    kv_cache_before = kv_cache.clone()
    kv_slot_mapping = torch.full(
        (tokens,), -1, dtype=torch.int64, device="cuda"
    )
    ape = torch.zeros((4, 256), dtype=torch.float32, device="cuda")

    # Keep every capture input at a fixed address; replay performs no allocation.
    q_norm_weight = torch.ones(1536, dtype=torch.bfloat16, device="cuda")
    kv_norm_weight = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    capture_tensors = (
        hidden_input, hidden_fp8, hidden_scale, fused_wqa_wkv,
        fused_wqa_wkv_scale, qkv, q_norm_weight, kv_norm_weight, q_norm,
        kv_norm, q_norm_fp8, q_norm_scale, projection_barrier,
        index_compressor_weight, index_kv_score, index_weight_projection,
        index_weights, index_barrier, row_ready, c4_workspace, state_cache,
        token_to_request, positions, state_slot_mapping, state_block_table,
        rms_weight, cos_sin_cache, kv_cache, kv_slot_mapping, ape,
    )
    pointers_before = tuple(tensor.data_ptr() for tensor in capture_tensors)

    def fixed_launch(projection_sms: int = default_projection_sms) -> None:
        root_fusion(
            hidden_input, hidden_fp8, hidden_scale,
            fused_wqa_wkv, fused_wqa_wkv_scale, qkv,
            q_norm_weight, kv_norm_weight, q_norm, kv_norm,
            q_norm_fp8, q_norm_scale, projection_barrier,
            index_compressor_weight, index_kv_score,
            index_weight_projection, index_weights, index_barrier,
            row_ready, c4_workspace, state_cache, token_to_request,
            positions, state_slot_mapping, state_block_table, rms_weight,
            cos_sin_cache, kv_cache, kv_slot_mapping, ape, grid_sms, projection_sms,
        )

    fixed_launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fixed_launch()
    torch.cuda.synchronize()
    assert tuple(tensor.data_ptr() for tensor in capture_tensors) == pointers_before
    allocated_after_capture = torch.cuda.memory_allocated()
    reserved_after_capture = torch.cuda.memory_reserved()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert tuple(tensor.data_ptr() for tensor in capture_tensors) == pointers_before
    assert torch.cuda.memory_allocated() == allocated_after_capture

    # Exercise every other GB200 lane split (and one B200 split) without
    # rebuilding or changing any tensor address.
    for projection_sms in alternate_projection_sms:
        projection_barrier.zero_()
        index_barrier.zero_()
        c4_workspace.zero_()
        fixed_launch(projection_sms)
        torch.cuda.synchronize()
        assert tuple(tensor.data_ptr() for tensor in capture_tensors) == pointers_before
        assert torch.cuda.memory_reserved() == reserved_after_capture

    c4_workspace.zero_()
    assert torch.count_nonzero(hidden_fp8.view(torch.uint8)) == 0
    assert bool((hidden_scale == 0x55555555).all())
    assert torch.count_nonzero(qkv) == 0
    assert torch.count_nonzero(q_norm) == 0
    assert torch.count_nonzero(kv_norm) == 0
    assert torch.count_nonzero(q_norm_fp8.view(torch.uint8)) == 0
    assert bool((q_norm_scale == 0x55555555).all())
    # The accepted no-request route intentionally leaves this dense auxiliary
    # output unconsumed and untouched.
    assert bool((index_kv_score == 13.0).all())
    assert torch.count_nonzero(index_weights) == 0
    assert torch.equal(kv_cache, kv_cache_before)

    for backing, sentinel in (
        (hidden_fp8_backing, 1.0),
        (qkv_backing, 7.0),
        (q_norm_backing, 7.0),
        (kv_norm_backing, 7.0),
        (q_norm_fp8_backing, 1.0),
        (index_score_backing, 13.0),
        (index_weight_backing, 11.0),
    ):
        _assert_outer_guards(backing, sentinel)
    _assert_mn_guards(hidden_scale_backing, -77)
    _assert_mn_guards(q_norm_scale_backing, -99)
    _assert_mn_guards(fused_scale_backing, 0x7F7F7F7F)

    # Exercise the production state-cache and paged-KV routes with every
    # output value analytically determined by the all-zero projection.
    valid_state_cache = torch.full(
        (2049, 4, 512), 17.0, dtype=torch.float32, device="cuda"
    )
    valid_state_cache[0].zero_()
    valid_row_ready = torch.zeros(8196, dtype=torch.int64, device="cuda")
    valid_positions = torch.arange(4, 8196, dtype=torch.int64, device="cuda")
    valid_state_slots = torch.arange(4, 8196, dtype=torch.int64, device="cuda")
    valid_block_table = torch.arange(
        2049, dtype=torch.int32, device="cuda"
    ).view(1, -1)
    valid_kv_backing, valid_kv = _guarded_contiguous(
        (32, 4352), torch.uint8, 0xA5
    )
    valid_kv_slots = torch.full(
        (tokens,), -1, dtype=torch.int64, device="cuda"
    )
    valid_kv_slots[3::4] = torch.arange(2048, dtype=torch.int64, device="cuda")
    projection_barrier.zero_()
    index_barrier.zero_()

    root_fusion(
        hidden_input, hidden_fp8, hidden_scale,
        fused_wqa_wkv, fused_wqa_wkv_scale, qkv,
        q_norm_weight, kv_norm_weight, q_norm, kv_norm,
        q_norm_fp8, q_norm_scale, projection_barrier,
        index_compressor_weight, index_kv_score,
        index_weight_projection, index_weights, index_barrier,
        valid_row_ready, c4_workspace, valid_state_cache, token_to_request,
        valid_positions, valid_state_slots, valid_block_table, rms_weight,
        cos_sin_cache, valid_kv, valid_kv_slots, ape,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(valid_state_cache) == 0
    assert bool((index_kv_score == 13.0).all())
    assert torch.count_nonzero(valid_kv[:, :4096]) == 0
    # CUDA's approximate log2f may choose either adjacent scale for an exact
    # zero block. Both encodings dequantize the all-zero payload exactly.
    assert bool(
        ((valid_kv[:, 4096:] == 1) | (valid_kv[:, 4096:] == 2)).all()
    )
    _assert_outer_guards(valid_kv_backing, 0xA5)

    # A sparse deterministic impulse gives a nonzero pure-PyTorch oracle for
    # the right-hand index-compression and index-weight projections without
    # relying on any assumed FP8/MXFP4 rounding behavior.
    hidden_input.zero_()
    hidden_input[0, 0] = 1
    index_compressor_weight.zero_()
    index_compressor_weight[0, 0] = 3
    index_compressor_weight[256, 0] = 5
    index_weight_projection.zero_()
    index_weight_projection[0, 0] = 2
    index_weights.fill_(11)
    valid_state_cache.fill_(17)
    valid_state_cache[0].zero_()
    valid_row_ready.zero_()
    valid_kv_slots.fill_(-1)
    projection_barrier.zero_()
    index_barrier.zero_()

    c0_reference = torch.matmul(
        hidden_input[0].float(), index_compressor_weight.float().t()
    )
    iw0_reference = torch.matmul(
        hidden_input[0].float(), index_weight_projection.float().t()
    )
    root_fusion(
        hidden_input, hidden_fp8, hidden_scale,
        fused_wqa_wkv, fused_wqa_wkv_scale, qkv,
        q_norm_weight, kv_norm_weight, q_norm, kv_norm,
        q_norm_fp8, q_norm_scale, projection_barrier,
        index_compressor_weight, index_kv_score,
        index_weight_projection, index_weights, index_barrier,
        valid_row_ready, c4_workspace, valid_state_cache, token_to_request,
        valid_positions, valid_state_slots, valid_block_table, rms_weight,
        cos_sin_cache, valid_kv, valid_kv_slots, ape,
    )
    torch.cuda.synchronize()

    state_reference = torch.zeros_like(valid_state_cache)
    state_reference[1, 0] = c0_reference
    weight_reference = torch.zeros_like(index_weights)
    weight_reference[0] = iw0_reference.to(torch.bfloat16)
    torch.testing.assert_close(valid_state_cache, state_reference, rtol=0, atol=0)
    torch.testing.assert_close(index_weights, weight_reference, rtol=0, atol=0)
    assert bool((index_kv_score == 13.0).all())
    assert torch.count_nonzero(qkv) == 0
    assert torch.count_nonzero(q_norm) == 0
    assert torch.count_nonzero(kv_norm) == 0
