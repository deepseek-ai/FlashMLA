"""SM100 acceptance gates for the persistent sparse-attention extension.

These tests are intentionally opt-in: they require the extension to have been
built in place and a CUDA device with the accepted Blackwell grid sizes.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from reference.leaf_ops import (
    combine_indices_reference,
    index_cache_gather_reference,
    swa_cache_insert_reference,
)
from reference.learned_topk import learned_topk_reference
from reference.quant_ops import (
    attention_output_quant_reference,
    output_quant_reference,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_MLA_CSA_ACCEPTANCE") != "1"
    or not torch.cuda.is_available(),
    reason="set FLASH_MLA_CSA_ACCEPTANCE=1 on an SM100 GPU",
)


def load_extension():
    from flash_mla import persistent_sparse_attention_cuda as csa_cuda

    return csa_cuda


def _graph_replay(operation) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()


def _guarded(numel: int, dtype: torch.dtype, canary: int):
    backing = torch.full(
        (32 + numel + 32,), canary, dtype=dtype, device="cuda"
    )
    return backing, backing[32 : 32 + numel]


def _assert_guards(backing: torch.Tensor, canary: int) -> None:
    assert bool((backing[:32] == canary).all())
    assert bool((backing[-32:] == canary).all())


def test_swa_cache_insert_native_exact_repeat_graph_and_canary() -> None:
    extension = load_extension()
    torch.manual_seed(3)
    tokens, block_size = 5, 64
    kv = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64, device="cuda")
    slots = torch.tensor([0, 63, 64, -1, 3], dtype=torch.int64, device="cuda")
    cos_sin = torch.randn(4, 64, dtype=torch.float32, device="cuda")
    page_bytes = block_size * 584
    backing, view = _guarded(2 * page_bytes, torch.uint8, 0xA5)
    cache = view.reshape(2, page_bytes)
    expected = cache.clone()
    swa_cache_insert_reference(
        kv, expected, slots, positions, cos_sin, block_size
    )
    _graph_replay(
        lambda: extension.swa_cache_insert(
            kv, cache, slots, positions, cos_sin, block_size, 4
        )
    )
    assert torch.equal(cache, expected)
    _assert_guards(backing, 0xA5)


def test_index_cache_gather_native_exact_repeat_graph_and_canary() -> None:
    extension = load_extension()
    page_bytes, capacity = 64 * 68, 130
    cache = (
        torch.arange(3 * page_bytes, device="cuda", dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(3, page_bytes)
    )
    table = torch.tensor([[2, 0], [1, -1]], dtype=torch.int32, device="cuda")
    cu_lens = torch.tensor([0, 65, 130], dtype=torch.int32, device="cuda")
    k_backing, k_view = _guarded(capacity * 64, torch.uint8, 0xA5)
    s_backing, s_view = _guarded(capacity * 4, torch.uint8, 0x5A)
    dst_k = k_view.reshape(capacity, 64)
    dst_s = s_view.reshape(capacity, 4)
    ref_k = dst_k.clone()
    ref_s = dst_s.clone()
    index_cache_gather_reference(
        cache, table, cu_lens, capacity, dst_k=ref_k, dst_scale=ref_s
    )
    _graph_replay(
        lambda: extension.index_cache_gather(
            cache, dst_k, dst_s, table, cu_lens, 40, True
        )
    )
    assert torch.equal(dst_k, ref_k)
    assert torch.equal(dst_s, ref_s)
    _assert_guards(k_backing, 0xA5)
    _assert_guards(s_backing, 0x5A)


def test_learned_topk_tie_aware_sets_and_scores_graph() -> None:
    extension = load_extension()
    torch.manual_seed(7)
    logits = torch.randn(3, 4096, dtype=torch.float32, device="cuda")
    logits[0, :700] = 6.0
    logits[0, 700:1700] = 5.0
    logits[1, :2048] = 3.0
    starts = torch.tensor([0, 17, 8], dtype=torch.int32, device="cuda")
    ends = torch.tensor([4096, 17 + 2048, 900], dtype=torch.int32, device="cuda")
    expected = learned_topk_reference(logits, starts, ends)
    backing, view = _guarded(3 * 1024, torch.int32, -77)
    output = view.reshape(3, 1024)
    _graph_replay(lambda: extension.learned_topk(logits, starts, ends, output, 40))
    assert torch.equal(output.sort(dim=-1).values, expected.sort(dim=-1).values)
    for row in range(logits.size(0)):
        valid_actual = output[row][output[row] >= 0].to(torch.int64)
        valid_expected = expected[row][expected[row] >= 0].to(torch.int64)
        actual_bits = logits[row, starts[row] + valid_actual].view(torch.int32)
        expected_bits = logits[row, starts[row] + valid_expected].view(torch.int32)
        assert torch.equal(
            actual_bits.sort().values,
            expected_bits.sort().values,
        )
    _assert_guards(backing, -77)


def test_learned_topk_forced_cutoff_is_canonical_and_repeat_exact() -> None:
    extension = load_extension()
    row_start, row_length = 37, 4096
    logits = torch.full(
        (2, row_start + row_length), 1.0, dtype=torch.float32, device="cuda"
    )
    starts = torch.full((2,), row_start, dtype=torch.int32, device="cuda")
    ends = torch.full(
        (2,), row_start + row_length, dtype=torch.int32, device="cuda"
    )
    output = torch.empty((2, 1024), dtype=torch.int32, device="cuda")
    expected = torch.arange(1024, dtype=torch.int32, device="cuda").expand(2, -1)
    for _ in range(3):
        output.fill_(-1)
        extension.learned_topk(logits, starts, ends, output, 40)
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
    _graph_replay(lambda: extension.learned_topk(logits, starts, ends, output, 40))
    assert torch.equal(output, expected)


def test_combine_indices_native_exact_repeat_graph_and_canary() -> None:
    extension = load_extension()
    topk = torch.arange(4 * 1024, dtype=torch.int32, device="cuda").reshape(4, 1024)
    expected, expected_lengths = combine_indices_reference(topk, 8192)
    out_backing, out_view = _guarded(4 * 1152, torch.int32, -77)
    len_backing, len_view = _guarded(4, torch.int32, -99)
    combined = out_view.reshape(4, 1152)
    _graph_replay(
        lambda: extension.combine_indices(topk, combined, len_view, 8192, 40)
    )
    assert torch.equal(combined, expected)
    assert torch.equal(len_view, expected_lengths)
    _assert_guards(out_backing, -77)
    _assert_guards(len_backing, -99)


def test_attention_output_quant_finite_raw_bits_graph_and_canary() -> None:
    extension = load_extension()
    torch.manual_seed(11)
    tokens = 3
    value = torch.randn(
        tokens, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    cos_sin = torch.randn(3, 64, dtype=torch.float32, device="cuda")
    ref_output, ref_scales, _ = attention_output_quant_reference(
        value, positions, cos_sin
    )
    out_backing, out_view = _guarded(tokens * 16 * 4096, torch.uint8, 0xA5)
    output = out_view.reshape(16, tokens, 4096).transpose(0, 1)
    aligned = (tokens + 3) & ~3
    scale_backing = torch.full(
        (32 + 16 * 8 * aligned + 32,),
        0x5A5A5A5A,
        dtype=torch.int32,
        device="cuda",
    )
    physical = scale_backing[32 : 32 + 16 * 8 * aligned]
    scales = physical.as_strided(
        (16, tokens, 8), (8 * aligned, 1, aligned)
    ).transpose(0, 1)
    _graph_replay(
        lambda: extension.attention_output_quant(
            value, positions, cos_sin, output, scales, 40
        )
    )
    assert torch.equal(output, ref_output)
    assert torch.equal(scales, ref_scales)
    _assert_guards(out_backing, 0xA5)
    assert bool((scale_backing[:32] == 0x5A5A5A5A).all())
    assert bool((scale_backing[-32:] == 0x5A5A5A5A).all())


def test_attention_output_quant_cuda_special_source_oracle() -> None:
    extension = load_extension()
    value = torch.empty((1, 128, 512), dtype=torch.bfloat16, device="cuda")
    value[..., :128] = float("nan")
    value[..., 128:256] = float("inf")
    value[..., 256:384] = -float("inf")
    pattern = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0,
         -0.0, 1.0, -1.0, 448.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    value[..., 384:] = pattern.repeat(16)
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    cos_sin = torch.cat(
        (torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")),
        dim=1,
    )
    out_backing, out_view = _guarded(16 * 4096, torch.uint8, 0xA5)
    output = out_view.reshape(16, 1, 4096).transpose(0, 1)
    scale_backing, scale_view = _guarded(16 * 8 * 4, torch.int32, 0x5A5A5A5A)
    scales = scale_view.as_strided((1, 16, 8), (1, 32, 4))
    _graph_replay(
        lambda: extension.attention_output_quant(
            value, positions, cos_sin, output, scales, 40
        )
    )
    expected_pattern = torch.tensor(
        [254, 126, 254, 0, 128, 64, 192, 126],
        dtype=torch.uint8,
        device="cuda",
    )
    heads = output.reshape(128, 512)
    assert bool((heads[:, :128] == 254).all())
    assert bool((heads[:, 128:256] == 126).all())
    assert bool((heads[:, 256:384] == 254).all())
    assert torch.equal(
        heads[:, 384:448].reshape(128, 8, 8),
        expected_pattern.expand(128, 8, 8),
    )
    expected_rope_pattern = torch.tensor(
        [254, 254, 254, 254, 0, 64, 192, 126],
        dtype=torch.uint8,
        device="cuda",
    )
    assert torch.equal(
        heads[:, 448:].reshape(128, 8, 8),
        expected_rope_pattern.expand(128, 8, 8),
    )
    assert bool((scales == 0x7E7E7E55).all())
    _assert_guards(out_backing, 0xA5)
    _assert_guards(scale_backing, 0x5A5A5A5A)


def test_output_quant_production_shape_raw_bits_specials_graph_canary() -> None:
    extension = load_extension()
    torch.manual_seed(13)
    value = torch.randn(
        8192, 16384, dtype=torch.bfloat16, device="cuda"
    )
    value[0, :5] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    ref_output, ref_scales = output_quant_reference(value)
    out_backing, out_view = _guarded(value.numel(), torch.uint8, 0xA5)
    output = out_view.reshape_as(value)
    scale_backing, scale_view = _guarded(32 * 8192, torch.int32, 0x5A5A5A5A)
    scales = scale_view.reshape(32, 8192)
    _graph_replay(lambda: extension.output_quant(value, output, scales, 40))
    assert torch.equal(output, ref_output)
    assert torch.equal(scales, ref_scales)
    _assert_guards(out_backing, 0xA5)
    _assert_guards(scale_backing, 0x5A5A5A5A)


@pytest.mark.parametrize("five_buffers", [False, True])
def test_reset_ready_exact_repeat_graph_and_canary(five_buffers: bool) -> None:
    extension = load_extension()
    n = 4097
    c_backing, counters = _guarded(n, torch.int32, 77)
    r_backing, ready = _guarded(n, torch.int32, 88)
    p_backing, producer = _guarded(n + 1, torch.int64, 99)
    counters.fill_(1)
    ready.fill_(2)
    producer.fill_(3)
    buffers = [counters, ready, producer]
    guarded = [(c_backing, 77), (r_backing, 88), (p_backing, 99)]
    if five_buffers:
        x_backing, consumer = _guarded(n, torch.int64, 111)
        q_backing, qnorm = _guarded(n, torch.int64, 122)
        consumer.fill_(4)
        qnorm.fill_(5)
        buffers.extend((consumer, qnorm))
        guarded.extend(((x_backing, 111), (q_backing, 122)))
    _graph_replay(lambda: extension.reset_ready(*buffers, 40))
    assert all(not bool(buffer.any()) for buffer in buffers)
    for backing, canary in guarded:
        _assert_guards(backing, canary)
