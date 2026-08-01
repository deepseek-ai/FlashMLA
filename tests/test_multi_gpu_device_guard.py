"""
Regression tests for issue #158: kernels must honor the input tensor's device,
not the process's current CUDA device.

Before the fix, ``Arch arch = Arch();`` was constructed in the C++ API entry
points before ``at::cuda::CUDAGuard`` was installed.  ``Arch()`` reads the
*current* device's properties via ``at::cuda::getCurrentDeviceProperties()``,
so when the process's current device differed from ``q.device()`` (a common
setup in multi-GPU inference servers and distributed training), the dispatcher
could:

  * pick the wrong SM-specific impl (SM90 vs SM100) in heterogeneous boxes,
  * compute ``num_sm_parts`` from the wrong SM count, and
  * launch kernels on the wrong stream.

These tests pin ``torch.cuda.set_device(0)`` but place every tensor on
``cuda:1`` and assert that the result is bit-/numerically-equal to running
the same workload with matched current device.  They are skipped when fewer
than two CUDA devices are visible.
"""

import pytest
import torch

import flash_mla


def _require_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices")


def _dense_decode_inputs(device: torch.device, dtype=torch.bfloat16):
    torch.manual_seed(0)
    b, s_q, h_q, h_kv, d_qk, d_v = 1, 1, 128, 1, 576, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = 512

    q = (torch.randn(b, s_q, h_q, d_qk, dtype=dtype, device=device) / 10).clamp_(-1, 1)
    kcache = (torch.randn(num_blocks, page_block_size, h_kv, d_qk, dtype=dtype, device=device) / 10).clamp_(-1, 1)
    cache_seqlens = torch.full((b,), s_kv, dtype=torch.int32, device=device)
    block_table = torch.arange(b * (s_kv // page_block_size), dtype=torch.int32, device=device).view(b, -1)
    return q, kcache, cache_seqlens, block_table, d_v


def _run_dense_decode(q, kcache, cache_seqlens, block_table, d_v):
    sched_meta, _ = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=kcache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=d_v,
        tile_scheduler_metadata=sched_meta,
        causal=False,
    )
    return out, lse


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_dense_decode_respects_input_device_when_current_device_differs():
    """Regression: dense decode must work when q lives on cuda:1 but current device is 0."""
    _require_two_gpus()

    # Reference: current device == tensor device.
    torch.cuda.set_device(1)
    q_ref, k_ref, sq_ref, bt_ref, d_v = _dense_decode_inputs(torch.device("cuda:1"))
    out_ref, lse_ref = _run_dense_decode(q_ref, k_ref, sq_ref, bt_ref, d_v)
    torch.cuda.synchronize(1)

    # Under test: current device mismatches tensor device.  Without the fix this
    # path either picked the wrong SM impl, queried the wrong SM count, or
    # launched on cuda:0's stream.
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    q_mm, k_mm, sq_mm, bt_mm, _ = _dense_decode_inputs(torch.device("cuda:1"))
    out_mm, lse_mm = _run_dense_decode(q_mm, k_mm, sq_mm, bt_mm, d_v)
    torch.cuda.synchronize(1)

    assert out_mm.device == torch.device("cuda:1"), f"output landed on {out_mm.device}"
    assert lse_mm.device == torch.device("cuda:1")
    torch.testing.assert_close(out_mm, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(lse_mm, lse_ref, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_dense_decode_current_device_unchanged_after_call():
    """The guard must restore the caller's current device on exit."""
    _require_two_gpus()

    torch.cuda.set_device(0)
    q, kcache, sq, bt, d_v = _dense_decode_inputs(torch.device("cuda:1"))
    _run_dense_decode(q, kcache, sq, bt, d_v)
    assert torch.cuda.current_device() == 0, "CUDAGuard leaked: current device changed"
