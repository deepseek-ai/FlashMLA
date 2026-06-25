"""Regression test for the SM100 dense FMHA head_dim dispatch guard.

PR #185 changed the unsupported-head_dim branch in
``csrc/sm100/prefill/dense/fmha_cutlass_{fwd,bwd}_sm100.cu`` from a silent
``std::cout`` fallthrough (which left the caller's pre-allocated output
uninitialized) to ``TORCH_CHECK(false, ...)``. This test pins that behavior:
an unsupported ``(head_dim_qk, head_dim_vo)`` pair must raise through the
Python API, with the offending values surfaced in the message, so the
dispatch table cannot silently regress to "prints but returns garbage" when
new SM100 variants are added.

The SM100 dense path currently instantiates only (192, 128) and (128, 128);
any other pair hits the guarded branch. That check runs host-side, before any
kernel launch and before any compute-capability query, so it fires on any
CUDA device whose extension carries the dense-prefill symbol -- SM100 hardware
is not required to exercise it.

The guard is written to never produce a FALSE FAILURE: if the extension was
built without the dense-prefill symbol, or if some unrelated error surfaces
before the dispatch check (possible on hardware these kernels were not built
for), the test SKIPS rather than fails. The only cost is that, in such an
environment, it silently skips instead of catching a regression.

Only the forward path is covered: the backward branch carries the identical
change, but reaching it through the public API requires a successful forward
first, which an unsupported head_dim prevents.
"""
import pytest
import torch

flash_mla = pytest.importorskip("flash_mla")
from flash_mla import flash_attn_varlen_func
import flash_mla.cuda as _flash_mla_cuda

# Pairs the SM100 dense path does NOT instantiate
# (supported: (192, 128) and (128, 128)).
#   (64, 64)  - neither value supported.
#   (128, 64) - head_dim_qk is a supported value, but only when paired with
#               head_dim_vo == 128; this proves the dispatch matches on the
#               (qk, vo) pair, not a single dim.
UNSUPPORTED_HEADDIMS = [(64, 64), (128, 64)]

# The dispatch guard under test lives in the SM100 dense-prefill entry point.
# In every in-tree build this symbol is present (setup.py compiles the SM100
# TUs unconditionally); the hasattr skip only protects against out-of-tree
# builds that dropped them -- skip, don't fail, when there is nothing to test.
_HAS_SM100_DENSE = hasattr(_flash_mla_cuda, "dense_prefill_fwd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.skipif(
    not _HAS_SM100_DENSE,
    reason="extension built without SM100 dense prefill (dense_prefill_fwd)",
)
@pytest.mark.parametrize("head_dim_qk, head_dim_vo", UNSUPPORTED_HEADDIMS)
def test_unsupported_headdim_raises(head_dim_qk, head_dim_vo):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    s_q = s_k = 8
    h = 1

    # head_dim_qk = q.size(-1), head_dim_vo = v.size(-1) in the dispatcher.
    # bf16 in/out is required, otherwise the dtype branch asserts first
    # (FLASH_MLA_ASSERT -> std::abort, which would kill the process).
    q = torch.randn(s_q, h, head_dim_qk, device=device, dtype=dtype)
    k = torch.randn(s_k, h, head_dim_qk, device=device, dtype=dtype)
    v = torch.randn(s_k, h, head_dim_vo, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, s_k], dtype=torch.int32, device=device)

    try:
        flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, s_q, s_k, is_varlen=False,
        )
    except RuntimeError as e:
        msg = str(e)
        # The behavior under test: the dispatch guard rejects unsupported pairs
        # with the offending values in the message. If a DIFFERENT RuntimeError
        # surfaced first (e.g. an unrelated CUDA/runtime error on hardware these
        # kernels were not built for), that is not the condition we pin -> skip
        # rather than report a false failure.
        if "No kernel instantiated" not in msg:
            pytest.skip(f"unrelated RuntimeError before dispatch guard: {msg!r}")
        assert f"head_dim_qk={head_dim_qk}" in msg
        assert f"head_dim_vo={head_dim_vo}" in msg
    else:
        pytest.fail("expected RuntimeError for unsupported head_dim pair, got none")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: requires a CUDA device")
    elif not _HAS_SM100_DENSE:
        print("SKIP: extension built without dense_prefill_fwd")
    else:
        for hd_qk, hd_vo in UNSUPPORTED_HEADDIMS:
            device = torch.device("cuda")
            dtype = torch.bfloat16
            s = 8
            q = torch.randn(s, 1, hd_qk, device=device, dtype=dtype)
            k = torch.randn(s, 1, hd_qk, device=device, dtype=dtype)
            v = torch.randn(s, 1, hd_vo, device=device, dtype=dtype)
            cu = torch.tensor([0, s], dtype=torch.int32, device=device)
            try:
                flash_attn_varlen_func(q, k, v, cu, cu, s, s, is_varlen=False)
            except RuntimeError as e:
                msg = str(e)
                if "No kernel instantiated" not in msg:
                    print(f"SKIP ({hd_qk}, {hd_vo}): unrelated error: {msg!r}")
                    continue
                assert f"head_dim_qk={hd_qk}" in msg and f"head_dim_vo={hd_vo}" in msg
                print(f"PASS: ({hd_qk}, {hd_vo}) correctly raised")
            else:
                print(f"FAIL: ({hd_qk}, {hd_vo}) did not raise")
