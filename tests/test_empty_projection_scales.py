"""An empty block-scale dimension must be rejected instead of dividing by zero.

`permute_q_b_proj` / `permute_wv_proj` derive the scale granularity from the last
dimension of the scale-factor tensors. With a zero there the host side divides by
zero (SIGFPE) before any device or dtype check runs.

Run: python tests/test_empty_projection_scales.py
"""

import torch

from flash_mla import fused_norm_rope_attn_rope_cast as fused


def expect_clean_error(name, call, match):
    try:
        call()
    except RuntimeError as error:
        assert match in str(error), f"{name}: unexpected error: {error}"
        print(f"{name}: PASS ({error})", flush=True)
        return
    raise AssertionError(f"{name}: empty scale dimension was accepted")


def check_q_b_proj():
    weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((64, 0), dtype=torch.int32, device="cuda")
    expect_clean_error(
        "q_b_proj-empty-scale-dim",
        lambda: fused.permute_q_b_proj((weight, scales), 1, 64),
        "q_b_proj scale_factors must have a nonzero second dimension",
    )


def check_wv_proj():
    weight = torch.empty((1, 64, 512), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((1, 64, 0), dtype=torch.int32, device="cuda")
    expect_clean_error(
        "wv_proj-empty-scale-dim",
        lambda: fused.permute_wv_proj((weight, scales), 1, 512),
        "wv_proj scale_factors must have a nonzero third dimension",
    )


def check_valid_scales_still_work():
    weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((64, 1), dtype=torch.int32, device="cuda")
    out_weight, out_scales = fused.permute_q_b_proj((weight, scales), 1, 64)
    assert out_weight.shape == weight.shape
    assert out_scales.shape == scales.shape
    print("q_b_proj-valid-shapes: PASS", flush=True)


if not torch.cuda.is_available():
    print("SKIP: CUDA is required", flush=True)
    raise SystemExit(0)

check_q_b_proj()
check_wv_proj()
check_valid_scales_still_work()
print("all empty-scale checks passed", flush=True)
