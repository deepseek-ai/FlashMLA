"""Cross-device permute arguments must be rejected before the kernel launches.

The permute kernels receive raw pointers, so a scale-factor tensor from another
device would be read through a pointer from a foreign address space.

Run: python tests/test_permute_device.py
Requires two CUDA devices.
"""

import torch

from flash_mla import fused_norm_rope_attn_rope_cast as fused


def require_two_devices():
    if torch.cuda.device_count() < 2:
        print("SKIP: needs at least two CUDA devices", flush=True)
        raise SystemExit(0)


def expect_rejected(name, call):
    try:
        call()
    except RuntimeError as error:
        assert "same device" in str(error), f"{name}: unexpected error: {error}"
        print(f"{name}: PASS ({error})", flush=True)
        return
    raise AssertionError(f"{name}: cross-device argument was accepted")


def check_q_b_proj():
    weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty((64, 1), dtype=torch.int32, device="cuda:1")
    expect_rejected(
        "q_b_proj-cross-device",
        lambda: fused.permute_q_b_proj((weight, scales), 1, 64),
    )


def check_wv_proj():
    weight = torch.empty((1, 64, 512), dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty((1, 64, 4), dtype=torch.int32, device="cuda:1")
    expect_rejected(
        "wv_proj-cross-device",
        lambda: fused.permute_wv_proj((weight, scales), 1, 512),
    )


def check_same_device_still_works():
    weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty((64, 1), dtype=torch.int32, device="cuda:0")
    out_weight, out_scales = fused.permute_q_b_proj((weight, scales), 1, 64)
    assert out_weight.shape == weight.shape
    assert out_scales.shape == scales.shape
    assert out_weight.device == weight.device and out_scales.device == weight.device
    print("q_b_proj-same-device: PASS", flush=True)


require_two_devices()
check_q_b_proj()
check_wv_proj()
check_same_device_still_works()
print("all permute device checks passed", flush=True)
