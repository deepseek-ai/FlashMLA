"""Tensors from another device must be rejected before the fused kernel launches.

Run: python tests/test_fused_devices.py
Requires two CUDA devices and an SM100/SM103 GPU.
"""

import torch

from flash_mla import fused_norm_rope_attn_rope_cast as fused


def require_two_devices():
    if torch.cuda.device_count() < 2:
        print("SKIP: needs at least two CUDA devices", flush=True)
        raise SystemExit(0)


def make(dev):
    q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device=dev)
    kv = torch.zeros((64, 1, 512), dtype=torch.bfloat16, device=dev)
    indices = torch.zeros((1, 1, 64), dtype=torch.int32, device=dev)
    token_positions = torch.zeros((1,), dtype=torch.int32, device=dev)
    cos_sin_cache = torch.zeros((1, 64), dtype=torch.float32, device=dev)
    return q, kv, indices, token_positions, cos_sin_cache


def call_prefill(q, kv, indices, token_positions, cos_sin_cache):
    return fused.prefill(
        enable_q_norm=False, rms_norm_eps=1e-6,
        token_positions=token_positions, is_rope_neox_style=False, rope_dim=64,
        cos_sin_cache=cos_sin_cache,
        n_wv_group=8, num_per_channels=32,  # h_q=64 with wv_group_size=8
        use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True,
        q=q, kv=kv, indices=indices, sm_scale=0.044,
    )


def expect_rejected(name, call):
    try:
        call()
    except RuntimeError as error:
        assert "same device" in str(error), f"{name}: unexpected error: {error}"
        print(f"{name}: PASS ({error})", flush=True)
        return
    raise AssertionError(f"{name}: cross-device tensor was accepted")


require_two_devices()

# control: everything on one device still runs
q, kv, indices, pos, cache = make("cuda:0")
call_prefill(q, kv, indices, pos, cache)
print("control: PASS", flush=True)

# kv from the other device
q, kv, indices, pos, cache = make("cuda:0")
_, kv2, _, _, _ = make("cuda:1")
expect_rejected("kv-cross-device", lambda: call_prefill(q, kv2, indices, pos, cache))

# positions from the other device
_, _, _, pos2, _ = make("cuda:1")
expect_rejected("positions-cross-device", lambda: call_prefill(q, kv, indices, pos2, cache))

print("all fused-device checks passed", flush=True)
