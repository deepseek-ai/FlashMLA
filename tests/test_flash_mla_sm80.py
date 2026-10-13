import math
import random

import pytest
import torch
import triton

from flash_mla import get_mla_metadata, flash_mla_with_kvcache


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    # # print nan coordinates of x
    # print("nan x", torch.isnan(x).nonzero(as_tuple=False))
    # print("nan y", torch.isnan(y).nonzero(as_tuple=False))
    assert cos_diff < 8e-5


@torch.inference_mode()
def run_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, impl="legacy", bench=True):
    """The fork's original test body (page size 32, bf16, softmax scale 576**-0.5, NaN-padded KV, cos_diff < 8e-5).
    `impl` selects the kernel: "legacy" (the fork's sm80 kernel, BlockM=16 fallback on sm_86), "legacy_ws" or "v2" (sm86 kernels)."""
    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}, {impl=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 32
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
            impl=impl,
        )

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    cal_diff(out_flash, out_torch, "out")
    cal_diff(lse_flash, lse_torch, "lse")

    if bench:
        t = triton.testing.do_bench(flash_mla)
        FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
        bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(q.dtype).bits // 8)
        print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s")


@pytest.fixture
def bf16_cuda_defaults():
    """The test body builds its tensors with the default dtype / device (bf16, cuda:0 of CUDA_VISIBLE_DEVICES)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(torch.device("cuda:0"))
    torch.manual_seed(0)
    random.seed(0)
    yield
    torch.set_default_device(None)
    torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("impl", ["legacy", "v2"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("s_q", [1, 2])
@pytest.mark.parametrize("h_q", [16, 32, 64, 128])  # TP = 8, 4, 2, 1
@pytest.mark.parametrize("mean_sk", [4096, 8192])
def test_flash_mla(bf16_cuda_defaults, mean_sk, h_q, s_q, varlen, impl):
    """b=128 (about 3-6 GB of KV per case): needs a mostly free GPU."""
    run_flash_mla(128, s_q, mean_sk, h_q, 1, 576, 512, True, varlen, impl=impl, bench=False)


if __name__ == "__main__":
    import os

    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    d, dv = 576, 512
    causal = True

    # FLASH_MLA_TEST_IMPL=legacy (default, the fork's kernel) | v2 | legacy_ws
    impl = os.environ.get("FLASH_MLA_TEST_IMPL", "legacy")
    # run_flash_mla(1, 1, 4096, 16, 1, d, dv, True, True, impl=impl)
    for b in [128]:
        for s in [4096, 8192]:
            for h_q in [16, 32, 64, 128]:  # TP = 8, 4, 2, 1
                for s_q in [1, 2]:  # MTP = 1, 2
                    for varlen in [False, True]:
                        run_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, impl=impl)
