import os, sys
import torch
import math
import random
import triton
import itertools
from functools import partial
from typing import Callable
sys.path.append(f"{os.path.dirname(__file__)}/..")
import flash_mla
print(f"{flash_mla.__path__=}")

# import importlib
# import importlib.util
# spec1 = importlib.util.spec_from_file_location("flash_mla", "FlashMLA_ref/flash_mla/__init__.py")
# flash_mla_ref = importlib.util.module_from_spec(spec1)
# spec1.loader.exec_module(flash_mla_ref)
# print(f"{flash_mla_ref.__path__=}")

def scaled_dot_product_attention(
    query, key, value, h_q, h_kv,
    is_causal=False, window_left=-1
):
    """
    query: [h_q, s_q, d]
    key: [h_kv, s_k, d]
    value: [h_kv, s_k, d_v]
    """
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    s_q = query.shape[-2]
    s_k = key.shape[-2]
    attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
    if is_causal:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if window_left >= 0:
        if s_k >= window_left:
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool)\
                .tril(diagonal=s_k - s_q - window_left)
            attn_bias.masked_fill_(temp_mask, float("-inf"))
    attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    # 1 - 2*x \cdot y / x^2+y^2
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    get_cos_diff = lambda a, b: 1 - 2 * (a * b).sum().item() / max((a * a + b * b).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if not cos_diff < 1e-5: # maybe nan
        for i in range(x.shape[0]):
            print(f"{i=}")
            print(get_cos_diff(x[i], y[i]))
            print(x[i].flatten()[:10])
            print(y[i].flatten()[:10])
    assert cos_diff < 1e-5, f"{cos_diff=} {x.flatten()[0].item()}, {y.flatten()[0].item()}"

@torch.inference_mode()
def test_flash_mla_window(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen, window_left):
    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=} {window_left=}"
    )
    
    def gen_data(window_left=-1):
        cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
        if varlen:
            for i in range(b):
                seq_len = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
                cache_seqlens[i] = seq_len
        # cache_seqlens = torch.tensor([17], dtype=torch.int32)

        print(f"{cache_seqlens=}")
        total_seqlens = cache_seqlens.sum().item()
        mean_seqlens = cache_seqlens.float().mean().int().item()
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

        q = torch.randn(b, s_q, h_q, d)
        block_size = 64
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32
        ).view(b, max_seqlen_pad // block_size)
        # [b*max_seq_len//block_size, block_size, h_kv, d]
        blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
        for i in range(b):
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
                float("nan")
            )
        blocked_v = blocked_k[..., :dv]

        tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
            cache_seqlens, s_q * h_q // h_kv, h_kv,
            window_left=window_left, causal=causal, s_q = s_q
        )
        # print(tile_scheduler_metadata[..., :5])
        # print(f"{num_splits=}")
        return q, blocked_k, blocked_v, block_table, max_seqlen_pad, total_seqlens, cache_seqlens,\
            tile_scheduler_metadata, num_splits

    def run_flash_mla_window(window_left=-1):
        return flash_mla.flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            # win_tile_scheduler_metadata,
            num_splits,
            # win_num_splits,
            causal=causal,
            window_left=window_left
        )

    def ref_mla_window(window_left=-1):
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
                window_left=window_left,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    (
        q, blocked_k, blocked_v, block_table, max_seqlen_pad, total_seqlens,
        cache_seqlens, tile_scheduler_metadata, num_splits
    ) = gen_data(window_left=window_left)
    # print(tile_scheduler_metadata[..., :5])

    out_flash_win, lse_flash_win = run_flash_mla_window(window_left=window_left)
    out_torch, lse_torch = ref_mla_window(window_left=window_left)
    cal_diff(out_flash_win, out_torch, "out")
    cal_diff(lse_flash_win, lse_torch, "lse")

    def run_bench(func: Callable, name: str):
        t = triton.testing.do_bench(func)
        FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
        bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (
            torch.finfo(q.dtype).bits // 8
        )
        print(
            f"{name}: {t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
        )
    # run_bench(partial(run_flash_mla_window, window_left), "run_flash_mla_window")

def main():
    device = torch.device("cuda:0")
    torch_dtype = torch.bfloat16
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    
    # torch.set_printoptions(threshold=float('inf'))
    # torch.set_printoptions(sci_mode=False)
    # torch.set_printoptions(linewidth=99999)
    h_kv = 1
    d, dv = 576, 512
    causal = True
    varlen = True

    # qlen larger than 4, the cos diff may be greater than 1e-5.
    for b, s, h_q, s_q, window_left in itertools.product(
        [64, 128],
        [4096, 8192, 16384],
        [16, 32, 64, 128],
        [1, 2, 4],
        [1, 7, 64, 128, 512]
    ):
        test_flash_mla_window(b, s_q, s, h_q, h_kv, d, dv, causal, varlen, window_left)
        print("===" * 32)


if __name__ == "__main__":
    main()

# sudo ncu -o mla -f --import-source yes --launch-skip 11 python3 tests/test_window.py
