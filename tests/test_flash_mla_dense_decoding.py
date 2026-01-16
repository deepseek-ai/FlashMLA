import argparse
import math
import random
import dataclasses
from typing import Tuple

import torch

import kernelkit as kk
import flash_mla

@dataclasses.dataclass
class TestParam:
    b: int    # Batch size
    s_q: int  # Number of queries for one request
    s_k: int  # Seq len, or mean seq len if varlen == True
    is_varlen: bool
    is_causal: bool
    test_performance: bool = True
    have_zero_seqlen_k: bool = False
    block_size: int = 64
    h_q: int = 128    # Number of q heads
    h_kv: int = 1     # Number of kv heads
    d: int = 576      # Q/K head dim (= dv + RoPE dim)
    dv: int = 512     # V head dim
    seed: int = 0


def generate_test_data(t: TestParam) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test data from a given configuration
    Return: [cache_seqlens, q, block_table, blocked_k]
    Pay attention: This function changes the random seed
    """
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    torch.backends.cudnn.deterministic = True

    assert t.h_q % t.h_kv == 0

    cache_seqlens_cpu = torch.full((t.b,), t.s_k, dtype=torch.int32, device='cpu')
    if t.is_varlen:
        for i in range(t.b):
            cache_seqlens_cpu[i] = max(random.normalvariate(t.s_k, t.s_k / 2), t.s_q)

    if t.have_zero_seqlen_k:
        zeros_mask = torch.randn(t.b, dtype=torch.float32, device='cpu') > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = int(cache_seqlens_cpu.max().item())
    max_seqlen_pad = kk.cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(t.b, t.s_q, t.h_q, t.d) / 10
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(t.b * max_seqlen_pad // t.block_size, dtype=torch.int32).view(t.b, max_seqlen_pad // t.block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(t.b, -1)
    blocked_k = torch.randn(block_table.numel(), t.block_size, t.h_kv, t.d) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    for i in range(t.b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = kk.cdiv(cur_len, t.block_size)
        blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
        if cur_len % t.block_size != 0:
            blocked_k[block_table[i][cur_num_blocks - 1]][cur_len % t.block_size:] = float("nan")
        block_table[i][cur_num_blocks:] = 2147480000
    return cache_seqlens, q, block_table, blocked_k


def reference_torch(
    cache_seqlens: torch.Tensor,    # [batch_size]
    block_table: torch.Tensor,      # [batch_size, ?]
    q: torch.Tensor,    # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,    # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,    # [h_q, s_q, d]
        kv: torch.Tensor,      # [h_kv, s_k, d]
        dv: int,
        is_causal,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float()
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if is_causal and query.size(1) > 1:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                mask = mask.tril(diagonal=s_k - s_q)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]    # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = (lse == float("-inf"))
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output, lse

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = kk.cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0: cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out, cur_lse = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            is_causal
        )
        out_ref[i] = cur_out.transpose(0, 1)
        lse_ref[i] = cur_lse
    out_ref = out_ref.to(q.dtype)
    return out_ref, lse_ref


@torch.inference_mode()
def test_flash_mla(t: TestParam):
    print('-------------------------------')
    print(f"Running on {t}...")

    # Generating test data
    torch.cuda.synchronize()
    cache_seqlens, q, block_table, blocked_k, = generate_test_data(t)

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

    def run_flash_mla():
        return flash_mla.flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            t.dv,
            tile_scheduler_metadata,
            num_splits,
            causal=t.is_causal
        )

    out_ans, lse_ans = run_flash_mla()
    out_ref, lse_ref = reference_torch(cache_seqlens, block_table, q, blocked_k, t.dv, t.is_causal)
    is_correct = True
    is_correct &= kk.check_is_allclose("out", out_ans, out_ref, abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
    is_correct &= kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536)
    assert is_correct

    if t.test_performance:
        time_usage = kk.bench_kineto(run_flash_mla, 10).get_kernel_time("flash_fwd_splitkv_mla_kernel")

        mean_attended_seqlens = cache_seqlens.float().mean().item()
        compute_volume_flop = t.b * t.h_q * t.s_q * sum([
            2 * t.d * mean_attended_seqlens,   # Q * K^T
            2 * mean_attended_seqlens * t.dv,  # attention * V
        ])
        q_elem_size = torch.bfloat16.itemsize
        kv_token_size = t.d * torch.bfloat16.itemsize
        memory_volume_B = t.b * sum([
            t.s_q * t.h_q * (t.d * q_elem_size),    # Q
            mean_attended_seqlens * t.h_kv * kv_token_size,    # K/V
            t.s_q * t.h_q * (t.dv * q_elem_size),   # Output
        ])
        achieved_tflops = compute_volume_flop / time_usage / 1e12
        achieved_gBps = memory_volume_B / time_usage / 1e9

        print(f"{time_usage * 1000:.3f} ms, {achieved_tflops:.0f} TFLOPS, {achieved_gBps:.0f} GB/s")


def main(torch_dtype):
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cc_major, cc_minor = torch.cuda.get_device_capability()
    assert cc_major == 9, "Dense MLA decoding is only supported on sm90 (Hopper) currently."

    correctness_cases = [
        TestParam(b, s_q, s_k, is_varlen, is_causal, test_performance=False, have_zero_seqlen_k=False, block_size=64, h_q=h_q, h_kv=h_kv)
        for b in [1, 2, 6, 64]
        for s_q in [1, 2, 4]
        for s_k in [20, 140, 4096]
        for h_q in [1, 3, 9, 63, 64, 126, 128]
        for h_kv in [1, 2, 3, 8]
        for is_varlen in [False, True]
        for is_causal in [False, True]
        if h_q % h_kv == 0
    ]

    corner_cases = [
        # Cases where some kv cache have zero length
        TestParam(128, 2, 4096, is_varlen=True, is_causal=is_causal, test_performance=False, have_zero_seqlen_k=True, h_q=h_q, h_kv=h_kv)
        for h_q in [1, 3, 9, 63, 64, 126, 128]
        for h_kv in [1, 2, 3, 8]
        for is_causal in [False, True]
        if h_q % h_kv == 0
    ]

    performance_cases = [
        TestParam(128, s_q, s_k, is_varlen=True, is_causal=is_causal, test_performance=True)
        for is_causal in [False, True]
        for s_q in [1, 2]
        for s_k in [4096, 8192, 16384, 32768]
    ]

    testcases = correctness_cases + corner_cases + performance_cases

    for testcase in testcases:
        test_flash_mla(testcase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype)