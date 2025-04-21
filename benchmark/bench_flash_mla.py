# MLA Triton kernel is from: https://github.com/monellz/vllm/commit/feebaa7c063be6bfb590a876741aeef1c5f58cf8#diff-7b2e1c9032522f7266051b9887246a65753871dfb3625a258fee40109fe6e87a
import argparse
import math
import random

import flashinfer
import torch
import triton
import triton.language as tl

# pip install flashinfer-python
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


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


@torch.inference_mode()
def run_torch_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

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
                h_q, h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_torch, lse_torch = ref_mla()
    t = triton.testing.do_bench(ref_mla)
    return out_torch, lse_torch, t

@torch.inference_mode()
def run_flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
        )

    out_flash, lse_flash = flash_mla()
    t = triton.testing.do_bench(flash_mla)
    return out_flash, lse_flash, t


@torch.inference_mode()
def run_flash_infer(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()
    
    
    kv_indptr = [0]
    kv_indices = []
    for i in range(b):
        seq_len = cache_seqlens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_table[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
    for seq_len in cache_seqlens[1:]:
        kv_indptr.append((seq_len + block_size - 1) // block_size + kv_indptr[-1])
        
    q_indptr = torch.arange(0, b + 1).int() * s_q
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)

    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8),
        backend="fa3"
    )
    mla_wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        cache_seqlens,
        h_q,
        dv,
        d-dv,
        block_size,
        causal,
        1 / math.sqrt(d),
        q.dtype,
        blocked_k.dtype,
    )

    def flash_infer():
        output, lse = mla_wrapper.run(q_nope.view(-1, h_q, dv), q_pe.view(-1, h_q, d-dv), blocked_k_nope, blocked_k_pe, return_lse=True)
        return output.view(b, -1, h_q, dv), lse.view(b, h_q, 1)

    out_flash, lse_flash = flash_infer()
    t = triton.testing.do_bench(flash_infer)
    return out_flash, lse_flash, t


@triton.jit
def _mla_attn_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    cur_batch = tl.program_id(1)
    cur_head_id = tl.program_id(0)
    split_kv_id = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = cur_batch * stride_q_pe_bs + cur_head[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(
            Req_to_tokens + stride_req_to_tokens_bs * cur_batch + offs_n // PAGE_SIZE,
            mask=offs_n < split_kv_end,
            other=0,
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv[:, None]
        k_c = tl.load(Kv_c_cache + offs_k_c, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

        offs_k_pe = kv_loc[None, :] * stride_k_pe_bs + offs_d_kpe[:, None]
        k_pe = tl.load(K_pe_cache + offs_k_pe, mask=offs_n[None, :] < split_kv_end, other=0.0)

        qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
        qk *= sm_scale

        qk = tl.where(offs_n[None, :] < split_kv_end, qk, float("-inf"))

        v_c = tl.trans(k_c)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v_c.dtype), v_c)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    offs_o = cur_batch * stride_o_b + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_s + offs_d_ckv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None])
    offs_o_1 = cur_batch * stride_o_b + cur_head * stride_o_h + split_kv_id * stride_o_s + HEAD_DIM_CKV
    tl.store(O + offs_o_1, e_max + tl.log(e_sum))


def _mla_attn(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    attn_logits,
    req_to_tokens,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    page_size,
):
    batch_size, head_num = q_nope.shape[0], q_nope.shape[1]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]

    BLOCK_H = 16
    BLOCK_N = 64
    grid = (
        triton.cdiv(head_num, BLOCK_H),
        batch_size,
        num_kv_splits,
    )
    _mla_attn_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        req_to_tokens,
        b_seq_len,
        attn_logits,
        sm_scale,
        # stride
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c_cache.stride(-2),
        k_pe_cache.stride(-2),
        req_to_tokens.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N, 
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
    )

@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    B_seq_len,
    O,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_o_b,
    stride_o_h,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_l_1 = cur_batch * stride_l_b + cur_head * stride_l_h + HEAD_DIM_CKV

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            logits = tl.load(Logits + offs_l + split_kv_id * stride_l_s)
            logits_1 = tl.load(Logits + offs_l_1 + split_kv_id * stride_l_s)

            n_e_max = tl.maximum(logits_1, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(logits_1 - n_e_max)
            acc += exp_logic * logits

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max
    
    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )


def _mla_softmax_reducev(
    logits,
    o,
    b_seq_len,
    num_kv_splits,
):
    batch_size, head_num, head_dim_ckv = o.shape[0], o.shape[1], o.shape[2]
    grid = (batch_size, head_num)
    _mla_softmax_reducev_kernel[grid](
        logits,
        b_seq_len,
        o,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        num_warps=4,
        num_stages=2,
    )

def mla_decode_triton(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    o,
    req_to_tokens,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
):
    assert num_kv_splits == attn_logits.shape[2]
    _mla_attn(
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        attn_logits,
        req_to_tokens,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
    )
    _mla_softmax_reducev(
        attn_logits,
        o,
        b_seq_len,
        num_kv_splits,
    )
    

@torch.inference_mode()
def run_flash_mla_triton(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]
    
    assert d > dv, "mla with rope dim should be larger than no rope dim"
    q_nope, q_pe = q[..., :dv].contiguous(), q[..., dv:].contiguous()
    blocked_k_nope, blocked_k_pe = blocked_k[..., :dv].contiguous(), blocked_k[..., dv:].contiguous()

    def flash_mla_triton():
        num_kv_splits = 32
        o = torch.empty([b * s_q, h_q, dv])
        attn_logits = torch.empty([b * s_q, h_q, num_kv_splits, dv + 1])
        mla_decode_triton(q_nope.view(-1, h_q, dv), q_pe.view(-1, h_q, d-dv), blocked_k_nope.view(-1, dv), blocked_k_pe.view(-1, d-dv), o, block_table, cache_seqlens, attn_logits, num_kv_splits, 1 / math.sqrt(d), block_size)
        return o.view([b, s_q, h_q, dv])

    out_flash = flash_mla_triton()
    t = triton.testing.do_bench(flash_mla_triton)
    return out_flash, None, t


FUNC_TABLE = {
    "torch": run_torch_mla,
    "flash_mla": run_flash_mla,
    "flash_infer": run_flash_infer,
    "flash_mla_triton": run_flash_mla_triton,
}
    
def compare_ab(baseline, target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(f"comparing {baseline} vs {target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert baseline in FUNC_TABLE
    assert target in FUNC_TABLE
    baseline_func = FUNC_TABLE[baseline]
    target_func = FUNC_TABLE[target]
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    
    out_a, lse_a, perf_a = baseline_func(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)
    out_b, lse_b, perf_b = target_func(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)
    
    torch.testing.assert_close(out_b.float(), out_a.float(), atol=1e-2, rtol=1e-2), "out"
    if target not in ["flash_infer", "flash_mla_triton"] and baseline not in ["flash_infer", "flash_mla_triton"]:
        # flash_infer has a different lse return value
        # flash_mla_triton doesn't return lse
        torch.testing.assert_close(lse_b.float(), lse_a.float(), atol=1e-2, rtol=1e-2), "lse"

    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"perf {baseline}: {perf_a:.3f} ms, {FLOPS / 10 ** 9 / perf_a:.0f} TFLOPS, {bytes / 10 ** 6 / perf_a:.0f} GB/s")
    print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10 ** 9 / perf_b:.0f} TFLOPS, {bytes / 10 ** 6 / perf_b:.0f} GB/s")
    return bytes / 10 ** 6 / perf_a, bytes / 10 ** 6 / perf_b


def compare_a(target, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype):
    print(f"{target}: {b=}, {s_q=}, mean_seqlens={cache_seqlens.float().mean()}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {dtype=}")
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    assert target in FUNC_TABLE
    target_func = FUNC_TABLE[target]
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    
    out_b, lse_b, perf_b = target_func(q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal, dtype)

    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"perf {target}: {perf_b:.3f} ms, {FLOPS / 10 ** 9 / perf_b:.0f} TFLOPS, {bytes / 10 ** 6 / perf_b:.0f} GB/s")
    return bytes / 10 ** 6 / perf_b


available_targets = [
    "torch",
    "flash_mla",
    "flash_infer",
    "flash_mla_triton",
]

shape_configs = [
    {"b": batch, "s_q": 1, "cache_seqlens": torch.tensor([seqlen + 2 * i for i in range(batch)], dtype=torch.int32, device="cuda"), "h_q": head, "h_kv": 1, "d": 512+64, "dv": 512, "causal": True, "dtype": torch.bfloat16}
    for batch in [128] for seqlen in [1024, 2048, 4096, 8192, 8192*2, 8192*4] for head in [128]
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="torch")
    parser.add_argument("--target", type=str, default="flash_mla")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = get_args()
    benchmark_type = "all" if args.all else f"{args.baseline}_vs_{args.target}" if args.compare else args.target
    with open(f"{benchmark_type}_perf.csv", "w") as fout:
        fout.write("name,batch,seqlen,head,bw\n")
        for shape in shape_configs:
            if args.all:
                for target in available_targets:
                    perf = compare_a(target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                    fout.write(f'{target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perf:.0f}\n')
            elif args.compare:
                perfa, prefb = compare_ab(args.baseline, args.target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                fout.write(f'{args.baseline},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perfa:.0f}\n')
                fout.write(f'{args.target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{prefb:.0f}\n')
            elif args.one:
                perf = compare_a(args.target, shape["b"], shape["s_q"], shape["cache_seqlens"], shape["h_q"], shape["h_kv"], shape["d"], shape["dv"], shape["causal"], shape["dtype"])
                fout.write(f'{args.target},{shape["b"]},{shape["cache_seqlens"].float().mean().cpu().item():.0f},{shape["h_q"]},{perf:.0f}\n')