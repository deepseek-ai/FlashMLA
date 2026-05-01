"""Profile dense_decode_fwd's share of an MLA decode step.

Uses a DeepSeek-V3-shaped 1-layer attention block:
  x (b, H) -> Q proj -> q (b, hq, dk)
  q + KV cache -> dense_decode_fwd -> o (b, hq, dv)
  o -> O proj -> y (b, H)

This is an under-estimate of full decode step time (no FFN / MoE / layernorm),
which means dense_decode's measured share here is an UPPER bound on its share
of a full step. If decode is < 30% even here, BLOCK_M=8 redesign (which gives
+3-5pp on decode itself) won't move the full-step needle meaningfully."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import flash_mla.cuda as cuda


def time_fn(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def profile(batch, seqlen, hidden, num_q_heads, num_kv_heads, head_dim_k, head_dim_v, dtype):
    device = 'cuda'
    pbs = 64
    softmax_scale = 1.0 / (head_dim_k ** 0.5)

    # Linear projections (Q absorbed: hidden -> num_q_heads*head_dim_k).
    x   = torch.randn(batch, hidden, dtype=dtype, device=device) * 0.1
    W_q = torch.randn(hidden, num_q_heads * head_dim_k, dtype=dtype, device=device) * 0.01
    W_o = torch.randn(num_q_heads * head_dim_v, hidden, dtype=dtype, device=device) * 0.01

    # KV cache (paged, num_kv_heads heads).
    nb = (seqlen + pbs - 1) // pbs
    total_blocks = nb * batch
    kcache = torch.randn(total_blocks, pbs, num_kv_heads, head_dim_k, dtype=dtype, device=device) * 0.1
    seqlens_k = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(batch, nb)

    def attn_block():
        q = (x @ W_q).view(batch, 1, num_q_heads, head_dim_k)
        out, _, _, _ = cuda.dense_decode_fwd(
            q, kcache, head_dim_v, seqlens_k, block_table,
            softmax_scale, False, None, None,
        )
        out_flat = out.contiguous().view(batch, num_q_heads * head_dim_v)
        return out_flat @ W_o

    def attn_no_oproj():
        q = (x @ W_q).view(batch, 1, num_q_heads, head_dim_k)
        out, _, _, _ = cuda.dense_decode_fwd(
            q, kcache, head_dim_v, seqlens_k, block_table,
            softmax_scale, False, None, None,
        )
        return out

    def decode_only():
        q = torch.randn(batch, 1, num_q_heads, head_dim_k, dtype=dtype, device=device) * 0.1
        out, _, _, _ = cuda.dense_decode_fwd(
            q, kcache, head_dim_v, seqlens_k, block_table,
            softmax_scale, False, None, None,
        )
        return out

    full   = time_fn(attn_block)
    no_op  = time_fn(attn_no_oproj)
    only   = time_fn(decode_only)
    qproj  = no_op - only
    oproj  = full - no_op
    share  = only / full * 100.0
    return full, qproj, only, oproj, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    # DeepSeek-V3 architectural constants (MLA absorbed mode).
    HIDDEN       = 7168
    NUM_Q_HEADS  = 128
    NUM_KV_HEADS = 1
    HEAD_DIM_K   = 576
    HEAD_DIM_V   = 512

    print(f'DeepSeek-V3-shaped 1-layer attention block, dtype={args.dtype}')
    print(f'  hidden={HIDDEN} hq={NUM_Q_HEADS} hk={NUM_KV_HEADS} dk={HEAD_DIM_K} dv={HEAD_DIM_V}')
    print()
    print(f'{"config":<22} {"attn(ms)":>10} {"qproj":>8} {"decode":>8} {"oproj":>8} {"decode%":>9}')
    print('-' * 70)

    configs = [
        # (batch, seqlen)
        (1,    1024), (1,    4096), (1,   16384), (1,   65536),
        (4,    1024), (4,    4096), (4,   16384),
        (16,   1024), (16,   4096), (16,  16384),
        (64,   1024), (64,   4096), (64,  16384),
        (128,  4096),
    ]
    for b, sk in configs:
        try:
            full, qproj, only, oproj, share = profile(b, sk, HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM_K, HEAD_DIM_V, dtype)
            tag = f'b={b} sk={sk}'
            print(f'{tag:<22} {full:>10.3f} {qproj:>8.3f} {only:>8.3f} {oproj:>8.3f} {share:>8.1f}%')
        except torch.cuda.OutOfMemoryError:
            print(f'b={b} sk={sk}: OOM (skipped)')

    print()
    print('Note: this 1-layer attention block excludes FFN/MoE/layernorm/residual,')
    print('which together typically dominate full step time. The "decode%" above')
    print('is therefore an UPPER bound on dense_decode share of a real decode step.')


if __name__ == '__main__':
    main()
