"""Benchmark for the SM80 dense MLA decode kernel.

Compares against a PyTorch eager (BMM-based) reference. The eager path is
slow for long sequences -- iteration counts shrink accordingly. Reports
latency, KV bandwidth, and speedup across a config sweep."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import flash_mla.cuda as cuda


def torch_eager_mla(q, kcache, block_table, seqlens_k, softmax_scale, head_size_v, is_causal=False):
    """PyTorch BMM-based MLA decode. Returns the same shape as the kernel output."""
    b, sq, hq, dk = q.shape
    _, pbs, hk, _ = kcache.shape
    nq_per_hk = hq // hk
    out = torch.zeros(b, sq, hq, head_size_v, dtype=q.dtype, device=q.device)
    for bi in range(b):
        sk = int(seqlens_k[bi].item())
        bt = block_table[bi]
        nb = (sk + pbs - 1) // pbs
        ks = [kcache[bt[bl].item()] for bl in range(nb)]
        kc = torch.cat(ks, dim=0)[:sk]                                # (sk, hk, dk)
        k_full = kc.transpose(0, 1).contiguous()                      # (hk, sk, dk)
        v_full = k_full[:, :, :head_size_v]                           # (hk, sk, dv)
        q_b = q[bi]                                                   # (sq, hq, dk)
        q_rs = (q_b.view(sq, hk, nq_per_hk, dk)
                    .permute(1, 0, 2, 3)
                    .reshape(hk, sq * nq_per_hk, dk))
        scores = torch.bmm(q_rs.float(), k_full.float().transpose(1, 2)) * softmax_scale
        if is_causal:
            for sq_idx in range(sq):
                rb = max(0, sk - (sq - sq_idx - 1))
                if rb < sk:
                    for nq_idx in range(nq_per_hk):
                        scores[:, sq_idx * nq_per_hk + nq_idx, rb:] = float('-inf')
        probs = torch.softmax(scores, dim=-1)
        o = torch.bmm(probs, v_full.float())                          # (hk, q_per_hk, dv)
        o_rs = (o.view(hk, sq, nq_per_hk, head_size_v)
                  .permute(1, 2, 0, 3)
                  .reshape(sq, hq, head_size_v))
        out[bi] = o_rs.to(q.dtype)
    return out


def bench(fn, iters, warmup):
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


def make_inputs(batch, sq, hq, hk, sk, dtype, device):
    head_size_k = 576
    page_block_size = 64
    q = torch.randn(batch, sq, hq, head_size_k, dtype=dtype, device=device) * 0.1
    nb = (sk + page_block_size - 1) // page_block_size
    kcache = torch.randn(nb * batch, page_block_size, hk, head_size_k, dtype=dtype, device=device) * 0.1
    seqlens_k = torch.full((batch,), sk, dtype=torch.int32, device=device)
    block_table = torch.arange(nb * batch, dtype=torch.int32, device=device).view(batch, nb)
    return q, kcache, seqlens_k, block_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--no-torch-baseline', action='store_true',
                        help='skip PyTorch eager reference (much faster sweep)')
    parser.add_argument('--check', action='store_true',
                        help='also run a one-shot correctness check vs eager')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = 'cuda'
    dtype = getattr(torch, args.dtype)
    head_size_k = 576
    head_size_v = 512
    softmax_scale = 1.0 / (head_size_k ** 0.5)

    configs = [
        # (batch, sq, hq, hk, sk)
        (1,  1, 16, 1,    256),
        (1,  1, 16, 1,   1024),
        (1,  1, 16, 1,   4096),
        (1,  1, 16, 1,  16384),
        (1,  1, 16, 1,  65536),
        (4,  1, 16, 1,   1024),
        (4,  1, 16, 1,   4096),
        (16, 1, 16, 1,   1024),
        (16, 1, 16, 1,   4096),
        (64, 1, 16, 1,   1024),
        (64, 1, 16, 1,   4096),
        (1,  1, 64, 1,   4096),
    ]

    print(f'{"config":<32} {"ours(ms)":>9} {"torch(ms)":>10} {"speedup":>8} {"ours BW(GB/s)":>14}')
    print('-' * 75)
    for batch, sq, hq, hk, sk in configs:
        q, kcache, seqlens_k, block_table = make_inputs(batch, sq, hq, hk, sk, dtype, device)

        ours_fn = lambda: cuda.dense_decode_fwd(
            q, kcache, head_size_v, seqlens_k, block_table, softmax_scale, False, None, None
        )
        ours_ms = bench(ours_fn, iters=200, warmup=20)
        kv_bytes = batch * hk * sk * head_size_k * 2
        bw = kv_bytes / (ours_ms * 1e-3) / 1e9

        if args.no_torch_baseline:
            torch_str = 'skip'
            speedup_str = '-'
        else:
            iters = 5 if sk * batch >= 8192 else (20 if sk * batch >= 1024 else 50)
            warmup = 2
            torch_fn = lambda: torch_eager_mla(q, kcache, block_table, seqlens_k, softmax_scale, head_size_v)
            torch_ms = bench(torch_fn, iters=iters, warmup=warmup)
            torch_str = f'{torch_ms:.3f}'
            speedup_str = f'{torch_ms / ours_ms:.1f}x'

            if args.check:
                out, _, _, _ = ours_fn()
                ref = torch_eager_mla(q, kcache, block_table, seqlens_k, softmax_scale, head_size_v)
                diff = (out.float() - ref.float()).abs().max().item()
                tag = 'OK' if diff < 0.02 else f'FAIL diff={diff:.4f}'
                speedup_str = f'{speedup_str} ({tag})'

        cfg = f'b={batch} sq={sq} hq={hq} hk={hk} sk={sk}'
        print(f'{cfg:<32} {ours_ms:>9.3f} {torch_str:>10} {speedup_str:>8} {bw:>14.1f}')


if __name__ == '__main__':
    main()
