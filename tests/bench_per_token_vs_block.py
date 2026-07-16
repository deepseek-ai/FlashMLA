"""Kernel microbench: per-token (scattered) vs block-aligned indices on the
EXISTING sparse MLA fwd+bwd (same kernel). Both gather `topk` keys via gather4;
the only difference is memory locality (block-aligned = contiguous runs). The
big speedup (contiguous block TMA, no scatter) needs the KV-outer Phase-2 kernel
-- this bench establishes the per-token baseline + the locality delta today.
Sparse-MLA shape h_q=64, d_qk=576, d_v=512.
"""
import torch
import triton
from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_bwd

DEV = "cuda"
H, DQK, DV = 64, 576, 512
LOG2E = 1.4426950408889634


def make_indices(s_q, s_kv, topk, mode, kv_block_size=128, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.full((s_q, topk), -1, dtype=torch.int32)
    for qi in range(s_q):
        hi = qi + 1  # causal: keys 0..qi
        if hi <= 0:
            continue
        if mode == "per_token":
            n = min(topk, hi)
            keys = torch.randperm(hi, generator=g)[:n].sort().values
        else:  # block-aligned: pick contiguous blocks <= qi
            nblk = (hi + kv_block_size - 1) // kv_block_size
            want_blocks = min(topk // kv_block_size, nblk)
            blocks = torch.randperm(nblk, generator=g)[:want_blocks].sort().values
            keys = []
            for b in blocks.tolist():
                keys += list(range(b * kv_block_size, min((b + 1) * kv_block_size, hi)))
            keys = torch.tensor(sorted(keys)[:topk], dtype=torch.int32)
        idx[qi, :len(keys)] = torch.as_tensor(keys, dtype=torch.int32)
    return idx.to(DEV)[:, None, :]


def bench(s_q, s_kv, topk, mode):
    sm = DQK ** -0.5
    q = (torch.randn(s_q, H, DQK, device=DEV, dtype=torch.bfloat16) / 8)
    kv = (torch.randn(s_kv, 1, DQK, device=DEV, dtype=torch.bfloat16) / 8)
    indices = make_indices(s_q, s_kv, topk, mode)
    grad_o = (torch.randn(s_q, H, DV, device=DEV, dtype=torch.bfloat16) / 8)
    dq_acc = torch.zeros(s_q * H * DQK, dtype=torch.float32, device=DEV)
    dkv_acc = torch.zeros(s_kv * DQK, dtype=torch.float32, device=DEV)

    def fwd():
        return flash_mla_sparse_fwd(q, kv, indices, sm, DV)

    o, _ml, lse = fwd()
    lse2 = lse * LOG2E

    def bwd():
        dq_acc.zero_(); dkv_acc.zero_()
        flash_mla_sparse_bwd(q, kv, o, grad_o, indices, lse2, dq_acc, dkv_acc, sm, DV, False)

    tf = triton.testing.do_bench(lambda: fwd(), warmup=3, rep=5)
    tb = triton.testing.do_bench(bwd, warmup=3, rep=5)
    return tf, tb


if __name__ == "__main__":
    print(f"# sparse-MLA h={H} d_qk={DQK} d_v={DV}; fwd/bwd ms; per-token vs block-aligned (same kernel)")
    for (s, topk) in [(2048, 1024), (4096, 2048), (8192, 2048)]:
        row = f"s={s:>5} topk={topk:>4} |"
        for mode in ["per_token", "block"]:
            tf, tb = bench(s, s, topk, mode)
            row += f"  {mode:>9}: fwd {tf:6.3f} bwd {tb:6.3f} |"
        print(row, flush=True)
    print("PT_VS_BLOCK_BENCH_DONE", flush=True)
