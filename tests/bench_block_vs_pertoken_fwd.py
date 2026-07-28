"""FWD speed: block-sparse (16 blocks x 128 = 2048 keys, block-aligned) vs
DSA per-token (2048 scattered top-k keys). Both select topk=2048 keys/query so
the COMPUTE is identical; the only difference is the gather pattern (contiguous
blocks vs scattered) -> a memory-locality test on the existing sparse fwd kernel
(flash_mla_sparse_fwd, gather4). Sweep seqlen 4K -> 256K (x2). Sparse-MLA
h_q=64, d_qk=576, d_v=512. Run on B200.
"""
import torch
import triton
from flash_mla import flash_mla_sparse_fwd

DEV = "cuda"
H, DQK, DV = 64, 576, 512
TOPK, KBS = 2048, 128
NSEL = TOPK // KBS  # 16 blocks


def gen_indices(s, mode, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qi = torch.arange(s, device=DEV, dtype=torch.int32).view(s, 1)
    if mode == "per_token":                          # int32-lean (avoid huge long temporaries)
        r = torch.rand(s, TOPK, generator=g, device=DEV)
        idx = (r * (qi + 1).float()).to(torch.int32)
        del r
        return torch.minimum(idx, qi)                # causal: keys <= qi
    else:  # block: NSEL random causal blocks, expanded to KBS contiguous keys each
        nblk = (qi // KBS) + 1
        r = torch.rand(s, NSEL, generator=g, device=DEV)
        blocks = (r * nblk.float()).to(torch.int32).clamp(max=(s // KBS))
        del r
        offs = torch.arange(KBS, device=DEV, dtype=torch.int32).view(1, 1, KBS)
        keys = (blocks.unsqueeze(-1) * KBS + offs).reshape(s, TOPK)
        del blocks
        return torch.minimum(keys, qi)               # causal clamp


@torch.no_grad()
def bench(s, mode, h):
    scale = DQK ** -0.5
    q = torch.randn(s, h, DQK, device=DEV, dtype=torch.bfloat16) / 8
    kv = torch.randn(s, 1, DQK, device=DEV, dtype=torch.bfloat16) / 8
    idx = gen_indices(s, mode)[:, None, :]
    reps = 10 if s <= 262144 else 4
    t = triton.testing.do_bench(lambda: flash_mla_sparse_fwd(q, kv, idx, scale, DV), warmup=2, rep=reps)
    del q, kv, idx
    torch.cuda.empty_cache()
    return t


if __name__ == "__main__":
    print(f"# sparse fwd d_qk={DQK} d_v={DV} topk={TOPK}; block(16x128) vs per-token(2048)")
    print(f"# h reduced at large seqlen to fit memory (ratio is h-independent)")
    print(f"# {'seqlen':>8} {'h':>3} {'block ms':>9} {'pertoken ms':>12} {'block/pt':>9}")
    for s in [262144, 524288, 1048576]:   # extend the prior 4K-256K sweep up to 1M
        h = 64                              # sparse fwd kernel only supports h_q in {64,128}
        try:
            tb = bench(s, "block", h)
            tp = bench(s, "per_token", h)
            print(f"  {s:>8} {h:>3} {tb:9.3f} {tp:12.3f} {tb/tp:8.2f}x", flush=True)
        except Exception as e:
            print(f"  {s:>8} {h:>3}  FAILED: {repr(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
    print("BLOCK_VS_PERTOKEN_FWD_DONE", flush=True)
