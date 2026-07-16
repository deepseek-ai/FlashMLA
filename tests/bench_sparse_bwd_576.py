"""Baseline (B): the EXISTING per-token sparse MLA bwd at 576/512 (Q-outer +
atomicAdd scatter, the production DSA bwd) with block-aligned indices, topk=2048.
This is the "before KV-outer" cost that the 576/512 block-sparse KV-outer port
(task A) will improve. h_q=64, d_qk=576, d_v=512. Existing .so. B200.
"""
import torch
import triton
from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_bwd

DEV = "cuda"
H, DQK, DV, TOPK, KBS = 64, 576, 512, 2048, 128
LOG2E = 1.4426950408889634
NSEL = TOPK // KBS  # 16 blocks


def gen_block_indices(s):  # block-aligned causal: 16 random causal blocks x 128 keys
    qi = torch.arange(s, device=DEV, dtype=torch.int32).view(s, 1)
    nblk = (qi // KBS) + 1
    r = torch.rand(s, NSEL, device=DEV)
    blocks = (r * nblk.float()).to(torch.int32).clamp(max=s // KBS)
    offs = torch.arange(KBS, device=DEV, dtype=torch.int32).view(1, 1, KBS)
    keys = (blocks.unsqueeze(-1) * KBS + offs).reshape(s, TOPK)
    return torch.minimum(keys, qi)[:, None, :]


@torch.no_grad()
def bench(s):
    sm = DQK ** -0.5
    q = torch.randn(s, H, DQK, device=DEV, dtype=torch.bfloat16) / 8
    kv = torch.randn(s, 1, DQK, device=DEV, dtype=torch.bfloat16) / 8
    do = torch.randn(s, H, DV, device=DEV, dtype=torch.bfloat16) / 8
    idx = gen_block_indices(s)
    o, _ml, lse = flash_mla_sparse_fwd(q, kv, idx, sm, DV)
    lse2 = lse * LOG2E
    dq_acc = torch.zeros(s * H * DQK, dtype=torch.float32, device=DEV)
    dkv_acc = torch.zeros(s * DQK, dtype=torch.float32, device=DEV)

    def bwd():
        flash_mla_sparse_bwd(q, kv, o, do, idx, lse2, dq_acc, dkv_acc, sm, DV, False)
    reps = 10 if s <= 32768 else 4
    t = triton.testing.do_bench(bwd, warmup=2, rep=reps)
    del q, kv, do, idx, o, lse, lse2, dq_acc, dkv_acc; torch.cuda.empty_cache()
    return t


if __name__ == "__main__":
    print(f"# existing per-token sparse bwd (Q-outer scatter) 576/512 h={H} topk={TOPK}, block-aligned")
    for s in [4096, 16384, 32768, 65536, 131072, 262144]:
        try:
            print(f"  sparse-bwd {s:>7}: {bench(s):9.3f} ms", flush=True)
        except Exception as e:
            print(f"  sparse-bwd {s:>7}: FAILED {repr(e)[:70]}", flush=True); torch.cuda.empty_cache()
    print("SPARSE_BWD_576_DONE", flush=True)
