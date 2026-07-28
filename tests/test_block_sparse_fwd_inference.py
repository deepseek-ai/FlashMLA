"""Block-sparse MLA FORWARD inference test (forward-only, no autograd).

Runs the existing sparse-prefill fwd (flash_mla_sparse_fwd) with causal
block-aligned selection as an inference forward pass, and reports:
  (1) correctness vs the block-sparse torch oracle (cos, should be ~1.0),
  (2) output delta vs FULL dense-causal attention (the effect of dropping
      unselected blocks -- what block-sparsity actually changes at inference),
  (3) prefill forward latency at several sequence lengths.
Sparse-MLA shape h_q=64, d_qk=576, d_v=512.
"""
import torch
import triton
from flash_mla import flash_mla_sparse_fwd
from ref_block_sparse_mla import block_sparse_mla_fwd

DEV = "cuda"
H, DQK, DV = 64, 576, 512


def block_select_to_indices(q2k, kbs, s_q, s_kv, topk_keys):
    """q2k [s_q, topk_blocks] (KV-block ids, -1 pad) -> indices [s_q, topk_keys] int32,
    causal + block-aligned, -1 padded (per-query selection)."""
    nkb = (s_kv + kbs - 1) // kbs
    idx = torch.full((s_q, topk_keys), -1, dtype=torch.int32)
    q2k_cpu = q2k.cpu()
    for qi in range(s_q):
        keys = []
        for b in q2k_cpu[qi].tolist():
            if 0 <= b < nkb:
                for k in range(b * kbs, min((b + 1) * kbs, s_kv)):
                    if k <= qi:
                        keys.append(k)
        keys = sorted(set(keys))[:topk_keys]
        for j, k in enumerate(keys):
            idx[qi, j] = k
    return idx.to(DEV)


def full_causal_mla(q, kv, sm, d_v):
    k = kv[:, 0, :].float()
    v = kv[:, 0, :d_v].float()
    s_q = q.shape[0]
    sc = torch.einsum("qhd,kd->hqk", q.float(), k) * sm
    qi = torch.arange(s_q, device=q.device).view(-1, 1)
    kj = torch.arange(kv.shape[0], device=q.device).view(1, -1)
    sc = sc.masked_fill(~(kj <= qi).view(1, s_q, -1), float("-inf"))
    return torch.einsum("hqk,kv->qhv", torch.softmax(sc, -1), v)


@torch.no_grad()
def run(s, kbs=128, topk_blocks=8):
    sm = DQK ** -0.5
    q = torch.randn(s, H, DQK, device=DEV, dtype=torch.bfloat16) / 8
    kv = torch.randn(s, 1, DQK, device=DEV, dtype=torch.bfloat16) / 8
    nkb = (s + kbs - 1) // kbs

    # causal block selection per query: most-recent (local) blocks + block 0 (global sink)
    q2k = torch.full((s, topk_blocks), -1, dtype=torch.int32)
    for qi in range(s):
        myb = qi // kbs
        local = list(range(max(0, myb - topk_blocks + 2), myb + 1))   # recent window
        sel = sorted(set([0] + local))[-topk_blocks:]                 # + global block 0
        q2k[qi, :len(sel)] = torch.tensor(sel, dtype=torch.int32)
    q2k = q2k.to(DEV)
    topk_keys = topk_blocks * kbs
    indices = block_select_to_indices(q2k, kbs, s, s, topk_keys)[:, None, :]

    o_k, _ml, _lse = flash_mla_sparse_fwd(q, kv, indices, sm, DV)
    o_ref, _ = block_sparse_mla_fwd(q.float(), kv.float(), q2k, 1, kbs, DV, sm, causal=True)
    o_full = full_causal_mla(q, kv, sm, DV)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    c_ref = cos(o_k, o_ref)
    c_full = cos(o_k, o_full)
    lat = triton.testing.do_bench(lambda: flash_mla_sparse_fwd(q, kv, indices, sm, DV), warmup=3, rep=10)
    print(f"s={s:>6} blocks={topk_blocks}x{kbs}(~{topk_blocks*kbs:>4}/{s} keys, {100*topk_blocks*kbs/s:4.0f}%) | "
          f"O-vs-oracle cos={c_ref:.5f} | O-vs-FULL cos={c_full:.4f} | fwd {lat:6.3f} ms", flush=True)
    return c_ref


if __name__ == "__main__":
    print("# block-sparse MLA forward inference (local-window + global-sink selection)")
    ok = True
    for s in [2048, 4096, 8192]:
        ok &= run(s) > 0.99
    assert ok, "block-sparse fwd must match the oracle (cos>0.99)"
    print("BLOCK_SPARSE_FWD_INFERENCE_OK", flush=True)
