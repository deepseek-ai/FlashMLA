"""Block-sparse MLA via the EXISTING per-token sparse kernels (no new kernel).

Proves (c)-step-(a): block-sparse training fwd+bwd works TODAY on the existing
flash_mla_sparse_fwd/flash_mla_sparse_bwd by expressing the block selection as
causal, block-aligned per-token key indices. Validated vs the torch oracle
(autograd) ref_block_sparse_mla. Sparse-MLA shape: h_q=64, d_qk=576, d_v=512.
Run with a built flash_mla .so.
"""
import torch
from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_bwd
from ref_block_sparse_mla import block_sparse_mla_fwd


def block_select_to_indices(q2k_blocks, q_block_size, kv_block_size, s_q, s_kv, topk_keys):
    """q2k_blocks [num_q_blocks, topk] (KV-block ids, -1 pad) -> indices [s_q, topk_keys]
    int32, causal + block-aligned, -1 padded (the existing kernel's contract)."""
    dev = q2k_blocks.device
    num_kv_blocks = (s_kv + kv_block_size - 1) // kv_block_size
    idx = torch.full((s_q, topk_keys), -1, dtype=torch.int32, device="cpu")
    q2k_cpu = q2k_blocks.cpu()
    for qi in range(s_q):
        qb = qi // q_block_size
        keys = []
        for b in q2k_cpu[qb].tolist():
            if 0 <= b < num_kv_blocks:
                for k in range(b * kv_block_size, min((b + 1) * kv_block_size, s_kv)):
                    if k <= qi:                       # causal: kernel does NOT mask causality
                        keys.append(k)
        keys = sorted(set(keys))[:topk_keys]
        for j, k in enumerate(keys):
            idx[qi, j] = k
    return idx.to(dev)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    s_q = s_kv = 512
    h_q, d_qk, d_v = 64, 576, 512
    qbs, kbs = 64, 128
    nqb = (s_q + qbs - 1) // qbs
    nkb = (s_kv + kbs - 1) // kbs
    topk_blocks = 3
    topk_keys = topk_blocks * kbs                      # 384 <= 2048 capacity
    sm = d_qk ** -0.5

    q = (torch.randn(s_q, h_q, d_qk, device=dev, dtype=torch.bfloat16) / 8)
    kv = (torch.randn(s_kv, 1, d_qk, device=dev, dtype=torch.bfloat16) / 8)

    # causal-respecting random block selection per Q-block (blocks <= last query's block)
    q2k = torch.full((nqb, topk_blocks), -1, dtype=torch.int32)
    g = torch.Generator().manual_seed(1)
    for qb in range(nqb):
        max_kb = ((qb + 1) * qbs - 1) // kbs           # last K-block the Q-block can see
        choices = torch.randperm(max_kb + 1, generator=g)[:topk_blocks]
        q2k[qb, :len(choices)] = choices.int()
    q2k = q2k.to(dev)
    indices = block_select_to_indices(q2k, qbs, kbs, s_q, s_kv, topk_keys)[:, None, :]  # [s_q,1,topk_keys]

    # ---- torch oracle (autograd) ----
    qref = q.float().clone().requires_grad_(True)
    kvref = kv.float().clone().requires_grad_(True)
    o_ref, lse_ref = block_sparse_mla_fwd(qref, kvref, q2k, qbs, kbs, d_v, sm, causal=True)
    grad_o = torch.randn_like(o_ref) / 8
    o_ref.backward(grad_o)
    dq_ref, dkv_ref = qref.grad, kvref.grad

    # ---- existing sparse kernel fwd+bwd ----
    o_k, max_logits, lse_k = flash_mla_sparse_fwd(q, kv, indices, sm, d_v)
    # sparse fwd emits BASE-E lse (phase1: mi*ln2 + ln(li)); the bwd wants BASE-2.
    LOG2E = 1.4426950408889634
    lse_k_base2 = lse_k * LOG2E
    dq_acc = torch.zeros(s_q * h_q * d_qk, dtype=torch.float32, device=dev)
    dkv_acc = torch.zeros(s_kv * d_qk, dtype=torch.float32, device=dev)
    out = flash_mla_sparse_bwd(q, kv, o_k, grad_o.to(torch.bfloat16), indices, lse_k_base2,
                               dq_acc, dkv_acc, sm, d_v, fuse_reducesum=False)
    dq_k, dkv_k = out[0], out[1]

    def rep(name, a, b):
        a = a.float(); b = b.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        mae = (a - b).abs().mean().item()
        mx = (a - b).abs().max().item()
        print(f"  {name:4} cos={cos:.5f} mae={mae:.4e} max={mx:.4e}", flush=True)
        return cos

    print(f"=== block-sparse via existing kernel (s={s_q}, blocks={topk_blocks}x{kbs}) ===", flush=True)
    c_o = rep("O",   o_k, o_ref)
    c_dq = rep("dQ", dq_k, dq_ref)
    c_dkv = rep("dKV", dkv_k[:, :, :], dkv_ref)
    assert c_o > 0.99, f"O cos {c_o}"
    assert c_dq > 0.97 and c_dkv > 0.97, f"grad cos dq={c_dq} dkv={c_dkv}"
    print("BLOCK_SPARSE_EXISTING_OK", flush=True)


if __name__ == "__main__":
    main()
