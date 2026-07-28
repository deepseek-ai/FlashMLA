"""Verify the setup.py-integrated block-sparse bwd: flash_mla.block_sparse_prefill_bwd
(registered in the main flash_mla_cuda module) produces correct dQ/dK/dV vs the
separate-KV 192/128 autograd oracle. Run after a full build.
"""
import torch
import flash_mla
from ref_block_sparse_mla import build_block_mask


def oracle(q, k, v, q2k, qbs, kbs, scale):
    s_q, h, _ = q.shape
    s_kv = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    mask = build_block_mask(s_q, s_kv, q2k, qbs, kbs, True, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = scores.logsumexp(-1)
    p = torch.softmax(scores, -1).nan_to_num(0.0)
    o = torch.einsum("hqk,khv->qhv", p, v)
    return o, lse


def main():
    assert hasattr(flash_mla, "block_sparse_prefill_bwd"), "API not exported"
    torch.manual_seed(0)
    dev = "cuda"
    s, h, d, dv = 512, 4, 192, 128
    qbs, kbs = 64, 128
    nqb, nkb = s // qbs, (s + kbs - 1) // kbs
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8
    grad_o = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    q2k = torch.full((nqb, nkb), -1, dtype=torch.int32)
    for qb in range(nqb):
        mx = ((qb + 1) * qbs - 1) // kbs
        q2k[qb, :mx + 1] = torch.arange(mx + 1, dtype=torch.int32)
    q2k = q2k.to(dev)

    qf = q.float().clone().requires_grad_(True)
    kf = k.float().clone().requires_grad_(True)
    vf = v.float().clone().requires_grad_(True)
    o_ref, lse_ref = oracle(qf, kf, vf, q2k, qbs, kbs, scale)
    o_ref.backward(grad_o.float())

    dq, dk, dv_ = flash_mla.block_sparse_prefill_bwd(
        q, k, v, o_ref.detach().to(torch.bfloat16).contiguous(), grad_o,
        lse_ref.detach(), q2k, scale, -1, kbs, qbs)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    cdq, cdk, cdv = cos(dq, qf.grad), cos(dk, kf.grad), cos(dv_, vf.grad)
    print(f"integrated API: dQ={cdq:.5f} dK={cdk:.5f} dV={cdv:.5f}", flush=True)
    assert cdq > 0.98 and cdk > 0.98 and cdv > 0.98
    print("BLOCK_SPARSE_INTEGRATED_OK", flush=True)


if __name__ == "__main__":
    main()
