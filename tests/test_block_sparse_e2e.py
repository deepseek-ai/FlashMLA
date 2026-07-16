"""End-to-end 192/128 block-sparse training step via the integrated API:
  kernel fwd (q_block=256)  ->  expand selection ->  kernel bwd (q_block=64)
and validate O+LSE+dQ/dK/dV against a SINGLE autograd oracle for the q_block=256
block-sparse pattern. The bwd consumes the FWD's actual O+LSE (not the oracle's),
so this exercises the true fwd->bwd chain. Run after a full build on B200.
"""
import torch
import flash_mla
from ref_block_sparse_mla import build_block_mask

QBS_FWD, QBS_BWD, KBS = 256, 64, 128
FACTOR = QBS_FWD // QBS_BWD  # 4


def oracle(q, k, v, q2k256, scale):
    """differentiable block-sparse MHA fwd at the fwd's q_block=256 granularity."""
    s_q, h, _ = q.shape
    s_kv = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    mask = build_block_mask(s_q, s_kv, q2k256, QBS_FWD, KBS, True, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = scores.logsumexp(-1)
    p = torch.softmax(scores, -1).nan_to_num(0.0)
    o = torch.einsum("hqk,khv->qhv", p, v)
    return o, lse


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def make_q2k256(s, mode, window=4, dev="cuda"):
    nqb = (s + QBS_FWD - 1) // QBS_FWD
    rows = []
    for qb in range(nqb):
        mx = ((qb + 1) * QBS_FWD - 1) // KBS
        causal = list(range(mx + 1))
        sel = causal if mode == "full" else sorted(set([0] + causal[-window:]))
        rows.append(sel)
    topk = max(len(r) for r in rows)
    q2k = torch.full((nqb, topk), -1, dtype=torch.int32)
    for i, r in enumerate(rows):
        q2k[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    return q2k.to(dev)


def main():
    assert hasattr(flash_mla, "expand_block_selection"), "helper not exported"
    torch.manual_seed(0)
    dev = "cuda"
    s, h, d, dv = 1024, 8, 192, 128   # s multiple of 256
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8
    do = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    ok = True
    for mode in ["full", "sparse"]:
        q2k256 = make_q2k256(s, mode, dev=dev)

        # --- kernel fwd (q_block=256) ---
        o, lse = flash_mla.block_sparse_prefill_fwd(q, k, v, q2k256, scale)   # o[s,h,128], lse[h,s]

        # --- reconcile: expand 256-block selection to 64-block for the bwd ---
        q2k64 = flash_mla.expand_block_selection(q2k256, FACTOR)
        assert q2k64.shape[0] == (s + QBS_BWD - 1) // QBS_BWD, q2k64.shape

        # --- kernel bwd (q_block=64), consuming the fwd's OWN O+LSE ---
        dq, dk, dv = flash_mla.block_sparse_prefill_bwd(
            q, k, v, o, do, lse, q2k64, scale, -1, KBS, QBS_BWD)

        # --- single autograd oracle for the q_block=256 pattern ---
        qf = q.float().clone().requires_grad_(True)
        kf = k.float().clone().requires_grad_(True)
        vf = v.float().clone().requires_grad_(True)
        o_ref, lse_ref = oracle(qf, kf, vf, q2k256, scale)
        o_ref.backward(do.float())

        cO, cL = cos(o, o_ref), cos(lse, lse_ref)
        cQ, cK, cV = cos(dq, qf.grad), cos(dk, kf.grad), cos(dv, vf.grad)
        print(f"=== e2e [{mode:>6}] s={s} h={h} (fwd q_block=256 -> bwd q_block=64) ===", flush=True)
        print(f"    fwd  O cos={cO:.6f}  LSE cos={cL:.6f}", flush=True)
        print(f"    bwd dQ cos={cQ:.6f} dK cos={cK:.6f} dV cos={cV:.6f}", flush=True)
        ok &= (cO > 0.99 and cL > 0.999 and cQ > 0.98 and cK > 0.98 and cV > 0.98)

    # determinism of the full chain (bwd is the nondeterminism-sensitive part)
    q2k256 = make_q2k256(s, "sparse", dev=dev)
    o, lse = flash_mla.block_sparse_prefill_fwd(q, k, v, q2k256, scale)
    q2k64 = flash_mla.expand_block_selection(q2k256, FACTOR)
    _, dk1, dv1 = flash_mla.block_sparse_prefill_bwd(q, k, v, o, do, lse, q2k64, scale, -1, KBS, QBS_BWD)
    _, dk2, dv2 = flash_mla.block_sparse_prefill_bwd(q, k, v, o, do, lse, q2k64, scale, -1, KBS, QBS_BWD)
    det = torch.equal(dk1, dk2) and torch.equal(dv1, dv2)
    print(f"  full-chain dK/dV bitwise-deterministic: {det}", flush=True)

    assert ok, "e2e grad/output mismatch"
    print("BLOCK_SPARSE_E2E_OK", flush=True)


if __name__ == "__main__":
    main()
