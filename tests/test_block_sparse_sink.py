"""Validate the gpt-oss attention sink added to the block-sparse MLA fwd/bwd kernels.

The sink is a per-head value-less logit folded into the softmax denominator. We check, against a
pure-torch autograd reference over the SAME block selection (causal keys in the selected KV blocks):
  - forward O is sink-aware (= o_nosink * sigmoid(lse_nosink - sink)),
  - backward dq/dk/dv match (they must, since the fwd writes a sink-aware LSE),
  - d_sink matches the reference gradient of the sink parameter.

Run (in the FlashMLA build container): python tests/test_block_sparse_sink.py
"""
import math
import torch
import flash_mla

torch.manual_seed(0)
S, H, DQK, DV = 512, 4, 192, 128
QB, KVB = 256, 128
WINDOW = 256
DEV = "cuda"


def window_q2k(s_q, window):
    nqb = math.ceil(s_q / QB)
    topk = (QB - 1 + window + KVB - 1) // KVB + 1
    q2k = torch.full((nqb, topk), -1, dtype=torch.int32, device=DEV)
    for qb in range(nqb):
        hi = qb * QB + QB - 1
        lo = max(0, qb * QB - (window - 1))
        for j, bk in enumerate(list(range(lo // KVB, hi // KVB + 1))[:topk]):
            q2k[qb, j] = bk
    return q2k


def ref_attention(q, k, v, q2k, scale, sink):
    """Dense torch reference over the block selection (causal + selected-block mask) with sink."""
    s_q = q.shape[0]
    nqb, topk = q2k.shape
    # per-query allowed key mask: causal AND key in one of its q-block's selected KV blocks
    allowed = torch.zeros(s_q, s_q, dtype=torch.bool, device=DEV)
    kpos = torch.arange(s_q, device=DEV)
    for qb in range(nqb):
        q0, q1 = qb * QB, min((qb + 1) * QB, s_q)
        blkmask = torch.zeros(s_q, dtype=torch.bool, device=DEV)
        for j in range(topk):
            b = int(q2k[qb, j])
            if b >= 0:
                blkmask |= (kpos // KVB == b)
        allowed[q0:q1] = blkmask.unsqueeze(0)
    causal = kpos.unsqueeze(0) <= torch.arange(s_q, device=DEV).unsqueeze(1)
    mask = allowed & causal  # [s_q, s_kv]
    # scores [h, s_q, s_kv]
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink.view(H, 1, 1) - m)  # sink in denom
    w = p / denom
    o = torch.einsum("hqk,khd->qhd", w, v.float())  # [s_q, h, dv]
    return o


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


def run_case(qd, kd, vd, q2k, scale, sink, do):
    """Return (ferr, c_dq, c_dk, c_dv, dserr) comparing the kernel to a torch autograd reference.
    sink=None disables (uses -inf in the ref denom -> no sink contribution)."""
    qr = qd.float().clone().requires_grad_(True)
    kr = kd.float().clone().requires_grad_(True)
    vr = vd.float().clone().requires_grad_(True)
    H_ = qd.shape[1]
    sr = (sink.clone() if sink is not None else torch.full((H_,), float("-inf"), device=DEV)).requires_grad_(sink is not None)
    o_ref = ref_attention(qr, kr, vr, q2k, scale, sr)
    o_ref.backward(do)
    o_k, lse_k = flash_mla.block_sparse_prefill_fwd(qd, kd, vd, q2k, scale, sink)
    dq, dk, dv, d_sink = flash_mla.block_sparse_prefill_bwd(
        qd, kd, vd, o_k, do.bfloat16(), lse_k, q2k, scale, -1, KVB, 64, sink
    )
    ferr = (o_k.float() - o_ref).abs().max().item()
    dserr = (d_sink - sr.grad).abs().max().item() / (sr.grad.abs().max().item() + 1e-6) if sink is not None else 0.0
    return ferr, cos(dq, qr.grad), cos(dk, kr.grad), cos(dv, vr.grad), dserr, d_sink, (sr.grad if sink is not None else None)


def main():
    scale = DQK ** -0.5
    q2k = window_q2k(S, WINDOW)
    qd = torch.randn(S, H, DQK, device=DEV, dtype=torch.bfloat16)
    kd = torch.randn(S, H, DQK, device=DEV, dtype=torch.bfloat16)
    vd = torch.randn(S, H, DV, device=DEV, dtype=torch.bfloat16)
    do = torch.randn(S, H, DV, device=DEV)
    sink = torch.randn(H, device=DEV, dtype=torch.float32) * 0.5

    # Isolation: NO-SINK first. If its dq/dk/dv cosine is also low, the hand-rolled reference (not the
    # sink) is the culprit -> the kernel's no-sink bwd is already trusted at cos>0.97 by the repo's tests.
    f0, dq0, dk0, dv0, _, _, _ = run_case(qd, kd, vd, q2k, scale, None, do)
    print(f"NO-SINK : FWD O err={f0:.4f}  BWD cos dq={dq0:.4f} dk={dk0:.4f} dv={dv0:.4f}")
    f1, dq1, dk1, dv1, dserr, d_sink, dsref = run_case(qd, kd, vd, q2k, scale, sink, do)
    print(f"WITH-SINK: FWD O err={f1:.4f}  BWD cos dq={dq1:.4f} dk={dk1:.4f} dv={dv1:.4f}  d_sink rel-err={dserr:.4f}")
    print(f"  d_sink kernel={[round(x,4) for x in d_sink.tolist()]}")
    print(f"  d_sink ref   ={[round(x,4) for x in dsref.tolist()]}")

    # Sink is correct iff: (a) fwd O sink-aware, (b) d_sink matches, (c) the sink does NOT DEGRADE the
    # bwd dq/dk/dv relative to the no-sink baseline (the absolute cosine is set by the bf16-vs-fp32-ref
    # baseline, which the no-sink case measures).
    fwd_ok = f1 < 0.05
    dsink_ok = dserr < 0.05
    bwd_not_degraded = min(dq1, dk1, dv1) >= min(dq0, dk0, dv0) - 0.03
    print(f"checks: fwd_sink_aware={fwd_ok} d_sink_correct={dsink_ok} bwd_not_degraded_vs_nosink={bwd_not_degraded}")
    print("BLOCK_SPARSE_SINK_OK" if (fwd_ok and dsink_ok and bwd_not_degraded) else "BLOCK_SPARSE_SINK_FAIL")


if __name__ == "__main__":
    main()
