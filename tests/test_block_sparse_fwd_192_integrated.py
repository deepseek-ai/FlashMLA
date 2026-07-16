"""Verify the setup.py-integrated block-sparse FORWARD: flash_mla.block_sparse_prefill_fwd
(registered in the main flash_mla_cuda module) produces correct O + LSE vs the
separate-KV 192/128 block-sparse autograd oracle, then a scaling bench (full=dense vs
sparse) through the same integrated API. Run after a full build on B200.
"""
import torch
import triton
import flash_mla
from ref_block_sparse_mla import build_block_mask

QBS, KBS = 256, 128  # q_block = TileShape Q, kv_block = TileShapeQK K (fixed by the kernel)


def oracle(q, k, v, q2k, scale):
    s_q, h, _ = q.shape
    s_kv = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    mask = build_block_mask(s_q, s_kv, q2k, QBS, KBS, True, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = scores.logsumexp(-1)                       # [h,s_q] base-e
    p = torch.softmax(scores, -1).nan_to_num(0.0)
    o = torch.einsum("hqk,khv->qhv", p, v)           # [s_q,h,128]
    return o, lse


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def make_q2k(s, mode, window=4, dev="cuda"):
    nqb = (s + QBS - 1) // QBS
    rows = []
    for qb in range(nqb):
        mx = ((qb + 1) * QBS - 1) // KBS
        causal = list(range(mx + 1))
        sel = causal if mode == "full" else sorted(set([0] + causal[-window:]))
        rows.append(sel)
    topk = max(len(r) for r in rows)
    q2k = torch.full((nqb, topk), -1, dtype=torch.int32)
    for i, r in enumerate(rows):
        q2k[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    return q2k.to(dev), sum(len(r) for r in rows) / len(rows)


def main():
    assert hasattr(flash_mla, "block_sparse_prefill_fwd"), "API not exported"
    torch.manual_seed(0)
    dev = "cuda"
    s, h, d, dv = 512, 8, 192, 128
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    ok = True
    for name in ["full", "sparse"]:
        q2k, avg = make_q2k(s, name, dev=dev)
        o_ref, lse_ref = oracle(q.float(), k.float(), v.float(), q2k, scale)
        o, lse = flash_mla.block_sparse_prefill_fwd(q, k, v, q2k, scale)  # o[s,h,128], lse[h,s]
        co, cl = cos(o, o_ref), cos(lse, lse_ref)
        mae = (lse.float() - lse_ref.float()).abs().mean().item()
        print(f"integrated fwd [{name:>6}] O cos={co:.6f}  LSE cos={cl:.6f} mae={mae:.2e}  avgblk={avg:.1f}", flush=True)
        ok &= (co > 0.99 and cl > 0.999)
    assert ok, "integrated fwd O/LSE mismatch"
    print("BLOCK_SPARSE_FWD_INTEGRATED_OK", flush=True)

    # scaling bench through the integrated API (full=dense vs sparse)
    print("\n# integrated fwd scaling bench (full=dense causal vs sparse=sink+window4) h=8", flush=True)
    print(f"# {'s':>7} {'full ms':>9} {'sparse ms':>10} {'speedup':>8}", flush=True)
    for sb in [8192, 16384, 32768, 65536, 131072, 262144]:
        qb = torch.randn(sb, h, d, device=dev, dtype=torch.bfloat16) / 8
        kb = torch.randn(sb, h, d, device=dev, dtype=torch.bfloat16) / 8
        vb = torch.randn(sb, h, dv, device=dev, dtype=torch.bfloat16) / 8
        row = f"  {sb:>7}"
        tf = ts = None
        try:
            q2kf, _ = make_q2k(sb, "full", dev=dev)
            tf = triton.testing.do_bench(lambda: flash_mla.block_sparse_prefill_fwd(qb, kb, vb, q2kf, scale), warmup=5, rep=20)
            row += f" {tf:9.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:12]):>9}"
        try:
            q2ks, _ = make_q2k(sb, "sparse", dev=dev)
            ts = triton.testing.do_bench(lambda: flash_mla.block_sparse_prefill_fwd(qb, kb, vb, q2ks, scale), warmup=5, rep=20)
            row += f" {ts:10.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:12]):>10}"
        if tf and ts:
            row += f" {tf/ts:7.2f}x"
        del qb, kb, vb; torch.cuda.empty_cache()
        print(row, flush=True)
    print("BLOCK_SPARSE_FWD_INTEGRATED_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
