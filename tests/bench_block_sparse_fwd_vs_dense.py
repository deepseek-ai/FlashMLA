"""Rigorous final fwd bench: 192/128 block-sparse MLA fwd vs the STOCK FlashMLA dense
MLA fwd (dense_prefill_fwd, the kIsMla 192/128 path) -- the analogue of the 3-way bwd
table. Columns: stock dense / my-full(=dense causal, q2k all blocks) / my-sparse, and
sparse-vs-stock speedup. Same shape (causal, per-head K/V h_k=h). Run after a full
build on B200."""
import torch
import triton
import flash_mla
from flash_mla.flash_mla_interface import _flash_attn_varlen_forward

DEV = "cuda"
QBS, KBS = 256, 128


def make_q2k(s, mode, window=4):
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
    return q2k.to(DEV), sum(len(r) for r in rows) / len(rows)


def bench_stock_dense(s, h, d=192, dv=128):
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    cu = torch.tensor([0, s], dtype=torch.int32, device=DEV)
    t = triton.testing.do_bench(
        lambda: _flash_attn_varlen_forward(q, k, v, cu, cu, s, s, causal=True,
                                           softmax_scale=scale, is_varlen=True, window_size=-1),
        warmup=5, rep=20)
    del q, k, v; torch.cuda.empty_cache()
    return t


def bench_mine(s, h, mode, d=192, dv=128, window=4):
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    q2k, avg = make_q2k(s, mode, window)
    t = triton.testing.do_bench(
        lambda: flash_mla.block_sparse_prefill_fwd(q, k, v, q2k, scale), warmup=5, rep=20)
    del q, k, v, q2k; torch.cuda.empty_cache()
    return t, avg


if __name__ == "__main__":
    h = 8
    print(f"# 192/128 fwd: STOCK dense MLA (dense_prefill_fwd) vs block-sparse (mine) h={h}")
    print(f"# my-full = my kernel, full causal selection (== dense); sparse = sink+window(4)")
    print(f"# {'s':>7} {'stock ms':>9} {'myfull ms':>10} {'sparse ms':>10} {'sparse/stock':>13} {'avgblk':>7}", flush=True)
    for s in [8192, 16384, 32768, 65536, 131072, 262144]:
        row = f"  {s:>7}"
        td = tf = ts = None; av = 0
        try:
            td = bench_stock_dense(s, h); row += f" {td:9.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:12]):>9}"
        try:
            tf, _ = bench_mine(s, h, "full"); row += f" {tf:10.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:12]):>10}"
        try:
            ts, av = bench_mine(s, h, "sparse"); row += f" {ts:10.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:12]):>10}"
        if td and ts:
            row += f" {td/ts:12.2f}x {av:6.1f}"
        print(row, flush=True)
    print("BLOCK_SPARSE_FWD_VS_DENSE_DONE", flush=True)
