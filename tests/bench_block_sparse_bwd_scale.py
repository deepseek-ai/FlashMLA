"""Perf-at-scale: block-sparse KV-outer MLA bwd, SPARSE vs FULL(=dense causal)
selection at large seqlens, to show the sparsity compute saving (invisible at the
tiny correctness-test shape, which is launch-overhead bound). Timing only (random
O/LSE -- runtime depends on the CSR iteration count, not values). 192/128.
Run on B200 after a full build."""
import os
import torch
import triton
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")
CSRC = os.path.join(HERE, "..", "csrc"); CUT = os.path.join(CSRC, "cutlass")
mod = load(
    name="block_sparse_bwd_cutlass_ext",
    sources=[os.path.join(SRC, "block_sparse_bwd_pybind.cu")],
    extra_include_paths=[CSRC, os.path.join(CSRC, "sm90"), os.path.join(CSRC, "kerutils", "include"),
                         os.path.join(CUT, "include"), os.path.join(CUT, "tools", "util", "include")],
    extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-DBLOCK_SPARSE_BWD_STANDALONE",
                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                       "-gencode", "arch=compute_100f,code=sm_100f"],
    verbose=False,
)
from flash_mla import cuda as flash_mla_cuda  # stock FlashMLA dense MLA bwd
DEV = "cuda"


def bench_dense_stock(s, h):
    """Stock FlashMLA dense MLA bwd (dense_prefill_bwd), causal, same 192/128 shape."""
    d, dv = 192, 128
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    do = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    o = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    lse = torch.randn(h, s, device=DEV, dtype=torch.float32).transpose(0, 1)  # [s,h] stride(0)==1
    cu = torch.tensor([0, s], dtype=torch.int32, device=DEV)
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv_ = torch.zeros_like(v)
    ma = (s + 7) // 8 * 8
    ws = torch.empty(4 * ma * h * d + 4 * ma * h * 2, dtype=torch.uint8, device=DEV)

    def run():
        flash_mla_cuda.dense_prefill_bwd(ws, do, q, k, v, o, lse, cu, cu, dq, dk, dv_,
                                         1, float(scale), s, s, False, -1)
    t = triton.testing.do_bench(run, warmup=5, rep=20)
    del q, k, v, do, o, lse, dq, dk, dv_, ws; torch.cuda.empty_cache()
    return t


def make_q2k(nqb, nkb, qbs, kbs, mode, window=4):
    """full = all causal KV-blocks; sparse = global block 0 + last `window` causal blocks."""
    rows = []
    for qb in range(nqb):
        mx = ((qb + 1) * qbs - 1) // kbs
        causal = list(range(mx + 1))
        if mode == "full":
            sel = causal
        else:
            sel = sorted(set([0] + causal[-window:]))
        rows.append(sel)
    topk = max(len(r) for r in rows)
    q2k = torch.full((nqb, topk), -1, dtype=torch.int32)
    for i, r in enumerate(rows):
        q2k[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    avg = sum(len(r) for r in rows) / len(rows)
    return q2k.to(DEV), avg


def bench(s, h, mode, window=4):
    d, dv, qbs, kbs = 192, 128, 64, 128
    nqb, nkb = s // qbs, (s + kbs - 1) // kbs
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    grad_o = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    o = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    lse = (torch.randn(h, s, device=DEV, dtype=torch.float32)).transpose(0, 1)  # [s,h] stride(0)==1
    q2k, avg = make_q2k(nqb, nkb, qbs, kbs, mode, window)
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv_ = torch.zeros_like(v)
    ws = torch.empty(mod.workspace_size(s, h, 1, d), dtype=torch.uint8, device=DEV)

    def run():
        mod.block_sparse_bwd(ws, grad_o, q, k, v, o, lse, q2k, dq, dk, dv_, float(scale), -1, kbs, qbs)
    t = triton.testing.do_bench(run, warmup=5, rep=20)
    return t, avg


if __name__ == "__main__":
    h = 16
    print(f"# block-sparse KV-outer MLA bwd 192/128 h={h}; sparse = sink+window(4 blocks)")
    print(f"# stock = stock FlashMLA dense_prefill_bwd; full = my kernel, full causal selection")
    print(f"# {'s':>7} {'stock ms':>9} {'myfull ms':>10} {'sparse ms':>10} {'sparse/stock':>12}", flush=True)
    for s in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        row = f"  {s:>7}"
        try:
            td = bench_dense_stock(s, h); row += f" {td:9.3f}"
        except Exception as e:
            td = None; row += f"  {('FAIL:'+repr(e)[:14]):>9}"
        try:
            tf, af = bench(s, h, "full"); torch.cuda.empty_cache(); row += f" {tf:10.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:14]):>10}"
        try:
            ts, as_ = bench(s, h, "sparse"); torch.cuda.empty_cache(); row += f" {ts:10.3f}"
            if td: row += f" {td/ts:11.2f}x"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:14]):>10}"
        print(row, flush=True)
    print("BLOCK_SPARSE_BWD_SCALE_DONE", flush=True)
