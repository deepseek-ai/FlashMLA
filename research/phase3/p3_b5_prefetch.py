"""Phase-3 research (B5): does an L2-warm start-up chain shorten the small-batch v2 decode?

Per "layer": thrash (16 MB read-only stream = the weight stream that evicts L2 in a real layer) -> P (prefetch kernel, variants) -> v2 decode
+ combine (flash_mla_with_kvcache with out / lse / workspace).  L layers, each with its own KV cache; metadata and block table
shared by all layers (as in a model).  One CUDA graph; time per layer; variants differ only in what P does:
  noop      P launched, does nothing                        (launch cost reference)
  meta      P reads each part's metadata row + block-table entry (both become L2 resident)
  first     meta + prefetch.global.L2 of the first `ntok` tokens of each part
  part      meta + prefetch of all tokens of each part (the whole KV of the step)
  none      no P at all
  spinX     P also spins X us (models a latency-bound prologue that the prefetch could hide behind)
Outputs are compared bitwise with the `none` variant.
usage: python p3_b5_prefetch.py --b 1 --sk 4096 [--kv native|fp8]
"""
import argparse
import statistics as stt

import torch
from torch.utils.cpp_extension import load_inline

from p3_common import build_dir

import flash_mla

CU = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void pf_kernel(const int *md, const int *bt, long bt_stride, int b, int page, const char *kv, long row_bytes,
                          long page_bytes, int mode, int ntok, int spin_ns) {
    const int part = blockIdx.x;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (mode != 0) {
    const int4 m0 = *reinterpret_cast<const int4 *>(md + part * 8);
    const int4 m1 = *reinterpret_cast<const int4 *>(md + part * 8 + 4);
    const int bi = m0.x, bs = m0.y, ei = m0.z, es = m0.w;
    if (bi < b) {
    // tokens of the first segment of this part: [bs, end)
    const int seg_end = (bi == ei) ? min(es, m1.y) : m1.y;
    const int t1 = mode == 2 ? min(seg_end, bs + ntok) : (mode == 3 ? seg_end : bs);
    const int pg = bt[(long)bi * bt_stride + bs / page];
    if (threadIdx.x == 0 && pg == -12345) asm volatile("trap;");
    // all 128 threads issue: line i of the part = (token i / lpr, line i % lpr); one prefetch may block its thread
    const int lpr = (int)((row_bytes + 127) / 128);
    for (int i = threadIdx.x; mode >= 2 && i < (t1 - bs) * lpr; i += blockDim.x) {
        const int tok = bs + i / lpr, l = i % lpr;
        const int p = (tok / page == bs / page) ? pg : bt[(long)bi * bt_stride + tok / page];
        const char *row = kv + (long)p * page_bytes + (long)(tok % page) * row_bytes;
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(row + l * 128));
    }
    }
    }
    if (spin_ns > 0) {   // latency-bound work of a prologue-like kernel
        unsigned long long t;
        do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < (unsigned long long)spin_ns);
    }
}
__global__ void thrash_kernel(const float4 *a, long n4, float *out) {
    float acc = 0.f;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
        const float4 v = __ldcg(a + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.f) out[0] = acc;
}
void thrash(torch::Tensor a, torch::Tensor out) {
    thrash_kernel<<<328, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const float4 *)a.data_ptr(), a.numel() / 4,
                                                                    out.data_ptr<float>());
}
void pf(torch::Tensor md, torch::Tensor bt, torch::Tensor kv, int mode, int ntok, int spin_ns) {
    const int parts = md.size(0), b = bt.size(0), page = kv.size(1);
    const long row_bytes = kv.stride(1) * kv.element_size();
    const long page_bytes = kv.stride(0) * kv.element_size();
    pf_kernel<<<parts, 128, 0, at::cuda::getCurrentCUDAStream()>>>(md.data_ptr<int>(), bt.data_ptr<int>(), bt.stride(0), b, page,
        (const char *)kv.data_ptr(), row_bytes, page_bytes, mode, ntok, spin_ns);
}
"""
CPP = """void pf(torch::Tensor md, torch::Tensor bt, torch::Tensor kv, int mode, int ntok, int spin_ns);
void thrash(torch::Tensor a, torch::Tensor out);"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--kv", default="native", choices=["native", "fp8"])
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--thrash-mb", type=int, default=16)
    ap.add_argument("--ntok", type=int, default=32)
    a = ap.parse_args()
    ext = load_inline("p3_pf_ext3", CPP, cuda_sources=CU, functions=["pf", "thrash"], extra_cuda_cflags=["-O3", "-arch=sm_86"],
                      build_directory=build_dir("build_pf"), verbose=False)
    torch.manual_seed(0)
    dev, dt, page, H, L = "cuda", torch.bfloat16, 64, 16, a.layers
    nblk_per = -(-a.sk // page)
    nblk = a.b * nblk_per
    bt = torch.randperm(nblk, device=dev).view(a.b, nblk_per).to(torch.int32).contiguous()
    lens = torch.full((a.b,), a.sk, dtype=torch.int32, device=dev)
    caches, scales = [], []
    for _ in range(L):
        kv = torch.randn(nblk, page, 1, 576, dtype=dt, device=dev) * 0.5
        if a.kv == "fp8":
            kvq, sc = flash_mla.quantize_kv(kv.view(-1, page, 576), "fp8")
            caches.append(kvq)
            scales.append(sc)
        else:
            caches.append(kv)
            scales.append(None)
    hint = {"kv_dtype": "fp8"} if a.kv == "fp8" else {}
    md, nsplit = flash_mla.get_mla_metadata(lens, H, 1, **hint)
    ws = flash_mla.alloc_mla_workspace(a.b, H, 1, md.shape[0])
    q = torch.randn(a.b, 1, H, 576, dtype=dt, device=dev)
    outs = [torch.empty(a.b, 1, H, 512, dtype=dt, device=dev) for _ in range(L)]
    lse = torch.empty(a.b, H, 1, dtype=torch.float32, device=dev)
    src = torch.randn((a.thrash_mb << 20) // 4, dtype=torch.float32, device=dev)
    tout = torch.zeros(1, dtype=torch.float32, device=dev)
    kw = {"kv_scale": None, "kv_dtype": "fp8"} if a.kv == "fp8" else {}

    def step(mode, spin):
        for l in range(L):
            ext.thrash(src, tout)
            if mode >= 0:
                ext.pf(md, bt, caches[l].view(nblk, page, -1), mode, a.ntok, spin)
            k = dict(kw)
            if a.kv == "fp8":
                k["kv_scale"] = scales[l]
            flash_mla.flash_mla_with_kvcache(q, caches[l], bt, lens, 512, md, nsplit, softmax_scale=192 ** -0.5, causal=False,
                                             out=outs[l], lse=lse, workspace=ws, impl="v2", **k)

    def thrash_only():
        for _ in range(L):
            ext.thrash(src, tout)

    def timeit(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / 20 / L)
        return stt.median(ts)

    base_t = timeit(thrash_only)
    print(f"b={a.b} s_k={a.sk} kv={a.kv} parts={md.shape[0]} ntok={a.ntok}: thrash only {base_t:.2f} us/layer")
    ref = None
    for name, mode, spin in [("none", -1, 0), ("noop", 0, 0), ("meta", 1, 0), ("first", 2, 0),
                             ("spin2.4", 0, 2400), ("spin2.4+meta", 1, 2400), ("spin2.4+first", 2, 2400), ("spin2.4+part", 3, 2400),
                             ("spin4", 0, 4000), ("spin4+first", 2, 4000), ("spin4+part", 3, 4000), ("none_again", -1, 0)]:
        t = timeit(lambda: step(mode, spin))
        o = torch.stack(outs).clone()
        if ref is None:
            ref = o
        print(f"  {name:<11} {t:7.2f} us/layer  attention part (minus thrash) {t - base_t:6.2f}  bitwise_equal={torch.equal(o, ref)}")


if __name__ == "__main__":
    main()
