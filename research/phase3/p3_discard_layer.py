"""Phase-3 research: cost of the fp32 partial-O write-back in the real decode layer, and what discard.global.L2 saves.

27-layer fused graph (MLADecodeLayer, skinny GEMM, v2).  After each layer's attention (decode + combine) an extra kernel runs over
the workspace's partial-O region:
  orig     no extra kernel
  noop     the extra kernel launches and does nothing     (launch-cost reference)
  discard  the extra kernel issues discard.global.L2 on every 128-B line of the partial-O region: the dirty partials (already
           consumed by the combine) are dropped from L2 instead of being written back to DRAM during W_UV / o_proj
The difference discard - noop is what a discard inside the combine kernel would save.  Outputs are compared bitwise.
usage: python p3_discard_layer.py --b 1 --sk 4096 [--kv fp8]
"""
import argparse
import statistics as stt

import torch
from torch.utils.cpp_extension import load_inline

from p3_common import build_dir

from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void post_kernel(char *base, long nlines, int mode) {
    if (mode == 0) return;
    for (long l = blockIdx.x * (long)blockDim.x + threadIdx.x; l < nlines; l += (long)gridDim.x * blockDim.x)
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(base + l * 128) : "memory");
}
void post(torch::Tensor ws, long off, long bytes, int mode) {
    post_kernel<<<82, 256, 0, at::cuda::getCurrentCUDAStream()>>>((char *)ws.data_ptr() + off, bytes / 128, mode);
}
"""
CPP = "void post(torch::Tensor ws, long off, long bytes, int mode);"
COUNTER_BYTES = 64 * 1024   # mla_sm86::kCounterBytes: the partial-O region starts right after the combine counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--kv", default="native", choices=["native", "fp8"])
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--seq", default="orig,noop,discard,noop,discard,orig")
    a = ap.parse_args()
    ext = load_inline("p3_discard_ext", CPP, cuda_sources=CU, functions=["post"], extra_cuda_cflags=["-O3", "-arch=sm_86"],
                      build_directory=build_dir("build_discard"), verbose=False)
    cfg, page = MLAConfig(), 64
    torch.manual_seed(0)
    st, nblk = make_state(cfg, a.b, a.sk, page, 1, attn="v2", kv_dtype=a.kv)
    layers = [MLADecodeLayer(cfg, fused=True, attn="v2", kv_dtype=a.kv, seed=l, gemm="auto") for l in range(a.layers)]
    cs = [alloc_cache(nblk, page, a.kv, seed=l) for l in range(a.layers)]
    caches, scales = [c[0] for c in cs], [c[1] for c in cs]
    hidden = torch.randn(a.b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    mode = {"v": 0}
    with torch.no_grad():
        layers[0](hidden, caches[0], st, scales[0])   # allocates the workspace (eager warm-up)
    parts = st.meta.shape[0]
    M = cfg.n_heads   # s_q * h_q, h_k = 1
    oacc_bytes = (a.b + parts) * M * 512 * 4
    for layer in layers:
        orig = layer._attend

        def attend(q_full, kv_cache, kv_scale, st_, _orig=orig):
            o = _orig(q_full, kv_cache, kv_scale, st_)
            if mode["v"] >= 0:
                ext.post(st_.workspace, COUNTER_BYTES, oacc_bytes, mode["v"])
            return o
        layer._attend = attend

    def run(v):
        mode["v"] = v
        with torch.no_grad():
            g, out = capture_layer_graph(layers, caches, scales, st, hidden)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(9):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / 20 / a.layers)
        res = out.clone()
        del g
        return stt.median(ts), res

    print(f"b={a.b} s_k={a.sk} kv={a.kv} layers={a.layers} parts={parts} partial-O region {oacc_bytes / 2**20:.2f} MB")
    ref = None
    codes = {"orig": -1, "noop": 0, "discard": 1}
    for name, v in [(n, codes[n]) for n in a.seq.split(",")]:
        t, o = run(v)
        if ref is None:
            ref = o
        print(f"  {name:<8} {t:7.2f} us/layer  bitwise_equal={torch.equal(o, ref)}", flush=True)


if __name__ == "__main__":
    main()
