"""Phase-3 research: GEMM -> GEMM L2 prefetch chain inside the real decode layer (the shipped kernels are not modified).

27-layer fused graph (MLADecodeLayer, v2 attention).  flash_mla.skinny_gemm.skinny_bmm is monkeypatched to a copy of the skinny
GEMM kernel (p3_gemm_pf.cuh = the kernel of p3_pf_chain.cu: bf16, splits = 1, T <= 8) that can
  ef   load its weights with an L2::evict_first policy
  pf   at its last weight-load issue, prefetch (L2::evict_last) the first lines of the NEXT GEMM's weight rows (budget bytes):
       qkv_a -> W_UK,  W_UV -> o_proj,  o_proj -> next layer's qkv_a      (W_UK has no GEMM successor: attention follows)
Optionally the partial-O discard kernel of p3_discard_layer.py runs after the attention (noop / discard).
variants: repo (shipped kernel), copy, ef, pf (= pf + ef, 1 MB), pf_noef, pf2 (2 MB), copy+noop, copy+discard, pf+noop, pf+discard
usage: python p3_layer_gemm_pf.py --b 1 --sk 4096 [--kv fp8]
"""
import argparse
import statistics as stt

import torch
from torch.utils.cpp_extension import load_inline

from p3_common import HERE, build_dir

import p3_discard_layer as dl
from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph
from flash_mla import skinny_gemm as _sg

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "p3_gemm_pf.cuh"
template <int NT> int smem_of(int K) {
    constexpr int sbytes = (NT + 8) * 128;
    const int fit = kRing / sbytes, s = fit < 3 ? 3 : (fit > 8 ? 8 : fit), nc = K / 64;
    const int ring = (nc < s ? nc : s) * sbytes, red = 4 * 8 * (NT + 4) * 4;
    return ring > red ? ring : red;
}
void gemm_pf(torch::Tensor x, torch::Tensor w, torch::Tensor y, int n_tile, torch::Tensor nxt, long pf_rows, long pf_pitch,
             int pf_lines, int hint, int ef) {
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(gemm<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CK(cudaFuncSetAttribute(gemm<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        init = true;
    }
    TORCH_CHECK(x.size(0) <= 8 && x.scalar_type() == at::kBFloat16 && w.is_contiguous(), "gemm_pf: bf16, T <= 8");
    P p{};
    p.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
    p.w = reinterpret_cast<const __nv_bfloat16 *>(w.data_ptr());
    p.y = reinterpret_cast<__nv_bfloat16 *>(y.data_ptr());
    p.T = x.size(0); p.G = x.size(1); p.K = x.size(2); p.N = w.size(1);
    p.x_st = x.stride(0); p.x_sg = x.stride(1); p.y_st = y.stride(0); p.y_sg = y.stride(1);
    if (pf_lines > 0) {
        p.pf_ptr = reinterpret_cast<const char *>(nxt.data_ptr());
        p.pf_rows = pf_rows; p.pf_pitch = pf_pitch; p.pf_lines = pf_lines; p.pf_at = 1; p.pf_hint = hint;
    }
    p.ef = ef;
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    if (n_tile == 16) gemm<16><<<dim3(p.N / 16, 1, p.G), kThreads, smem_of<16>(p.K), s>>>(p);
    else gemm<32><<<dim3(p.N / 32, 1, p.G), kThreads, smem_of<32>(p.K), s>>>(p);
}
"""
CPP = ("void gemm_pf(torch::Tensor x, torch::Tensor w, torch::Tensor y, int n_tile, torch::Tensor nxt, long pf_rows, "
       "long pf_pitch, int pf_lines, int hint, int ef);")

#           kernel  pf     ef  hint budget    post (None / 0 noop / 1 discard)
VARIANTS = {
    "repo":          ("repo", False, 0, 0, 0, None),
    "copy":          ("copy", False, 0, 0, 0, None),
    "ef":            ("copy", False, 1, 0, 0, None),
    "pf":            ("copy", True, 1, 1, 1 << 20, None),
    "pf_noef":       ("copy", True, 0, 1, 1 << 20, None),
    "pf2":           ("copy", True, 1, 1, 2 << 20, None),
    "copy+noop":     ("copy", False, 0, 0, 0, 0),
    "copy+discard":  ("copy", False, 0, 0, 0, 1),
    "pf+noop":       ("copy", True, 1, 1, 1 << 20, 0),
    "pf+discard":    ("copy", True, 1, 1, 1 << 20, 1),
    "pfh":           ("copy", True, 1, 1, 1 << 19, None),
    "pf3":           ("copy", True, 1, 1, 3 << 20, None),
    "pfh+discard":   ("copy", True, 1, 1, 1 << 19, 1),
    "pf2+discard":   ("copy", True, 1, 1, 2 << 20, 1),
    "pf3+discard":   ("copy", True, 1, 1, 3 << 20, 1),
    "ef+discard":    ("copy", False, 1, 0, 0, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--kv", default="native", choices=["native", "fp8"])
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--seq", default="repo,copy,ef,pf,pf_noef,pf2,copy+noop,copy+discard,pf+noop,pf+discard,repo,copy,pf")
    a = ap.parse_args()
    ext = load_inline("p3_gemm_pf_ext", CPP, cuda_sources=CU, functions=["gemm_pf"], extra_cuda_cflags=["-O3", "-arch=sm_86"],
                      extra_include_paths=[HERE],
                      build_directory=build_dir("build_gemm_pf"), verbose=False)
    dext = load_inline("p3_discard_ext", dl.CPP, cuda_sources=dl.CU, functions=["post"], extra_cuda_cflags=["-O3", "-arch=sm_86"],
                       build_directory=build_dir("build_discard"), verbose=False)
    cfg, page = MLAConfig(), 64
    torch.manual_seed(0)
    st, nblk = make_state(cfg, a.b, a.sk, page, 1, attn="v2", kv_dtype=a.kv)
    layers = [MLADecodeLayer(cfg, fused=True, attn="v2", kv_dtype=a.kv, seed=l, gemm="auto") for l in range(a.layers)]
    cs = [alloc_cache(nblk, page, a.kv, seed=l) for l in range(a.layers)]
    caches, scales = [c[0] for c in cs], [c[1] for c in cs]
    hidden = torch.randn(a.b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        layers[0](hidden, caches[0], st, scales[0])   # allocates the workspace (eager warm-up)
    oacc_bytes = (a.b + st.meta.shape[0]) * cfg.n_heads * 512 * 4

    nxt = {}   # weight data_ptr -> the next GEMM's weight
    for l, layer in enumerate(layers):
        nxt[layer.w_qkv_a.data_ptr()] = layer.w_uk_t
        nxt[layer.w_uv_t.data_ptr()] = layer.w_o
        if l + 1 < len(layers):
            nxt[layer.w_o.data_ptr()] = layers[l + 1].w_qkv_a
    mode = {"kernel": "repo", "pf": False, "ef": 0, "hint": 0, "budget": 0, "post": None}
    orig_bmm = _sg.skinny_bmm

    def bmm(x, w, out=None, n_tile=None, splits=None):
        if mode["kernel"] == "repo":
            return orig_bmm(x, w, out, n_tile, splits)
        T, G, K = x.shape
        N = w.shape[1]
        nt = _sg.plan(T, G, N, K)[0]
        if out is None:
            out = torch.empty(T, G, N, dtype=x.dtype, device=x.device)
        nw = nxt.get(w.data_ptr()) if mode["pf"] else None
        if nw is not None:
            rows, pitch = nw.numel() // nw.shape[-1], nw.shape[-1] * 2
            lines = max(1, min(pitch // 128, mode["budget"] // (rows * 128)))
            ext.gemm_pf(x, w, out, nt, nw, rows, pitch, lines, mode["hint"], mode["ef"])
        else:
            ext.gemm_pf(x, w, out, nt, w, 0, 0, 0, 0, mode["ef"])
        return out
    _sg.skinny_bmm = bmm
    for layer in layers:
        orig = layer._attend

        def attend(q_full, kv_cache, kv_scale, st_, _orig=orig):
            o = _orig(q_full, kv_cache, kv_scale, st_)
            if mode["post"] is not None:
                dext.post(st_.workspace, dl.COUNTER_BYTES, oacc_bytes, mode["post"])
            return o
        layer._attend = attend

    def run(name):
        k, pf, ef, hint, budget, post = VARIANTS[name]
        mode.update(kernel=k, pf=pf, ef=ef, hint=hint, budget=budget, post=post)
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

    print(f"b={a.b} s_k={a.sk} kv={a.kv} layers={a.layers} partial-O {oacc_bytes / 2**20:.2f} MB")
    ref = None
    for name in a.seq.split(","):
        t, o = run(name)
        if ref is None:
            ref = o
        print(f"  {name:<13} {t:7.2f} us/layer  bitwise_equal={torch.equal(o, ref)}", flush=True)


if __name__ == "__main__":
    main()
