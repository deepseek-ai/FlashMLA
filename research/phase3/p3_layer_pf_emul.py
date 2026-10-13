"""Phase-3 research: KV first-tile L2 prefetch inside the real decode layer (timing emulation with a side graph branch).

27-layer fused graph (MLADecodeLayer, skinny GEMM, v2).  A prefetch kernel P (p3_b5_prefetch.pf: per part, the first `ntok`
tokens of the KV range, issued by all threads) runs on a side stream with no dependency on the layer kernels:
  serial          as shipped
  A_noop / A_pf   P concurrent with the prologue (side branch forks after W_UK, joins before the decode)
  B_noop / B_pf   P concurrent with W_UK + prologue (forks after qkv_a)
  F / F_pf        prologue itself on the side branch, concurrent with W_UK (the fusion emulation of p3_fuse_emul.py), without / with P
                  queued behind it on the side branch
Layer outputs are only meaningful for serial / A / B (prefetch is a hint); F reads a stale q_abs.
usage: python p3_layer_pf_emul.py --b 1 --sk 4096 --ntok 16 [--kv fp8]
"""
import argparse
import statistics as stt

import torch
from torch.utils.cpp_extension import load_inline

from p3_common import build_dir

import p3_b5_prefetch as pfm
from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state
from flash_mla import skinny_gemm as _sg
from flash_mla.mla_prologue import mla_decode_prologue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--kv", default="native", choices=["native", "fp8"])
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--ntok", type=int, default=16)
    a = ap.parse_args()
    ext = load_inline("p3_pf_ext3", pfm.CPP, cuda_sources=pfm.CU, functions=["pf", "thrash"],
                      extra_cuda_cflags=["-O3", "-arch=sm_86"], build_directory=build_dir("build_pf"), verbose=False)
    cfg, page = MLAConfig(), 64
    torch.manual_seed(0)
    st, nblk = make_state(cfg, a.b, a.sk, page, 1, attn="v2", kv_dtype=a.kv)
    layers = [MLADecodeLayer(cfg, fused=True, attn="v2", kv_dtype=a.kv, seed=l, gemm="auto") for l in range(a.layers)]
    cs = [alloc_cache(nblk, page, a.kv, seed=l) for l in range(a.layers)]
    caches, scales = [c[0] for c in cs], [c[1] for c in cs]
    T, H = st.T, cfg.n_heads
    hidden = torch.randn(T, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    q_abs_buf = torch.zeros(T, H, 512, device="cuda", dtype=torch.bfloat16)
    q_full_buf = torch.zeros(T, H, 576, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()

    def fwd(layer, h, kv_cache, kv_scale, variant):
        kind, pf = variant[0], variant.endswith("_pf")
        cur = torch.cuda.current_stream()
        kvv = kv_cache.view(nblk, page, -1)
        qkv = _sg.skinny_linear(h, layer.w_qkv_a)
        nq = H * cfg.q_head
        q = qkv[:, :nq].view(T, H, cfg.q_head)
        kv_c, k_pe = qkv[:, nq:nq + cfg.kv_lora], qkv[:, nq + cfg.kv_lora:]
        pro = lambda qa: mla_decode_prologue(qa, q[..., cfg.qk_nope:], kv_c, k_pe, st.cos_sin, st.block_table, st.cache_seqlens,
                                             st.s_q, kv_cache, kv_scale=kv_scale, kv_norm_weight=layer.kv_norm_w, eps=cfg.rms_eps,
                                             kv_dtype=layer.kv_dtype, q_out=q_full_buf)
        P = lambda: ext.pf(st.meta, st.block_table, kvv, 2 if pf else 0, a.ntok, 0)
        if kind == "s":
            q_full = pro(_sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf))
        elif kind == "A":
            _sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                P()
            q_full = pro(q_abs_buf)
            cur.wait_stream(side)
        elif kind == "B":
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                P()
            q_full = pro(_sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf))
            cur.wait_stream(side)
        else:  # F: prologue (+ P) on the side branch, W_UK on the main one
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                pro(q_abs_buf)
                if pf:
                    P()
            _sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf)
            cur.wait_stream(side)
            q_full = q_full_buf
        o = layer._attend(q_full, kv_cache, kv_scale, st)
        y = torch.empty(T, H * cfg.v_head, dtype=o.dtype, device=o.device)
        _sg.skinny_bmm(o, layer.w_uv_t, out=y.view(T, H, cfg.v_head))
        return _sg.skinny_linear(y, layer.w_o)

    def step(variant):
        st.refresh(page)
        h = hidden
        for l, layer in enumerate(layers):
            h = fwd(layer, h, caches[l], scales[l], variant)
        return h

    def run(variant):
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                step(variant)
                step(variant)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = step(variant)
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

    print(f"b={a.b} s_k={a.sk} kv={a.kv} layers={a.layers} ntok={a.ntok}")
    ref = None
    for v in ["serial", "A_noop", "A_pf", "B_noop", "B_pf", "F", "F_pf", "serial", "A_noop", "A_pf", "B_noop", "B_pf", "F", "F_pf"]:
        t, o = run(v)
        if ref is None:
            ref = o
        print(f"  {v:<7} {t:7.2f} us/layer  bitwise_equal={torch.equal(o, ref)}", flush=True)


if __name__ == "__main__":
    main()
