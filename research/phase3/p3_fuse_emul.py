"""Phase-3 research: what a W_UK + prologue horizontal fusion could save (timing emulation, NOT a correct layer).

27-layer fused graph (MLADecodeLayer, skinny GEMM, v2), forward re-implemented with three variants of the W_UK / prologue pair:
  serial      as shipped: W_UK bmm -> prologue (prologue copies q_abs into q_full)
  concurrent  W_UK bmm and prologue on two graph branches with no dependency (the prologue reads a stale q_abs buffer): both
              kernels run side by side, as the CTAs of one fused kernel would (one extra launch remains)
  no_prologue prologue removed (q_full stale; the cache write is skipped): upper bound of the saving
Only times are meaningful for concurrent / no_prologue.
usage: python p3_fuse_emul.py --b 1 --sk 4096 [--kv fp8]
"""
import argparse
import statistics as stt

import torch

from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state
from flash_mla import skinny_gemm as _sg
from flash_mla.mla_prologue import mla_decode_prologue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--kv", default="native", choices=["native", "fp8"])
    ap.add_argument("--layers", type=int, default=27)
    a = ap.parse_args()
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
        qkv = _sg.skinny_linear(h, layer.w_qkv_a)
        nq = H * cfg.q_head
        q = qkv[:, :nq].view(T, H, cfg.q_head)
        kv_c, k_pe = qkv[:, nq:nq + cfg.kv_lora], qkv[:, nq + cfg.kv_lora:]
        pro = lambda qa: mla_decode_prologue(qa, q[..., cfg.qk_nope:], kv_c, k_pe, st.cos_sin, st.block_table, st.cache_seqlens,
                                             st.s_q, kv_cache, kv_scale=kv_scale, kv_norm_weight=layer.kv_norm_w, eps=cfg.rms_eps,
                                             kv_dtype=layer.kv_dtype, q_out=q_full_buf)
        if variant == "serial":
            q_abs = _sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t)
            q_full = pro(q_abs)
        elif variant == "concurrent":
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                pro(q_abs_buf)
            _sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf)
            cur.wait_stream(side)
            q_full = q_full_buf
        else:
            _sg.skinny_bmm(q[..., :cfg.qk_nope], layer.w_uk_t, out=q_abs_buf)
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
                step(variant)
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
        del g
        return stt.median(ts)

    print(f"b={a.b} s_k={a.sk} kv={a.kv} layers={a.layers}")
    for v in ["serial", "concurrent", "no_prologue", "serial", "concurrent", "no_prologue"]:
        print(f"  {v:<12} {run(v):7.2f} us/layer", flush=True)


if __name__ == "__main__":
    main()
