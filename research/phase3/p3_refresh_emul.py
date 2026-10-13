"""Phase-3 research: per-step bookkeeping in the fused path.

DecodeState.refresh computes token positions and cache slots with ~13 small torch kernels every step; the fused layer does not
read them (the prologue kernel derives the slot from seqlens + block table).  Variants of the 27-layer fused graph:
  orig   refresh as shipped
  lean   refresh computes only the scheduler metadata
Outputs are compared bitwise.
usage: python p3_refresh_emul.py --b 1 --sk 4096 [--kv fp8]
"""
import argparse
import statistics as stt

import torch

import flash_mla
from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state, capture_layer_graph


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
    hidden = torch.randn(a.b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    orig_refresh = st.refresh
    hint = {"kv_dtype": a.kv} if a.kv != "native" else {}

    def lean_refresh(page_):
        st.page = page_
        st.meta, st.nsplit = flash_mla.get_mla_metadata(st.cache_seqlens, st.s_q * cfg.n_heads, 1, **hint)

    def run(lean):
        st.refresh = lean_refresh if lean else orig_refresh
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
            ts.append(e0.elapsed_time(e1) * 1000 / 20)
        res = out.clone()
        del g
        return stt.median(ts), res

    print(f"b={a.b} s_k={a.sk} kv={a.kv} layers={a.layers}")
    ref = None
    for name, lean in [("orig", False), ("lean", True), ("orig", False), ("lean", True)]:
        t, o = run(lean)
        if ref is None:
            ref = o
        print(f"  {name:<5} {t:9.1f} us/step  {t / a.layers:7.2f} us/layer  bitwise_equal={torch.equal(o, ref)}", flush=True)


if __name__ == "__main__":
    main()
