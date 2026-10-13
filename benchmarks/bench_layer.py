#!/usr/bin/env python
"""Single MLA decode-layer benchmark (DeepSeek-V2-Lite shapes): unfused PyTorch vs fused prologue, eager vs CUDA Graph.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> \\
    python benchmarks/bench_layer.py --b 1,8,32,64 --sk 4096 --attn legacy --page 32 [--layers 27] [--kv native]

Rows per (variant, mode):
    unfused/eager      HF-style PyTorch ops around the attention kernel, python launch per op
    fused/eager        fused prologue (1 kernel) + merged q/kv_a GEMM, still one python launch per op
    {unfused,fused}/graph   whole step (metadata + N chained layers, each with its own weights and KV cache) in one CUDA graph
time is per layer (graph: total / N).  `kernels` = GPU kernels launched by one layer (torch.profiler).
Profile with nsys:  tools/prof_nsys.sh layer_fused_graph -- python benchmarks/bench_layer.py --b 8 --sk 4096 --variants fused --modes graph --nvtx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import (CsvOut, cooldown, cos_diff, default_out, enable_cooldown, flush_buffer, parse_int_list, smi_sample, time_events,
                    time_graph_replay)
from flash_mla.mla_decode_layer import MLAConfig, MLADecodeLayer, alloc_cache, capture_layer_graph, make_state


def count_kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default="1,8,32,64")
    ap.add_argument("--sk", default="4096")
    ap.add_argument("--sq", type=int, default=1)
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--attn", default="v2", help="v2 | legacy | pkg | triton | torch")
    ap.add_argument("--kv", default="native", choices=["native", "fp8", "int8"])
    ap.add_argument("--variants", default="unfused,fused")
    ap.add_argument("--modes", default="eager,graph")
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--mem-gb", type=float, default=14.0)
    ap.add_argument("--nvtx", action="store_true", help="run a short marked loop for nsys (decode_step ranges)")
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--cool", action="store_true", help="wait for the GPU to cool (FLASHMLA_COOL_HI/LO) before every variant")
    ap.add_argument("--tag", default="layer")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    enable_cooldown(a.cool)
    cfg = MLAConfig()
    out = CsvOut(a.out or default_out(a.tag))
    print(f"[layer] torch {torch.__version__}; {torch.cuda.get_device_name(0)}; attn={a.attn} page={a.page} kv={a.kv}; out={out.path}", flush=True)
    fb = flush_buffer()
    for b in parse_int_list(a.b):
        for s_k in parse_int_list(a.sk):
            T = b * a.sq
            max_blocks = -(-s_k // a.page)
            nblk = b * max_blocks
            cache_bytes = nblk * a.page * (1152 if a.kv == "native" else 644)
            n_layers = int(max(1, min(a.layers, a.mem_gb * 1e9 // (cache_bytes + 28e6))))
            hidden0 = torch.randn(T, cfg.hidden, device="cuda", dtype=torch.bfloat16)
            results = {}
            outputs = {}
            for variant in a.variants.split(","):
                fused = variant == "fused"
                cooldown()
                st, _ = make_state(cfg, b, s_k, a.page, a.sq, attn=a.attn, kv_dtype=a.kv)
                layers = [MLADecodeLayer(cfg, fused=fused, attn=a.attn, kv_dtype=a.kv, seed=l) for l in range(n_layers)]
                cs = [alloc_cache(nblk, a.page, a.kv, seed=l) for l in range(n_layers)]
                caches, scales = [c[0] for c in cs], [c[1] for c in cs]
                hidden = hidden0.clone()
                base = dict(h_q=cfg.n_heads, s_q=a.sq, b=b, s_k=s_k, page=a.page, dtype="bfloat16", kv=a.kv,
                            impl=f"{variant}/{a.attn}", extra=f"layers={n_layers}")
                with torch.no_grad():
                    y = layers[0](hidden, caches[0], st, scales[0])
                    torch.cuda.synchronize()
                    outputs[variant] = y.float().clone()
                    for mode in a.modes.split(","):
                        if mode == "eager":
                            fn = lambda: layers[0](hidden, caches[0], st, scales[0])
                            kern = count_kernels(fn)
                            r = time_events(fn)
                            per_layer, nl = r["median"], 1
                            p10, p90, reps = r["p10"], r["p90"], r["reps"]
                        else:
                            g, o = capture_layer_graph(layers, caches, scales, st, hidden)
                            r = time_graph_replay(g, n_layers)
                            per_layer, nl = r["median"], n_layers
                            p10, p90, reps = r["p10"], r["p90"], r["reps"]
                            kern = ""
                            if a.nvtx:
                                for _ in range(3):
                                    g.replay()
                                torch.cuda.synchronize()
                                for _ in range(8):
                                    fb.zero_()
                                    torch.cuda.nvtx.range_push("decode_step")
                                    g.replay()
                                    torch.cuda.nvtx.range_pop()
                                torch.cuda.synchronize()
                            del g, o
                        smi = smi_sample() if not a.no_smi else None
                        results[(variant, mode)] = per_layer
                        out.write(mode=mode, n_layers=nl, time_us=per_layer, p10_us=p10, p90_us=p90, reps=reps,
                                  sm_mhz=smi[0] if smi else "", power_w=smi[1] if smi else "", status="ok",
                                  s_k_total=b * s_k, cos_out="", cos_lse="", **{**base, "extra": base["extra"] + f" kernels={kern}"})
                        print(f"b={b:3d} s_k={s_k:5d} {variant:8s} {mode:6s} {per_layer:9.1f} us/layer"
                              + (f"  kernels/layer={kern}" if kern != "" else "") + f"  (layers={nl})", flush=True)
                del layers, caches, scales
                torch.cuda.empty_cache()
            if "unfused" in outputs and "fused" in outputs:
                cd = cos_diff(outputs["fused"], outputs["unfused"])
                print(f"   fused vs unfused output cos_diff = {cd:.2e}", flush=True)
            for mode in a.modes.split(","):
                if ("unfused", mode) in results and ("fused", mode) in results:
                    print(f"   {mode}: fused speedup over unfused = {results[('unfused', mode)] / results[('fused', mode)]:.2f}x", flush=True)
            if "graph" in a.modes and "eager" in a.modes and ("fused", "graph") in results:
                print(f"   fused graph vs unfused eager = {results[('unfused', 'eager')] / results[('fused', 'graph')]:.2f}x", flush=True)
    out.close()
    print(f"[layer] done -> {out.path}")


if __name__ == "__main__":
    main()
