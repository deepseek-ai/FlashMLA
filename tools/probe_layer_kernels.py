#!/usr/bin/env python
"""One DeepSeek-V2-Lite decode-layer forward per NVTX range "probe", for per-kernel ncu / nsys measurement.

    tools/prof_ncu.sh <tag> --nvtx --nvtx-include "probe/" -- python tools/probe_layer_kernels.py --b 8
    (NCU_KERNEL='.*' NCU_SKIP=0 NCU_COUNT=<kernels per layer> to profile every kernel of exactly one layer)
"""
import argparse

import torch

from flash_mla import MLAConfig, MLADecodeLayer, alloc_cache, make_state

ap = argparse.ArgumentParser()
ap.add_argument("--b", type=int, default=8)
ap.add_argument("--sk", type=int, default=4096)
ap.add_argument("--sq", type=int, default=1)
ap.add_argument("--page", type=int, default=64)
ap.add_argument("--attn", default="v2")
ap.add_argument("--kv", default="native", choices=["native", "fp8", "int8"])
ap.add_argument("--unfused", action="store_true")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--reps", type=int, default=1)
a = ap.parse_args()

cfg = MLAConfig()
st, nblk = make_state(cfg, a.b, a.sk, a.page, a.sq, attn=a.attn, kv_dtype=a.kv)
layer = MLADecodeLayer(cfg, fused=not a.unfused, attn=a.attn, kv_dtype=a.kv, seed=0)
cache, scale = alloc_cache(nblk, a.page, a.kv, seed=0)
hidden = torch.randn(a.b * a.sq, cfg.hidden, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    for _ in range(a.warmup):
        layer(hidden, cache, st, scale)
    torch.cuda.synchronize()
    for _ in range(a.reps):
        torch.cuda.nvtx.range_push("probe")
        layer(hidden, cache, st, scale)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
print("probe done", a)
