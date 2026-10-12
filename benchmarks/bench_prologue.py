#!/usr/bin/env python
"""Micro-benchmark of the fused decode prologue (RoPE + q concat + paged KV write [+ RMSNorm] [+ quantize]) against the unfused
PyTorch op sequence (flash_mla.mla_prologue.mla_decode_prologue_ref) and a bare memcpy-class lower bound.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> python benchmarks/bench_prologue.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import time_events, time_graph_replay
from flash_mla.kv_quant import ROW_BYTES
from flash_mla.mla_prologue import make_rope_cache, mla_decode_prologue, mla_decode_prologue_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="1,8,32,64,256")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()
    dev, H, page = "cuda", a.H, a.page
    print(f"{'T':>5} {'variant':14s} {'eager us':>9} {'graph us':>9}   (graph = 20 back-to-back calls in one graph, per call; L2 flushed once per replay)")
    for T in map(int, a.T.split(",")):
        B = T
        nblk_per = -(-a.ctx // page)
        bt = torch.arange(B * nblk_per, device=dev, dtype=torch.int32).view(B, nblk_per)
        lens = torch.full((B,), a.ctx, dtype=torch.int32, device=dev)
        cs = make_rope_cache(a.ctx + 8, device=dev)
        q_nope = torch.randn(T, H, 512, device=dev, dtype=torch.bfloat16)
        q_pe = torch.randn(T, H, 64, device=dev, dtype=torch.bfloat16)
        kv_c = torch.randn(T, 512, device=dev, dtype=torch.bfloat16)
        k_pe = torch.randn(T, 64, device=dev, dtype=torch.bfloat16)
        nw = (torch.rand(512, device=dev) + 0.5).to(torch.bfloat16)
        cache = torch.zeros(B * nblk_per, page, 1, 576, device=dev, dtype=torch.bfloat16)
        kq = torch.zeros(B * nblk_per, page, ROW_BYTES, device=dev, dtype=torch.uint8)
        sc = torch.ones(B * nblk_per, page, device=dev, dtype=torch.float32)
        q_out = torch.empty(T, H, 576, device=dev, dtype=torch.bfloat16)
        base = dict(q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe, cos_sin=cs, block_table=bt, cache_seqlens=lens, s_q=1)
        variants = {
            "torch ref": lambda: mla_decode_prologue_ref(**base, kv_cache=cache, kv_norm_weight=nw),
            "fused bf16": lambda: mla_decode_prologue(**base, kv_cache=cache, kv_norm_weight=nw, q_out=q_out),
            "fused fp8": lambda: mla_decode_prologue(**base, kv_cache=kq, kv_scale=sc, kv_norm_weight=nw, kv_dtype="fp8", q_out=q_out),
            "fused int8": lambda: mla_decode_prologue(**base, kv_cache=kq, kv_scale=sc, kv_norm_weight=nw, kv_dtype="int8", q_out=q_out),
        }
        for name, fn in variants.items():
            fn()
            torch.cuda.synchronize()
            e = time_events(fn)["median"]
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                for _ in range(20):
                    fn()
            gt = time_graph_replay(g, 20)["median"]
            print(f"{T:5d} {name:14s} {e:9.1f} {gt:9.1f}")


if __name__ == "__main__":
    main()
