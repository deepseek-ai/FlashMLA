#!/usr/bin/env python
"""Run ONE decode-attention shape for a fixed number of iterations: the target for ncu / nsys.

    python benchmarks/bench_one.py --impl v2 --hq 16 --sq 1 --b 64 --sk 4096 --page 64 [--iters 12] [--graph] [--layers 8]

Every iteration is preceded by an L2 flush (so each launch looks like the benchmark), wrapped in an NVTX range `decode_step`.
ncu:  tools/prof_ncu.sh <tag> -- python benchmarks/bench_one.py ...        (kernel regex flash_fwd, skip 5, profile 1)
nsys: tools/prof_nsys.sh <tag> -- python benchmarks/bench_one.py ... --graph (graph nodes traced individually)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from bench_decode import capture_graph, make_impl
from common import Case, Data, flush_buffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="pkg")
    ap.add_argument("--hq", type=int, default=16); ap.add_argument("--sq", type=int, default=1)
    ap.add_argument("--b", type=int, default=64); ap.add_argument("--sk", type=int, default=4096)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--varlen", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv", default="native")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--graph", action="store_true", help="replay a CUDA graph of --layers layers instead of eager calls")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--no-flush", action="store_true")
    a = ap.parse_args()

    case = Case(a.hq, a.sq, a.b, a.sk, a.page, a.varlen, a.dtype, a.kv)
    n_layers = a.layers if a.graph else 1
    data = Data(case, n_layers, True, False, 0)
    impl = make_impl(a.impl)
    state = impl.prepare(case, data)
    fn = impl.runner(case, data, state, 0)
    fn()
    torch.cuda.synchronize()
    fb = None if a.no_flush else flush_buffer()
    if a.graph:
        g = capture_graph([impl.runner(case, data, state, l) for l in range(n_layers)])
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        for _ in range(a.iters):
            if fb is not None:
                fb.zero_()
            torch.cuda.nvtx.range_push("decode_step")
            g.replay()
            torch.cuda.nvtx.range_pop()
    else:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        for _ in range(a.iters):
            if fb is not None:
                fb.zero_()
            torch.cuda.nvtx.range_push("decode_step")
            fn()
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"bench_one done: {case.tag()} impl={a.impl} iters={a.iters} graph={a.graph}")


if __name__ == "__main__":
    main()
