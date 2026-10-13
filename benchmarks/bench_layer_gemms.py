#!/usr/bin/env python
"""Per-op timing of the four DeepSeek-V2-Lite decode-layer GEMMs (qkv_a, W_UK bmm, W_UV bmm, o_proj), several
implementations, CUDA-graph timed with a rotating set of weight copies (so weights stream from DRAM as in a real layer).

    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_layer_gemms.py --b 1,8,64 [--impls torch,torch_t,skinny] [--sweep]
--sweep adds one skinny row per (n_tile, splits) candidate ("skinny_<n_tile>x<splits>") to tune flash_mla.skinny_gemm.plan.
Output: one CSV row per (op, impl, b) in /data2/LiuJun/tmp/flashmla/logs (time_us, GB/s of weight bytes, % of 873.8).
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

from flash_mla import skinny_gemm as sg

PEAK = 873.8
H, HID, QH, NOPE, LORA, ROPE, VH = 16, 2048, 192, 128, 512, 64, 128
NQKV = H * QH + LORA + ROPE   # 3648


def make_ops(T, copies, dev="cuda", dt=torch.bfloat16, sweep=False):
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    r = lambda *s: (torch.randn(*s, generator=g, device=dev) * 0.02).to(dt)
    hidden = r(T, HID)
    qkv = r(T, NQKV)
    o = r(T, H, LORA)
    y = r(T, H * VH)
    W = dict(qkv=[r(NQKV, HID) for _ in range(copies)], uk=[r(H, NOPE, LORA) for _ in range(copies)],
             uv=[r(H, LORA, VH) for _ in range(copies)], o=[r(HID, H * VH) for _ in range(copies)])
    WT = dict(uk=[w.transpose(1, 2).contiguous() for w in W["uk"]], uv=[w.transpose(1, 2).contiguous() for w in W["uv"]])
    q = qkv[:, :H * QH].view(T, H, QH)
    q_abs = torch.empty(T, H, LORA, device=dev, dtype=dt)
    y_out = torch.empty(T, H * VH, device=dev, dtype=dt)
    ops = {}
    # name -> (weight bytes, {impl: fn(i)})
    ops["qkv_a"] = (NQKV * HID * 2, {
        "torch": lambda i: F.linear(hidden, W["qkv"][i]),
    })
    ops["w_uk"] = (H * NOPE * LORA * 2, {
        "torch": lambda i: torch.bmm(q[..., :NOPE].transpose(0, 1), W["uk"][i]).transpose(0, 1),
        "torch_t": lambda i: torch.bmm(q[..., :NOPE].transpose(0, 1), WT["uk"][i].transpose(1, 2)).transpose(0, 1),
        "torch_c": lambda i: torch.bmm(q[..., :NOPE].transpose(0, 1).contiguous(), W["uk"][i]).transpose(0, 1),
    })
    ops["w_uv"] = (H * LORA * VH * 2, {
        "torch": lambda i: torch.bmm(o.transpose(0, 1), W["uv"][i]).transpose(0, 1).reshape(T, H * VH),
        "torch_t": lambda i: torch.bmm(o.transpose(0, 1), WT["uv"][i].transpose(1, 2)).transpose(0, 1).reshape(T, H * VH),
        "torch_noreshape": lambda i: torch.bmm(o.transpose(0, 1), W["uv"][i]),
    })
    ops["o_proj"] = (HID * H * VH * 2, {
        "torch": lambda i: F.linear(y, W["o"][i]),
    })
    if sg.available():
        y3 = y_out.view(T, H, VH)
        shapes = dict(qkv_a=(1, NQKV, HID), w_uk=(H, LORA, NOPE), w_uv=(H, VH, LORA), o_proj=(1, HID, H * VH))
        calls = dict(qkv_a=lambda i, **k: sg.skinny_linear(hidden, W["qkv"][i], **k),
                     w_uk=lambda i, **k: sg.skinny_bmm(q[..., :NOPE], WT["uk"][i], out=q_abs, **k),
                     w_uv=lambda i, **k: sg.skinny_bmm(o, WT["uv"][i], out=y3, **k),
                     o_proj=lambda i, **k: sg.skinny_linear(y, W["o"][i], **k))
        for name, (G, N, K) in shapes.items():
            ops[name][1]["skinny"] = (lambda f: lambda i: f(i))(calls[name])
            if sweep:
                for nt in (16, 32, 64):
                    if N % nt:
                        continue
                    for spl in sorted({1, 2, 3, 4, 6, 8, 12, 16}):
                        if spl <= K // 64:
                            ops[name][1][f"skinny_{nt}x{spl}"] = (lambda f, a, b: lambda i: f(i, n_tile=a, splits=b))(calls[name], nt, spl)
    return ops


def time_graph(fn, copies, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(copies):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(copies):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / copies)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default="1,8,64")
    ap.add_argument("--impls", default="all")
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--tag", default="gemms")
    ap.add_argument("--sweep", action="store_true", help="time every skinny (n_tile, splits) candidate")
    a = ap.parse_args()
    out = f"/data2/LiuJun/tmp/flashmla/logs/bench_{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []
    with torch.no_grad():
        for T in [int(x) for x in a.b.split(",")]:
            for op, (wb, impls) in make_ops(T, a.copies, sweep=a.sweep).items():
                for impl, fn in impls.items():
                    if a.impls != "all" and impl not in a.impls.split(",") and not (a.sweep and impl.startswith("skinny_")):
                        continue
                    us = time_graph(fn, a.copies)
                    gbps = wb / us / 1e3
                    rows.append(dict(op=op, impl=impl, b=T, time_us=round(us, 2), weight_mb=round(wb / 1e6, 2),
                                     gbps=round(gbps, 1), pct_bw=round(100 * gbps / PEAK, 1)))
                    print(f"b={T:3d} {op:7s} {impl:16s} {us:8.2f} us  {gbps:6.1f} GB/s ({100 * gbps / PEAK:4.1f}%)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("csv:", out)


if __name__ == "__main__":
    main()
