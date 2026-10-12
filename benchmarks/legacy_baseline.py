"""Benchmark the legacy (A100-oriented) sm80 MLA decode kernels on whatever GPU CUDA_VISIBLE_DEVICES selects.

Usage: CUDA_VISIBLE_DEVICES=1 python benchmarks/legacy_baseline.py [--out docs/legacy_baseline.md] [--ws 0|1|both]
Reports ms, GB/s (KV + Q + O bytes), TFLOPS (2*s_q*h_q*sum(s_k)*(d+dv)) per shape, plus cos_diff vs a fp32 reference
for a small subset.  page_block_size is fixed to 32 (legacy kernel requirement).
"""
import argparse, math, sys

import torch
import triton

from flash_mla import get_mla_metadata, flash_mla_with_kvcache

D, DV, PAGE = 576, 512, 32


def ref_attn(q, k, v, h_q, h_kv):
    q, k, v = q.float(), k.float(), v.float()
    k = k.repeat_interleave(h_q // h_kv, dim=0); v = v.repeat_interleave(h_q // h_kv, dim=0)
    w = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    return torch.softmax(w, dim=-1) @ v, w.logsumexp(dim=-1)


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


@torch.inference_mode()
def run_case(b, s_q, s_k, h_q, warp_spec, check):
    dev = torch.device("cuda")
    h_kv = 1
    seqlens = torch.full((b,), s_k, dtype=torch.int32, device=dev)
    pad = triton.cdiv(s_k, 256) * 256
    q = torch.randn(b, s_q, h_q, D, device=dev, dtype=torch.bfloat16)
    nblk = b * pad // PAGE
    block_table = torch.arange(nblk, dtype=torch.int32, device=dev).view(b, pad // PAGE)
    kc = torch.randn(nblk, PAGE, h_kv, D, device=dev, dtype=torch.bfloat16)
    meta, nsplit = get_mla_metadata(seqlens, s_q * h_q // h_kv, h_kv)

    def f():
        return flash_mla_with_kvcache(q, kc, block_table, seqlens, DV, meta, nsplit, causal=False, warp_spec=warp_spec)

    out, lse = f()
    cd = None
    if check:
        bb = min(b, 2)
        cds = []
        for i in range(bb):
            kk = kc.view(b, pad, h_kv, D)[i, :s_k].transpose(0, 1)
            O, L = ref_attn(q[i].transpose(0, 1), kk, kk[..., :DV], h_q, h_kv)
            cds.append(cos_diff(out[i].transpose(0, 1), O))
        cd = max(cds)
    t = triton.testing.do_bench(f, warmup=25, rep=100) * 1e-3  # seconds, L2 flushed by do_bench
    flops = 2 * s_q * h_q * b * s_k * (D + DV)
    nbytes = (b * s_k * h_kv * D + b * s_q * h_q * D + b * s_q * h_q * DV) * 2
    return t, nbytes / t / 1e9, flops / t / 1e12, cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0", choices=["0", "1", "both"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--hq", default="16,128"); ap.add_argument("--b", default="1,64,128"); ap.add_argument("--sk", default="1024,4096,8192")
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = []
    for ws in ([False, True] if a.ws == "both" else [a.ws == "1"]):
        for h_q in map(int, a.hq.split(",")):
            for b in map(int, a.b.split(",")):
                for s_k in map(int, a.sk.split(",")):
                    try:
                        t, gbs, tf, cd = run_case(b, 1, s_k, h_q, ws, check=(b == 1 or (b == 64 and s_k == 1024)))
                        rows.append((ws, h_q, b, s_k, t * 1e6, gbs, tf, cd))
                        print(f"ws={int(ws)} h_q={h_q:3d} b={b:3d} s_k={s_k:5d}  {t*1e6:9.1f} us  {gbs:7.1f} GB/s  {tf:6.2f} TFLOPS  cos_diff={cd}", flush=True)
                    except Exception as ex:  # keep going, record failures
                        print(f"ws={int(ws)} h_q={h_q} b={b} s_k={s_k} FAILED: {ex}", flush=True)
                        rows.append((ws, h_q, b, s_k, None, None, None, str(ex)[:80]))
    if a.out:
        with open(a.out, "w") as f:
            f.write("| kernel | h_q | b | s_k | time (us) | GB/s | TFLOPS | cos_diff |\n|---|---|---|---|---|---|---|---|\n")
            for ws, h_q, b, s_k, us, gbs, tf, cd in rows:
                k = "legacy-ws" if ws else "legacy-coop"
                if us is None:
                    f.write(f"| {k} | {h_q} | {b} | {s_k} | FAILED | | | {cd} |\n")
                else:
                    f.write(f"| {k} | {h_q} | {b} | {s_k} | {us:.1f} | {gbs:.0f} | {tf:.2f} | {'' if cd is None else format(cd, '.1e')} |\n")


if __name__ == "__main__":
    main()
