#!/usr/bin/env python
"""End-to-end single-layer accuracy of the quantized KV cache (FP8 e4m3 / INT8, per-token scale) vs the bf16 cache.

For every (regime, context length, sharpness, seed): build a context of N tokens by pushing synthetic hidden states through the
layer's own kv_a_proj + RMSNorm + RoPE (so the cache holds real MLA-shaped latents, not iid noise), then run one decode token
through the full layer (q_proj -> absorption -> fused prologue [quantize-on-write for the new token] -> attention -> W_UV ->
o_proj) with
    reference : bf16 cache,                         fp32 reference attention
    fp8/int8  : quantized cache (context quantized with flash_mla.kv_quant, new token by the CUDA write path),
                fp32 attention over the dequantized cache  (= what an exact in-kernel dequant computes)
and report cos_diff = 1 - 2<x,y>/(|x|^2+|y|^2) and the relative L2 error of (a) the latent attention output o and (b) the layer
output. No real DeepSeek weights are available on the server, so the statistics are synthetic; the regimes are chosen to be
adversarial for per-token scaling (massive-activation channels, heavy-tailed per-channel latent scales).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> python benchmarks/eval_kv_quant.py --ctx 1024,4096,16384
"""
import argparse
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from flash_mla.kv_quant import bytes_per_token, dequantize_kv, quantize_kv
from flash_mla.mla_decode_layer import MLAConfig, MLADecodeLayer, make_state
from flash_mla.mla_prologue import apply_rope_ref, rmsnorm_ref


def cos_diff(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-30)


def rel_l2(x, y):  # error of x relative to the reference y
    x, y = x.double().flatten(), y.double().flatten()
    return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()


def hidden_states(n, d, regime, gen):
    h = torch.randn(n, d, device="cuda", generator=gen)
    if regime in ("outlier", "heavy"):  # LLM-style massive activations: a few channels 30x larger, token-dependent magnitude
        ch = torch.randperm(d, device="cuda", generator=gen)[:4]
        h[:, ch] *= 30.0
        h *= torch.exp(0.5 * torch.randn(n, 1, device="cuda", generator=gen))
    return h.to(torch.bfloat16)


def run_one(regime, n_ctx, sharp, seed, kinds, page=32, b=2, attn="torch"):
    cfg = MLAConfig()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    layer = MLADecodeLayer(cfg, fused=True, merge_proj=False, attn=attn, seed=1000 + seed)
    layer.w_q = (layer.w_q.float() * sharp).to(torch.bfloat16)  # sharper attention = fewer tokens carry the mass
    if regime == "heavy":  # heavy-tailed per-channel latent scales (log-normal rows of kv_a_proj)
        sc = torch.exp(0.8 * torch.randn(layer.w_kv_a.shape[0], 1, device="cuda", generator=gen))
        layer.w_kv_a = (layer.w_kv_a.float() * sc).to(torch.bfloat16)
    st, nblk = make_state(cfg, b, n_ctx, page, 1, attn=attn, seed=seed)
    # context rows through the layer's own projections
    rows = []
    for i in range(b):
        h = hidden_states(n_ctx, cfg.hidden, regime, gen)
        kv_a = F.linear(h, layer.w_kv_a)
        kv_c = rmsnorm_ref(kv_a[:, :cfg.kv_lora], layer.kv_norm_w, cfg.rms_eps)
        k_pe = apply_rope_ref(kv_a[:, cfg.kv_lora:], st.cos_sin, torch.arange(n_ctx, device="cuda"), False).to(torch.bfloat16)
        rows.append(torch.cat([kv_c, k_pe], -1))
    flat = torch.zeros(nblk * page, 576, device="cuda", dtype=torch.bfloat16)
    for i in range(b):
        slots = st.block_table[i].long().repeat_interleave(page) * page + torch.arange(page, device="cuda").repeat(st.block_table.shape[1])
        flat[slots[:n_ctx]] = rows[i]
    h_new = hidden_states(b, cfg.hidden, regime, gen)
    out = {}
    res = {}
    with torch.no_grad():
        # reference: bf16 cache
        layer.kv_dtype = "native"
        cache = flat.view(nblk, page, 1, 576).clone()
        y_ref = layer(h_new, cache, st, None).float()
        o_ref = layer.last_o.float()
        for kind in kinds:
            layer.kv_dtype = kind
            kvq, sc = quantize_kv(flat.view(nblk, page, 576), kind)
            y_q = layer(h_new, kvq, st, sc).float()  # new token is quantized by the CUDA write path
            o_q = layer.last_o.float()
            out[kind] = dict(cos_o=cos_diff(o_q, o_ref), rel_o=rel_l2(o_q, o_ref), cos_y=cos_diff(y_q, y_ref), rel_y=rel_l2(y_q, y_ref))
        # attention sharpness diagnostic: effective number of attended tokens (exp(entropy)) of head 0 for the new token
        q_full = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="1024,4096,16384")
    ap.add_argument("--regimes", default="gauss,outlier,heavy")
    ap.add_argument("--sharp", default="1,4", help="multipliers of q_proj (higher = peakier attention)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", default="fp8,int8")
    ap.add_argument("--page", type=int, default=32)
    ap.add_argument("--md", default=None, help="write the table to this markdown file")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    lines = ["| regime | ctx | q sharpness | kind | cos_diff(o) mean | cos_diff(o) max | rel L2(o) mean | cos_diff(layer out) mean | cos_diff(layer out) max | rel L2(layer out) mean |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for regime in a.regimes.split(","):
        for ctx in map(int, a.ctx.split(",")):
            for sharp in map(float, a.sharp.split(",")):
                per = {k: [] for k in kinds}
                for seed in range(a.seeds):
                    r = run_one(regime, ctx, sharp, seed, kinds, a.page)
                    for k in kinds:
                        per[k].append(r[k])
                for k in kinds:
                    m = lambda key: statistics.mean(x[key] for x in per[k])
                    mx = lambda key: max(x[key] for x in per[k])
                    line = (f"| {regime} | {ctx} | {sharp:g} | {k} | {m('cos_o'):.2e} | {mx('cos_o'):.2e} | {m('rel_o'):.2e} | "
                            f"{m('cos_y'):.2e} | {mx('cos_y'):.2e} | {m('rel_y'):.2e} |")
                    lines.append(line)
                    print(line, flush=True)
    if a.md:
        open(a.md, "w").write("\n".join(lines) + "\n")
    print(f"bytes/token: bf16 {bytes_per_token('native')} B, fp8/int8 {bytes_per_token('fp8')} B ({bytes_per_token('fp8') / bytes_per_token('native'):.3f}x)")


if __name__ == "__main__":
    main()
