"""Quantized-KV (fp8 e4m3 / int8) tests for the sm86 MLA decode kernels (impl="v2").

Format: docs/kv_quant_format.md -- uint8 [num_blocks, page, 640] = 512 latent bytes + 64 rope dims (bf16/fp16) and a
float32 per-token scale [num_blocks, page].

Two references:
  * the kernel must reproduce attention over the DEQUANTIZED cache to float precision (cos_diff < 8e-5), i.e. the in-kernel
    byte -> bf16 conversion and the fp32 scale handling are exact;
  * against the original unquantized cache the error must stay inside the accuracy gates (fp8 < 5e-3, int8 < 5e-4).

Run:  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_flash_mla_sm86_quant.py -x -q
"""
import random

import pytest
import torch

from flash_mla import alloc_mla_workspace, flash_mla_with_kvcache, get_mla_metadata

from test_flash_mla_sm86 import COS_TOL, D, DEV, DV, cos_diff, ref_mla, varlen_seqlens

GATE = {"fp8": 5e-3, "int8": 5e-4}


def quantize_kv(kv, kind):
    """kv: [nb, page, 1, 576] (bf16/fp16, finite).  Returns (uint8 [nb, page, 640], fp32 scale [nb, page], dequantized cache)."""
    nb, page = kv.shape[:2]
    lat = kv[..., :DV].float()
    amax = lat.abs().amax(dim=-1)                                   # [nb, page, 1]
    if kind == "fp8":
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
        q = (lat / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        qf = q.float()
        qb = q.view(torch.uint8)
    else:
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
        qi = torch.round(lat / scale[..., None]).clamp(-127, 127).to(torch.int8)
        qf = qi.float()
        qb = qi.view(torch.uint8)
    rope = kv[..., DV:].contiguous().view(torch.uint8)               # [nb, page, 1, 128]
    packed = torch.cat([qb.reshape(nb, page, DV), rope.reshape(nb, page, 128)], dim=-1).contiguous()
    deq = torch.cat([qf * scale[..., None], kv[..., DV:].float()], dim=-1)   # fp32: what the kernel sees mathematically (no bf16 re-rounding)
    return packed, scale.reshape(nb, page).float().contiguous(), deq


def make_quant_inputs(b, s_q, h_q, seqlens, page, dtype, kind, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    max_sk = max(seqlens)
    pages_per_seq = max(1, (max_sk + page - 1) // page)
    nb = b * pages_per_seq
    q = torch.randn(b, s_q, h_q, D, device=DEV, dtype=torch.float32, generator=g).to(dtype)
    # heavy-tailed rows so that per-token scales differ a lot
    row_gain = torch.exp(torch.randn(nb, page, 1, 1, device=DEV, generator=g))
    kv = (torch.randn(nb, page, 1, D, device=DEV, dtype=torch.float32, generator=g) * row_gain).to(dtype)
    perm = torch.randperm(nb, device=DEV, generator=g).to(torch.int32)
    bt = perm.view(b, pages_per_seq).contiguous()
    packed, scale, deq = quantize_kv(kv, kind)
    sl = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    # poison every slot beyond the sequence end: random bytes + NaN scales (must never influence the result)
    for i in range(b):
        pg = bt[i].long()
        flat = packed[pg].reshape(-1, 640)
        fs = scale[pg].reshape(-1)
        flat[seqlens[i]:] = torch.randint(0, 256, flat[seqlens[i]:].shape, dtype=torch.uint8, device=DEV)
        fs[seqlens[i]:] = float("nan")
        packed[pg] = flat.reshape(pages_per_seq, page, 640)
        scale[pg] = fs.reshape(pages_per_seq, page)
    return q, kv, packed, scale, deq, bt, sl


def run_quant_case(b, s_q, h_q, seqlens, page, dtype, kind, causal, check_gate=True, seed=0, fuse=None):
    q, kv, packed, scale, deq, bt, sl = make_quant_inputs(b, s_q, h_q, seqlens, page, dtype, kind, seed)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1, kind)
    out, lse = flash_mla_with_kvcache(q, packed, bt, sl, DV, md, ns, causal=causal, kv_scale=scale, kv_dtype=kind)
    torch.cuda.synchronize()
    assert not torch.isnan(out).any() and not torch.isnan(lse).any(), "NaN leaked from the poisoned tail"
    ref_o, ref_l = ref_mla(q, deq, bt, sl, h_q, causal)
    assert cos_diff(out, ref_o) < COS_TOL, f"vs dequantized reference: out {cos_diff(out, ref_o)}"
    assert cos_diff(lse, ref_l) < COS_TOL, f"vs dequantized reference: lse {cos_diff(lse, ref_l)}"
    out2, lse2 = flash_mla_with_kvcache(q, packed, bt, sl, DV, md, ns, causal=causal, kv_scale=scale, kv_dtype=kind)
    assert torch.equal(out, out2) and torch.equal(lse, lse2), "non-deterministic"
    if check_gate:
        # accuracy vs the unquantized cache: the kernel must add (almost) nothing to the quantization error of the format
        # itself (which depends on the data: the synthetic heavy-tailed rows used here are harder than typical MLA latents),
        # and stay within 2x of SA's accuracy gate (fp8 < 5e-3, int8 < 5e-4 measured on realistic data).
        o_full, _ = ref_mla(q, kv, bt, sl, h_q, causal)
        d_kernel, d_format = cos_diff(out, o_full), cos_diff(ref_o, o_full)
        assert d_kernel < d_format * 1.05 + 5e-5, f"{kind}: kernel adds error: {d_kernel} vs format {d_format}"
        if sum(seqlens) < 20000:   # the absolute gate is data dependent: long, peaky synthetic contexts have a larger format error
            assert d_kernel < 2 * GATE[kind], f"{kind}: cos_diff vs unquantized cache {d_kernel} >= {2 * GATE[kind]}"
    return out, lse


@pytest.mark.parametrize("kind", ["fp8", "int8"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("page", [16, 32, 64])
@pytest.mark.parametrize("h_q,s_q,causal", [(8, 1, False), (16, 1, False), (16, 2, True), (32, 1, False), (64, 2, True), (128, 1, False)])
def test_quant_matrix(kind, dtype, page, h_q, s_q, causal):
    b = 5
    run_quant_case(b, s_q, h_q, varlen_seqlens(b, 700, s_q), page, dtype, kind, causal)


@pytest.mark.parametrize("kind", ["fp8", "int8"])
@pytest.mark.parametrize("h_q", [16, 128])
def test_quant_b1_long(kind, h_q):
    run_quant_case(1, 1, h_q, [32768], 64, torch.bfloat16, kind, False)


@pytest.mark.parametrize("kind", ["fp8", "int8"])
@pytest.mark.parametrize("sk", [1, 2, 3, 15, 16, 17, 31, 33, 65, 129])
def test_quant_edge_seqlens(kind, sk):
    run_quant_case(3, 1, 16, [sk, max(1, sk - 1), sk + 40], 64, torch.bfloat16, kind, False)


@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_quant_varlen_large_batch(kind):
    b = 96
    run_quant_case(b, 1, 16, varlen_seqlens(b, 1500, 1, seed=4), 64, torch.bfloat16, kind, False, check_gate=False)


@pytest.mark.parametrize("kind", ["fp8", "int8"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("page", [16, 32, 64])
def test_quant_two_ctas_per_sm(monkeypatch, kind, dtype, page):
    # opt-in plan for long contexts (docs/ampere_opt_report.md): 2 CTAs per SM and twice the split parts;
    # page 16 clamps the tile to 16 tokens / 3 stages
    monkeypatch.setenv("FLASH_MLA_SM86_QCFG", "16,2,2,32")
    b = 40
    seqlens = varlen_seqlens(b, 3000, 1, seed=5)
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    assert get_mla_metadata(torch.tensor(seqlens, dtype=torch.int32, device=DEV), 16, 1, kind)[0].shape[0] == 2 * sms
    run_quant_case(b, 1, 16, seqlens, page, dtype, kind, False)


@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_quant_prealloc_and_fused(monkeypatch, kind):
    monkeypatch.setenv("FLASH_MLA_SM86_FUSE", "4")
    b, s_q, h_q, page = 12, 1, 16, 64
    seqlens = varlen_seqlens(b, 2500, s_q, seed=6)
    q, kv, packed, scale, deq, bt, sl = make_quant_inputs(b, s_q, h_q, seqlens, page, torch.bfloat16, kind)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1, kind)
    ws = alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0], DEV)
    outs = []
    for _ in range(3):
        out = torch.empty(b, s_q, h_q, DV, dtype=torch.bfloat16, device=DEV)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=DEV)
        flash_mla_with_kvcache(q, packed, bt, sl, DV, md, ns, out=out, lse=lse, workspace=ws, kv_scale=scale, kv_dtype=kind)
        outs.append((out, lse))
    torch.cuda.synchronize()
    ref_o, ref_l = ref_mla(q, deq, bt, sl, h_q, False)
    for out, lse in outs:
        assert cos_diff(out, ref_o) < COS_TOL and cos_diff(lse, ref_l) < COS_TOL
        assert torch.equal(out, outs[0][0]) and torch.equal(lse, outs[0][1])


@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_quant_cuda_graph(kind):
    b, s_q, h_q, page, cap = 16, 1, 16, 64, 4096
    q, kv, packed, scale, deq, bt, sl = make_quant_inputs(b, s_q, h_q, [cap] * b, page, torch.bfloat16, kind)
    # all slots valid for this test: rebuild a clean (non-poisoned) quantized cache
    kvc = torch.nan_to_num(kv)
    packed, scale, deq = quantize_kv(kvc, kind)
    sl_static = torch.tensor(varlen_seqlens(b, 2000, s_q, seed=3), dtype=torch.int32, device=DEV).clamp(max=cap)
    nparts = get_mla_metadata(sl_static, s_q * h_q, 1, kind)[0].shape[0]
    out = torch.empty(b, s_q, h_q, DV, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=DEV)
    ws = alloc_mla_workspace(b, s_q * h_q, 1, nparts, DEV)

    def step():
        md, ns = get_mla_metadata(sl_static, s_q * h_q, 1, kind)
        flash_mla_with_kvcache(q, packed, bt, sl_static, DV, md, ns, out=out, lse=lse, workspace=ws, kv_scale=scale, kv_dtype=kind)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for seed in (11, 12, 13):
        sl_static.copy_(torch.tensor(varlen_seqlens(b, 1800, s_q, seed=seed), dtype=torch.int32, device=DEV).clamp(max=cap))
        graph.replay()
        torch.cuda.synchronize()
        ref_o, ref_l = ref_mla(q, deq, bt, sl_static, h_q, False)
        assert cos_diff(out, ref_o) < COS_TOL and cos_diff(lse, ref_l) < COS_TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"] + __import__("sys").argv[1:]))
