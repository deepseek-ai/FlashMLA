"""Quantization-error / format tests of the quantized MLA KV cache (docs/kv_quant_format.md).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> python -m pytest tests/test_kv_quant.py -q
"""
import math

import pytest
import torch

from flash_mla.kv_quant import (E4M3_MAX, INT8_MAX, LATENT, ROW_BYTES, bytes_per_token, dequantize_kv, quant_error,  # noqa: E402
                                quantize_kv)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"
KINDS = ["fp8", "int8"]


def rows(n=4096, dist="gauss", seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    x = torch.randn(n, 576, device=DEV, generator=g)
    if dist == "heavy":      # outlier channels: per-channel log-normal scales (typical for MLA latents), plus Student-t-ish tails
        ch = torch.exp(1.5 * torch.randn(576, device=DEV, generator=g))
        x = x * ch
    elif dist == "laplace":
        x = torch.distributions.Laplace(0.0, 1.0).sample((n, 576)).to(DEV)
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("kind", KINDS)
def test_layout_and_scale_convention(kind):
    x = rows(64)
    kvq, sc = quantize_kv(x, kind)
    assert kvq.dtype == torch.uint8 and kvq.shape == (64, ROW_BYTES) and sc.dtype == torch.float32 and sc.shape == (64,)
    amax = x[:, :LATENT].float().abs().amax(-1)
    qmax = E4M3_MAX if kind == "fp8" else INT8_MAX
    assert torch.allclose(sc, amax / qmax, rtol=2e-7, atol=0)  # (quantize_kv uses IEEE division, python scalar division may differ by 1 ulp)
    # rope dims travel as raw bf16 bytes
    assert torch.equal(kvq[:, LATENT:].contiguous().view(torch.bfloat16), x[:, LATENT:])
    # the largest latent magnitude maps to the top code
    d = dequantize_kv(kvq, sc, kind, torch.float32)
    assert torch.allclose(d[:, :LATENT].abs().amax(-1), amax, rtol=2 ** -4 if kind == "fp8" else 1 / 127)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_row_and_tiny_values(kind):
    x = torch.zeros(3, 576, device=DEV, dtype=torch.bfloat16)
    x[1, :LATENT] = 1e-30   # far below the e4m3 subnormal range once divided by a normal scale? scale = 1e-30/448 -> still exact path
    x[2, LATENT:] = 3.0
    kvq, sc = quantize_kv(x, kind)
    assert sc[0].item() == 1.0 and (kvq[0, :LATENT] == 0).all(), "amax == 0 -> scale 1, q 0"
    assert torch.isfinite(sc).all() and (sc > 0).all()
    d = dequantize_kv(kvq, sc, kind, torch.float32)
    assert torch.allclose(d[1, :LATENT], x[1, :LATENT].float(), rtol=0.07)
    assert (d[2, LATENT:] == 3.0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_idempotent(kind):
    """quantize(dequantize(q)) reproduces q (the grid is a fixed point)."""
    x = rows(512)
    kvq, sc = quantize_kv(x, kind)
    d = dequantize_kv(kvq, sc, kind, torch.float32).to(torch.bfloat16)
    # bf16 rounding of the dequantized value can move a code by one step; compare through dequantization instead
    kvq2, sc2 = quantize_kv(d, kind)
    d2 = dequantize_kv(kvq2, sc2, kind, torch.float32)
    d1 = dequantize_kv(kvq, sc, kind, torch.float32)
    assert (d2 - d1).abs().amax(-1).max() <= 1.01 * (sc.max())


@pytest.mark.parametrize("dist", ["gauss", "heavy", "laplace"])
def test_error_bounds(dist):
    x = rows(4096, dist)
    e8, ei = quant_error(x, "fp8"), quant_error(x, "int8")
    for e in (e8, ei):
        assert e["rope_exact"]
    # e4m3: 3 mantissa bits -> relative step 2^-3, RMS rounding error per element ~ 2^-4/sqrt(3) of the element
    assert e8["rel_rms_mean"] < 0.04, e8
    assert e8["max_abs_over_amax"] <= 2 ** -4 + 1e-3 + (2 ** -9 / 448) * 16, e8   # <= half an e4m3 step at the top binade
    # int8: step amax/127, error <= half a step; RMS relative error depends on crest factor (amax/rms)
    assert ei["max_abs_over_amax"] <= 0.5 / 127 * 1.001 + 1e-6, ei
    crest = (x[:, :LATENT].float().abs().amax(-1) / x[:, :LATENT].float().pow(2).mean(-1).sqrt()).mean().item()
    assert ei["rel_rms_mean"] < crest / 127 / math.sqrt(12) * 1.3, (ei, crest)


def test_bytes_per_token():
    assert bytes_per_token("native") == 1152
    assert bytes_per_token("fp8") == bytes_per_token("int8") == 644
    assert abs(644 / 1152 - 0.559) < 1e-3


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_write_path_is_bit_exact(kind):
    """The fused prologue's quantize-on-write must produce exactly the bytes / scales of the torch reference."""
    pytest.importorskip("flash_mla_fused_cuda")
    from flash_mla.mla_prologue import mla_decode_prologue  # noqa: E402
    T, H, page = 256, 4, 64
    x = rows(T, "heavy", seed=3)
    kv_c, k_pe = x[:, :LATENT].contiguous(), x[:, LATENT:].contiguous()
    nblk = T // page + 1
    bt = torch.arange(nblk - 1, device=DEV, dtype=torch.int32).view(1, -1).repeat(1, 1)
    # one sequence, T new tokens written to positions 0..T-1 -> s_q = T
    seqlens = torch.tensor([T], dtype=torch.int32, device=DEV)
    cs = torch.cat([torch.ones(T + 1, 32), torch.zeros(T + 1, 32)], -1).to(DEV)  # identity rotation (cos=1, sin=0)
    kvq = torch.zeros(nblk, page, ROW_BYTES, dtype=torch.uint8, device=DEV)
    sc = torch.zeros(nblk, page, dtype=torch.float32, device=DEV)
    qn = torch.zeros(T, H, 512, dtype=torch.bfloat16, device=DEV)
    qp = torch.zeros(T, H, 64, dtype=torch.bfloat16, device=DEV)
    mla_decode_prologue(qn, qp, kv_c, k_pe, cs, bt, seqlens, T, kvq, kv_scale=sc, kv_dtype=kind)
    ref_q, ref_s = quantize_kv(x, kind)  # identity rotation -> rope bytes equal the input bytes, too
    assert torch.equal(kvq.view(-1, ROW_BYTES)[:T], ref_q), "bytes differ from the torch quantizer"
    assert torch.equal(sc.view(-1)[:T], ref_s), "scales differ from the torch quantizer"


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_write_path_codes_and_ties(kind):
    """Every representable code, every rounding tie between neighbouring codes and +-1 bf16 ulp around the ties."""
    pytest.importorskip("flash_mla_fused_cuda")
    from flash_mla.mla_prologue import mla_decode_prologue  # noqa: E402
    if kind == "fp8":
        codes = torch.arange(0, 127, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn).float()
        top = 448.0
    else:
        codes = torch.arange(0, 128, dtype=torch.float32, device=DEV)
        top = 127.0
    mid = (codes[:-1] + codes[1:]) / 2
    vals = torch.cat([codes, mid, mid * (1 + 2 ** -7), mid * (1 - 2 ** -7)]).to(torch.bfloat16)
    vals = torch.cat([vals, -vals]).clamp(-top, top)  # keep amax == top so that the scale stays exactly 1.0
    rows = (vals.numel() + 510) // 511
    flat = torch.zeros(rows * 511, device=DEV, dtype=torch.bfloat16)
    flat[: vals.numel()] = vals
    x = torch.zeros(rows, 576, device=DEV, dtype=torch.bfloat16)
    x[:, 0] = top  # amax == qmax -> scale exactly 1.0, so y == x
    x[:, 1:512] = flat.view(rows, 511)
    x[:, 512:] = torch.randn(rows, 64, device=DEV).to(torch.bfloat16)
    page = 64
    nblk = (rows + page - 1) // page
    cs = torch.cat([torch.ones(rows + 1, 32), torch.zeros(rows + 1, 32)], -1).to(DEV)
    bt = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, -1)
    kvq = torch.zeros(nblk, page, ROW_BYTES, dtype=torch.uint8, device=DEV)
    sc = torch.zeros(nblk, page, dtype=torch.float32, device=DEV)
    z = torch.zeros(rows, 2, 512, device=DEV, dtype=torch.bfloat16)
    zp = torch.zeros(rows, 2, 64, device=DEV, dtype=torch.bfloat16)
    mla_decode_prologue(z, zp, x[:, :512].contiguous(), x[:, 512:].contiguous(), cs, bt,
                        torch.tensor([rows], dtype=torch.int32, device=DEV), rows, kvq, kv_scale=sc, kv_dtype=kind)
    ref_q, ref_s = quantize_kv(x, kind)
    assert torch.equal(sc.view(-1)[:rows], ref_s) and (ref_s == 1.0).all()
    assert torch.equal(kvq.view(-1, ROW_BYTES)[:rows], ref_q)


# --------------------------------------------------------------------------------------------------------------------------------
# End-to-end accuracy gate (single decode layer: dequant-reference attention over the quantized cache vs the bf16 cache).
# Thresholds ~2x the worst cos_diff measured by benchmarks/eval_kv_quant.py over {gauss, outlier, heavy} x ctx {1k, 4k, 16k} x
# q sharpness {1, 4} x 3 seeds (docs/perf_log_sys.md section 6): fp8 worst 2.4e-03, int8 worst 2.6e-04.
# The in-kernel dequant of the attention kernel must stay within these same bounds against the bf16 reference.
# --------------------------------------------------------------------------------------------------------------------------------
GATE_COS_DIFF = {"fp8": 5e-3, "int8": 5e-4}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("regime,sharp", [("gauss", 4.0), ("outlier", 1.0)])
def test_end_to_end_accuracy_gate(kind, regime, sharp):
    pytest.importorskip("flash_mla_fused_cuda")
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
    from eval_kv_quant import run_one
    r = run_one(regime, 2048, sharp, 0, [kind])[kind]
    assert r["cos_o"] < GATE_COS_DIFF[kind], r
    assert r["cos_y"] < GATE_COS_DIFF[kind], r
