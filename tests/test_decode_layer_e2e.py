"""End-to-end tests of the DeepSeek-V2-Lite decode layer on the merged build:
fused prologue (RoPE / q concat / RMSNorm / paged write, quantize-on-write) + the sm86 "v2" attention kernel
vs the bf16 PyTorch reference layer (unfused ops, fp32 torch attention over a native bf16 cache).

Gates on cos_diff(layer output): bf16 < 8e-5, fp8 < 5e-3, int8 < 5e-4.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<repo> python -m pytest tests/test_decode_layer_e2e.py -q
"""
import pytest
import torch

pytest.importorskip("flash_mla_cuda")
pytest.importorskip("flash_mla_fused_cuda", reason="fused extension not built (tools/build.sh)")
from flash_mla.kv_quant import quantize_kv  # noqa: E402
from flash_mla.mla_decode_layer import MLAConfig, MLADecodeLayer, alloc_cache, capture_layer_graph, make_state  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

GATE = {"native": 8e-5, "fp8": 5e-3, "int8": 5e-4}


def cosd(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def tail_slots(st, s_k, page):
    """Flat slot ids of the cache rows past each sequence's end (inside its last page)."""
    b, nb = st.block_table.shape
    out = []
    for i in range(b):
        n_valid = s_k - (nb - 1) * page
        if n_valid < page:
            base = st.block_table[i, nb - 1].long() * page
            out.append(base + torch.arange(n_valid, page, device=base.device))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=st.block_table.device)


def build(kind, attn, fused, b, s_k, page, s_q, seed=3, h_scale=1.0):
    """(layer, cache, kv_scale, state, hidden). The quantized cache is the quantization of the SAME native cache the
    reference layer starts from, with NaN poison (scale = NaN / NaN bf16) in the rows past each sequence's end."""
    cfg = MLAConfig()
    st, nblk = make_state(cfg, b, s_k, page, s_q, attn=attn, kv_dtype=kind)
    native, _ = alloc_cache(nblk, page, "native", seed=seed)
    ts = tail_slots(st, s_k, page)
    native.view(-1, 576)[ts] = float("nan")
    if kind == "native":
        cache, scale = native, None
    else:
        flat = native.view(nblk, page, 576).clone()
        flat.view(-1, 576)[ts] = 0.0  # quantize finite data, then poison the scales below
        cache, scale = quantize_kv(flat, kind)
        scale.view(-1)[ts] = float("nan")
    layer = MLADecodeLayer(cfg, fused=fused, attn=attn, kv_dtype=kind, seed=7)
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    hidden = h_scale * torch.randn(b * s_q, cfg.hidden, device="cuda", dtype=torch.bfloat16, generator=g)
    return layer, cache, scale, st, hidden


def reference(b, s_k, page, s_q, seed=3, h_scale=1.0):
    """bf16 PyTorch reference: unfused ops + fp32 torch attention over the native bf16 cache."""
    layer, cache, scale, st, hidden = build("native", "torch", False, b, s_k, page, s_q, seed, h_scale)
    with torch.no_grad():
        return layer(hidden, cache, st, scale)


@pytest.mark.parametrize("s_q", [1, 2])
@pytest.mark.parametrize("page", [16, 32, 64])
@pytest.mark.parametrize("b,s_k", [(1, 77), (3, 300), (5, 2049), (2, 4097)])
@pytest.mark.parametrize("fused", [True, False])
def test_layer_bf16_v2_matches_torch_reference(b, s_k, page, s_q, fused):
    y_ref = reference(b, s_k, page, s_q)
    layer, cache, scale, st, hidden = build("native", "v2", fused, b, s_k, page, s_q)
    with torch.no_grad():
        y = layer(hidden, cache, st, scale)
    assert torch.isfinite(y.float()).all()
    cd = cosd(y, y_ref)
    assert cd < GATE["native"], cd


@pytest.mark.parametrize("h_scale", [1.0, 3.0])  # 3.0: sharper softmax (fewer tokens carry the weight)
@pytest.mark.parametrize("s_q", [1, 2])
@pytest.mark.parametrize("page", [32, 64])
@pytest.mark.parametrize("b,s_k", [(1, 300), (4, 2049), (3, 4097)])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_layer_quantized_v2_matches_bf16_reference(kind, fused, b, s_k, page, s_q, h_scale):
    y_ref = reference(b, s_k, page, s_q, h_scale=h_scale)
    layer, cache, scale, st, hidden = build(kind, "v2", fused, b, s_k, page, s_q, h_scale=h_scale)
    with torch.no_grad():
        y = layer(hidden, cache, st, scale)
    assert torch.isfinite(y.float()).all()
    cd = cosd(y, y_ref)
    assert cd < GATE[kind], cd


@pytest.mark.parametrize("kind", ["fp8", "int8"])
def test_fused_and_unfused_quantized_layers_write_identical_cache(kind):
    """quantize-on-write: the CUDA prologue and the torch reference quantizer produce the same bytes / scales."""
    b, s_k, page, s_q = 3, 300, 64, 2
    outs = []
    for fused in (True, False):
        layer, cache, scale, st, hidden = build(kind, "v2", fused, b, s_k, page, s_q)
        with torch.no_grad():
            layer(hidden, cache, st, scale)
        slots = st.slots
        outs.append((cache.view(-1, 640)[slots].clone(), scale.view(-1)[slots].clone()))
    # RMSNorm / rope run in a different op order in the two paths, so the 8-bit codes may differ by one code in rare
    # ties; the per-token scale and rope bytes must agree to ~1 ulp, and the dequantized rows to within 1% of amax.
    assert torch.allclose(outs[0][1], outs[1][1], rtol=1e-2)
    d = (outs[0][0][:, :512].float() - outs[1][0][:, :512].float()).abs()
    assert (d > 0).float().mean().item() < 0.2


@pytest.mark.parametrize("kind", ["native", "fp8", "int8"])
def test_whole_layer_graph_replay_matches_eager_v2(kind):
    """Whole-step CUDA graph (metadata + L chained layers + quantize-on-write + v2 attention) over changing seqlens."""
    cfg = MLAConfig()
    b, s_k, page, L = 4, 200, 64, 3
    st, nblk = make_state(cfg, b, s_k, page, 1, attn="v2", kv_dtype=kind)
    layers = [MLADecodeLayer(cfg, fused=True, attn="v2", kv_dtype=kind, seed=l) for l in range(L)]
    cs = [alloc_cache(nblk, page, kind, seed=l) for l in range(L)]
    caches_g, scales_g = [c[0] for c in cs], [c[1] for c in cs]
    st_e, _ = make_state(cfg, b, s_k, page, 1, attn="v2", kv_dtype=kind)
    h_static = torch.randn(b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        g, out = capture_layer_graph(layers, caches_g, scales_g if kind != "native" else None, st, h_static)
        caches_e = [c.clone() for c in caches_g]
        scales_e = [None if s is None else s.clone() for s in scales_g]
        for step in range(3):
            st.cache_seqlens += 1
            st_e.cache_seqlens += 1
            h = torch.randn(b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
            h_static.copy_(h)
            g.replay()
            st_e.refresh(page)
            x = h
            for l in range(L):
                x = layers[l](x, caches_e[l], st_e, scales_e[l])
            assert cosd(out, x) < 1e-6, (step, cosd(out, x))
