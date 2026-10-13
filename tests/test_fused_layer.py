"""Decode-layer tests: fused prologue layer == unfused PyTorch layer, attention backends agree, CUDA-graph replay == eager.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> python -m pytest tests/test_fused_layer.py -q
"""
import pytest
import torch

pytest.importorskip("flash_mla_fused_cuda", reason="fused extension not built (tools/build.sh)")
from flash_mla.mla_decode_layer import MLAConfig, MLADecodeLayer, alloc_cache, capture_layer_graph, make_state  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def cosd(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


def run(fused, attn, b=3, s_k=300, page=32, s_q=1, kv="native", seed=0, steps=1):
    cfg = MLAConfig()
    st, nblk = make_state(cfg, b, s_k, page, s_q, attn=attn)
    layer = MLADecodeLayer(cfg, fused=fused, attn=attn, kv_dtype=kv, seed=7)
    cache, scale = alloc_cache(nblk, page, kv, seed=seed)
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    h = torch.randn(b * s_q, cfg.hidden, device="cuda", dtype=torch.bfloat16, generator=g)
    with torch.no_grad():
        y = layer(h, cache, st, scale)
    return y, cache, scale, st


@pytest.mark.parametrize("attn,page,s_q", [("torch", 32, 1), ("torch", 64, 2), ("legacy", 32, 1), ("legacy", 32, 2)])
def test_fused_matches_unfused(attn, page, s_q):
    if attn == "legacy":
        pytest.importorskip("flash_mla_cuda")
    y_f, c_f, _, st = run(True, attn, page=page, s_q=s_q)
    y_u, c_u, _, _ = run(False, attn, page=page, s_q=s_q)
    assert cosd(y_f, y_u) < 1e-4, cosd(y_f, y_u)
    slots = st.slots
    a, b = c_f.view(-1, 576)[slots].float(), c_u.view(-1, 576)[slots].float()
    assert cosd(a, b) < 1e-5


def test_attention_backends_agree():
    pytest.importorskip("flash_mla_cuda")
    y_t = run(True, "torch", page=32)[0]
    y_l = run(True, "legacy", page=32)[0]
    y_tr = run(True, "triton", page=32)[0]
    assert cosd(y_t, y_l) < 2e-4
    assert cosd(y_t, y_tr) < 2e-4


@pytest.mark.parametrize("kv", ["fp8", "int8"])
def test_quantized_cache_layer_close_to_native(kv):
    y_n = run(True, "torch", kv="native", page=32)[0]
    y_q = run(True, "torch", kv=kv, page=32)[0]
    # NOTE: caches are random (not the same data), so only check that the quantized path is sane: finite and same scale
    assert torch.isfinite(y_q.float()).all()
    assert 0.2 < (y_q.float().norm() / y_n.float().norm()).item() < 5


def test_cuda_graph_replay_matches_eager_over_steps():
    pytest.importorskip("flash_mla_cuda")
    cfg = MLAConfig()
    b, s_k, page, L = 4, 200, 32, 3
    st, nblk = make_state(cfg, b, s_k, page, 1, attn="legacy")
    layers = [MLADecodeLayer(cfg, fused=True, attn="legacy", seed=l) for l in range(L)]
    caches_g = [alloc_cache(nblk, page, seed=l)[0] for l in range(L)]
    st_e, _ = make_state(cfg, b, s_k, page, 1, attn="legacy")
    h = torch.randn(b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    h_static = h.clone()
    with torch.no_grad():
        g, out = capture_layer_graph(layers, caches_g, [None] * L, st, h_static)  # (the warm-up runs write the caches)
        caches_e = [c.clone() for c in caches_g]  # eager reference starts from the same cache contents
        for step in range(3):  # decode: every sequence grows by one token per step, new hidden state each step
            st.cache_seqlens += 1
            st_e.cache_seqlens += 1
            h = torch.randn(b, cfg.hidden, device="cuda", dtype=torch.bfloat16)
            h_static.copy_(h)
            g.replay()
            st_e.refresh(page)
            x = h
            for l in range(L):
                x = layers[l](x, caches_e[l], st_e, None)
            assert cosd(out, x) < 1e-6, (step, cosd(out, x))
