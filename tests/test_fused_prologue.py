"""Unit tests of the fused MLA decode prologue against the PyTorch reference.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<worktree> python -m pytest tests/test_fused_prologue.py -q
"""
import itertools

import pytest
import torch

pytest.importorskip("flash_mla_fused_cuda", reason="fused extension not built (tools/build.sh)")
from flash_mla.mla_prologue import (apply_rope_ref, make_rope_cache, mla_decode_prologue, mla_decode_prologue_ref,  # noqa: E402
                                    rmsnorm_ref)
from flash_mla.kv_quant import ROW_BYTES, dequantize_kv  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"


def make_inputs(dtype, page, H, s_q, B=5, max_len=700, seed=0, qstride=False, kv_dtype="native"):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    T = B * s_q
    lens = torch.randint(max(s_q, 1), max_len, (B,), dtype=torch.int32, device=DEV, generator=g)
    lens[0] = s_q  # shortest legal sequence (all tokens new)
    lens[-1] = max_len + page  # crosses several pages
    max_blocks = int((lens.max().item() + page - 1) // page)
    nblk = B * max_blocks + 3
    bt = torch.randperm(nblk, device=DEV, generator=g)[: B * max_blocks].view(B, max_blocks).to(torch.int32)
    if qstride:  # q_nope as it comes out of the absorption bmm: (H, T, 512) transposed to (T, H, 512); q_pe a slice of q_proj
        q_nope = torch.randn(H, T, 512, device=DEV, dtype=dtype, generator=g).transpose(0, 1)
        q_proj = torch.randn(T, H, 192, device=DEV, dtype=dtype, generator=g)
        q_pe = q_proj[..., 128:]
        kv_a = torch.randn(T, 576, device=DEV, dtype=dtype, generator=g)
        kv_c, k_pe = kv_a[:, :512], kv_a[:, 512:]
    else:
        q_nope = torch.randn(T, H, 512, device=DEV, dtype=dtype, generator=g)
        q_pe = torch.randn(T, H, 64, device=DEV, dtype=dtype, generator=g)
        kv_c = torch.randn(T, 512, device=DEV, dtype=dtype, generator=g)
        k_pe = torch.randn(T, 64, device=DEV, dtype=dtype, generator=g)
    cos_sin = make_rope_cache(int(lens.max().item()) + 8, device=DEV)
    if kv_dtype == "native":
        cache = torch.randn(nblk, page, 1, 576, device=DEV, dtype=dtype, generator=g)
        scale = None
    else:
        cache = torch.randint(0, 256, (nblk, page, ROW_BYTES), device=DEV, dtype=torch.uint8, generator=g)
        scale = torch.rand(nblk, page, device=DEV, dtype=torch.float32, generator=g) + 0.5
    return dict(q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe, cos_sin=cos_sin, block_table=bt, cache_seqlens=lens, s_q=s_q,
                kv_cache=cache, kv_scale=scale)


def ulp_close(a, b, ulps=1):
    """bf16/fp16 values equal up to `ulps` units in the last place (relative 2^-7 for bf16, 2^-10 for fp16)."""
    rel = (2.0 ** -7 if a.dtype == torch.bfloat16 else 2.0 ** -10) * ulps
    a, b = a.float(), b.float()
    return bool(((a - b).abs() <= rel * torch.maximum(a.abs(), b.abs()) + 1e-6).all())


def run_both(inp, **kw):
    ref_in = {k: (v.clone() if isinstance(v, torch.Tensor) and k in ("kv_cache", "kv_scale") else v) for k, v in inp.items()}
    fused_in = {k: (v.clone() if isinstance(v, torch.Tensor) and k in ("kv_cache", "kv_scale") else v) for k, v in inp.items()}
    q_ref = mla_decode_prologue_ref(**ref_in, **kw)
    q_out = mla_decode_prologue(**fused_in, **kw)
    return q_ref, q_out, ref_in, fused_in


def written_slots(inp, T):
    page = inp["kv_cache"].shape[1]
    s_q = inp["s_q"]
    t = torch.arange(T, device=DEV)
    b, j = t // s_q, t % s_q
    pos = inp["cache_seqlens"][b].long() - s_q + j
    return inp["block_table"][b, pos // page].long() * page + pos % page


@pytest.mark.parametrize("dtype,page,H,s_q,interleaved", [
    (torch.bfloat16, 64, 16, 1, False), (torch.bfloat16, 32, 16, 2, False), (torch.bfloat16, 16, 128, 1, False),
    (torch.bfloat16, 64, 16, 1, True), (torch.float16, 64, 16, 1, False), (torch.float16, 16, 8, 2, True),
])
@pytest.mark.parametrize("norm", [False, True])
def test_native_cache(dtype, page, H, s_q, interleaved, norm):
    inp = make_inputs(dtype, page, H, s_q)
    kw = dict(interleaved=interleaved, eps=1e-6)
    if norm:
        kw["kv_norm_weight"] = (torch.rand(512, device=DEV) + 0.5).to(dtype)
    orig = inp["kv_cache"].clone()
    q_ref, q_out, ref_in, fused_in = run_both(inp, **kw)
    T = q_ref.shape[0]
    # q: nope part exact copy, rope part within 1 ulp of the fp32-then-round reference
    assert torch.equal(q_out[..., :512], inp["q_nope"])
    assert ulp_close(q_out[..., 512:], q_ref[..., 512:])
    # cache
    slots = written_slots(inp, T)
    flat_ref, flat_out, flat_orig = (x.view(-1, 576) for x in (ref_in["kv_cache"], fused_in["kv_cache"], orig))
    mask = torch.zeros(flat_out.shape[0], dtype=torch.bool, device=DEV)
    mask[slots] = True
    assert torch.equal(flat_out[~mask], flat_orig[~mask]), "slots outside the new tokens must stay untouched"
    if norm:
        assert ulp_close(flat_out[slots][:, :512], flat_ref[slots][:, :512], ulps=2)
    else:
        assert torch.equal(flat_out[slots][:, :512], flat_ref[slots][:, :512])
    assert ulp_close(flat_out[slots][:, 512:], flat_ref[slots][:, 512:])


@pytest.mark.parametrize("kv_dtype", ["fp8", "int8"])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("norm", [False, True])
def test_quantized_cache(kv_dtype, page, norm):
    inp = make_inputs(torch.bfloat16, page, 16, 1, kv_dtype=kv_dtype)
    kw = dict(kv_dtype=kv_dtype)
    if norm:
        kw["kv_norm_weight"] = (torch.rand(512, device=DEV) + 0.5).to(torch.bfloat16)
    orig_q, orig_s = inp["kv_cache"].clone(), inp["kv_scale"].clone()
    q_ref, q_out, ref_in, fused_in = run_both(inp, **kw)
    T = q_ref.shape[0]
    assert torch.equal(q_out[..., :512], inp["q_nope"])
    assert ulp_close(q_out[..., 512:], q_ref[..., 512:])
    slots = written_slots(inp, T)
    kq_out, kq_ref = fused_in["kv_cache"].view(-1, ROW_BYTES), ref_in["kv_cache"].view(-1, ROW_BYTES)
    sc_out, sc_ref = fused_in["kv_scale"].view(-1), ref_in["kv_scale"].view(-1)
    mask = torch.zeros(kq_out.shape[0], dtype=torch.bool, device=DEV)
    mask[slots] = True
    assert torch.equal(kq_out[~mask], orig_q.view(-1, ROW_BYTES)[~mask]), "unrelated cache rows modified"
    assert torch.equal(sc_out[~mask], orig_s.view(-1)[~mask]), "unrelated scales modified"
    if not norm:  # bit-exact quantization of identical bf16 inputs
        assert torch.equal(kq_out[slots][:, :512], kq_ref[slots][:, :512])
        assert torch.equal(sc_out[slots], sc_ref[slots])
    else:  # rsqrtf vs torch.rsqrt may differ by 1 bf16 ulp of the normed value -> a code can flip; bound the damage
        d_out = dequantize_kv(kq_out[slots], sc_out[slots], kv_dtype, torch.float32)[:, :512]
        d_ref = dequantize_kv(kq_ref[slots], sc_ref[slots], kv_dtype, torch.float32)[:, :512]
        amax = d_ref.abs().amax(-1, keepdim=True)
        assert ((d_out - d_ref).abs() <= 0.08 * amax).all()
        assert (kq_out[slots][:, :512] != kq_ref[slots][:, :512]).float().mean() < 0.02
    # rope bytes: bf16 within 1 ulp
    r_out = kq_out[slots][:, 512:].contiguous().view(torch.bfloat16)
    r_ref = kq_ref[slots][:, 512:].contiguous().view(torch.bfloat16)
    assert ulp_close(r_out, r_ref)


def test_strided_inputs():
    """q_nope as a transposed bmm output, q_pe / kv_c / k_pe as slices of larger tensors."""
    inp = make_inputs(torch.bfloat16, 64, 16, 1, qstride=True)
    assert not inp["q_nope"].is_contiguous()
    q_ref, q_out, ref_in, fused_in = run_both(inp)
    assert torch.equal(q_out[..., :512], inp["q_nope"])
    assert ulp_close(q_out[..., 512:], q_ref[..., 512:])
    slots = written_slots(inp, q_ref.shape[0])
    assert torch.equal(fused_in["kv_cache"].view(-1, 576)[slots][:, :512], ref_in["kv_cache"].view(-1, 576)[slots][:, :512])


def test_cuda_graph_replay_with_new_seqlens():
    """The call is capturable and re-reads cache_seqlens / inputs from device memory on every replay."""
    inp = make_inputs(torch.bfloat16, 64, 16, 1)
    fused_in = {k: (v.clone() if isinstance(v, torch.Tensor) and k in ("kv_cache",) else v) for k, v in inp.items()}
    q_out = torch.empty(inp["q_nope"].shape[0], 16, 576, dtype=torch.bfloat16, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mla_decode_prologue(**fused_in, q_out=q_out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mla_decode_prologue(**fused_in, q_out=q_out)
    for step in range(3):  # decode steps: sequences grow by one token, new inputs are copied in place
        fused_in["cache_seqlens"] += 1
        fused_in["q_nope"].normal_(); fused_in["q_pe"].normal_(); fused_in["kv_c"].normal_(); fused_in["k_pe"].normal_()
        g.replay()
        ref_in = {k: (v.clone() if isinstance(v, torch.Tensor) and k == "kv_cache" else v) for k, v in fused_in.items()}
        q_ref = mla_decode_prologue_ref(**ref_in)
        T = q_ref.shape[0]
        slots = written_slots(fused_in, T)
        assert ulp_close(q_out[..., 512:], q_ref[..., 512:])
        assert torch.equal(q_out[..., :512], fused_in["q_nope"])
        assert torch.equal(fused_in["kv_cache"].view(-1, 576)[slots][:, :512], ref_in["kv_cache"].view(-1, 576)[slots][:, :512])


def test_rope_reference_is_a_rotation():
    cs = make_rope_cache(64, device=DEV)
    x = torch.randn(7, 3, 64, device=DEV)
    pos = torch.arange(7, device=DEV)
    for il in (False, True):
        y = apply_rope_ref(x, cs, pos, il)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5)
    assert torch.allclose(apply_rope_ref(x, cs, torch.zeros(7, dtype=torch.long, device=DEV), False), x)


def test_bad_arguments_raise():
    inp = make_inputs(torch.bfloat16, 64, 16, 1)
    bad = dict(inp)
    bad["q_nope"] = inp["q_nope"].float()
    with pytest.raises(RuntimeError):
        mla_decode_prologue(**bad)
    bad = dict(inp)
    bad["s_q"] = 2  # T = 5 is not a multiple of 2
    with pytest.raises(RuntimeError):
        mla_decode_prologue(**bad)
