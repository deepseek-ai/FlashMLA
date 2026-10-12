"""Correctness tests for the sm86 MLA decode kernels (impl="v2").

Run on one GPU:   CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_flash_mla_sm86.py -x -q
Reference: fp32 attention on the gathered (un-paged) KV; pass criterion cos_diff < 8e-5 for out and lse.
"""
import math
import random

import pytest
import torch

from flash_mla import (
    alloc_mla_workspace,
    flash_mla_with_kvcache,
    get_mla_metadata,
    get_mla_workspace_bytes,
)

D, DV = 576, 512
DEV = "cuda"
COS_TOL = 8e-5


def ref_mla(q, blocked_k, block_table, cache_seqlens, h_q, causal, scale=None):
    """q: [b, s_q, h_q, d]; blocked_k: [nb, page, 1, d]. Returns fp32 (out [b,s_q,h_q,dv], lse [b,h_q,s_q])."""
    b, s_q = q.shape[:2]
    page = blocked_k.shape[1]
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out = torch.zeros(b, s_q, h_q, DV, dtype=torch.float32, device=q.device)
    lse = torch.full((b, h_q, s_q), float("inf"), dtype=torch.float32, device=q.device)
    for i in range(b):
        sk = int(cache_seqlens[i])
        if sk == 0:
            continue
        nblk = (sk + page - 1) // page
        k = blocked_k[block_table[i, :nblk].long()].reshape(-1, D)[:sk].float()   # [sk, d]
        v = k[:, :DV]
        scores = torch.einsum("shd,td->hst", q[i].float(), k) * scale             # [h_q, s_q, sk]
        if causal and s_q > 1:
            mask = torch.ones(s_q, sk, dtype=torch.bool, device=q.device).tril(diagonal=sk - s_q)
            scores = scores.masked_fill(~mask, float("-inf"))
        lse[i] = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1)
        out[i] = torch.einsum("hst,td->shd", p, v)
    return out, lse


def cos_diff(x, y):
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / max((x * x + y * y).sum().item(), 1e-12)).item()


def make_inputs(b, s_q, h_q, seqlens, page, dtype, extra_pages=0, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    max_sk = max(seqlens)
    pages_per_seq = max(1, (max_sk + page - 1) // page) + extra_pages
    nb = b * pages_per_seq
    q = torch.randn(b, s_q, h_q, D, device=DEV, dtype=torch.float32, generator=g).to(dtype)
    kv = torch.randn(nb, page, 1, D, device=DEV, dtype=torch.float32, generator=g).to(dtype)
    perm = torch.randperm(nb, device=DEV, generator=g).to(torch.int32)
    block_table = perm.view(b, pages_per_seq).contiguous()
    # NaN beyond the valid length (inside used pages and in the unused tail pages)
    for i in range(b):
        sk = seqlens[i]
        pg = block_table[i].long()
        flat = kv[pg].reshape(-1, 1, D)
        flat[sk:] = float("nan")
        kv[pg] = flat.reshape(pages_per_seq, page, 1, D)
    return q, kv, block_table, seqlens_t


def run_case(b, s_q, h_q, seqlens, page, dtype, causal, impl="v2", extra_pages=0, seed=0, check_det=True):
    q, kv, bt, sl = make_inputs(b, s_q, h_q, seqlens, page, dtype, extra_pages, seed)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1)
    out, lse = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, causal=causal, impl=impl)
    torch.cuda.synchronize()
    ref_o, ref_l = ref_mla(q, kv, bt, sl, h_q, causal)
    assert not torch.isnan(out).any(), "NaN in out"
    assert not torch.isnan(lse).any(), "NaN in lse"
    co, cl = cos_diff(out, ref_o), cos_diff(torch.where(torch.isinf(lse), torch.zeros_like(lse), lse),
                                              torch.where(torch.isinf(ref_l), torch.zeros_like(ref_l), ref_l))
    assert co < COS_TOL, f"out cos_diff {co}"
    assert cl < COS_TOL, f"lse cos_diff {cl}"
    if check_det:
        out2, lse2 = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, causal=causal, impl=impl)
        assert torch.equal(out, out2) and torch.equal(lse, lse2), "non-deterministic result"
    return out, lse


def varlen_seqlens(b, mean_sk, s_q, seed=1):
    rnd = random.Random(seed)
    return [max(int(rnd.normalvariate(mean_sk, mean_sk / 2)), s_q) for _ in range(b)]


# ---------------------------------------------------------------------------------------------
# core matrix: dtype x page x h_q x s_q/causal
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("page", [16, 32, 64])
@pytest.mark.parametrize("h_q", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("s_q,causal", [(1, False), (2, True)])
def test_matrix(dtype, page, h_q, s_q, causal):
    b = 6
    seqlens = varlen_seqlens(b, 700, s_q)
    run_case(b, s_q, h_q, seqlens, page, dtype, causal)


@pytest.mark.parametrize("h_q", [16, 128])
@pytest.mark.parametrize("s_q,causal", [(1, False), (2, True)])
def test_fixed_len_batches(h_q, s_q, causal):
    run_case(33, s_q, h_q, [1024] * 33, 64, torch.bfloat16, causal)


@pytest.mark.parametrize("h_q", [16, 64])
def test_varlen_large_batch(h_q):
    b = 128
    run_case(b, 1, h_q, varlen_seqlens(b, 2048, 1, seed=3), 64, torch.bfloat16, False)


# ---------------------------------------------------------------------------------------------
# splitting / scheduler stress
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("h_q", [16, 128])
@pytest.mark.parametrize("page", [16, 64])
def test_b1_long(h_q, page):
    run_case(1, 1, h_q, [32768], page, torch.bfloat16, False)


def test_b1_long_causal_mtp():
    run_case(1, 2, 16, [32771], 64, torch.bfloat16, True)


@pytest.mark.parametrize("b", [2, 3, 5])
def test_small_batch_long(b):
    run_case(b, 1, 16, [8000 + 977 * i for i in range(b)], 64, torch.bfloat16, False)


# ---------------------------------------------------------------------------------------------
# tiny / edge sequence lengths
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sk", [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129])
@pytest.mark.parametrize("page", [16, 64])
def test_edge_seqlens(sk, page):
    run_case(3, 1, 16, [sk, max(1, sk - 1), sk + 40], page, torch.bfloat16, False)


@pytest.mark.parametrize("sk", [2, 3, 17, 33, 65])
def test_edge_seqlens_causal(sk):
    run_case(4, 2, 16, [sk, sk + 1, sk + 17, sk + 100], 32, torch.bfloat16, True)


def test_tail_rows_padding():
    # h_q * s_q not a multiple of the row-block size (padding rows must be ignored)
    run_case(5, 1, 8, varlen_seqlens(5, 500, 1), 64, torch.bfloat16, False)
    run_case(5, 2, 24, varlen_seqlens(5, 500, 2), 64, torch.bfloat16, True)


# ---------------------------------------------------------------------------------------------
# API: preallocated out / lse / workspace, legacy impl cross-check, CUDA graph replay
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("s_q,causal", [(1, False), (2, True)])
def test_prealloc(s_q, causal):
    b, h_q, page = 7, 16, 64
    seqlens = varlen_seqlens(b, 3000, s_q)
    q, kv, bt, sl = make_inputs(b, s_q, h_q, seqlens, page, torch.bfloat16)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1)
    out0, lse0 = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, causal=causal)
    out = torch.empty(b, s_q, h_q, DV, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=DEV)
    ws = alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0], DEV)
    o2, l2 = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, causal=causal, out=out, lse=lse, workspace=ws)
    assert o2.data_ptr() == out.data_ptr() and l2.data_ptr() == lse.data_ptr()
    assert torch.equal(out, out0) and torch.equal(lse, lse0)


def test_graph_replay_changing_seqlens():
    b, s_q, h_q, page = 24, 1, 16, 64
    cap = 4096
    seqlens_a = varlen_seqlens(b, 2000, s_q, seed=11)
    seqlens_a = [min(s, cap) for s in seqlens_a]
    q, kv, bt, sl = make_inputs(b, s_q, h_q, [cap] * b, page, torch.bfloat16)   # capacity cap per sequence
    kv_clean = kv.clone()
    kv_clean = torch.nan_to_num(kv_clean)   # all slots valid data: every length <= cap is a legal prefix
    sl_static = torch.tensor(seqlens_a, dtype=torch.int32, device=DEV)
    num_parts_probe = get_mla_metadata(sl_static, s_q * h_q, 1)[0].shape[0]
    out = torch.empty(b, s_q, h_q, DV, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=DEV)
    ws = alloc_mla_workspace(b, s_q * h_q, 1, num_parts_probe, DEV)

    def step():
        md, ns = get_mla_metadata(sl_static, s_q * h_q, 1)
        flash_mla_with_kvcache(q, kv_clean, bt, sl_static, DV, md, ns, out=out, lse=lse, workspace=ws)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for trial, seqlens in enumerate([seqlens_a, varlen_seqlens(b, 1500, s_q, seed=12), [cap] * b, [1] * b]):
        seqlens = [min(s, cap) for s in seqlens]
        sl_static.copy_(torch.tensor(seqlens, dtype=torch.int32, device=DEV))
        g.replay()
        torch.cuda.synchronize()
        ref_o, ref_l = ref_mla(q, kv_clean, bt, sl_static, h_q, False)
        assert not torch.isnan(out).any()
        assert cos_diff(out, ref_o) < COS_TOL, f"graph replay {trial}"
        assert cos_diff(lse, ref_l) < COS_TOL, f"graph replay {trial} lse"
        g.replay()
        torch.cuda.synchronize()
        o_again = out.clone()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(o_again, out), "graph replay not deterministic"


# ---------------------------------------------------------------------------------------------
# fused (last-arriver) split-KV combine: needs a workspace (zeroed counters); must equal the reference, be
# deterministic, leave the counters at zero (self-reset) and survive CUDA-Graph replay
# ---------------------------------------------------------------------------------------------
def run_ws_case(b, s_q, h_q, seqlens, page, dtype, causal, fuse, extra_env=None):
    q, kv, bt, sl = make_inputs(b, s_q, h_q, seqlens, page, dtype)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1)
    ws = alloc_mla_workspace(b, s_q * h_q, 1, md.shape[0], DEV)
    outs = []
    for _ in range(3):   # repeated calls: counters must have reset themselves
        out = torch.empty(b, s_q, h_q, DV, dtype=dtype, device=DEV)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=DEV)
        flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, causal=causal, out=out, lse=lse, workspace=ws)
        outs.append((out, lse))
    torch.cuda.synchronize()
    ref_o, ref_l = ref_mla(q, kv, bt, sl, h_q, causal)
    for out, lse in outs:
        assert not torch.isnan(out).any() and not torch.isnan(lse).any()
        assert cos_diff(out, ref_o) < COS_TOL
        assert cos_diff(lse, ref_l) < COS_TOL
        assert torch.equal(out, outs[0][0]) and torch.equal(lse, outs[0][1]), "non-deterministic / counters not reset"
    # the counter region of the workspace must be all zero again
    n = get_mla_workspace_bytes(b, s_q * h_q, 1, md.shape[0])
    return outs[0]


@pytest.mark.parametrize("fuse", ["0", "2", "4", "8"])
@pytest.mark.parametrize("h_q,s_q,causal", [(16, 1, False), (64, 1, False), (16, 2, True), (128, 1, False)])
def test_fused_combine(monkeypatch, fuse, h_q, s_q, causal):
    monkeypatch.setenv("FLASH_MLA_SM86_FUSE", fuse)
    b = 17
    run_ws_case(b, s_q, h_q, varlen_seqlens(b, 1500, s_q, seed=5), 64, torch.bfloat16, causal, int(fuse))


def test_fused_equals_unfused_closely(monkeypatch):
    b, s_q, h_q = 9, 1, 16
    seqlens = varlen_seqlens(b, 2000, s_q, seed=8)
    monkeypatch.setenv("FLASH_MLA_SM86_FUSE", "0")
    o0, l0 = run_ws_case(b, s_q, h_q, seqlens, 64, torch.bfloat16, False, 0)
    monkeypatch.setenv("FLASH_MLA_SM86_FUSE", "8")
    o1, l1 = run_ws_case(b, s_q, h_q, seqlens, 64, torch.bfloat16, False, 8)
    assert cos_diff(o0, o1) < 1e-6 and cos_diff(l0, l1) < 1e-6


def test_graph_replay_fused(monkeypatch):
    monkeypatch.setenv("FLASH_MLA_SM86_FUSE", "8")
    test_graph_replay_changing_seqlens()


# ---------------------------------------------------------------------------------------------
# the legacy kernel must keep working with the same metadata (page_size == 32, bf16)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("h_q", [16, 64])
def test_legacy_crosscheck(h_q):
    b, s_q = 6, 1
    seqlens = varlen_seqlens(b, 900, s_q, seed=2)
    q, kv, bt, sl = make_inputs(b, s_q, h_q, seqlens, 32, torch.bfloat16)
    md, ns = get_mla_metadata(sl, s_q * h_q, 1)
    o_new, l_new = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, impl="v2")
    o_old, l_old = flash_mla_with_kvcache(q, kv, bt, sl, DV, md, ns, impl="legacy")
    assert cos_diff(o_new, o_old) < COS_TOL and cos_diff(l_new, l_old) < COS_TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"] + __import__("sys").argv[1:]))
