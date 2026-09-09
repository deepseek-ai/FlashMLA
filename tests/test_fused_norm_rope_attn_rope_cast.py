import math
import time
import sys
from typing import Tuple, Optional
import random
import functools

import argparse
import torch
import tile_kernels
import deep_gemm
import kernelkit as kk

from lib import TestParam, Testcase, RawTestParamForDecode, TestcaseForDecode, ExtraTestParamForDecode
import lib
import quant
import ref

import flash_mla

# FMA with double precision is necessary for numerical consistency of fused ref
_GET_FMA_KERNEL = None
def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    global _GET_FMA_KERNEL
    assert a.numel() == b.numel() == c.numel()
    if _GET_FMA_KERNEL is None:
        import tilelang
        from tilelang import language as T
        @tilelang.jit(
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            },
        )
        def _get_fma_kernel():
            num_threads = 256
            numel_per_cta = 1024
            numel = T.symbolic("numel")
            @T.prim_func
            def fma_kernel(
                out: T.Tensor[(numel,), torch.float32],
                a: T.Tensor[(numel,), torch.float32],
                b: T.Tensor[(numel,), torch.float32],
                c: T.Tensor[(numel,), torch.float32]
            ):
                with T.Kernel(numel // numel_per_cta, threads=num_threads) as (tid_x, ):
                    a_fragment = T.alloc_fragment((numel_per_cta, ), torch.float32)
                    b_fragment = T.alloc_fragment((numel_per_cta, ), torch.float32)
                    c_fragment = T.alloc_fragment((numel_per_cta, ), torch.float32)
                    out_fragment = T.alloc_fragment((numel_per_cta, ), torch.float32)

                    T.copy(a[tid_x * numel_per_cta], a_fragment)
                    T.copy(b[tid_x * numel_per_cta], b_fragment)
                    T.copy(c[tid_x * numel_per_cta], c_fragment)
                    for i in T.Parallel(numel_per_cta):
                        out_fragment[i] = a_fragment[i] * b_fragment[i] + c_fragment[i]
                    T.copy(out_fragment, out[tid_x * numel_per_cta])
            return fma_kernel
        _GET_FMA_KERNEL = _get_fma_kernel
    out = torch.empty_like(a).flatten()
    _GET_FMA_KERNEL()(out, a.flatten(), b.flatten(), c.flatten())
    return out.view_as(a)


def _rope_inplace(q_or_o: torch.Tensor, conjugate: bool, cos_sin_table: torch.Tensor, token_positions: torch.Tensor, rope_dim: int = 64):
    # cos_sin_table: [*, rope_dim]; token_positions: [s_q] (prefill) or [b, s_q] (decode)
    # q_or_o: [s_q, h_q, d_qk] (prefill) or [b, s_q, h_q, d_qk] (decode)
    # The unsqueeze dim depends on q_or_o's dimensionality:
    #   3D => dim=1, 4D => dim=2, i.e., unsqueeze_dim = q_or_o.ndim - 2
    unsqueeze_dim = q_or_o.ndim - 2
    cos = cos_sin_table[token_positions, :rope_dim//2].unsqueeze(unsqueeze_dim)
    sin = cos_sin_table[token_positions, rope_dim//2:].unsqueeze(unsqueeze_dim)
    if conjugate:
        sin = -sin
    rope_part = q_or_o[..., -rope_dim:]
    rope_part_x0 = rope_part[..., 0::2]
    rope_part_x1 = rope_part[..., 1::2]
    new_rope_part_x0 = _fma(rope_part_x0, cos.expand_as(rope_part_x0), -rope_part_x1 * sin)
    new_rope_part_x1 = _fma(rope_part_x0, sin.expand_as(rope_part_x0), rope_part_x1 * cos)
    rope_part[..., 0::2] = new_rope_part_x0
    rope_part[..., 1::2] = new_rope_part_x1

def ref_fused_norm_rope_attn_rope_cast_fwd(p: TestParam, t: Testcase, enable_q_norm: bool, cos_sin_table: torch.Tensor, token_positions: torch.Tensor):
    # Q norm
    if enable_q_norm:
        q_float32 = t.q.float()
        rms_norm_scale_factor = torch.rsqrt(torch.sum(q_float32*q_float32, dim=-1) / p.d_qk + rms_norm_eps)
    else:
        rms_norm_scale_factor = None

    # Q RoPE
    q = t.q.clone().float()
    _rope_inplace(q, False, cos_sin_table, token_positions)
    q = q.to(torch.bfloat16)

    # Core attention
    old_q = t.q
    t.q = q
    _, out, max_logits, lse = ref.ref_sparse_attn_fwd(p, t, rms_norm_scale_factor)
    t.q = old_q

    # O RoPE
    _rope_inplace(out, True, cos_sin_table, token_positions)

    return out, max_logits, lse

def ref_fused_norm_rope_attn_rope_cast_decode(
    p: TestParam,
    t: TestcaseForDecode,
    enable_q_norm: bool,
    cos_sin_table: torch.Tensor,
    token_positions: torch.Tensor   # [b*s_q]
):
    """
    Reference implementation for the fused Q norm + RoPE + Core Attn (decode) + O RoPE
    Returns: (out_bf16, lse)
    """
    b, s_q, h_q, d_qk = t.q.shape

    # Q norm
    if enable_q_norm:
        q_float32 = t.q.float()
        rms_norm_scale_factor = torch.rsqrt(torch.sum(q_float32*q_float32, dim=-1) / p.d_qk + rms_norm_eps)
    else:
        rms_norm_scale_factor = None

    # Q RoPE
    q = t.q.clone().reshape(b*s_q, h_q, d_qk).float()
    _rope_inplace(q, False, cos_sin_table, token_positions)
    q = q.to(torch.bfloat16).reshape(b, s_q, h_q, d_qk)

    # Core attention (decode)
    old_q = t.q
    t.q = q
    out, lse = ref.ref_sparse_attn_decode(p, t, rms_norm_scale_factor)
    t.q = old_q

    # O RoPE
    out = out.float().reshape(b*s_q, h_q, p.d_v)
    _rope_inplace(out, True, cos_sin_table, token_positions)

    return out.to(torch.bfloat16), lse.transpose(1, 2).reshape(b*s_q, h_q)

def build_cos_sin_cache(max_token_position):
    # Copied from vLLM
    base = 100000
    rope_dim = 64
    mscale = 1.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float, device="cuda") / rope_dim))
    freqs = torch.outer(
        torch.arange(0, max_token_position, dtype=torch.float, device="cuda"),
        inv_freq
    )
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cos = freqs_cis.real.cuda() * mscale
    sin = freqs_cis.imag.cuda() * mscale
    return torch.cat((cos, sin), dim=-1)
max_token_position = 131072
cos_sin_cache = build_cos_sin_cache(max_token_position)
rms_norm_eps = 1e-4

@functools.lru_cache(maxsize=None)  # To avoid generating q weight and permuted q weight for multiple times, we cache previously generated q_weight and use them later
def get_q_b_weight(h_q: int, d_q: int, q_lora_rank: int, scale_gran: int):
    kk.utils.set_random_seed(0)
    q_weight = torch.randn((h_q*d_q, q_lora_rank), dtype=torch.bfloat16, device='cuda') / 10
    q_weight_quanted = tile_kernels.quant.per_token_cast(q_weight, 'e4m3', scale_gran, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    q_weight_quanted_permuted = flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj(q_weight_quanted, h_q, d_q)
    return q_weight_quanted, q_weight_quanted_permuted

@functools.lru_cache(maxsize=None)
def get_wb_weight(n_wv_group: int, wv_group_size: int, d_o: int, scale_gran: int, wv_proj_out_dim: int):
    kk.utils.set_random_seed(0)
    o_weight = torch.randn((n_wv_group*wv_proj_out_dim, wv_group_size * d_o), dtype=torch.bfloat16, device='cuda') / 10
    o_weight_quanted = tile_kernels.quant.per_token_cast(o_weight, 'e4m3', scale_gran, use_tma_aligned_col_major_sf=False, round_sf=True, use_packed_ue8m0=False)
    o_weight_quanted_sf = deep_gemm.transform_sf_into_required_layout(
        o_weight_quanted[1].view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o // scale_gran),
        wv_proj_out_dim,
        wv_group_size * d_o,
        num_groups=n_wv_group,
        recipe=(1, 1, scale_gran),
        is_sfa=False,
    )
    o_weight_quanted = (o_weight_quanted[0].view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o), o_weight_quanted_sf)
    o_weight_quanted_permuted = flash_mla.fused_norm_rope_attn_rope_cast.permute_wv_proj(o_weight_quanted, wv_group_size, d_o)
    return o_weight_quanted, o_weight_quanted_permuted

_counter = kk.Counter()

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")

    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    q_lora_rank = random.choice([1024, 1536])
    q_b_proj_scale_gran = random.choice([32, 128])
    q_lora = torch.randn((p.s_q, q_lora_rank)) / 10
    q_lora = tile_kernels.quant.per_token_cast(q_lora, 'e4m3', q_b_proj_scale_gran, None, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    q_b_proj, q_b_proj_permuted = get_q_b_weight(p.h_q, p.d_qk, q_lora_rank, q_b_proj_scale_gran)
    # TODO Make q weight non-contiguous

    o_scale_gran = 32
    enable_q_norm = random.choice([False, True])
    token_positions = torch.randint(0, max_token_position, (p.s_q, ), device='cuda', dtype=torch.int32)

    wv_group_size = 8
    n_wv_group = p.h_q // 8
    wv_proj_out_dim = random.choice([256, 512, 1024])
    wv_proj_scale_gran = 32
    wv_proj, wv_proj_permuted = get_wb_weight(n_wv_group, wv_group_size, p.d_v, wv_proj_scale_gran, wv_proj_out_dim)
    
    def calculate_q_b_proj(weight) -> torch.Tensor:
        q = torch.empty((p.s_q, p.h_q*p.d_qk), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_gemm_nt(q_lora, weight, q, recipe_a=(1, q_b_proj_scale_gran), recipe_b=(1, q_b_proj_scale_gran))
        return q.view(p.s_q, p.h_q, p.d_qk)
    q_for_fused = calculate_q_b_proj(q_b_proj_permuted)

    def calculate_wv_proj(output, weight) -> torch.Tensor:
        wv_proj_out = torch.empty((p.s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            output,
            weight,
            wv_proj_out,
            recipe=(1, 1, 32),
        )
        return wv_proj_out
    
    def run_fused_norm_rope_attn_rope_cast_fwd():
        return flash_mla.fused_norm_rope_attn_rope_cast.prefill(
            enable_q_norm,
            rms_norm_eps,
            token_positions,
            False, 64, cos_sin_cache,
            n_wv_group, o_scale_gran, True, True, True,
            q_for_fused, t.kv, t.indices,
            sm_scale=t.sm_scale,
            attn_sink=t.attn_sink,
            topk_length=t.topk_length,
        )
    
    torch.cuda.synchronize()
    ans_out_fp8, ans_out_sf, ans_max_logits, ans_lse = run_fused_norm_rope_attn_rope_cast_fwd()
    torch.cuda.synchronize()
    ans_wv_proj_out = calculate_wv_proj((ans_out_fp8, ans_out_sf), wv_proj_permuted)

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        fused_time = kk.bench_kineto(run_fused_norm_rope_attn_rope_cast_fwd, num_tests=p.num_runs).get_kernel_time("fused_norm_rope_attn_rope_cast_fwd")
        fused_flops = flops_and_mem_vol.fwd_flop/fused_time/1e12
        fused_mem_bw = flops_and_mem_vol.fwd_prefill_with_fp8_out_mem_vol/fused_time/1e12
        print(f"Fused:    {fused_time*1e6:4.0f} us, {fused_flops:6.1f} TFlops, {fused_mem_bw:4.2f} TBps")

    if p.check_correctness:
        out_criteria = {'abs_tol': 1.1e-3, 'rel_tol': 1.01/8, 'cos_diff_tol': 1e-3} if p.k_amplifier_portion == 0.0 else {'abs_tol': 1.0, 'rel_tol': 1.0, 'cos_diff_tol': 1e-3}
        wv_proj_out_criteria = {'abs_tol': 1.0, 'rel_tol': 1.0, 'cos_diff_tol': 1e-3} if p.k_amplifier_portion == 0.0 else {'abs_tol': 100.0, 'rel_tol': 100.0, 'cos_diff_tol': 1e-3}
        max_logits_criteria = {'abs_tol': 1e-5, 'rel_tol': 4.01/65536} if p.k_amplifier_portion == 0.0 else {'abs_tol': 1.0, 'rel_tol': 1.0}
        lse_criteria = {'abs_tol': 1e-5, 'rel_tol': 4.01/65536} if p.k_amplifier_portion == 0 else {'abs_tol': 1e-4, 'rel_tol': 8.01/65536}
        t.q = calculate_q_b_proj(q_b_proj)
        ref_out, ref_max_logits, ref_lse = ref_fused_norm_rope_attn_rope_cast_fwd(p, t, enable_q_norm, cos_sin_cache, token_positions)
        ref_lse[ref_lse == float("-inf")] = float("+inf")

        ans_out_sf_2d = ans_out_sf.view(p.s_q, -1)
        if ans_out_sf_2d.stride(0) != 1:
            # PyTorch normalizes strides of size-1 dims during view(), which breaks
            # tile_kernels' col-major sf layout detection (`sf.stride(0) == 1`) when s_q == 1,
            # so we restore the token-dim stride to 1 manually here
            assert p.s_q == 1
            ans_out_sf_2d = ans_out_sf_2d.as_strided(ans_out_sf_2d.shape, (1, ans_out_sf_2d.stride(1)))
        ans_wv_out_dequantized = tile_kernels.quant.per_token_cast_back((ans_out_fp8.view(p.s_q, -1), ans_out_sf_2d), 'fp32', 32)
        ans_wv_out_dequantized = ans_wv_out_dequantized \
            .view(p.s_q, n_wv_group, p.d_v // o_scale_gran, wv_group_size, o_scale_gran) \
            .transpose(2, 3) \
            .reshape(p.s_q, p.h_q, p.d_v)

        assert o_scale_gran == 32
        ref_out_fp8, ref_out_sf = tile_kernels.quant.per_token_cast(ref_out.view(p.s_q, -1), 'e4m3', 32, round_sf=True, use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True)
        ref_wv_proj_out = calculate_wv_proj(
            (ref_out_fp8.view(p.s_q, n_wv_group, wv_group_size * p.d_v), ref_out_sf.view(p.s_q, n_wv_group, wv_group_size * p.d_v // (32*4))),
            wv_proj
        )
        
        is_correct = True
        is_correct &= kk.check_is_allclose("out", ans_wv_out_dequantized, ref_out, **out_criteria)
        is_correct &= kk.check_is_allclose("wv_proj_out", ans_wv_proj_out, ref_wv_proj_out, **wv_proj_out_criteria)
        is_correct &= kk.check_is_allclose("max_logits", ans_max_logits, ref_max_logits, **max_logits_criteria)
        is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, **lse_criteria)

        return is_correct
    else:
        return True


@torch.inference_mode()
def run_decode_test(p: TestParam) -> bool:
    assert p.decode is not None
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")

    t = lib.generate_testcase_for_decode(p)
    b = p.decode.b
    s_q = p.s_q

    q_lora_rank = random.choice([1024, 1536])
    q_b_proj_scale_gran = random.choice([32, 128])
    q_lora = torch.randn((b*s_q, q_lora_rank), device='cuda') / 10
    q_lora = tile_kernels.quant.per_token_cast(q_lora, 'e4m3', q_b_proj_scale_gran, None, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    q_b_proj, q_b_proj_permuted = get_q_b_weight(p.h_q, p.d_qk, q_lora_rank, q_b_proj_scale_gran)

    o_scale_gran = 32
    enable_q_norm = random.choice([False, True])
    wv_group_size = 8
    n_wv_group = p.h_q // 8

    def calculate_q_b_proj(weight) -> torch.Tensor:
        q = torch.empty((b*s_q, p.h_q*p.d_qk), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_gemm_nt(q_lora, weight, q, recipe_a=(1, q_b_proj_scale_gran), recipe_b=(1, q_b_proj_scale_gran))
        return q.view(b*s_q, p.h_q, p.d_qk)
    q_for_fused = calculate_q_b_proj(q_b_proj_permuted)
    token_positions = torch.randint(0, max_token_position, (b*s_q, ), device='cuda', dtype=torch.int32)

    def run_fused_norm_rope_attn_rope_cast_decode():
        # NOTE The kernel interface does not have the batch dimension (batch size is always 1),
        # so we squeeze the batch dimension out of every per-request tensor here
        return flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=enable_q_norm,
            rms_norm_eps=rms_norm_eps,
            token_positions=token_positions,
            is_rope_neox_style=False,
            rope_dim=64,
            cos_sin_cache=cos_sin_cache,
            n_wv_group=n_wv_group,
            num_per_channels=o_scale_gran,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
            q=q_for_fused,
            k_cache=t.kv_scope.get_kvcache_for_flash_mla(),
            indices_in_kvcache=t.kv_scope.indices_in_kvcache.reshape(b*s_q, p.topk),
            sm_scale=t.sm_scale,
            d_v=p.d_v,
            attn_sink=t.attn_sink,
            topk_length=t.kv_scope.topk_length.repeat_interleave(s_q, 0) if t.kv_scope.topk_length is not None else None,
            extra_k_cache=t.extra_kv_scope.get_kvcache_for_flash_mla() if t.extra_kv_scope is not None else None,
            extra_indices_in_kvcache=t.extra_kv_scope.indices_in_kvcache.reshape(b*s_q, p.decode.extra_topk) if t.extra_kv_scope is not None else None,
            extra_topk_length=t.extra_kv_scope.topk_length.repeat_interleave(s_q, 0) if t.extra_kv_scope is not None and t.extra_kv_scope.topk_length is not None else None,
        )

    torch.cuda.synchronize()
    ans_out_fp8, ans_out_sf, ans_lse = run_fused_norm_rope_attn_rope_cast_decode()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)
        fused_time = kk.bench_kineto(run_fused_norm_rope_attn_rope_cast_decode, num_tests=p.num_runs).get_kernel_time("fused_norm_rope_attn_rope_cast_fwd")
        fused_flops = flops_and_mem_vol.flop/fused_time/1e12
        fused_mem_bw = flops_and_mem_vol.mem_vol/fused_time/1e12
        print(f"Fused decode: {fused_time*1e6:4.0f} us, {fused_flops:6.1f} TFlops, {fused_mem_bw:4.2f} TBps")

    is_correct = True
    if p.check_correctness:
        out_criteria = {'abs_tol': 1.1e-3, 'rel_tol': 1.01/8, 'cos_diff_tol': 1e-3} if p.k_amplifier_portion == 0.0 else {'abs_tol': 1.0, 'rel_tol': 1.0, 'cos_diff_tol': 1e-3}
        lse_criteria = {'abs_tol': 1e-5, 'rel_tol': 4.01/65536} if p.k_amplifier_portion == 0 else {'abs_tol': 1e-5, 'rel_tol': 4.01/65536}

        t.q = calculate_q_b_proj(q_b_proj).view(b, s_q, p.h_q, p.d_qk)
        ref_out, ref_lse = ref_fused_norm_rope_attn_rope_cast_decode(p, t, enable_q_norm, cos_sin_cache, token_positions)

        # NOTE Tensors returned by the fused decode kernel do not have the batch dimension (batch size is always 1)
        ans_out_sf_2d = ans_out_sf.view(b*s_q, -1)
        if ans_out_sf_2d.stride(0) != 1:
            assert s_q == 1
            ans_out_sf_2d = ans_out_sf_2d.as_strided(ans_out_sf_2d.shape, (1, ans_out_sf_2d.stride(1)))
        ans_wv_out_dequantized = tile_kernels.quant.per_token_cast_back(
            (ans_out_fp8.view(b*s_q, -1), ans_out_sf_2d), 'fp32', 32
        )
        ans_wv_out_dequantized = ans_wv_out_dequantized \
            .view(b*s_q, n_wv_group, p.d_v // 32, wv_group_size, 32) \
            .transpose(2, 3) \
            .reshape(b*s_q, p.h_q, p.d_v)

        is_correct &= kk.check_is_allclose("out", ans_wv_out_dequantized, ref_out.to(torch.float32), **out_criteria)
        is_correct &= kk.check_is_allclose("lse", ans_lse, ref_lse, **lse_criteria)


    return is_correct


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    lib.stick_unit_test_args(parser)
    args = parser.parse_args()

    correctness_cases_prefill = []
    correctness_cases_decode = []

    for h_q in [128, 64]:
        for s_kv, topk in [
            # Regular shapes
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 120),
            (1840, 240),
            (1521, 600),
            (3412, 2896),

            # Irregular shapes with OOB TopK
            (95, 152),
            (153, 264),
            (2345, 5136),

            (32, 2048), # Some block may be fully invalid
        ]:
            for (have_attn_sink, is_all_indices_invalid, have_topk_length) in [
                (False, False, False),
                (True, False, False),
                (True, True, False),
                (True, False, True),
                (False, True, True)
            ]:
                for s_q in [1, 184, 2123]:
                    # Prefill
                    correctness_cases_prefill.extend([TestParam(
                        s_q, s_kv, topk, h_q, 
                        is_all_indices_invalid=is_all_indices_invalid,
                        have_topk_length=have_topk_length,
                        k_amplifier_portion=k_amplifier_portion,
                        k_amplifier_ratio=k_amplifier_ratio,
                        num_runs=0
                    )
                        for (k_amplifier_portion, k_amplifier_ratio) in [
                            (0.0, 1.0),
                            (0.02, 2**8)
                        ]
                    ])
                for (b, s_q) in [(1, 1), (69, 3), (19, 80)]:
                    for extra_s_kv, extra_topk, kvcache_layout, extra_kvcache_layout in [
                        (None, None, quant.KVCacheLayout.V4_FP8Sparse, None),
                        (None, None, quant.KVCacheLayout.V41_FP8Sparse, None),
                        (64, 64, quant.KVCacheLayout.V4_FP8Sparse, None),
                        (32, 2048, quant.KVCacheLayout.V4_FP8Sparse, None),
                        (32, 2048, quant.KVCacheLayout.V41_FP8Sparse, None),
                        (32, 2048, quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4),
                        (592, 120, quant.KVCacheLayout.V4_FP8Sparse, None),
                        (512, 512, quant.KVCacheLayout.V41_FP8Sparse, None),
                        (512, 512, quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4),
                        (334, 432, quant.KVCacheLayout.V4_FP8Sparse, None),
                        (3412, 2896, quant.KVCacheLayout.V41_FP8Sparse, None),
                        (3412, 2896, quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4),
                    ]:
                        # The fp4 extra KV cache requires topk % 8 == 0
                        if extra_kvcache_layout == quant.KVCacheLayout.V41_FP4 and topk % 8 != 0:
                            continue
                        for have_extra_topk_length in [False, True]:
                            if have_extra_topk_length and extra_s_kv is None:
                                continue
                            correctness_cases_decode.append(RawTestParamForDecode(
                                b, h_q, s_q, 1, s_kv, False, topk,
                                is_all_indices_invalid=is_all_indices_invalid,
                                have_zero_seqlen_k=False,
                                have_topk_length=have_topk_length,
                                enable_attn_sink=have_attn_sink,
                                extra_s_k=extra_s_kv,
                                extra_topk=extra_topk,
                                block_size=(64+topk%8),
                                extra_block_size=(128+(extra_topk%8) if extra_topk is not None else None),
                                have_extra_topk_length=have_extra_topk_length,
                                d_qk=512,
                                kvcache_layout=kvcache_layout,
                                extra_kvcache_layout=extra_kvcache_layout,
                                num_runs=0)
                            )

    performance_case_templates = [
        # V4 small
        # (512, 32, 512+128, [8192, 32768]),
        # V4 / V4.1
        (512, 64, 512, 128, [8192, 32768], [(quant.KVCacheLayout.V4_FP8Sparse, None), (quant.KVCacheLayout.V41_FP8Sparse, None), (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4)], [(140, 4), (320, 3)]),
        # V4 / V4.1 teachar
        (512, 128, 1024, 128, [8192, 32768], [(quant.KVCacheLayout.V4_FP8Sparse, None), (quant.KVCacheLayout.V41_FP8Sparse, None), (quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4)], [(80, 4)]),
    ]

    performance_cases_prefill = []
    performance_cases_decode = []
    for (d_qk, h_q, extra_topk, topk, s_kv_list, kvcache_layouts, decoding_bsz_and_s_q) in performance_case_templates:
        for s_kv in s_kv_list:
            for prefill_s_q in [4096]:
                performance_cases_prefill.append(TestParam(prefill_s_q, s_kv, topk+extra_topk, h_q, d_qk=d_qk, have_attn_sink=True, check_correctness=True))
            for (decoding_bsz, decoding_s_q) in decoding_bsz_and_s_q:
                for kvcache_layout, extra_kvcache_layout in kvcache_layouts:
                    performance_cases_decode.append(RawTestParamForDecode(
                        decoding_bsz, h_q, decoding_s_q, 1, topk, False, topk, have_topk_length=False,
                        extra_s_k=s_kv, extra_topk=extra_topk, extra_block_size=128,
                        d_qk=d_qk, kvcache_layout=kvcache_layout, extra_kvcache_layout=extra_kvcache_layout))

    # Prefill testcases
    testcases = correctness_cases_prefill + correctness_cases_decode + performance_cases_prefill + performance_cases_decode
    testcases = [
        (t.to_test_param() if isinstance(t, RawTestParamForDecode) else t)
        for t in testcases
    ]

    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    failed_cases = []
    for test_idx, test in enumerate(testcases):
        if test != testcases[0] and test.num_runs > 0 and not args.no_cooldown:
            time.sleep(0.3)
        print(f'[{test_idx:5d}/{len(testcases):5d} ({test_idx/len(testcases)*100:5.1f}%)] ', end='')
        is_correct = run_test(test) if test.decode is None else run_decode_test(test)
        if not is_correct:
            failed_cases.append(test)
            if not args.run_to_finish:
                sys.exit(1)
    
    total = len(testcases)
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {total} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {total} cases passed!\033[0m")
