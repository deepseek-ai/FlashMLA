import dataclasses
import enum
import functools
from typing import Tuple, List, Optional, overload
import random

import argparse
import torch
import kernelkit as kk
if kk.is_on_ascend_platform():
    import torch_npu    # noqa: F401  (importing torch_npu registers the NPU backend of torch)

import flash_mla

import quant


@functools.lru_cache(maxsize=1)
@kk.requires_platform(kk.Platform.CUDA)
def get_current_compute_capability() -> Tuple[int, int]:
    cc_major, cc_minor = torch.cuda.get_device_capability()
    return (cc_major, cc_minor)

def device_synchronize():
    if kk.is_on_cuda_platform():
        torch.cuda.synchronize()
    elif kk.is_on_ascend_platform():
        torch.npu.synchronize()
    else:
        assert False, "Unknown platform"

class TestTarget(enum.Enum):
    SPARSE_FWD = 0
    DECODE = 1

@dataclasses.dataclass
class ExtraTestParamForDecode:
    b: int
    is_varlen: bool
    have_zero_seqlen_k: bool
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False
    kvcache_layout: Optional[quant.KVCacheLayout] = None   # Layout of the main KV cache. Defaults to V41 fp8 (the only supported main-cache format)
    extra_kvcache_layout: Optional[quant.KVCacheLayout] = None   # Layout of the extra KV cache. None: the same as kvcache_layout

@dataclasses.dataclass
class TestParam:
    s_q: int
    s_kv: int
    topk: Optional[int] = None
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    seed: int = -1   # -1: to be filled automatically
    check_correctness: bool = True
    is_all_indices_invalid: bool = False    # All indices are invalid, i.e., all indices are set to a large number (e.g., 2147483647)
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    k_amplifier_portion: float = 0.0
    k_amplifier_ratio: float = 1.0
    decode: Optional[ExtraTestParamForDecode] = None

    def can_run_on_and_clamp(self, test_target: TestTarget) -> bool:
        if kk.is_on_ascend_platform():
            if test_target == TestTarget.SPARSE_FWD:
                return self.h_q == 64 and self.d_qk == 512
            elif test_target == TestTarget.DECODE:
                assert self.decode is not None
                if self.h_q != 64 or self.d_qk != 512:
                    return False
                if self.decode.block_size == 1 or self.decode.extra_block_size == 1:
                    return False
                if self.decode.kvcache_layout != quant.KVCacheLayout.V41_FP8Sparse:
                    return False
                if self.decode.extra_kvcache_layout not in (None, quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse):
                    return False
                return True
            else:
                raise RuntimeError(f"Unknown test_target: {test_target}")
        elif kk.is_on_cuda_platform():
            cc_major, cc_minor = get_current_compute_capability()
            if cc_major != 10:
                return False    # Only sm100a / sm103a is supported
            if self.h_q not in (64, 128) or self.d_qk != 512:
                return False

            if test_target == TestTarget.SPARSE_FWD:
                if self.topk < 128:
                    self.topk = 128
                return True
            elif test_target == TestTarget.DECODE:
                assert self.decode is not None
                return True
            else:
                raise RuntimeError(f"Unknown test_target: {test_target}")
        else:
            raise RuntimeError("Unknown platform")

@dataclasses.dataclass
class RawTestParamForDecode:
    """
    "Flattened" test parameters for decoding test
    
    In our test script, to maintain compatibility with TestParam, we embed decode-only parameters into TestParam.decode, which is not very convinient when construct testcases. So here we have a "flattened" version of test parameters for decoding test.
    """
    b: int
    h_q: int
    s_q: int
    h_kv: int
    s_kv: int
    is_varlen: bool
    topk: int
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    have_topk_length: bool = False
    enable_attn_sink: bool = True
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False
    d_qk: int = 512      # Q/K head dim (= dv + RoPE dim)
    d_v: int = 512     # V head dim
    kvcache_layout: Optional[quant.KVCacheLayout] = None
    extra_kvcache_layout: Optional[quant.KVCacheLayout] = None
    check_correctness: bool = True
    num_runs: int = 10
    seed: int = -1

    def to_test_param(self) -> TestParam:
        return TestParam(
            s_q=self.s_q,
            s_kv=self.s_kv,
            topk=self.topk,
            h_q=self.h_q,
            h_kv=self.h_kv,
            d_qk=self.d_qk,
            d_v=self.d_v,
            seed=self.seed,
            check_correctness=self.check_correctness,
            is_all_indices_invalid=self.is_all_indices_invalid,
            num_runs=self.num_runs,
            have_attn_sink=self.enable_attn_sink,
            have_topk_length=self.have_topk_length,
            decode=ExtraTestParamForDecode(
                b=self.b,
                is_varlen=self.is_varlen,
                have_zero_seqlen_k=self.have_zero_seqlen_k,
                extra_s_k=self.extra_s_k,
                extra_topk=self.extra_topk,
                block_size=self.block_size,
                extra_block_size=self.extra_block_size,
                have_extra_topk_length=self.have_extra_topk_length,
                kvcache_layout=self.kvcache_layout,
                extra_kvcache_layout=self.extra_kvcache_layout,
            )
        )


@dataclasses.dataclass
class Testcase:
    p: TestParam
    q: torch.Tensor     # [s_q, h_q, d_qk]
    kv: torch.Tensor    # [s_kv, h_kv, d_qk]
    indices: torch.Tensor   # [s_q, h_kv, topk]
    sm_scale: float
    attn_sink: Optional[torch.Tensor]   # [h_q]
    topk_length: Optional[torch.Tensor]  # [s_q]

def _randperm_batch(batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]) -> torch.Tensor:
    """
    Generate random permutations in batch
    The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]` holds.
    Values within each row are unique.
    If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
    """
    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
    rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
    rand[torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max) >= perm_range.view(batch_size, 1)] = float("-inf")    # Fill invalid positions, so that the following `topk` operators will select positions within `perm_range` first
    res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
    if len(paddings) == 1:
        res[res >= perm_range.view(batch_size, 1)] = paddings[0]
    else:
        fillers = torch.tensor(paddings, dtype=torch.int32).index_select(0, torch.randint(0, len(paddings), (res.numel(), ), dtype=torch.int32))
        res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
    torch.use_deterministic_algorithms(False)
    return res

def generate_testcase(t: TestParam) -> Testcase:
    kk.set_random_seed(t.seed)
    q = torch.randn((t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16)/10 + (random.random()-0.5)/10
    kv = torch.randn((t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16)/10 + (random.random()-0.5)/10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    invalid_indices_candidate = [-2147483648, -123456, -1, t.s_kv, 114514, 1919810, 2147480000, 2147483647]
    indices = _randperm_batch(t.s_q, torch.full((t.s_q, ), t.s_kv, dtype=torch.int32), t.topk, invalid_indices_candidate).view(t.s_q, t.h_kv, t.topk)
    if t.is_all_indices_invalid:
        all_indices_invalid_mask = torch.randn(t.s_q, device='cpu') < -2
        indices[all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)] = random.choice(invalid_indices_candidate)
    indices = indices.to(q.device)

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = torch.randn((t.h_q, ), dtype=torch.float32)
        mask = torch.randn((t.h_q, ), dtype=torch.float32)
        attn_sink[mask < -0.5] = float("-inf")
        attn_sink[mask > +0.5] = float("+inf")

    topk_length = None
    if t.have_topk_length:
        topk_length = torch.randint(0, max(t.topk + 1, 64), (t.s_q, ), dtype=torch.int32, device=q.device).clamp_max(t.topk)

    if t.k_amplifier_portion > 0.0:
        selected_indices = torch.randint(0, t.s_kv, (int(t.s_kv * t.k_amplifier_portion), ), device=kv.device)
        amplifier_coeffs = torch.rand((selected_indices.size(0), ), device=kv.device) * (t.k_amplifier_ratio - 1) + 1
        kv[selected_indices] *= amplifier_coeffs.unsqueeze(-1).unsqueeze(-1)

    q = kk.non_contiguousify(q)
    if not kk.is_on_ascend_platform():
        # Ascend requires `kv` to be contiguous
        kv = kk.non_contiguousify(kv)
    indices = kk.non_contiguousify(indices)

    return Testcase(
        p=t,
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.5,
        attn_sink=attn_sink,
        topk_length=topk_length
    )


@dataclasses.dataclass
class KVScope:
    t: TestParam
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    blocked_k: torch.Tensor
    abs_indices: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: Optional[torch.Tensor]
    kvcache_layout: Optional[quant.KVCacheLayout]
    blocked_k_quantized: Optional[torch.Tensor] = None

    def quant_and_dequant_(self):
        """
        For FP8 cases, we need to quantize the KV cache for Flash MLA.
        Besides, the quantization error may be too large to be distinguished from wrong kernels, so we de-quantize kvcache here to mitigate quantization error
        """
        kvcache_layout = self.kvcache_layout
        if kvcache_layout is None:
            kvcache_layout = quant.KVCacheLayout.V41_FP8Sparse
        self.blocked_k_quantized = quant.quantize_k_cache(self.blocked_k, kvcache_layout)
        blocked_k_dequantized = quant.dequantize_k_cache(self.blocked_k_quantized, kvcache_layout)
        self.blocked_k = blocked_k_dequantized

    def get_kvcache_for_flash_mla(self) -> torch.Tensor:
        """
        Return the quantized blocked_k for Flash MLA
        """
        assert self.blocked_k_quantized is not None, "Please call `quant_and_dequant_` first before calling `get_kvcache_for_flash_mla`"
        return self.blocked_k_quantized

@dataclasses.dataclass
class TestcaseForDecode:
    p: TestParam
    q: torch.Tensor     # [b, s_q, h_q, d_qk]
    attn_sink: Optional[torch.Tensor]   # [h_q]
    sm_scale: float
    kv_scope: KVScope
    extra_kv_scope: Optional[KVScope]

def generate_testcase_for_decode(t: TestParam) -> TestcaseForDecode:
    kk.set_random_seed(t.seed)
    assert t.h_q % t.h_kv == 0
    assert t.decode is not None

    q = torch.randn((t.decode.b, t.s_q, t.h_q, t.d_qk))
    q.clamp_(min=-1.0, max=1.0)
    q = kk.non_contiguousify(q)

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = torch.randn((t.h_q, ), dtype=torch.float32)
        inf_mask = torch.randn((t.h_q, ), dtype=torch.float32)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")

    def generate_one_k_scope(s_k: int, block_size: int, topk: int, is_varlen: bool, have_zero_seqlen: bool, is_all_indices_invalid: bool, have_topk_length: bool, kvcache_layout: Optional[quant.KVCacheLayout]) -> KVScope:
        b = t.decode.b  # type: ignore
        cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device='cpu')
        if is_varlen:
            for i in range(b):
                cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), t.s_q)

        if have_zero_seqlen:
            zeros_mask = torch.randn(b, dtype=torch.float32, device='cpu') > 0
            cache_seqlens_cpu[zeros_mask] = 0

        max_seqlen_alignment = 4 * block_size
        max_seqlen_pad = max(kk.cdiv(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment), 1) * max_seqlen_alignment
        cache_seqlens = cache_seqlens_cpu.to(torch.get_default_device())

        assert max_seqlen_pad % block_size == 0
        block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
        block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(b, -1)

        # NOTE On torch_npu, dividing a bf16 tensor by a Python scalar materialises a full-size fp32
        # intermediate inside the op (5x the tensor size, measured) -> OOM for multi-GiB caches.
        blocked_k = torch.randn((block_table.numel(), block_size, t.h_kv, t.d_qk)) * 0.1
        blocked_k.clamp_(min=-1.0, max=1.0)
    
        abs_indices = torch.empty((b, t.s_q, topk), dtype=torch.int32)
        if is_all_indices_invalid:
            abs_indices.fill_(-1)
        else:
            abs_indices = _randperm_batch(b*t.s_q, cache_seqlens.repeat_interleave(t.s_q), topk, [-1]).view(b, t.s_q, topk)
        indices_in_kvcache = quant.abs_indices2indices_in_kvcache(abs_indices, block_table, block_size)

        topk_length = torch.randint(0, topk+1, (b, ), dtype=torch.int32, device=q.device) if have_topk_length else None

        # Mask nonused KV as NaN
        if have_topk_length:
            indices_in_kvcache_masked = indices_in_kvcache.clone()
            indices_in_kvcache_masked[torch.arange(0, topk).view(1, 1, topk).broadcast_to(b, t.s_q, topk) >= (topk_length.view(b, 1, 1) if have_topk_length else topk)] = -1
        else:
            indices_in_kvcache_masked = indices_in_kvcache

        blocked_k = blocked_k.view(-1, t.h_kv, t.d_qk)
        nonused_indices_mask = torch.ones(blocked_k.size(0)*blocked_k.size(1), dtype=torch.bool, device='cpu')
        nonused_indices_mask[indices_in_kvcache_masked] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, block_size, t.h_kv, t.d_qk)
    
        block_table = kk.non_contiguousify(block_table)
        abs_indices = kk.non_contiguousify(abs_indices)
        indices_in_kvcache = kk.non_contiguousify(indices_in_kvcache)
        return KVScope(t, cache_seqlens, block_table, blocked_k, abs_indices, indices_in_kvcache, topk_length, kvcache_layout)

    kv_scope0 = generate_one_k_scope(t.s_kv, t.decode.block_size, t.topk, t.decode.is_varlen, t.decode.have_zero_seqlen_k, t.is_all_indices_invalid, t.have_topk_length, t.decode.kvcache_layout)
    kv_scope0.quant_and_dequant_()
    if t.decode.extra_topk is not None:
        if t.decode.extra_s_k is None:
            t.decode.extra_s_k = t.decode.extra_topk*2
        if t.decode.extra_block_size is None:
            t.decode.extra_block_size = t.decode.block_size
        extra_layout = t.decode.extra_kvcache_layout if t.decode.extra_kvcache_layout is not None else t.decode.kvcache_layout
        kv_scope1 = generate_one_k_scope(t.decode.extra_s_k, t.decode.extra_block_size, t.decode.extra_topk, t.decode.is_varlen, t.decode.have_zero_seqlen_k, t.is_all_indices_invalid, t.decode.have_extra_topk_length, extra_layout)
        kv_scope1.quant_and_dequant_()
    else:
        assert t.decode.extra_block_size is None and t.decode.extra_s_k is None and t.decode.have_extra_topk_length == False
        kv_scope1 = None
    
    sm_scale = t.d_qk ** -0.55

    return TestcaseForDecode(t, q, attn_sink, sm_scale, kv_scope0, kv_scope1)


def run_flash_mla_sparse_fwd(p: TestParam, t: Testcase):
    return flash_mla.flash_mla_sparse_fwd(
        t.q, t.kv, t.indices,
        sm_scale=t.sm_scale,
        d_v=p.d_v,
        attn_sink=t.attn_sink,
        topk_length=t.topk_length
    )

def run_flash_mla_decode(p: TestParam, t: TestcaseForDecode, tile_scheduler_metadata, num_splits, bsz_start: int = 0, bsz_end: Optional[int] = None):
    assert p.decode is not None
    b = bsz_end-bsz_start if bsz_end is not None else p.decode.b
    s_q = p.s_q
    squeeze_s_q_with_b = s_q != 1 and kk.is_on_ascend_platform()   # The Ascend decoding kernel only supports s_q=1, so we "squeeze" the s_q dimension and the batch_size dimension into one

    @overload
    def squeeze(t: None) -> None: ...
    @overload
    def squeeze(t: torch.Tensor) -> torch.Tensor: ...
    def squeeze(t: Optional[torch.Tensor]):
        # `t` is expected to have shape `[b, s_q, ...]` and will be reshaped as `[b*s_q, 1, ...]`, if `squeeze_s_q_with_b` is True
        if t is None or not squeeze_s_q_with_b:
            return t
        return t.reshape(b*s_q, 1, *t.shape[2:])
    
    @overload
    def repeat_b_by_s_q(t: None) -> None: ...
    @overload
    def repeat_b_by_s_q(t: torch.Tensor) -> torch.Tensor: ...
    def repeat_b_by_s_q(t: Optional[torch.Tensor]):
        # `t` is expected to have shape `[b, ...]` and will be reshaped as `[b*s_q, ...]`, if `squeeze_s_q_with_b` is True
        if t is None or not squeeze_s_q_with_b:
            return t
        return t.repeat_interleave(p.s_q, dim=0)
    
    out, lse = flash_mla.flash_mla_with_kvcache(
        squeeze(t.q[bsz_start: bsz_end]),
        t.kv_scope.get_kvcache_for_flash_mla(),
        None, None,
        p.d_v,
        tile_scheduler_metadata, num_splits,

        softmax_scale=t.sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=squeeze(t.kv_scope.indices_in_kvcache[bsz_start : bsz_end]),
        attn_sink=t.attn_sink,
        extra_k_cache=t.extra_kv_scope.get_kvcache_for_flash_mla() if t.extra_kv_scope is not None else None,
        extra_indices_in_kvcache=squeeze(t.extra_kv_scope.indices_in_kvcache[bsz_start: bsz_end] if t.extra_kv_scope is not None else None),
        topk_length=repeat_b_by_s_q(t.kv_scope.topk_length[bsz_start: bsz_end] if t.kv_scope.topk_length is not None else None),
        extra_topk_length=repeat_b_by_s_q(t.extra_kv_scope.topk_length[bsz_start: bsz_end] if t.extra_kv_scope is not None and t.extra_kv_scope.topk_length is not None else None)
    )

    if squeeze_s_q_with_b:
        out = out.reshape(b, s_q, *out.shape[2:])
        lse = lse.reshape(b, s_q, p.h_q).transpose(1, 2).contiguous()
    return out, lse


@dataclasses.dataclass
class FlopsAndMemVolStatistics:
    """
    FLOPs and memory volume statistics for prefilling
    """
    num_valid_indices: int
    fwd_flop: float
    fwd_prefill_mem_vol: float
    fwd_prefill_with_fp8_out_mem_vol: float

def count_flop_and_mem_vol(p: TestParam, t: Testcase) -> FlopsAndMemVolStatistics:
    total_topk = (p.s_q*p.topk) if t.topk_length is None else t.topk_length.sum().item()
    indices_valid_mask = (t.indices >= 0) & (t.indices < p.s_kv)
    if t.topk_length is not None:
        indices_valid_mask &= (torch.arange(p.topk)[None, None, :].broadcast_to(p.s_q, p.h_kv, p.topk)) < t.topk_length[:, None, None]
    num_valid_indices = int(indices_valid_mask.sum().item())

    fwd_flop = 2 * total_topk * p.h_q * (p.d_qk + p.d_v)
    fwd_prefill_mem_vol = num_valid_indices*p.d_qk*2 + p.s_q*p.h_q*(p.d_qk+p.d_v)*2
    return FlopsAndMemVolStatistics(
        num_valid_indices,
        fwd_flop,
        fwd_prefill_mem_vol,
        fwd_prefill_mem_vol - p.s_q*p.h_q*p.d_v,    # The FP8 output only stores d_v bytes per element (and no separate SF traffic is counted)
    )

@dataclasses.dataclass
class FlopsAndMemVolStatisticsForDecode:
    """
    FLOPs and memory volume statistics for decoding
    """
    flop: float
    mem_vol: float

def count_flop_and_mem_vol_for_decode(p: TestParam, t: TestcaseForDecode) -> FlopsAndMemVolStatisticsForDecode:
    assert p.decode
    b = p.decode.b

    def get_num_attended_tokens(kv_scope: KVScope) -> int:
        topk = kv_scope.indices_in_kvcache.shape[-1]
        if kv_scope.topk_length is None:
            return b * p.s_q * topk
        else:
            return int(kv_scope.topk_length.sum().item()) * p.s_q
        
    def get_num_retrieved_tokens(kv_scope: KVScope) -> int:
        if kv_scope.topk_length is None:
            indices = kv_scope.indices_in_kvcache
        else:
            indices = kv_scope.indices_in_kvcache.clone()
            batch, s_q, topk = indices.shape
            mask = torch.arange(0, topk, device=indices.device).view(1, 1, topk).broadcast_to(batch, s_q, topk) >= kv_scope.topk_length.view(batch, 1, 1)
            indices[mask] = -1
        num_unique_tokens = indices.unique().numel()    # type: ignore
        return num_unique_tokens

    num_attended_tokens = get_num_attended_tokens(t.kv_scope) + (get_num_attended_tokens(t.extra_kv_scope) if t.extra_kv_scope is not None else 0)
    def kv_bytes(kv_scope: KVScope) -> int:
        layout = kv_scope.kvcache_layout or quant.KVCacheLayout.V41_FP8Sparse
        return get_num_retrieved_tokens(kv_scope) * layout.get_bytes_per_token()

    compute_flop = 2 * p.h_q * num_attended_tokens * (p.d_qk + p.d_v)
    mem_vol = sum([
        2 * b * p.s_q * p.h_q * p.d_qk, # Q
        kv_bytes(t.kv_scope) + (kv_bytes(t.extra_kv_scope) if t.extra_kv_scope is not None else 0),   # K
        2 * b * p.s_q * p.h_q * p.d_v, # O
    ])
    return FlopsAndMemVolStatisticsForDecode(
        compute_flop,
        mem_vol
    )

    
def stick_unit_test_args(parser: argparse.ArgumentParser):
    parser.add_argument("-nc", "--no-cooldown", action="store_true", help="Don't call time.sleep() before performance testcases")
    parser.add_argument("-rf", "--run-to-finish", action="store_true", help="Don't exit when a testcase is failed")
