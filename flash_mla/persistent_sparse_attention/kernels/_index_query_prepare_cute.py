# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM's DeepSeek-V4 CuTeDSL indexer implementation.
"""CuTeDSL implementation for :mod:`index_query_prepare`."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    const_expr,
)

from .cute_utils import (
    bf16x2_abs,
    bf16x2_max,
    cvt,
    fmax,
    recast_val,
    shr_u32,
    warp_reduce,
)


MXFP4_BLOCK_SIZE = 32
_TORCH_TO_CUTE = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
}


def launch(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    q_packed: torch.Tensor,
    q_scale: torch.Tensor,
    weights_out: torch.Tensor,
    num_sms: int,
) -> None:
    _, num_heads, head_dim = index_q.shape
    rope_dim = cos_sin_cache.shape[-1]
    compiled = _IndexQueryPrepareKernel.compile(
        head_dim,
        rope_dim,
        num_heads,
        _TORCH_TO_CUTE[cos_sin_cache.dtype],
    )
    compiled(
        positions,
        index_q,
        cos_sin_cache,
        index_weights,
        q_packed,
        q_scale,
        weights_out,
        float(softmax_scale * head_scale),
        int(num_sms),
    )


class _IndexQueryPrepareKernel:
    """Eight tokens per 1024-thread CTA with a fixed-grid scheduler."""

    shared_sentinel_bytes = 1024

    def __init__(
        self,
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
    ):
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.num_heads = num_heads
        self.cos_sin_dtype = cos_sin_dtype
        self.coarsen = 4
        if num_heads % self.coarsen:
            raise ValueError("num_heads must be divisible by four")
        self.subwarp_size = head_dim // 16
        self.threads_per_token = (
            num_heads // self.coarsen
        ) * self.subwarp_size
        self.tb_size = 1024
        self.tokens_per_cta = self.tb_size // self.threads_per_token
        if self.tokens_per_cta != 8:
            raise ValueError(
                "accepted schedule requires 8 tokens per CTA "
                "(head_dim=128, num_heads=64)"
            )

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_quant: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
        num_sms: Int32,
        stream: CUstream,
    ):
        self.kernel(
            positions,
            q,
            cos_sin_cache,
            weights,
            q_quant,
            q_scale,
            weights_out,
            scale,
        ).launch(
            grid=(num_sms, 1, 1),
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_quant: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
    ):
        block_id, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        num_ctas, _, _ = cute.arch.grid_dim()

        token_lane = tid // self.threads_per_token
        token_thread = tid % self.threads_per_token
        sublane = token_thread % self.subwarp_size
        head_tile_id = token_thread // self.subwarp_size
        head_start = head_tile_id * self.coarsen

        # This live allocation makes one full CTA the occupancy limit and is
        # also used in the address calculation so it cannot be optimized away.
        smem = cutlass.utils.SmemAllocator()
        shared_sentinel = smem.allocate_tensor(
            Uint8,
            cute.make_layout((self.shared_sentinel_bytes,)),
            byte_alignment=128,
        )
        shared_sentinel[tid] = Uint8(1)
        cute.arch.sync_threads()
        token_lane += shared_sentinel[tid].to(Int32) - Int32(1)

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_u32x8 = cute.make_copy_atom(
            cp_op,
            Uint32,
            num_bits_per_copy=256,
        )
        cp_f32x8 = cute.make_copy_atom(
            cp_op,
            Float32,
            num_bits_per_copy=256,
        )
        cp_u32x4 = cute.make_copy_atom(
            cp_op,
            Uint32,
            num_bits_per_copy=128,
        )
        cp_u32x2 = cute.make_copy_atom(
            cp_op,
            Uint32,
            num_bits_per_copy=64,
        )

        for token_id in range(
            block_id * self.tokens_per_cta + token_lane,
            q.shape[0],
            num_ctas * self.tokens_per_cta,
        ):
            q_bf16x2 = cute.make_rmem_tensor(
                cute.make_layout((self.coarsen, 8), stride=(8, 1)),
                Uint32,
            )
            q_tile = cute.local_tile(
                q[token_id, None, None],
                tiler=(self.coarsen, 16),
                coord=(head_tile_id, sublane),
            )
            for i in cutlass.range_constexpr(self.coarsen):
                cute.copy(
                    cp_u32x8,
                    cute.recast_tensor(q_tile[i, None], Uint32),
                    q_bf16x2[i, None],
                )

            if sublane * 16 >= self.nope_dim:
                cos_values = cute.make_rmem_tensor((8,), Float32)
                sin_values = cute.make_rmem_tensor((8,), Float32)
                pos = positions[token_id]
                cos_id = sublane - self.nope_dim // 16
                sin_id = cos_id + self.rope_dim // 16
                cos_src = cute.local_tile(
                    cos_sin_cache[pos, None],
                    tiler=(8,),
                    coord=(cos_id,),
                )
                sin_src = cute.local_tile(
                    cos_sin_cache[pos, None],
                    tiler=(8,),
                    coord=(sin_id,),
                )
                if const_expr(self.cos_sin_dtype is Float32):
                    cute.copy(cp_f32x8, cos_src, cos_values)
                    cute.copy(cp_f32x8, sin_src, sin_values)
                else:
                    cos_bf16x2 = cute.make_rmem_tensor((4,), Uint32)
                    sin_bf16x2 = cute.make_rmem_tensor((4,), Uint32)
                    cute.copy(
                        cp_u32x4,
                        cute.recast_tensor(cos_src, Uint32),
                        cos_bf16x2,
                    )
                    cute.copy(
                        cp_u32x4,
                        cute.recast_tensor(sin_src, Uint32),
                        sin_bf16x2,
                    )
                    for j in cutlass.range_constexpr(4):
                        cos0, cos1 = cvt.bf16x2_to_fp32x2(cos_bf16x2[j])
                        sin0, sin1 = cvt.bf16x2_to_fp32x2(sin_bf16x2[j])
                        cos_values[j * 2] = cos0
                        cos_values[j * 2 + 1] = cos1
                        sin_values[j * 2] = sin0
                        sin_values[j * 2 + 1] = sin1

                for i in cutlass.range_constexpr(self.coarsen):
                    for j in cutlass.range_constexpr(8):
                        q0, q1 = cvt.bf16x2_to_fp32x2(q_bf16x2[i, j])
                        rot0 = q0 * cos_values[j] - q1 * sin_values[j]
                        rot1 = q0 * sin_values[j] + q1 * cos_values[j]
                        q_bf16x2[i, j] = cvt.fp32x2_to_bf16x2(rot0, rot1)

            q_fp4_tile = cute.local_tile(
                q_quant[token_id, None, None],
                tiler=(self.coarsen, 8),
                coord=(head_tile_id, sublane),
            )
            for i in cutlass.range_constexpr(self.coarsen):
                amax_bf16x2 = bf16x2_abs(q_bf16x2[i, 0])
                for j in cutlass.range_constexpr(1, 8):
                    amax_bf16x2 = bf16x2_max(
                        amax_bf16x2,
                        bf16x2_abs(q_bf16x2[i, j]),
                    )
                amax_bf16x2 = warp_reduce(
                    amax_bf16x2,
                    bf16x2_max,
                    width=MXFP4_BLOCK_SIZE // 16,
                )
                amax_pair = cvt.bf16x2_to_fp32x2(amax_bf16x2)
                amax = fmax(amax_pair[0], amax_pair[1])

                eps = cutlass.const_expr(float.fromhex("0x6p-126"))
                fp4_scale = fmax(amax, eps) * Float32(1.0 / 6.0)
                bits = recast_val(fp4_scale, Uint32)
                ue8m0 = shr_u32(
                    bits + Uint32(0x7FFFFF),
                    Uint32(23),
                ) & Uint32(0xFF)
                if token_thread % 2 == 0:
                    q_scale[
                        token_id,
                        head_start + i,
                        sublane // 2,
                    ] = Uint8(ue8m0)

                inv_scale_bits = (Uint32(254) - ue8m0) << Uint32(23)
                inv_fp4_scale = recast_val(inv_scale_bits, Float32)
                values = cute.make_rmem_tensor(16, Float32)
                for j in cutlass.range_constexpr(8):
                    q0, q1 = cvt.bf16x2_to_fp32x2(q_bf16x2[i, j])
                    values[j * 2] = q0 * inv_fp4_scale
                    values[j * 2 + 1] = q1 * inv_fp4_scale

                packed = cute.make_rmem_tensor((2,), Uint32)
                packed[0] = cvt.fp32x8_to_fp4x8(values, 0)
                packed[1] = cvt.fp32x8_to_fp4x8(values, 8)
                cute.copy(
                    cp_u32x2,
                    packed,
                    cute.recast_tensor(q_fp4_tile[i, None], Uint32),
                )

            if token_thread < self.num_heads:
                weights_out[token_id, token_thread] = (
                    weights[token_id, token_thread].to(Float32) * scale
                )

    @cache
    @staticmethod
    def compile(
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
    ):
        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()
        q = cute.runtime.make_fake_tensor(
            BFloat16,
            (num_tokens, num_heads, head_dim),
            stride=(num_heads * head_dim, head_dim, 1),
            assumed_align=32,
        )
        positions = cute.runtime.make_fake_tensor(
            Int64,
            (num_tokens,),
            stride=(1,),
            assumed_align=1,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            cos_sin_dtype,
            (max_pos, rope_dim),
            stride=(rope_dim, 1),
            assumed_align=32,
        )
        weights = cute.runtime.make_fake_tensor(
            BFloat16,
            (num_tokens, num_heads),
            stride=(num_heads, 1),
            assumed_align=8,
        )
        q_fp4 = cute.runtime.make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // 2),
            stride=(num_heads * (head_dim // 2), head_dim // 2, 1),
            assumed_align=32,
        )
        q_scale = cute.runtime.make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE),
            stride=(
                num_heads * (head_dim // MXFP4_BLOCK_SIZE),
                head_dim // MXFP4_BLOCK_SIZE,
                1,
            ),
            assumed_align=4,
        )
        weights_out = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, num_heads),
            stride=(num_heads, 1),
            assumed_align=4,
        )
        kernel = _IndexQueryPrepareKernel(
            head_dim,
            rope_dim,
            num_heads,
            cos_sin_dtype,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp4,
            q_scale,
            weights_out,
            Float32(0.0),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )


__all__ = ["launch"]
