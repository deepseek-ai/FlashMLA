# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM's DeepSeek-V4 CuTeDSL cache implementation.
"""CuTeDSL implementation for :mod:`kv_cache_gather`."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync

from .cute_utils import bf16x2_mul, cvt


def launch(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    num_sms: int,
) -> None:
    _KvCacheGatherKernel.compile(
        block_size=block_size,
        has_gather_lens=gather_lens is not None,
    )(out, k_cache, seq_lens, gather_lens, block_table, offset, num_sms)


class _KvCacheGatherKernel:
    head_dim = 512
    group_size = 64
    shared_sentinel_bytes = 48 * 1024

    def __init__(
        self,
        fp8_dim: int = 448,
        block_size: int = 64,
    ):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.data_dim = fp8_dim + self.bf16_dim * 2
        self.block_size = block_size
        self.num_warps = 32
        self.tb_size = self.num_warps * 32
        self.num_stages = 4

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,
        k_cache: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor | None,
        block_table: cute.Tensor,
        offset: Int32,
        num_sms: Int32,
        stream: CUstream,
    ):
        # A page stores all token data first and the per-token scale plane
        # second.  This logical split intentionally does not use k_cache's
        # nominal second-mode stride.
        k_data = cute.make_tensor(
            k_cache.iterator,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.data_dim),
                stride=(k_cache.stride[0], self.data_dim, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_cache.iterator + self.block_size * self.data_dim,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, 8),
                stride=(k_cache.stride[0], 8, 1),
            ),
        )
        self.kernel(
            out,
            k_data,
            k_scale,
            seq_lens,
            gather_lens,
            block_table,
            offset,
        ).launch(
            grid=(out.shape[0], num_sms, 1),
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load_g2s(
        self,
        k_data_slice: cute.Tensor,
        k_scale: cute.Tensor,
        block_table: cute.Tensor,
        s_kdata_slice: cute.Tensor,
        s_kscale: cute.Tensor,
        req_id,
        pos,
        lane_id,
        stage_id,
    ):
        op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
        cp16_atom = cute.make_copy_atom(op, Uint32, num_bits_per_copy=128)
        cp8_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            Uint8,
            num_bits_per_copy=64,
        )
        page_id = block_table[req_id, pos // self.block_size]
        block_offset = pos % self.block_size

        index = lane_id
        source = k_data_slice[page_id, block_offset, (None, index)]
        cute.copy(
            cp16_atom,
            cute.recast_tensor(source, Uint32),
            s_kdata_slice[(None, index), stage_id],
        )
        index += 32
        if index < cutlass.const_expr(self.data_dim // 16):
            source = k_data_slice[page_id, block_offset, (None, index)]
            cute.copy(
                cp16_atom,
                cute.recast_tensor(source, Uint32),
                s_kdata_slice[(None, index), stage_id],
            )
        elif index == cutlass.const_expr(self.data_dim // 16):
            cute.copy(
                cp8_atom,
                k_scale[page_id, block_offset, None],
                s_kscale[None, stage_id],
            )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        k_data: cute.Tensor,
        k_scale: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor | None,
        block_table: cute.Tensor,
        offset: Int32,
    ):
        req_id, worker_id, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        _, num_workers, _ = cute.arch.grid_dim()

        smem = cutlass.utils.SmemAllocator()
        s_kdata = smem.allocate_tensor(
            Uint32,
            cute.make_layout(
                (self.data_dim // 4, self.num_warps, self.num_stages)
            ),
            byte_alignment=16,
        )[None, warp_id, None]
        s_kscale = smem.allocate_tensor(
            Uint8,
            cute.make_layout((8, self.num_warps, self.num_stages)),
            byte_alignment=8,
        )[None, warp_id, None]
        sentinel = smem.allocate_tensor(
            Uint8,
            cute.make_layout((self.shared_sentinel_bytes,)),
            byte_alignment=128,
        )
        sentinel[tid] = Uint8(1)
        cute.arch.sync_threads()
        worker_id += sentinel[tid].to(Int32) - Int32(1)

        k_data_slice = cute.logical_divide(k_data, (None, None, 16))
        s_kdata_16b = cute.logical_divide(s_kdata, (4, None))
        s_kdata_8b = cute.logical_divide(s_kdata, (2, None))
        out_slice = cute.logical_divide(out, (None, None, 8))

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp8_atom = cute.make_copy_atom(
            cp_op,
            Uint32,
            num_bits_per_copy=64,
        )
        cp16_atom = cute.make_copy_atom(
            cp_op,
            Uint32,
            num_bits_per_copy=128,
        )

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]  # type: ignore[index]
        start_pos = seq_len - gather_len

        for stage in cutlass.range_constexpr(self.num_stages - 1):
            next_pos = (
                start_pos
                + worker_id * self.num_warps
                + warp_id
                + stage * num_workers * self.num_warps
            )
            if next_pos < seq_len:
                self.load_g2s(
                    k_data_slice,
                    k_scale,
                    block_table,
                    s_kdata_16b,
                    s_kscale,
                    req_id,
                    next_pos,
                    lane_id,
                    stage,
                )
            cute.arch.cp_async_commit_group()
        prefetch_stage = self.num_stages - 1
        compute_stage = 0

        for index in range(
            worker_id * self.num_warps + warp_id,
            gather_len,
            num_workers * self.num_warps,
        ):
            pos = start_pos + index
            next_pos = (
                pos
                + num_workers * self.num_warps * (self.num_stages - 1)
            )
            if next_pos < seq_len:
                self.load_g2s(
                    k_data_slice,
                    k_scale,
                    block_table,
                    s_kdata_16b,
                    s_kscale,
                    req_id,
                    next_pos,
                    lane_id,
                    prefetch_stage,
                )
                prefetch_stage = (prefetch_stage + 1) % self.num_stages
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.num_stages - 1)
            cute.arch.sync_warp()

            data0 = cute.make_rmem_tensor((2,), Uint32)
            data1 = cute.make_rmem_tensor((2,), Uint32)
            cute.copy(
                cp8_atom,
                s_kdata_8b[(None, lane_id), compute_stage],
                data0,
            )
            cute.copy(
                cp8_atom,
                s_kdata_8b[(None, lane_id + 32), compute_stage],
                data1,
            )

            scale0_u32 = Uint32(
                s_kscale[
                    lane_id * 8 // self.group_size,
                    compute_stage,
                ]
            )
            scale0_bf16x2 = (
                scale0_u32 << Uint32(23)
            ) | (scale0_u32 << Uint32(7))
            scale1_u32 = Uint32(
                s_kscale[
                    (lane_id + 32) * 8 // self.group_size,
                    compute_stage,
                ]
            )
            scale1_bf16x2 = (
                scale1_u32 << Uint32(23)
            ) | (scale1_u32 << Uint32(7))

            dequant0 = cute.make_rmem_tensor(4, Uint32)
            dequant1 = cute.make_rmem_tensor(4, Uint32)
            for j in cutlass.range_constexpr(2):
                tmp0 = cvt.fp8x4_to_bf16x4(data0[j])
                tmp1 = cvt.fp8x4_to_bf16x4(data1[j])
                dequant0[j * 2] = bf16x2_mul(
                    tmp0[0],
                    scale0_bf16x2,
                )
                dequant1[j * 2] = bf16x2_mul(
                    tmp1[0],
                    scale1_bf16x2,
                )
                dequant0[j * 2 + 1] = bf16x2_mul(
                    tmp0[1],
                    scale0_bf16x2,
                )
                dequant1[j * 2 + 1] = bf16x2_mul(
                    tmp1[1],
                    scale1_bf16x2,
                )

            if lane_id + 32 >= self.fp8_dim // 8:
                source_index = (
                    self.fp8_dim // 16
                    + lane_id
                    + 32
                    - self.fp8_dim // 8
                )
                cute.copy(
                    cp16_atom,
                    s_kdata_16b[(None, source_index), compute_stage],
                    dequant1,
                )

            destination = out_slice[
                req_id,
                offset + index,
                (None, lane_id),
            ]
            cute.copy(
                cp16_atom,
                dequant0,
                cute.recast_tensor(destination, Uint32),
            )
            destination = out_slice[
                req_id,
                offset + index,
                (None, lane_id + 32),
            ]
            cute.copy(
                cp16_atom,
                dequant1,
                cute.recast_tensor(destination, Uint32),
            )
            compute_stage = (compute_stage + 1) % self.num_stages

    @cache
    @staticmethod
    def compile(
        fp8_dim: int = 448,
        block_size: int = 64,
        has_gather_lens: bool = True,
    ):
        num_reqs = cute.sym_int()
        head_dim = _KvCacheGatherKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8
        output_capacity = cute.sym_int()
        num_blocks = cute.sym_int()
        table_width = cute.sym_int()

        out = cute.runtime.make_fake_tensor(
            BFloat16,
            (num_reqs, output_capacity, head_dim),
            stride=(output_capacity * head_dim, head_dim, 1),
            assumed_align=16,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (num_blocks, block_size, head_bytes),
            stride=(cute.sym_int64(divisibility=32), head_bytes, 1),
            assumed_align=32,
        )
        seq_lens = cute.runtime.make_fake_tensor(
            Int32,
            (num_reqs,),
            stride=(1,),
            assumed_align=4,
        )
        gather_lens = (
            cute.runtime.make_fake_tensor(
                Int32,
                (num_reqs,),
                stride=(1,),
                assumed_align=4,
            )
            if has_gather_lens
            else None
        )
        block_table = cute.runtime.make_fake_tensor(
            Int32,
            (num_reqs, table_width),
            stride=(table_width, 1),
            assumed_align=4,
        )
        kernel = _KvCacheGatherKernel(fp8_dim, block_size)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            Int32(0),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )


__all__ = ["launch"]
