#pragma once
//
// Sparse MLA bwd host wrapper.
//
// Builds the kernel's Arguments / Params from our SparseAttnBwdParams + fills in
// the 3 sparse CUtensorMap descriptors (gather kv, scatter dkv, store kl_target),
// then launches via cute::launch_kernel_on_cluster.
//
// This file MUST be included in a .cu compilation unit (not the .hpp) so that
// CUTLASS_CUDA_DRIVER_WRAPPER_CALL's cuTensorMapEncodeTiled shim resolves.
//

#include <cuda.h>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"

#include <ATen/ATen.h>     // at::zeros for PyTorch-allocator-managed
                           // dq_acc / dkv_acc workspaces.
#include "params.h"   // SparseAttnBwdParams
#include "sum_OdO.cuh"
#include "sparse_bwd_kernel.hpp"
#include "config.h"

namespace sm100::sparse_bwd {

using namespace cute;

// FP32 dq_acc -> BF16 cast with softmax_scale multiplication.
// Standard FA backward: dQ = softmax_scale * dS @ K (chain through s = scale * Q @ K^T).
// Dense MLA bwd applies scale to dK in-kernel (line ~1202) but leaves dQ unscaled
// for the in-kernel accumulator; we apply it here in the post-pass.
__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    cutlass::bfloat16_t* __restrict__ dst,
    int total_elems,
    float softmax_scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total_elems; i += stride) {
        dst[i] = cutlass::bfloat16_t(src[i] * softmax_scale);
    }
}

// FP32 dkv_acc -> bf16 dkv cast. softmax_scale was already applied
// in-kernel for dK contributions; dV has no scale. So this cast is straight
// fp32 -> bf16 (no scale).
__global__ void cast_fp32_to_bf16_kernel_noscale(
    const float* __restrict__ src,
    cutlass::bfloat16_t* __restrict__ dst,
    int total_elems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total_elems; i += stride) {
        dst[i] = cutlass::bfloat16_t(src[i]);
    }
}

// The dense-MLA-bwd-derived kernel struct with our shape + cluster constants.
// TileShape from config.h: <_64, _64, _192, _128>.
using SparseKernel = ::cutlass::fmha::kernel::Sm100SparseBwdMlaKernelTmaWarpSpecialized<
    /*ProblemShape=*/cute::tuple<int, int, int, int, cute::tuple<int, int>>,
    /*Element=*/Element,
    /*ElementAcc=*/ElementAcc,
    /*TileShape=*/TileShape,
    /*Mask=*/::cutlass::fmha::collective::CausalForBackwardMask<false>
>;

// __global__ wrapper that adapts the device-side operator() to launch_kernel_on_cluster.
//
// CRITICAL: `__grid_constant__` is required for params containing raw CUtensorMap
// fields. Without it, TMA instructions (cp.async.bulk.tensor) that dereference
// the descriptor via kernel-parameter memory hit illegal memory access. Sparse
// FWD uses the same annotation in phase1.cuh's sparse_attn_fwd_kernel.
__global__ __launch_bounds__(SparseKernel::MaxThreadsPerBlock, SparseKernel::MinBlocksPerMultiprocessor)
void sparse_bwd_kernel_entry(__grid_constant__ const typename SparseKernel::Params params) {
    extern __shared__ char smem[];
    SparseKernel{}(params, smem);
}

// Build the 3 sparse CUtensorMap descriptors. Called from `run_sparse_bwd_kernel`
// after to_underlying_arguments produces the Params struct.
inline void build_sparse_descriptors(
    typename SparseKernel::Params& params,
    const SparseAttnBwdParams& bwd_params)
{
    using bf16 = ::cutlass::bfloat16_t;

    // tensor_map_kv -- bf16 [d_qk, s_kv], gather-aligned.
    {
        uint64_t size[2]        = {(uint64_t)bwd_params.d_qk, (uint64_t)bwd_params.s_kv};
        uint64_t stride[1]      = {(uint64_t)bwd_params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2]    = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &params.mainloop_params.tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            /*tensorRank=*/2,
            (void*)bwd_params.kv,
            size, stride, box_size, elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "FATAL sparse_bwd: tensor_map_kv encode failed: %d\n", (int)res);
            std::abort();
        }
    }

    // tensor_map_dkv -- bf16 [d_qk, s_kv], scatter target.
    {
        uint64_t size[2]        = {(uint64_t)bwd_params.d_qk, (uint64_t)bwd_params.s_kv};
        uint64_t stride[1]      = {(uint64_t)bwd_params.stride_dkv_s_kv * sizeof(bf16)};
        uint32_t box_size[2]    = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &params.mainloop_params.tensor_map_dkv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            /*tensorRank=*/2,
            (void*)bwd_params.dkv,
            size, stride, box_size, elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "FATAL sparse_bwd: tensor_map_dkv encode failed: %d\n", (int)res);
            std::abort();
        }
    }

    // tensor_map_kl_target -- FP32 [topk, s_q]. SWIZZLE_NONE because the
    // 128B-swizzled FP32 box would need box[0]=32 (not 64).
    if (bwd_params.kl_target != nullptr) {
        uint64_t size[2]        = {(uint64_t)bwd_params.topk, (uint64_t)bwd_params.s_q};
        uint64_t stride[1]      = {(uint64_t)bwd_params.topk * sizeof(float)};
        uint32_t box_size[2]    = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &params.mainloop_params.tensor_map_kl_target,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            /*tensorRank=*/2,
            (void*)bwd_params.kl_target,
            size, stride, box_size, elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "FATAL sparse_bwd: tensor_map_kl_target encode failed: %d\n", (int)res);
            std::abort();
        }
    }
}

// Use the same ProblemShape type the SparseKernel template instance is parameterised
// on (it doesn't re-expose it as an inner typedef).
using KernelProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;

// Host entry: build Arguments + Params, encode sparse descriptors, launch.
inline void run_sparse_bwd_kernel(SparseAttnBwdParams& bwd_params) {
    using bf16 = Element;
    using PS   = KernelProblemShape;

    // Stage 1: precompute d_row = sum_d(O * dO) via the standalone kernel.
    launch_sum_OdO<D_V>(bwd_params);
    {
        cudaError_t e = cudaStreamSynchronize(bwd_params.stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "FATAL sparse_bwd: sum_OdO failed: %s\n", cudaGetErrorString(e));
            std::abort();
        }
    }

    // Stage 2: build dense-MLA-bwd-style Arguments. Pointers and strides come
    // from our SparseAttnBwdParams; the kernel's load() fires
    // `ku::tma_gather4_cta_group_1` instead of cute::copy for K and V.
    // with TileShapeQ=B_H=h_q=64, each Q-block IS one (s_q_idx)
    // token's full head batch. Dense's tiling expects Q (seqlen) / TileShapeQ
    // = num_Q_blocks. For sparse we want num_Q_blocks == s_q so that
    // get<1>(blk_coord) ranges 0..s_q-1 and indexes Q tokens directly.
    // Achieve this by passing Q = s_q * h_q (flatten heads into the Q seqlen
    // axis; HB carries H=1 since h_q is now in the Q dim's stride layout).
    PS problem_shape = cute::make_tuple(
        /*Q=*/bwd_params.s_q * bwd_params.h_q,
        /*K=*/bwd_params.s_kv,
        /*D=*/bwd_params.d_qk,
        /*D_VO=*/bwd_params.d_v,
        /*HB=*/cute::make_tuple(/*H=*/1, /*B=*/1));

    using TensorStride    = typename SparseKernel::TensorStride;
    using IndicesStride   = typename SparseKernel::IndicesStride;
    using RowTensorStride = typename SparseKernel::RowTensorStride;

    // Stride layout matches dense MLA bwd's `TensorStrideContiguousK = Stride<int,_1,Stride<int,int>>`
    // i.e. (seq_stride, _1, (head_stride, batch_stride)). For B==1 (DSA always single-batch),
    // batch_stride is _0 (treated as such by the helper at line 155-157 of dense's
    // fmha_cutlass_bwd_sm100.cuh -- we mirror).
    // With Q axis = s_q * h_q (flattened), consecutive Q rows are at stride
    // q.stride(1) (the h-stride = d_qk). H is degenerate (H=1).
    auto stride_q  = cute::make_stride(bwd_params.stride_q_h,    cute::_1{},
                                       cute::make_stride(0,                       0));
    auto stride_kv = cute::make_stride(bwd_params.stride_kv_s_kv, cute::_1{},
                                       cute::make_stride(bwd_params.stride_kv_h, 0));
    // O, dO, dQ also flatten heads into the Q axis (same s_q*h_q convention).
    auto stride_o  = cute::make_stride(bwd_params.stride_o_h,    cute::_1{},
                                       cute::make_stride(0,                       0));
    auto stride_do = cute::make_stride(bwd_params.stride_do_h,   cute::_1{},
                                       cute::make_stride(0,                       0));
    // Indices stays per-s_q (one int32 vec per s_q token, indep of head).
    auto stride_indices = cute::make_stride(bwd_params.stride_indices_s_q, cute::_1{},
                                            cute::make_stride(bwd_params.stride_indices_h, 0));
    // LSE / sum_OdO are [s_q, h_q] -- with Q axis flattened, stride between
    // consecutive Q-elements is 1 (contiguous in (s_q, h_q) row-major).
    auto stride_lse     = cute::make_stride(cute::_1{},
                                            cute::make_stride(0, 0));
    auto stride_sumodo  = cute::make_stride(cute::_1{},
                                            cute::make_stride(0, 0));
    // dQ stride mirrors Q.
    auto stride_dq      = cute::make_stride(bwd_params.stride_dq_h,  cute::_1{},
                                            cute::make_stride(0,                          0));
    auto stride_dkv     = cute::make_stride(bwd_params.stride_dkv_s_kv, cute::_1{},
                                            cute::make_stride(bwd_params.stride_dkv_h, 0));

    // caller-supplied workspaces (dense FMHA BWD pattern).
    // Python wrapper allocates persistent fp32 buffers at
    // module level and passes them via SparseAttnBwdParams. Eliminates:
    //   - cudaMallocAsync illegal access (887/895/926/932 random rank FATAL)
    //   - at::zeros OOM (940/943: PyTorch caching allocator accumulation)
    // Kernel just memsets and uses; alloc lifetime is caller responsibility.
    const size_t dq_acc_bytes  = (size_t)bwd_params.s_q * bwd_params.h_q
                                  * bwd_params.d_qk * sizeof(ElementAcc);
    const size_t dkv_acc_bytes = (size_t)bwd_params.s_kv * bwd_params.d_qk
                                  * sizeof(ElementAcc);
    ElementAcc* ptr_dq_acc  = bwd_params.dq_acc_workspace;
    ElementAcc* ptr_dkv_acc = bwd_params.dkv_acc_workspace;
    cudaMemsetAsync(ptr_dq_acc,  0, dq_acc_bytes,  bwd_params.stream);
    cudaMemsetAsync(ptr_dkv_acc, 0, dkv_acc_bytes, bwd_params.stream);

    typename SparseKernel::Arguments args{
        problem_shape,
        typename SparseKernel::MainloopArguments{
            (const Element*)bwd_params.q,        stride_q,
            (const Element*)bwd_params.kv,       stride_kv,    // ptr_kv, stride_kv
            (const Element*)bwd_params.kv,       stride_kv,    // ptr_k alias
            (const Element*)bwd_params.kv,       stride_kv,    // ptr_v alias
            (const Element*)bwd_params.do_grad,  stride_do,
            bwd_params.indices,                  stride_indices,
            bwd_params.topk,
            bwd_params.lse,                      stride_lse,
            bwd_params.d_row,                    stride_sumodo,
            ptr_dq_acc,                          stride_dq,    // FP32 dq accumulator (TMA REDUCE_ADD target)
            bwd_params.sm_scale,
        },
        typename SparseKernel::EpilogueArguments{
            (Element*)((char*)bwd_params.dkv + 0),                              stride_dkv,  // ptr_dk
            (Element*)((char*)bwd_params.dkv + 0),                              stride_dkv,  // ptr_dv (same buffer; V occupies dkv[:,:,:D_V])
            ptr_dkv_acc,                                                                       // ptr_dkv_acc (restored: FP32 workspace, deterministic atomic noise floor)
            bwd_params.kl_target,                                                              // ptr_kl_target (FP32 or nullptr)
        },
        cutlass::KernelHardwareInfo{0 /*device_id*/, bwd_params.num_sms},
    };

    auto params = SparseKernel::to_underlying_arguments(args, /*workspace=*/nullptr);
    build_sparse_descriptors(params, bwd_params);

    // D path: single CTA per Q-token. Each CTA loops over 3 d_qk chunks
    // (chunk_idx 0..2 -> cols [0,192), [192,384), [384,576)) for S/dK/dQ.
    // No cluster, no cross-CTA partial-S. SMEM unchanged (1 chunk at a time).
    dim3 grid(/*x=*/bwd_params.s_q, /*y=*/1, /*z=*/1);
    dim3 block(SparseKernel::MaxThreadsPerBlock, 1, 1);
    dim3 cluster(1, 1, 1);
    size_t smem_size = SparseKernel::SharedStorageSize;

    // Print kernel config once per process (was per-call, flooded 8-node logs).
    {
        static bool printed = false;
        if (!printed) {
            fprintf(stderr, "[sparse_bwd] SharedStorageSize = %zu bytes, sizeof(Params) = %zu bytes, "
                            "grid=(%d, %d, %d), block=%d\n",
                    smem_size, sizeof(typename SparseKernel::Params),
                    grid.x, grid.y, grid.z, block.x);
            printed = true;
        }
    }
    cudaError_t set_err = cudaFuncSetAttribute(
        (void*)&sparse_bwd_kernel_entry,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (set_err != cudaSuccess) {
        fprintf(stderr, "[sparse_bwd] FATAL cudaFuncSetAttribute(MaxDynamicSMEM=%zu) failed: %s\n",
                smem_size, cudaGetErrorString(set_err));
        // ATen tensor cleanup is automatic on scope exit.
        std::abort();
    }
    // Reverted experiment: direct cudaLaunchKernel with cluster<1,1,1>.
    void* kernel_args[1] = { (void*)&params };
    cudaError_t launch_err = cudaLaunchKernel(
        (void*)&sparse_bwd_kernel_entry,
        grid, block, kernel_args, smem_size, bwd_params.stream);
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[sparse_bwd] FATAL cudaLaunchKernel: %s (grid=(%d,%d,%d), block=%d, smem=%zu)\n",
                cudaGetErrorString(launch_err),
                grid.x, grid.y, grid.z, block.x, smem_size);
        // ATen tensor cleanup is automatic on scope exit.
        std::abort();
    }

    // Diagnostic sync.
    cudaError_t e = cudaStreamSynchronize(bwd_params.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "FATAL sparse_bwd: sparse_bwd_kernel_entry failed: %s\n",
                cudaGetErrorString(e));
        // ATen tensor cleanup is automatic on scope exit.
        std::abort();
    }

    // Debug diagnostics: peek at dq_acc + LSE + d_row. Env-gated because the
    // CPU loop over the full dq_acc + 3 stream-sync cudaMemcpys per call
    // dominates iter time and floods logs in multi-rank training runs.
    if (getenv("FLASH_MLA_SPARSE_BWD_DIAG") &&
        atoi(getenv("FLASH_MLA_SPARSE_BWD_DIAG")) != 0) {
        cudaStreamSynchronize(bwd_params.stream);
        size_t dq_total = (size_t)bwd_params.s_q * bwd_params.h_q * bwd_params.d_qk;
        float* h_full = (float*)malloc(dq_total * sizeof(float));
        cudaMemcpy(h_full, ptr_dq_acc, dq_total*sizeof(float), cudaMemcpyDeviceToHost);
        int nan_cnt=0, inf_cnt=0, nz=0; float maxabs=0;
        for (size_t i=0; i<dq_total; ++i) {
            float v = h_full[i];
            if (v != v) ++nan_cnt;
            else if (!isfinite(v)) ++inf_cnt;
            else if (v != 0.f) { ++nz; if (fabsf(v) > maxabs) maxabs = fabsf(v); }
        }
        fprintf(stderr, "[sparse_bwd dq_acc] total=%zu nan=%d inf=%d nz=%d maxabs=%g\n",
                dq_total, nan_cnt, inf_cnt, nz, maxabs);
        free(h_full);
        float h_lse[8] = {0};
        cudaMemcpy(h_lse, bwd_params.lse, 8*sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[sparse_bwd lse] %g %g %g %g %g %g %g %g\n",
                h_lse[0], h_lse[1], h_lse[2], h_lse[3],
                h_lse[4], h_lse[5], h_lse[6], h_lse[7]);
        float h_drow[8] = {0};
        cudaMemcpy(h_drow, bwd_params.d_row, 8*sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[sparse_bwd d_row] %g %g %g %g %g %g %g %g\n",
                h_drow[0], h_drow[1], h_drow[2], h_drow[3],
                h_drow[4], h_drow[5], h_drow[6], h_drow[7]);
    }

    // Cast FP32 dq_acc -> bf16 bwd_params.dq.
    int cast_total_elems = bwd_params.s_q * bwd_params.h_q * bwd_params.d_qk;
    int cast_block = 256;
    int cast_grid = (cast_total_elems + cast_block * 4 - 1) / (cast_block * 4);
    cast_fp32_to_bf16_kernel<<<cast_grid, cast_block, 0, bwd_params.stream>>>(
        ptr_dq_acc, bwd_params.dq, cast_total_elems, bwd_params.sm_scale);

    // caller manages workspace lifetime — DO NOT cudaFreeAsync here.
    // The ptr came from PyTorch caching allocator (torch.empty in Python);
    // calling cudaFreeAsync on it = cudaErrorInvalidValue.

    // restored: cast FP32 dkv_acc -> bf16 bwd_params.dkv (no scale; scale
    // was applied in-kernel for dK contributions, dV has no scale).
    int dkv_cast_total = bwd_params.s_kv * bwd_params.d_qk;
    int dkv_cast_block = 256;
    int dkv_cast_grid  = (dkv_cast_total + dkv_cast_block * 4 - 1) / (dkv_cast_block * 4);
    cast_fp32_to_bf16_kernel_noscale<<<dkv_cast_grid, dkv_cast_block, 0, bwd_params.stream>>>(
        ptr_dkv_acc, bwd_params.dkv, dkv_cast_total);

    // caller manages workspace lifetime — no cudaFreeAsync.
}

}  // namespace sm100::sparse_bwd
