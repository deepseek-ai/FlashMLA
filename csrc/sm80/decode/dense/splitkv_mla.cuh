#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/cuda/CUDAContext.h>

#include "params.h"
#include "utils.h"
#include "config.h"
#include "traits.h"
#include "../../utils.cuh"

namespace sm80 {

static constexpr float LOG2_E = 1.44269504088896340736f;

// =============================================================================
// Phase 3 -- BLOCK_M=16 + 4-wg V-quarter + double sK buffer.
//
// Threading: 128 threads / CTA = 4 warpgroups x 1 warp x 32 lanes.
//   Each wg = 1 warp covers the M dim (16 rows) entirely via mma.m16n8k16.
//   Wgs split V columns into 4 quarters: wg w -> V[w*128 : (w+1)*128].
//   All wgs compute QK^T (4x duplicated, but per-warp QK^T is short).
//
// SMEM layout (162 KB / 164 KB cap):
//   sQ              : 16 x 576 BF16 =  18 KB
//   sK[0]           : 64 x 576 BF16 =  72 KB
//   sK[1]           : 64 x 576 BF16 =  72 KB
//
// Cross-block prefetch: while iter i computes on sK[stage], iter i+1's K is
// already cp.async-issued into sK[1-stage]. wait_group<1> at iter start blocks
// only on the current stage's load, leaving the next stage's load in flight.
// =============================================================================

template<typename T>
__global__ void __launch_bounds__(cfg::NUM_THREADS, 1)
flash_fwd_splitkv_mla_kernel_sm80(__grid_constant__ const DenseAttnDecodeParams params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using sm80::cfg::BLOCK_SIZE_M;
    using sm80::cfg::PAGE_BLOCK_SIZE;
    using sm80::cfg::HEAD_DIM_K;
    using sm80::cfg::HEAD_DIM_V;
    using sm80::cfg::NUM_THREADS;
    using sm80::cfg::NUM_WARPGROUPS;
    using sm80::cfg::HEAD_DIM_V_PER_WG;
    using sm80::cfg::SMEM_STRIDE_K;
    using sm80::cfg::SK_STAGES;

    constexpr int N_TILES_PER_WG = HEAD_DIM_V_PER_WG / 8;  // 16 PV N-tiles per wg
    constexpr int QK_K_TILES     = HEAD_DIM_K / 16;        // 36
    constexpr int QK_N_TILES     = PAGE_BLOCK_SIZE / 8;    // 8
    constexpr int PV_K_TILES     = PAGE_BLOCK_SIZE / 16;   // 4

    extern __shared__ char smem_buf[];
    T* sQ    = reinterpret_cast<T*>(smem_buf);
    T* sK[SK_STAGES];
    {
        char* sK_base = smem_buf + cfg::smem_q_bytes<T>();
        #pragma unroll
        for (int s = 0; s < SK_STAGES; ++s) {
            sK[s] = reinterpret_cast<T*>(sK_base + s * cfg::smem_k_bytes<T>());
        }
    }

    const int tid          = threadIdx.x;
    const int warp_idx     = tid / 32;        // 0..3
    const int lane_idx     = tid % 32;
    const int wg_idx       = warp_idx;        // 1 warp / wg, so wg_idx == warp_idx

    const int m_block_idx   = blockIdx.x;
    const int k_head_idx    = blockIdx.y;
    const int partition_idx = blockIdx.z;

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    // ---- cp.async tiling (16-byte chunks) ----
    constexpr int Q_BYTES_PER_ROW   = HEAD_DIM_K * sizeof(T);
    constexpr int CHUNK_BYTES       = 16;
    constexpr int CHUNKS_PER_ROW    = Q_BYTES_PER_ROW / CHUNK_BYTES;     // 72
    constexpr int Q_TOTAL_CHUNKS    = BLOCK_SIZE_M * CHUNKS_PER_ROW;     // 16 * 72 = 1152
    constexpr int Q_PER_TID_BASE    = Q_TOTAL_CHUNKS / NUM_THREADS;      // 4 for 256 thread
    constexpr int Q_REMAINDER       = Q_TOTAL_CHUNKS - Q_PER_TID_BASE * NUM_THREADS;  // 128
    constexpr int K_TOTAL_CHUNKS    = PAGE_BLOCK_SIZE * CHUNKS_PER_ROW;  // 64 * 72 = 4608
    constexpr int K_CHUNKS_PER_TID  = K_TOTAL_CHUNKS / NUM_THREADS;      // 18
    constexpr int ELEMS_PER_CHUNK   = CHUNK_BYTES / sizeof(T);           // 8
    static_assert(Q_BYTES_PER_ROW % CHUNK_BYTES == 0,                "Q row not 16B aligned");
    static_assert(K_TOTAL_CHUNKS % NUM_THREADS == 0,                 "K chunks not divisible");

    // ---- per-block-iter K load helper ----
    auto issue_k_load = [&](int block_idx, int stage, const int* block_table_ptr) {
        int kv_block_index = __ldg(block_table_ptr + block_idx);
        const T* gK_block = (const T*)params.k_ptr
            + (int64_t)kv_block_index * params.k_batch_stride
            + k_head_idx * params.k_head_stride;
        T* sK_dst = sK[stage];
        #pragma unroll
        for (int i = 0; i < K_CHUNKS_PER_TID; ++i) {
            int chunk_idx    = tid * K_CHUNKS_PER_TID + i;
            int row          = chunk_idx / CHUNKS_PER_ROW;
            int chunk_in_row = chunk_idx % CHUNKS_PER_ROW;
            int elem_offset  = chunk_in_row * ELEMS_PER_CHUNK;
            int swiz_off     = swizzle_col_bf16(row, elem_offset);
            const T* g_src = gK_block + row * params.k_row_stride + elem_offset;
            uint32_t s_dst = cvta_to_shared_u32(sK_dst + row * SMEM_STRIDE_K + swiz_off);
            cp_async_16_cg(s_dst, g_src);
        }
        cp_async_commit_group();
    };

    // -----------------------------------------------------------------
    // batch loop
    // -----------------------------------------------------------------
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        const int seqlen_k       = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx= batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        const int end_block_idx  = batch_idx == sched_meta.end_req_idx
                                    ? sched_meta.end_block_idx
                                    : (seqlen_k + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE;

        const T* gQ = (const T*)params.q_ptr
            + batch_idx  * params.q_batch_stride
            + m_block_idx* BLOCK_SIZE_M * params.q_row_stride
            + k_head_idx * params.q_head_stride;
        T* gO = (T*)params.o_ptr
            + batch_idx  * params.o_batch_stride
            + m_block_idx* BLOCK_SIZE_M * params.o_row_stride
            + k_head_idx * params.o_head_stride;
        float* gLse = params.softmax_lse_ptr
            + (batch_idx * params.h_k + k_head_idx) * params.q_seq_per_hk
            + m_block_idx * BLOCK_SIZE_M;
        const int* block_table_ptr = params.block_table + batch_idx * params.block_table_batch_stride;

        const int num_valid_seq_q = min(params.q_seq_per_hk - m_block_idx * BLOCK_SIZE_M, BLOCK_SIZE_M);

        // ---- per-row right border (causal + OOB) ----
        int rRightBorder[2];
        {
            int base_row = m_block_idx * BLOCK_SIZE_M;
            int row_lo   = base_row + (lane_idx / 4);
            int row_hi   = row_lo + 8;
            auto rb = [&](int row) -> int {
                if (params.is_causal && row < params.q_seq_per_hk) {
                    int s_q_idx  = row / params.q_head_per_hk;
                    int mask_len = params.s_q - s_q_idx - 1;
                    return max(0, seqlen_k - mask_len);
                }
                return seqlen_k;
            };
            rRightBorder[0] = rb(row_lo);
            rRightBorder[1] = rb(row_hi);
        }

        // ---- per-batch register state ----
        // rO[N_TILES_PER_WG][4]: 16 N-tiles x 4 fp32 = 64 fp32/thread (half of Phase 2's 128)
        float rO[N_TILES_PER_WG][4];
        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WG; ++n) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) rO[n][j] = 0.0f;
        }
        float rL[2] = {0.0f, 0.0f};
        float rM[2] = {-INFINITY, -INFINITY};
        const float scale_log2 = params.scale_softmax_log2;

        // ---- Load Q tile ----
        // Each thread loads Q_PER_TID_BASE chunks; the first Q_REMAINDER threads
        // load one extra chunk to cover Q_TOTAL_CHUNKS that isn't a multiple of NUM_THREADS.
        #pragma unroll
        for (int i = 0; i < Q_PER_TID_BASE; ++i) {
            int chunk_idx    = tid * Q_PER_TID_BASE + i;
            int row          = chunk_idx / CHUNKS_PER_ROW;
            int chunk_in_row = chunk_idx % CHUNKS_PER_ROW;
            int elem_offset  = chunk_in_row * ELEMS_PER_CHUNK;
            int swiz_off     = swizzle_col_bf16(row, elem_offset);
            const T* g_src = gQ + row * params.q_row_stride + elem_offset;
            uint32_t s_dst = cvta_to_shared_u32(sQ + row * SMEM_STRIDE_K + swiz_off);
            cp_async_16(s_dst, g_src);
        }
        if (tid < Q_REMAINDER) {
            int chunk_idx    = NUM_THREADS * Q_PER_TID_BASE + tid;
            int row          = chunk_idx / CHUNKS_PER_ROW;
            int chunk_in_row = chunk_idx % CHUNKS_PER_ROW;
            int elem_offset  = chunk_in_row * ELEMS_PER_CHUNK;
            int swiz_off     = swizzle_col_bf16(row, elem_offset);
            const T* g_src = gQ + row * params.q_row_stride + elem_offset;
            uint32_t s_dst = cvta_to_shared_u32(sQ + row * SMEM_STRIDE_K + swiz_off);
            cp_async_16(s_dst, g_src);
        }
        cp_async_commit_group();

        // ---- Prologue: issue first K block load (stage 0) ----
        if (start_block_idx < end_block_idx) {
            issue_k_load(start_block_idx, 0, block_table_ptr);
        }
        // Wait for both Q and the first K to finish before starting compute.
        cp_async_wait_all();
        __syncthreads();

        // ---- K-block loop with double-buffered prefetch ----
        // sK[stage] holds K_i during iter i. We issue K_{i+1} into sK[1-stage]
        // during the compute of iter i. wait_group<1> at iter start waits for
        // K_i to be ready, leaving K_{i+1} (if any) in flight.
        int stage = 0;
        // Issue prefetch for block_idx + 1 if it exists, before entering the loop.
        if (start_block_idx + 1 < end_block_idx) {
            issue_k_load(start_block_idx + 1, 1, block_table_ptr);
        }

        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            const int start_token = block_idx * PAGE_BLOCK_SIZE;
            T* sK_cur = sK[stage];

            // === QK^T + softmax + rPb pack (rP fp32 inner scope to free regs before PV) ===
            uint32_t rPb[PV_K_TILES][4];
            {
                float rP[QK_N_TILES][4];
                #pragma unroll
                for (int n = 0; n < QK_N_TILES; ++n) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) rP[n][j] = 0.0f;
                }

                #pragma unroll 1
                for (int k_tile = 0; k_tile < QK_K_TILES; ++k_tile) {
                    int k_offset = k_tile * 16;

                    // Load A (Q tile, M=16, K=16). m_block has only one warp -> ROWS_PER_WARP=16.
                    // ldmatrix.x4 mat layout: mat 0..3 covering (M_lo/M_hi x K_lo/K_hi).
                    int mat        = lane_idx / 8;
                    int m_lo_or_hi = mat & 1;
                    int k_half     = mat >> 1;
                    int q_row      = m_lo_or_hi * 8 + (lane_idx % 8);
                    int q_col      = k_offset + k_half * 8;
                    int q_swiz     = swizzle_col_bf16(q_row, q_col);
                    uint32_t rQ[4];
                    uint32_t s_addr_q = cvta_to_shared_u32(sQ + q_row * SMEM_STRIDE_K + q_swiz);
                    ldmatrix_x4(rQ, s_addr_q);

                    #pragma unroll
                    for (int n = 0; n < QK_N_TILES; ++n) {
                        int n_offset = n * 8;
                        // K^T B operand: K stored row-major -> ldmatrix without .trans gives col-major B.
                        int b_mat = (lane_idx / 8) & 1;
                        int row_b = lane_idx & 7;
                        int n_row = n_offset + row_b;
                        int k_col = k_offset + b_mat * 8;
                        int k_swiz= swizzle_col_bf16(n_row, k_col);
                        uint32_t s_addr_b = cvta_to_shared_u32(sK_cur + n_row * SMEM_STRIDE_K + k_swiz);
                        uint32_t rKT[2];
                        ldmatrix_x2(rKT, s_addr_b);
                        mma_m16n8k16_acc<T>(rP[n], rQ, rKT);
                    }
                }

                // Mask + scale.
                #pragma unroll
                for (int n = 0; n < QK_N_TILES; ++n) {
                    int base_token = start_token + n * 8 + (lane_idx % 4) * 2;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) rP[n][j] *= scale_log2;
                    if (base_token + 0 >= rRightBorder[0]) rP[n][0] = -INFINITY;
                    if (base_token + 1 >= rRightBorder[0]) rP[n][1] = -INFINITY;
                    if (base_token + 0 >= rRightBorder[1]) rP[n][2] = -INFINITY;
                    if (base_token + 1 >= rRightBorder[1]) rP[n][3] = -INFINITY;
                }

                // new rowmax.
                float new_rM[2] = {rM[0], rM[1]};
                #pragma unroll
                for (int n = 0; n < QK_N_TILES; ++n) {
                    new_rM[0] = max(new_rM[0], max(rP[n][0], rP[n][1]));
                    new_rM[1] = max(new_rM[1], max(rP[n][2], rP[n][3]));
                }
                new_rM[0] = max(new_rM[0], __shfl_xor_sync(0xffffffff, new_rM[0], 1));
                new_rM[0] = max(new_rM[0], __shfl_xor_sync(0xffffffff, new_rM[0], 2));
                new_rM[1] = max(new_rM[1], __shfl_xor_sync(0xffffffff, new_rM[1], 1));
                new_rM[1] = max(new_rM[1], __shfl_xor_sync(0xffffffff, new_rM[1], 2));

                float scale_for_old[2];
                scale_for_old[0] = (rM[0] == -INFINITY) ? 0.0f : exp2f(rM[0] - new_rM[0]);
                scale_for_old[1] = (rM[1] == -INFINITY) ? 0.0f : exp2f(rM[1] - new_rM[1]);

                #pragma unroll
                for (int n = 0; n < N_TILES_PER_WG; ++n) {
                    rO[n][0] *= scale_for_old[0];
                    rO[n][1] *= scale_for_old[0];
                    rO[n][2] *= scale_for_old[1];
                    rO[n][3] *= scale_for_old[1];
                }
                rL[0] *= scale_for_old[0];
                rL[1] *= scale_for_old[1];
                rM[0] = new_rM[0];
                rM[1] = new_rM[1];

                // exp + accumulate L.
                #pragma unroll
                for (int n = 0; n < QK_N_TILES; ++n) {
                    rP[n][0] = exp2f(rP[n][0] - new_rM[0]);
                    rP[n][1] = exp2f(rP[n][1] - new_rM[0]);
                    rP[n][2] = exp2f(rP[n][2] - new_rM[1]);
                    rP[n][3] = exp2f(rP[n][3] - new_rM[1]);
                    rL[0] += rP[n][0] + rP[n][1];
                    rL[1] += rP[n][2] + rP[n][3];
                }

                // pack rP -> rPb.
                #pragma unroll
                for (int kt = 0; kt < PV_K_TILES; ++kt) {
                    int n0 = kt * 2;
                    int n1 = kt * 2 + 1;
                    rPb[kt][0] = pack_2xfp32_to_b32<T>(rP[n0][0], rP[n0][1]);
                    rPb[kt][1] = pack_2xfp32_to_b32<T>(rP[n0][2], rP[n0][3]);
                    rPb[kt][2] = pack_2xfp32_to_b32<T>(rP[n1][0], rP[n1][1]);
                    rPb[kt][3] = pack_2xfp32_to_b32<T>(rP[n1][2], rP[n1][3]);
                }
            }  // rP fp32 destroyed

            // === PV: rO[16x128] += rPb[16x64] @ V[64x128] ===
            // V[k, n] = sK_cur[k, wg_idx*128 + n].
            #pragma unroll 1
            for (int kt = 0; kt < PV_K_TILES; ++kt) {
                int k_off_pv = kt * 16;
                uint32_t rA[4] = { rPb[kt][0], rPb[kt][1], rPb[kt][2], rPb[kt][3] };

                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WG; ++nt) {
                    int v_global_col = wg_idx * HEAD_DIM_V_PER_WG + nt * 8;
                    int b_mat   = (lane_idx / 8) & 1;
                    int row_b   = lane_idx & 7;
                    int k_row   = k_off_pv + b_mat * 8 + row_b;
                    int v_swiz  = swizzle_col_bf16(k_row, v_global_col);
                    uint32_t s_addr_v = cvta_to_shared_u32(sK_cur + k_row * SMEM_STRIDE_K + v_swiz);
                    uint32_t rB[2];
                    ldmatrix_x2_trans(rB, s_addr_v);
                    mma_m16n8k16_acc<T>(rO[nt], rA, rB);
                }
            }

            // ---- Prefetch K_{i+2} into sK[stage] (the buffer we just finished reading) ----
            // PV is done with sK_cur, so the *current* stage is now safe to overwrite.
            // We use it as the buffer for K_{i+2}, leaving sK[1-stage] (=K_{i+1}) intact.
            __syncthreads();
            if (block_idx + 2 < end_block_idx) {
                issue_k_load(block_idx + 2, stage, block_table_ptr);
            }
            // Swap stage: next iter will compute on the buffer we previously prefetched.
            stage = 1 - stage;

            // Wait for the *new* current stage (K_{i+1}) to be ready before next iter's compute.
            // Because we always have at most 2 commit_groups in flight (one per stage), we wait
            // for the older one to complete using wait_group<1>.
            if (block_idx + 1 < end_block_idx) {
                cp_async_wait_group<1>();
                __syncthreads();
            }
        }

        // ---- Reduce rL within warp ----
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];
        }

        // ---- Store rO ----
        const int n_split_idx_local = batch_idx == sched_meta.begin_req_idx
                                        ? sched_meta.begin_split_idx : 0;
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx
                                    ? !sched_meta.is_first_req_splitted
                                    : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        const float rL_inv[2] = { 1.0f / rL[0], 1.0f / rL[1] };
        const int row_lo_local = lane_idx / 4;
        const int row_hi_local = row_lo_local + 8;

        if (is_no_split) {
            T* gO_row_lo = gO + row_lo_local * params.o_row_stride;
            T* gO_row_hi = gO + row_hi_local * params.o_row_stride;
            #pragma unroll
            for (int nt = 0; nt < N_TILES_PER_WG; ++nt) {
                int v_col = wg_idx * HEAD_DIM_V_PER_WG + nt * 8 + (lane_idx % 4) * 2;
                T v0 = static_cast<T>(rO[nt][0] * rL_inv[0]);
                T v1 = static_cast<T>(rO[nt][1] * rL_inv[0]);
                T v2 = static_cast<T>(rO[nt][2] * rL_inv[1]);
                T v3 = static_cast<T>(rO[nt][3] * rL_inv[1]);
                if (row_lo_local < num_valid_seq_q) {
                    gO_row_lo[v_col + 0] = v0;
                    gO_row_lo[v_col + 1] = v1;
                }
                if (row_hi_local < num_valid_seq_q) {
                    gO_row_hi[v_col + 0] = v2;
                    gO_row_hi[v_col + 1] = v3;
                }
            }
            // LSE: only wg 0 writes (all wgs have the same rL/rM up to noise; pick one).
            if (wg_idx == 0 && (lane_idx % 4) == 0) {
                int row_lo = lane_idx / 4;
                int row_hi = row_lo + 8;
                float lse_lo = (rL[0] == 0.0f || rL[0] != rL[0])
                                ? INFINITY : (logf(rL[0]) + rM[0] / LOG2_E);
                float lse_hi = (rL[1] == 0.0f || rL[1] != rL[1])
                                ? INFINITY : (logf(rL[1]) + rM[1] / LOG2_E);
                if (row_lo < num_valid_seq_q) gLse[row_lo] = lse_lo;
                if (row_hi < num_valid_seq_q) gLse[row_hi] = lse_hi;
            }
        } else {
            const int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx_local;
            float* gOAccum   = (float*)params.oaccum_ptr
                + ((int64_t)(split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk
                   + m_block_idx * BLOCK_SIZE_M) * HEAD_DIM_V;
            float* gLseAccum = params.softmax_lseaccum_ptr
                + (split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk
                + m_block_idx * BLOCK_SIZE_M;
            float* gOA_row_lo = gOAccum + row_lo_local * HEAD_DIM_V;
            float* gOA_row_hi = gOAccum + row_hi_local * HEAD_DIM_V;
            #pragma unroll
            for (int nt = 0; nt < N_TILES_PER_WG; ++nt) {
                int v_col = wg_idx * HEAD_DIM_V_PER_WG + nt * 8 + (lane_idx % 4) * 2;
                float v0 = rO[nt][0] * rL_inv[0];
                float v1 = rO[nt][1] * rL_inv[0];
                float v2 = rO[nt][2] * rL_inv[1];
                float v3 = rO[nt][3] * rL_inv[1];
                if (row_lo_local < num_valid_seq_q) {
                    gOA_row_lo[v_col + 0] = v0;
                    gOA_row_lo[v_col + 1] = v1;
                }
                if (row_hi_local < num_valid_seq_q) {
                    gOA_row_hi[v_col + 0] = v2;
                    gOA_row_hi[v_col + 1] = v3;
                }
            }
            if (wg_idx == 0 && (lane_idx % 4) == 0) {
                int row_lo = lane_idx / 4;
                int row_hi = row_lo + 8;
                float lse_lo = (rL[0] == 0.0f || rL[0] != rL[0])
                                ? -INFINITY : (log2f(rL[0]) + rM[0]);
                float lse_hi = (rL[1] == 0.0f || rL[1] != rL[1])
                                ? -INFINITY : (log2f(rL[1]) + rM[1]);
                if (row_lo < num_valid_seq_q) gLseAccum[row_lo] = lse_lo;
                if (row_hi < num_valid_seq_q) gLseAccum[row_hi] = lse_hi;
            }
        }

        if (batch_idx != sched_meta.end_req_idx) __syncthreads();
    }
#endif
}

template<typename T>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    using namespace sm80::cfg;
    FLASH_ASSERT(params.d == HEAD_DIM_K);
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);

    constexpr size_t smem_size = smem_total_bytes<T>();
    auto kernel = &flash_fwd_splitkv_mla_kernel_sm80<T>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    const int num_m_block = (params.q_seq_per_hk + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    dim3 grid(num_m_block, params.h_k, params.num_sm_parts);
    dim3 block(NUM_THREADS, 1, 1);

    kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
