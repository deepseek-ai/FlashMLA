#pragma once

#include "splitkv_mla.h"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "components/dequant.h"
#include "components/helpers.h"
#include "config.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using fp8_e8m0 = __nv_fp8_e8m0;

template<
    typename Tensor0,
    typename Tensor1,
    typename Tensor2
>
__forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup
) {
    float scale_for_olds[2];
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])
                cur_rP(i) = -INFINITY;
            cur_max = max(cur_max, cur_rP(i));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        cur_max *= scale_softmax_log2;
        float old_max = rM[local_row_idx];
        rM[local_row_idx] = max(cur_max, old_max);
        float scale_for_old = exp2f(old_max - rM[local_row_idx]);
        scale_for_olds[local_row_idx] = scale_for_old;

        CUTE_UNROLL
        for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= scale_for_old;
        }

        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            cur_rP(i) = exp2f(cur_rP(i)*scale_softmax_log2 - rM[local_row_idx]);
            cur_rS(i) = (bf16)cur_rP(i);
            cur_sum += cur_rP(i);
        }

        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
    if (idx_in_warpgroup%4 == 0)
        *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;
    
    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    }
    
    // Initialize TMA barriers
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q.init(1);
        if constexpr (CLUSTER_SIZE == 2) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_remote_ready[i].init(1);
                plan.bar_k_avail[i].init(4);
            }
        } else {
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_avail[i].init(256);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    ku::barrier_cluster_arrive_relaxed();

    int bar_phase_k = 0; // Don't use array here to prevent using local memory

    // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // Don't use PDL because of compiler bugs!
    // cudaGridDependencySynchronize();
    
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    if (sched_meta.begin_req_idx >= params.b) return;

    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ = flat_divide(
            tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
        plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
    }

    ku::barrier_cluster_wait_acquire();

    struct MainloopArgs {
        int start_block_idx, end_block_idx;
        bool is_no_split;

        // The following fields are only valid for MODEL1
        int topk_length, extra_topk_length, num_orig_kv_blocks;
    };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            total_topk_padded = params.topk;
        } else {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<192>();

        TiledMMA tiled_mma_QK = TiledMMA_QK{};
        ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
        TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        
        float rL[2], rM[2];
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

        float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
        if (params.attn_sink != nullptr) {
            for (int i = 0; i < 2; ++i) {
                int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
                rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
            }
        }

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);

            rL[0] = rL[1] = 0.0f;
            rM[0] = rM[1] = MAX_INIT_VAL;
            cute::fill(rO, 0.);

            // Wait for Q
            plan.bar_q.wait((sched_meta.begin_req_idx-batch_idx)&1);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

                // Wait, issue WGMMA
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                }

                gemm<true, -1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sQ),
                    thr_mma_QK.partition_fragment_B(sK),
                    rP
                );

                bar_phase_k ^= 1<<buf_idx;

                cute::warpgroup_wait<0>();
                
                // Calculate S = softmax(mask(scale(P)))
                if (block_idx != args.start_block_idx)
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);  // Make sure that sScale and sS is free

                // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
                scale_softmax(rP, rS, rO, params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], block_idx, idx_in_warpgroup);

                // Store S into shared, inform warpgroup 1
                save_rPb_to_sP(rS, sS, idx_in_warpgroup);
                fence_view_async_shared();

                // Issue O += S @ V
                gemm<false, -1>(
                    tiled_mma_PV,
                    rS,
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );

                NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

                cute::warpgroup_wait<0>();

                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }
            }

            // Copy the next q
            if (threadIdx.x/32 == 0 && elect_one_sync()) {
                if (batch_idx != sched_meta.end_req_idx) {
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx+1),
                        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
                    )(_, _, head_block_idx, _0{});
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
                } else {
                    // This kernel is followed by the combine kernel, so we signal PDL here
                    cudaTriggerProgrammaticLaunchCompletion();
                }
            }

            // Synchronize L and M across warpgroups
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

            if (idx_in_warpgroup%4 == 0) {
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    sL[row] = rL[i];
                    sM[row] = rM[i];
                }
            }
            
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                if (args.is_no_split) {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
                } else {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
                }
                if (idx_in_warpgroup%4 == 0) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    plan.sOScale[row] = o_scales[i];
                }
            }

            // This is a synchronization point for warpgroup 0/1.
            // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
            // Warpgroup 1 should wait wg 0 for sL to be ready
            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
            
            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
            if (args.is_no_split) {
                bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_h_q, _1{})
                ));
                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;	// (BLOCK_M) : (1)

                store_o<true>(rO, gO, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_accum_h_q, _1{})
                ));
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }

                cute::tma_store_wait<0>();
            }
            
            sync_all_threads_in_cluster();
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<160>();

        TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            cute::fill(rO, 0.);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

                // Wait for S and sScale
                NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

                // Scale O
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rO); ++i) {
                        cur_rO(i) *= cur_scales[local_row_idx];
                    }
                }
                
                // Issue O += S @ V, and wait
                gemm<false, -1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sS),
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );
                cute::warpgroup_wait<0>();
                
                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }
                
                if (block_idx != args.end_block_idx-1)
                    NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);   // Tell WG0 that sScale and sS are available
            }

            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = get_AorC_row_idx(i, idx_in_warpgroup);
                o_scales[i] = plan.sOScale[row];
            }
                
            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
            if (args.is_no_split) {
                bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_h_q, _1{})
                ));

                store_o<true>(rO, gO, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_accum_h_q, _1{})
                ));
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            }

            sync_all_threads_in_cluster();
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<152>();

        static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
        static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
        static constexpr int NUM_TOKENS_PER_ROUND = 32; // If head is 128, each CTA is responsible for dequantizing 32 tokens (1 rounds); if head is 64, each CTA is responsible for dequantizing 64 tokens (2 rounds)
        int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
        int lane_idx = idx_in_warpgroup % 32;
        int my_token_idx_base = warp_idx*8 + lane_idx%8;
        
        CUTE_NO_UNROLL
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            int* gIndices = params.indices + batch_idx*params.stride_indices_b + s_q_idx*params.stride_indices_s_q; // (topk) : (1)
            int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q; // (extra_topk) : (1)
            
            int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
            CUTE_UNROLL
            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                if (MODEL_TYPE == ModelType::V32 || args.start_block_idx < args.num_orig_kv_blocks)
                    nxt_token_indexs[round] = __ldg(gIndices + args.start_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + round*NUM_TOKENS_PER_ROUND + my_token_idx_base);
            }

            struct IsOrigBlock {};
            struct IsExtraBlock {};

            struct IsFirstExtraBlock {};
            struct IsNotFirstExtraBlock {};
            auto process_one_block = [&](int block_idx, auto is_extra_block_t, auto is_first_extra_block_t) {
                static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                static constexpr bool IS_FIRST_EXTRA_BLOCK = std::is_same_v<decltype(is_first_extra_block_t), IsFirstExtraBlock>;
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;

                int* indices_base;
                int page_block_size;
                int64_t k_block_stride, k_row_stride;
                fp8* k_ptr;
                if constexpr (!IS_EXTRA_BLOCK) {
                    indices_base = gIndices + (block_idx)*TOPK_BLOCK_SIZE;
                    page_block_size = params.page_block_size;
                    k_block_stride = params.stride_kv_block;
                    k_row_stride = params.stride_kv_row;
                    k_ptr = (fp8*)params.kv;
                } else {
                    indices_base = gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE;
                    page_block_size = params.extra_page_block_size;
                    k_block_stride = params.stride_extra_kv_block;
                    k_row_stride = params.stride_extra_kv_row;
                    k_ptr = (fp8*)params.extra_kv;
                }
                [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
                [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;
                transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&(plan.bar_k_remote_ready[buf_idx]));

                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                    int my_token_idx = my_token_idx_base + round*NUM_TOKENS_PER_ROUND;
                    bf16* sK_nope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                    bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

                    // Get prefetched token index
                    int token_index;
                    if constexpr (!IS_EXTRA_BLOCK) {
                        token_index = nxt_token_indexs[round];
                        if (block_idx+1 != (MODEL_TYPE == ModelType::V32 ? args.end_block_idx : args.num_orig_kv_blocks))
                            nxt_token_indexs[round] = __ldg(gIndices + (block_idx+1)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    } else {
                        if constexpr (IS_FIRST_EXTRA_BLOCK) {
                            token_index = __ldg(gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                        } else {
                            token_index = nxt_token_indexs[round];
                        }
                        if (block_idx+1 != args.end_block_idx)
                            nxt_token_indexs[round] = __ldg(gExtraIndices + (block_idx+1-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    }
                    
                    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                        // For MODEL1, we need to check whether the token_index is within topk_length
                        if (rel_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx >= topk_length) {
                            token_index = -1;   // To prevent IMA when we have invalid (e.g. INT_MAX) topk indexes outside topk_length
                        }
                    }

                    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);   // Use uint32_t division and mod to improve performance
                    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size < page_block_size, so there will be no illegal-memory-access error

                    fp8* gK_base;
                    bf16 scales[NUM_SCALES];
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        static_assert(NUM_SCALES == 4);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride;
                        float scales_float[NUM_SCALES];
                        *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i) {
                            scales[i] = (bf16)scales_float[i];
                        }
                    } else {
                        static_assert(NUM_SCALES == 8);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
                        fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
                        fp8_e8m0 scales_e8m0[NUM_SCALES];
                        *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; i += 2) {
                            *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
                        }
                    }

                    // Wait for the nope buffer to be available
                    if (round == 0) {
                        plan.bar_k_avail[buf_idx].wait((bar_phase_k>>buf_idx&1)^1);
                    }
                    
                    if (CLUSTER_SIZE == 2 && round == 0 && idx_in_warpgroup == 0) {
                        plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx((TOPK_BLOCK_SIZE/2)*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*sizeof(bf16));
                    }

                    // Collectively copy from global memory and dequant
                    // For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
                    
                    fp8* gK_nope = gK_base + (lane_idx/8)*16;
                    if (token_index == -1) {
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i)
                            scales[i] = (bf16)0.0f;
                    }
                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE/64; dim_idx += 1) {
                        fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim_idx*64);   // We use EVICT_LAST here since gK_base may not be aligned to 32B (for V3.2) and the performance is the best among all cache hints (for MODEL1)
                        bf16 scale = scales[MODEL_TYPE == ModelType::V32 ? dim_idx/2 : dim_idx];
                        auto dequant_and_save_bf16x8 = [&](const fp8x8 &data, int offset) {
                            int smem_offset = (dim_idx*64 + offset) * TOPK_BLOCK_SIZE;
                            bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
                            *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                            if constexpr (CLUSTER_SIZE == 2) {
                                st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                            }
                        };
                        if (token_index == -1)
                            *(uint128_t*)(&cur_fp8x16) = uint128_t();
                        dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
                        dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
                    }

                    bf16* gK_rope;
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE+NUM_SCALES*sizeof(float)) + (lane_idx/8)*8;
                    } else {
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE) + (lane_idx/8)*8;
                    }
                    bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
                    bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                        bf16x8 cur_bf16x8 = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx*32);
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            // NOTE We do not need to mask the RoPE part for V3.2 since it isn't involved in the SV gemm
                        } else {
                            if (token_index == -1)
                                *(uint128_t*)(&cur_bf16x8) = uint128_t();
                        }
                        int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                        *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                        if constexpr (CLUSTER_SIZE == 2) {
                            st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                        }
                    }
                }

                fence_view_async_shared();

                if (idx_in_warpgroup < 32) {
                    // We put this after fence_view_async_shared() since this won't be read by async proxy
                    auto is_index_valid = [&](int index, int offset_within_thread) -> bool {
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            return index != -1;
                        } else {
                            return index != -1 && rel_block_idx*TOPK_BLOCK_SIZE + lane_idx*2 + offset_within_thread < topk_length;
                        }
                    };
                    int2 indices = __ldg((int2*)(indices_base + lane_idx*2));
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = {
                        is_index_valid(indices.x, 0),
                        is_index_valid(indices.y, 1)
                    };
                }

                // Signal the barrier
                plan.bar_k_local_ready[buf_idx].arrive();
                bar_phase_k ^= 1 << buf_idx;
            };

            if constexpr (MODEL_TYPE == ModelType::V32) {
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }
            } else {
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }

                if (args.num_orig_kv_blocks < args.end_block_idx) {
                    process_one_block(max(args.start_block_idx, args.num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
                }
                CUTE_NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks)+1; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{}, IsNotFirstExtraBlock{});
                }
            }

            sync_all_threads_in_cluster();
        }
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif

}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);
    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        if (params.extra_kv != nullptr) {
            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        }
    } else {
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQ{}
    );
    
    CUtensorMap tensor_map_o;
    {
        // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_o,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_o
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // NOTE Don't use PDL because of potential compiler bugs!
    // cudaLaunchAttribute mla_kernel_attributes[1];
    // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(NUM_THREADS, 1, 1),
    //     smem_size,
    //     stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    cutlass::ClusterLaunchParams launch_params = {
        dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)mla_kernel, params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
