/*
Sparse Attention Forward Pass (Small TopK) — SM100, h_q=128

Specialized forward kernel optimized for small topk values (topk <= 1280), with 128
query heads. Uses a different tiling strategy than the standard forward kernel.
Supports both prefill and decode modes (including split-KV decoding). Uses FP8 KV
cache for decoding mode with dequantization and RoPE/NoPE separation.

Template parameters:
  FWD_MODE — Forward mode (Prefill / Decode / DecodeWithSplitKV)
  D_QK     — Head dimension for QK (only 512 supported)

Grid: [2*s_q, 1, 1] (prefill) or [2*b, 1, 1] (decode), Cluster: [2, 1, 1]
NUM_THREADS=512 (4 warpgroups)

I/O: See SparseAttnFwdParams or SparseAttnDecodeParams in kernels/params.h
*/
#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>

#include "kernels/params.h"
#include "kernels/utils.h"
#include "kernels/sm100/common_subroutine.h"
#include "kernels/sm100/helpers.h"

#include "config.h"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

using namespace cute;
using FwdMode = SparseAttnFwdMode;

// NOTES. We found out that, if we use kerutils::st_shared, warpgroup 0 suffers from register spilling. However, if we change to this dump implementation with `__cvta_generic_to_shared`, there is no register spilling.
CUTE_DEVICE
void st_shared_with_cvta(void* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

template<FwdMode FWD_MODE, int D_QK, ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>
__device__ void
KernelTemplate<FWD_MODE, D_QK, MODEL_TYPE, EXTRA_MODEL_TYPE>::sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params) {
#ifdef KERUTILS_ENABLE_SM100A
    // Grid shape: [2*s_q, 1, 1] for prefilling, [2*s_q, num_sm_parts, 1] for decoding w/ splitKV, [2*s_q, batch, 1] for decoding w/o splitKV
    // Cluster shape: [2, 1, 1]
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    int cta_idx = blockIdx.x % 2;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
        if constexpr (IS_DECODE) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_quant_part[cta_idx]);
            if constexpr (D_BF16 > 0) {
                cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_bf16_part);
            }
        } else {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        smem.bar_sQ_full.init(1);
        smem.bar_tQ_empty.init(1);
        smem.bar_tQ_full.init(1);
        smem.bar_tOut_full.init(1);
        smem.bar_tOut_empty.init(256);
        smem.bar_P_empty.init(256);
        smem.bar_QK_done.init(1);
        smem.bar_SV_done.init(1);
        smem.bar_S_O_full.init(256);
        smem.bar_li_full.init(H_Q/2);
        smem.bar_li_empty.init(128);
        if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
            smem.bar_clc_full.init(1);
            smem.bar_clc_empty.init(NUM_WORKER_THREADS);
        }
        fence_barrier_init();
    } else if (warp_idx == 2) {
        cute::TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (warp_idx == 3 && elect_one_sync()) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            smem.bar_KV_full[i].init(IS_PREFILL ? 1 : (128/32)*2+1);
            smem.bar_KV_empty[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
            smem.bar_valid_coord_scales_full[i].init(IS_PREFILL ? B_TOPK/8 : 32);
            smem.bar_valid_coord_scales_empty[i].init(IS_PREFILL ? 128 : (128 + (D_BF16 > 0 && block_id_in_cluster().x==1) + 2 + 128));   // We use `block_id_in_cluster().x` instead of cta_idx to prevent reg spilling
        }
        if constexpr (IS_DECODE) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_RAW_K_BUFS; ++i) {
                smem.bar_raw_KV_full[i].init(1);
                smem.bar_raw_KV_empty[i].init(128);
            }
        }
        fence_barrier_init();
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

    struct OuterloopArgs {
        bool outer_loop_phase;
        int batch_idx, s_q_idx;
        int start_block_idx, end_block_idx;
        int topk_length;

        int extra_topk_length, num_orig_kv_blocks;  // extra-KV related
        bool is_no_split; int n_split_idx;  // splitkv related
    };

    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            int s_q_idx = blockIdx.x / 2;
            DecodingSchedMeta sched_meta;
            KU_LDG_256(
                params.tile_scheduler_metadata_ptr + blockIdx.y,
                &sched_meta,
                ".nc",
                "no_allocate",
                "evict_normal",
                "256B"
            );
            if (sched_meta.begin_req_idx >= params.b) {
                return 0;
            }

            #pragma unroll 1
            for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
                int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
                int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0
                int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
                bool is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);

                OuterloopArgs args = {
                    (bool)outer_loop_phase,
                    batch_idx, s_q_idx,
                    start_block_idx, end_block_idx,
                    topk_length,

                    extra_topk_length, orig_topk_padded / B_TOPK,
                    !is_split, n_split_idx
                };

                loop_body(args);
                outer_loop_phase ^= 1;
            }
        } else {
            // Prefill (or decoding w/o splitKV) mode. Use CLC to allocate different s_q (for decoding, different batches + s_q) to different workers
            ku::CLCResult next_job = {true, (int)blockIdx.x, IS_PREFILL ? 0 : (int)blockIdx.y, 0};
            CUTE_NO_UNROLL
            while (next_job.is_valid) {
                int s_q_idx = next_job.x / 2;
                int batch_idx = IS_PREFILL ? 0 : next_job.y;
                int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + (IS_PREFILL?s_q_idx:batch_idx)) : params.topk;
                
                if constexpr (IS_PREFILL) {
                    int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1
                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        0, s_q_idx,
                        0, num_k_blocks,
                        topk_length
                    };
                    loop_body(args);
                } else {
                    int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                    int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0

                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        batch_idx, s_q_idx,
                        0, total_topk_padded / B_TOPK,
                        topk_length,

                        extra_topk_length, orig_topk_padded / B_TOPK,
                        false, 0
                    };
                    loop_body(args);
                }

                smem.bar_clc_full.wait(outer_loop_phase);
                next_job = ku::get_clc_query_response<true>(smem.clc_response_obj);
                smem.bar_clc_empty.arrive(0u);

                outer_loop_phase ^= 1;
            }
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        // Q fetching and O writing back warpgroup
        cutlass::arch::warpgroup_reg_alloc<176>();

        bf16* sO_addrs[B_EPI/8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            Tensor sO = make_tensor(make_smem_ptr(smem.Q.data()), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128>());
            sO_addrs[i] = &sO(idx_in_warpgroup%64, (idx_in_warpgroup/64)*(D_V/2) + i*8);
        }

        float* sO_accum_addrs[B_EPI_SPLITKV/4];
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            // If split-KV is enabled, we need to store back O in float32
            // We view Q buffer (with shape 64 x 512, bf16) as 4 buffers with shape (H_Q/2) x (B_EPI_SPLITKV*2), float32
            Tensor sO_accum = make_tensor(make_smem_ptr((float*)smem.Q.data()), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128, float>());
            CUTE_UNROLL
            for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                sO_accum_addrs[i] = &sO_accum(idx_in_warpgroup%64, i*4) + (idx_in_warpgroup >= 64 ? (H_Q/2)*B_EPI_SPLITKV : 0);
            }
        }

        auto perform_o_copy_out = [&](const OuterloopArgs &args, bool is_last_o) {
            // outer_loop_phase is the loop phase corresponding to s_q_idx

            // Get li (output_scale actually)
            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale = smem.rowwise_li_buf[idx_in_warpgroup%64];
            float2 output_scale_float2 = float2 {output_scale, output_scale};
            smem.bar_li_empty.arrive();

            // Retrieve and store O
            smem.bar_tOut_full.wait(args.outer_loop_phase);
            if (is_last_o && elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI; ++k) {
                    float2 o[B_EPI/2];
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI/8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        CUTE_UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                        }
                        bf16* o_addr = sO_addrs[i] + k*B_EPI*(H_Q/2);
                        if (k == 0 && i == 0) {
                            smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                        }
                        st_shared_with_cvta(o_addr, *(__int128_t*)o_bf16);
                    }
                }

                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && elect_one_sync()) {
                    SM90_TMA_STORE_5D::copy(
                        &tma_params.tensor_map_o, 
                        smem.Q.data(),
                        0, cta_idx*(H_Q/2), 0, args.s_q_idx, IS_DECODE ? args.batch_idx : 0
                    );
                    cute::tma_store_arrive();
                }
            } else {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI_SPLITKV; ++k) {
                    int cur_buf_idx = k % NUM_EPI_SPLITKV_BUFS;
                    if (k == 0) {
                        cute::tma_store_wait<0>();
                    } else {
                        cute::tma_store_wait<NUM_EPI_SPLITKV_BUFS-1>();
                    }
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);

                    float o[B_EPI_SPLITKV];
                    ku::tmem_ld_32dp32bNx<B_EPI_SPLITKV>(tmem_cols::O + k*B_EPI_SPLITKV, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI_SPLITKV-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                        CUTE_UNROLL
                        for (int j = 0; j < 4; j += 2) {
                            *(float2*)(o + i*4 + j) = ku::float2_mul(float2 {o[i*4+j], o[i*4+j+1]}, output_scale_float2);
                        }
                        if (k == 0 && i == 0) {
                            smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                        }
                        st_shared_with_cvta(
                            sO_accum_addrs[i] + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2),
                            *(__int128_t*)(o + i*4)
                        );
                    }

                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    if constexpr (IS_DECODE) {  // Otherwise nvcc complains about `tma_params` doesn't have `tensor_map_o_accum`
                        float* cur_buf_base = (float*)smem.Q.data() + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2);
                        if (warp_idx == 0 && elect_one_sync()) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum, 
                                cur_buf_base,
                                0, cta_idx*(H_Q/2), k*(B_EPI_SPLITKV/32), args.s_q_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        } else if (warp_idx == 1 && elect_one_sync()) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum, 
                                cur_buf_base + (H_Q/2)*B_EPI_SPLITKV,
                                0, cta_idx*(H_Q/2), k*(B_EPI_SPLITKV/32) + (D_V/2)/32, args.s_q_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        }
                    }
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        bool final_outer_loop_phase = \
        run_outer_loop([&](const OuterloopArgs &args) {
            // Copy Q for this round
            if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
                cute::tma_store_wait<0>();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);  // Since we use two warps to issue TMA during FwdMode::DecodeWithSplitKV
            }
            if (warp_idx == 0 && elect_one_sync()) {
                // Wait for sQ to become empty, and issue G -> S copy for Q
                if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
                    cute::tma_store_wait<0>();  // This thread must be the same one as o copy out thread (since `elect_one_sync()` always returns the same thread for the same `mask`, according to PTX document)
                }
                int stride_q_b_div_stride_q_s_q = 0;
                if constexpr (IS_DECODE) {
                    stride_q_b_div_stride_q_s_q = params.stride_q_b / params.stride_q_s_q;
                }
                SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                    &tma_params.tensor_map_q,
                    (uint64_t*)&smem.bar_sQ_full,
                    (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                    smem.Q.data(),
                    0, cta_idx*(H_Q/2), 0, 0, (IS_DECODE ? args.batch_idx*stride_q_b_div_stride_q_s_q : 0) + args.s_q_idx
                );

                // Wait for sQ to be ready, and issue S -> T copy for Q
                if (cta_idx == 0) {
                    smem.bar_sQ_full.arrive_and_expect_tx(H_Q*D_Q*sizeof(bf16));
                    smem.bar_sQ_full.wait(args.outer_loop_phase);

                    smem.bar_tQ_empty.wait(args.outer_loop_phase^1);
                    ku::tcgen05_after_thread_sync();
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(smem.Q.data()),
                            ku::make_umma_canonical_k_major_layout<(H_Q/2)*2, 64, 128>()
                        )
                    );
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q/64/2; ++tile_idx) {
                        // A tile is 128 rows * 64 cols in UTCCP's view, or 64 rows * 128 cols in our view
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                            // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                            // NOTE Using `sQ_desc+((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4)` leads to IMA, doesn't know why
                            UMMA::SmemDescriptor cur_sQ_desc = sQ_desc;
                            // cur_sQ_desc.lo += ((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4);
                            asm volatile ("add.u32 %0, %0, %1;" : "+r"(cur_sQ_desc.lo) : "r"((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4));
                            // uint64_t cur_sQ_desc = sQ_desc;
                            // cur_sQ_desc += ((tile_idx*((H_Q/2)*128*2) + subtile_idx*32) >> 4);
                            SM100_UTCCP_128dp256bit_2cta::copy(
                                cur_sQ_desc,
                                tmem_cols::Q + tile_idx*32 + subtile_idx*8
                            );
                        }
                    }
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full, 1|2);
                }
            }

            if (last_args.batch_idx != -1) {
                perform_o_copy_out(last_args, false);
            } else {
                smem.bar_tQ_full.wait(args.outer_loop_phase);   // To prevent double arrive
            }
            last_args = args;
        });
        if (last_args.batch_idx != -1) {
            cute::tma_store_wait<0>();
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            perform_o_copy_out(last_args, true);
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // KV fetching threads for prefill, dequant threads for decoding
        cutlass::arch::warpgroup_reg_dealloc<80>();
        RingBufferState rs;

        if constexpr (!IS_DECODE) {
            const int warp_idx = cutlass::canonical_warp_idx();    // Using `warp_idx` without `__shfl_sync` is faster
            if (elect_one_sync()) {
                // KV fetching threads
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                    int64_t cache_hint = ku::create_simple_cache_policy<ku::CacheHint::EVICT_LAST>();

                    static constexpr int NUM_ROWS_PER_THREAD = B_TOPK / 4;

                    CUTE_NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();

                        int cur_indices[NUM_ROWS_PER_THREAD];
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/8; local_row += 1) {
                            int row = local_row*(4*8) + (warp_idx-4)*8;
                            KU_LDG_256(
                                gIndices + k*B_TOPK + row, 
                                cur_indices + local_row*8, 
                                ".nc", 
                                "no_allocate", 
                                "evict_first", 
                                "256B"
                            );
                        }
                        smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);

                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/4; local_row += 1) {
                            int row = (warp_idx-4)*8 + (local_row/2)*(4*8) + (local_row%2)*4;
                            int4 indices = *(int4*)(cur_indices+local_row*4);
                            static_assert(D_K == 512);
                            CUTE_UNROLL
                            for (int local_col = 0; local_col < (D_K/64)/2; ++local_col) {
                                ku::tma_gather4_cta_group_2<true>(
                                    &tma_params.tensor_map_kv,
                                    smem.bar_KV_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + row*64 + local_col*64*B_TOPK,
                                    local_col*64 + cta_idx*(D_K/2),
                                    indices,
                                    cache_hint
                                );
                            }
                        }
                        rs.update();
                    }
                });
            }
                
        } else {
            struct IsCTA0 {};
            struct IsCTA1 {};

            auto launch_dequant_wg = [&](auto cta_id_t) {
                static constexpr bool IS_CTA1 = std::is_same<decltype(cta_id_t), IsCTA1>::value;
                const DequantizerT<OrigKVFormat, IS_CTA1> dequant_orig(idx_in_warpgroup);
                const DequantizerT<ExtraKVFormat, IS_CTA1> dequant_extra(idx_in_warpgroup);   // Identical to dequant_orig unless the extra cache is fp4
                const uint8_t *raw_0 = smem.K_raw[0].data(), *raw_1 = smem.K_raw[1].data();
                static_assert(NUM_RAW_K_BUFS == 2);
                run_outer_loop([&](const OuterloopArgs &args) {
                    for_each_kv_block<OrigKVFormat, ExtraKVFormat>(args.start_block_idx, args.end_block_idx, args.num_orig_kv_blocks, [&]<typename F>(int, bool) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();

                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_full[raw_k_buf_idx].wait(raw_k_bar_phase);

                        const auto *dequant = [&] {
                            if constexpr (std::is_same_v<F, OrigKVFormat>) return &dequant_orig; else return &dequant_extra;
                        }();
                        dequant->run(
                            raw_k_buf_idx == 0 ? raw_0 : raw_1,
                            smem.scales[index_buf_idx][0],
                            cute::cast_smem_ptr_to_uint(smem.K[k_buf_idx].data()),
                            [&] { smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1); }
                        );

                        fence_view_async_shared();  // NOTE Should we use shared::cluster here?
                        __syncwarp();
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        smem.bar_raw_KV_empty[raw_k_buf_idx].arrive();
                        if (elect_one_sync()) {
                            smem.bar_KV_full[k_buf_idx].arrive(0u);
                        }
                        rs.update();
                    });
                });
            };
            if (cta_idx == 0) {
                launch_dequant_wg(IsCTA0{});
            } else {
                launch_dequant_wg(IsCTA1{});
            }
        }
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<80>();

        RingBufferState rs;
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // UMMA thread
            TiledMMA tiled_mma_P = TiledMMA_P{};
            TiledMMA tiled_mma_O = TiledMMA_O{};
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<H_Q/2>, Int<B_TOPK*2>>{});
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<H_Q/2>, Int<D_V>>{});
            Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<H_Q/2>, Int<D_Q/2>>{})
            );
            tP.data().get() = tmem_cols::P;
            tO.data().get() = tmem_cols::O;
            tQ.data().get() = tmem_cols::Q;
            
            run_outer_loop([&](const OuterloopArgs &args) {
                smem.bar_tQ_full.wait(args.outer_loop_phase);

                // Issue P = Q K^T
                auto issue_P = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_P_empty.wait(bar_phase^1);
                    if constexpr (IS_PREFILL) {
                        smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_K*sizeof(bf16));
                    } else {
                        if constexpr (D_BF16 > 0) {
                            smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_BF16*sizeof(bf16));
                        } else {
                            smem.bar_KV_full[k_buf_idx].arrive();
                        }
                    }
                    smem.bar_KV_full[k_buf_idx].wait(k_bar_phase);
                    ku::tcgen05_after_thread_sync();
                    Tensor sK = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128>()
                    );
                    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_QK_done, 1|2);
                };

                // Issue O += S V
                auto issue_O = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_S_O_full.wait(bar_phase);
                    if (k == args.start_block_idx) {
                        smem.bar_tOut_empty.wait(args.outer_loop_phase^1);
                    }
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(
                        make_smem_ptr(smem.S.data()),
                        ku::make_umma_canonical_k_major_layout<H_Q/2, B_TOPK, 0>()
                    );
                    Tensor sV = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_mn_major_layout<D_V/2, B_TOPK, 128>()
                    );
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == args.start_block_idx);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_SV_done, 1|2);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_KV_empty[k_buf_idx], 1|2);
                };

                CUTE_NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx+1; ++k) {
                    if (k < args.end_block_idx) {
                        issue_P(k, 0);
                    }
                    if (k == args.end_block_idx-1) {
                        ku::umma_arrive_2x1SM_noelect(smem.bar_tQ_empty);
                    }

                    if (k > args.start_block_idx) {
                        issue_O(k-1, -1);
                    }
                    
                    if (k != args.end_block_idx) {
                        rs.update();
                    }
                }
                ku::tcgen05_before_thread_sync();
                ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tOut_full, 1|2);
            });
        } else if (warp_idx == 8 && cta_idx == 1 && elect_one_sync()) {
            if constexpr (IS_DECODE && D_BF16 > 0) {
                // KV BF16 part fetching warp
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);
                        CUTE_UNROLL
                        for (int row = 0; row < B_TOPK; row += 4) {
                            int4 cur_indices = *(int4*)(smem.tma_coord[index_buf_idx] + row);
                            ku::tma_gather4_cta_group_2<true>(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_bf16_part : &tma_params.tensor_map_kv_bf16_part,
                                smem.bar_KV_full[k_buf_idx],
                                smem.K[k_buf_idx].data() + (D_NOPE-D_K/2)*B_TOPK + row*D_ROPE,
                                0,
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                            );
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    }
                });
            }
        } else if (warp_idx == 9) {
            // KV validness loading warp (for prefill), Indices transformation warp (for decode, Responsible for generating: TMA coordinates, scale factors, and valid masks)
            if constexpr (IS_PREFILL) {
                if (lane_idx < B_TOPK/8) {
                    run_outer_loop([&](const OuterloopArgs &args) {
                        int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                        CUTE_NO_UNROLL
                        for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                            char k_validness_mask = load_indices_and_generate_mask(
                                lane_idx,
                                gIndices + k*B_TOPK,
                                params.s_kv,
                                k*B_TOPK,
                                args.topk_length
                            );
                            
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                            smem.bar_valid_coord_scales_empty[indices_buf_idx].wait(indices_bar_phase^1);
                            smem.is_k_valid[indices_buf_idx][lane_idx] = k_validness_mask;
                            smem.bar_valid_coord_scales_full[indices_buf_idx].arrive();
                            
                            rs.update();
                        }
                    });
                }
            } else {
                static_assert(B_TOPK == 64);
                // Each thread is responsible for 2 tokens
                static constexpr int tma_coords_step_per_token = 1;   // A token's data is exactly one TMA_K_STRIDE row, for all formats
                int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE_FOR_DECODING; // must < 2G since k_batch_stride < 1T and TMA_K_STRIDE_FOR_DECODING > 512
                int tma_coords_step_per_extra_block = params.stride_extra_kv_block / ExtraKVFormat::TMA_K_STRIDE;
                uint8_t* k_scales_ptr = (uint8_t*)params.kv + params.page_block_size*TMA_K_STRIDE_FOR_DECODING;
                uint8_t* extra_k_scales_ptr = (uint8_t*)params.extra_kv + params.extra_page_block_size*ExtraKVFormat::TMA_K_STRIDE;
                
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*args.s_q_idx;
                    int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*args.s_q_idx;
                    
                    struct IsOrigBlock {};
                    struct IsExtraBlock {};
                    // Prefetch the next block's indices while processing the current one, so that the
                    // index LDG latency overlaps with the scale LDG and the computation of the current block.
                    auto load_block_indices = [&](int block_idx, auto is_extra_block_t) -> int2 {
                        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                        if constexpr (!IS_EXTRA_BLOCK) {
                            return __ldg((int2*)(indices + block_idx*B_TOPK + lane_idx*2));
                        } else {
                            return __ldg((int2*)(extra_indices + (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2));
                        }
                    };
                    auto process_one_block = [&](int block_idx, int2 my_indices, auto is_extra_block_t) {
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                        using F = std::conditional_t<IS_EXTRA_BLOCK, ExtraKVFormat, OrigKVFormat>;
                        int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                        int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
                        [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
                        uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                        int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                        int abs_pos = IS_EXTRA_BLOCK ? (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2 : block_idx*B_TOPK + lane_idx*2;

                        // Issue the scale LDGs before waiting for the index buffer, so their latency
                        // overlaps with the (usually long) empty-barrier wait.
                        alignas(16) uint8_t scales[2][SCALE_SMEM_STRIDE_PER_CTA];
                        int kv_block_idx_arr[2], idx_in_block_arr[2];
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                            int kv_block_idx = (unsigned int)cur_idx / cur_block_size;
                            int idx_in_block = (unsigned int)cur_idx % cur_block_size;
                            kv_block_idx_arr[i] = kv_block_idx;
                            idx_in_block_arr[i] = idx_in_block;
                            // This CTA's half of the token's row of 1 B scales
                            int64_t offset = kv_block_idx*cur_k_block_stride + (idx_in_block*F::NUM_SCALES_EACH_TOKEN + (cta_idx == 1 ? F::NUM_SCALES_EACH_TOKEN/2 : 0));
                            bool is_token_valid = cur_idx != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                            ldg_or_zero<F::NUM_SCALES_EACH_TOKEN/2>(scales[i], cur_k_scales_ptr + offset, is_token_valid);
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].wait(index_bar_phase^1);

                        int tma_coords[2];
                        char valid_mask = 0;
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                            bool is_token_valid = cur_idx != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                            valid_mask |= is_token_valid << i;
                            tma_coords[i] = is_token_valid ? kv_block_idx_arr[i]*cur_tma_coords_step_per_block + idx_in_block_arr[i]*tma_coords_step_per_token : -1; // If the token is invalid because it topk position exceeds topk_length, we must manually fill tma_coords with -1 to avoid copying-in NaN.
                        }
                        valid_mask <<= lane_idx%4*2;
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);

                        if constexpr (SCALE_SMEM_STRIDE_PER_CTA == F::NUM_SCALES_EACH_TOKEN/2) {
                            copy_bytes<2*SCALE_SMEM_STRIDE_PER_CTA>(smem.scales[index_buf_idx][lane_idx*2], scales[0]);
                        } else {
                            // Mixed-format kernel, fp8 tokens: each token uses the first bytes of its 16 B row
                            CUTE_UNROLL
                            for (int i = 0; i < 2; ++i)
                                copy_bytes<F::NUM_SCALES_EACH_TOKEN/2>(smem.scales[index_buf_idx][lane_idx*2 + i], scales[i]);
                        }
                        *(int2*)(smem.tma_coord[index_buf_idx] + lane_idx*2) = *(int2*)tma_coords;
                        if (lane_idx%4 == 0)
                            smem.is_k_valid[index_buf_idx][lane_idx/4] = valid_mask;

                        smem.bar_valid_coord_scales_full[index_buf_idx].arrive();
                        rs.update();
                    };

                    const int orig_end = min(args.num_orig_kv_blocks, args.end_block_idx);
                    int2 my_indices = args.start_block_idx < orig_end ? load_block_indices(args.start_block_idx, IsOrigBlock{}) : int2{};
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < orig_end; ++block_idx) {
                        bool has_next = block_idx+1 < orig_end;
                        int2 next_indices = has_next ? load_block_indices(block_idx+1, IsOrigBlock{}) : int2{};
                        process_one_block(block_idx, my_indices, IsOrigBlock{});
                        if (has_next) my_indices = next_indices;
                    }

                    const int extra_start = max(args.start_block_idx, args.num_orig_kv_blocks);
                    my_indices = extra_start < args.end_block_idx ? load_block_indices(extra_start, IsExtraBlock{}) : int2{};
                    CUTE_NO_UNROLL
                    for (int block_idx = extra_start; block_idx < args.end_block_idx; ++block_idx) {
                        bool has_next = block_idx+1 < args.end_block_idx;
                        int2 next_indices = has_next ? load_block_indices(block_idx+1, IsExtraBlock{}) : int2{};
                        process_one_block(block_idx, my_indices, IsExtraBlock{});
                        if (has_next) my_indices = next_indices;
                    }
                });
            }
        } else if (warp_idx >= 10 && elect_one_sync()) {
            if constexpr (IS_PREFILL) {
                if (warp_idx == 10) {
                    // CLC Producer thread
                    run_outer_loop([&](const OuterloopArgs &args) {
                        if (cta_idx == 0) {
                            smem.bar_clc_empty.wait(args.outer_loop_phase^1);
                            ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                        }
                        smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                    });
                }
            } else {
                // Raw (quantized) KV part Producer thread (+CLC Producer if splitKV is not enabled)
                run_outer_loop([&](const OuterloopArgs &args) {
                    if (warp_idx == 10 && FWD_MODE == FwdMode::Decode) {
                        if (cta_idx == 0) {
                            smem.bar_clc_empty.wait(args.outer_loop_phase^1);
                            ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                        }
                        smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                    }
                    for_each_kv_block<OrigKVFormat, ExtraKVFormat>(args.start_block_idx, args.end_block_idx, args.num_orig_kv_blocks, [&]<typename F>(int block_idx, bool is_extra_block) {
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_empty[raw_k_buf_idx].wait(raw_k_bar_phase^1);
                        const CUtensorMap *tensor_map = is_extra_block ? &tma_params.tensor_map_extra_kv_quant_part[cta_idx] : &tma_params.tensor_map_kv_quant_part[cta_idx];

                        int4 nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + (warp_idx == 10 ? 0 : 4));
                        CUTE_UNROLL
                        for (int row = (warp_idx == 10 ? 0 : 4); row < B_TOPK; row += 8) {
                            int4 cur_indices = nxt_indices;
                            if (row+8 < B_TOPK)
                                nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + row + 8);
                            ku::tma_gather4(
                                tensor_map,
                                smem.bar_raw_KV_full[raw_k_buf_idx],
                                smem.K_raw[raw_k_buf_idx].data() + row*RAW_TOKEN_SMEM_STRIDE<F>,
                                0,
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                            );
                        }
                        if (warp_idx == 10) {
                            smem.bar_raw_KV_full[raw_k_buf_idx].arrive_and_expect_tx(B_TOPK*RAW_TOKEN_SMEM_STRIDE<F>);
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    });
                });
            }
        }
    } else {
        // Scale & Exp threads
        cutlass::arch::warpgroup_reg_alloc<176>();

        int local_warp_idx = warp_idx - 12;
        bf16* sS_base = smem.S.data() + (local_warp_idx >= 2 ? (H_Q/2)*(B_TOPK/2) : 0) + (idx_in_warpgroup%64)*8;

        RingBufferState rs;
        run_outer_loop([&](const OuterloopArgs &args) {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

            CUTE_NO_UNROLL
            for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                auto [_, bar_phase] = rs.get<1>();
                // NOTE We don't need to sync for Prefill mode, since we have two synchronizations inside the loop body (one for p_exchange_buf sync, another one for rowwise_max_buf sync). The latter one guarantees the emptyness of p_exchange_buf and the former one guarantees the emptyness of rowwise_max_buf
                smem.bar_valid_coord_scales_full[indices_buf_idx].wait(indices_bar_phase);

                // Get P from TMEM
                float p[NUM_ELEMS_PER_THREAD];
                smem.bar_QK_done.wait(bar_phase);
                ku::tcgen05_after_thread_sync();
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG2_WARP02_SYNC,
                    barrier_ids::WG2_WARP13_SYNC,
                    false  // We never write P back through the exchange buffer
                >(
                    tmem_cols::P,
                    smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {smem.bar_P_empty.arrive(0u);},
                    smem.P_exchange,
                    p
                );

                // Get rowwise max of P
                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(64, barrier_ids::WG2_WARP02_SYNC + (local_warp_idx&1));
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                // Calc scale factor, and scale li
                float new_max, scale_for_old;
                if (!should_scale_o) {
                    // Don't scale O
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;   // mi is still identical within each row

                // Calculate S
                nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SV_done.wait(bar_phase^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*(H_Q/2), *(__int128_t*)(s + i*4));
                }

                // Rescale O
                if (k > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_V/2, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                smem.bar_S_O_full.arrive(0u);
                smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();

                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Reduce li
            smem.bar_li_empty.wait(args.outer_loop_phase^1);
            smem.rowwise_li_buf[idx_in_warpgroup^64] = li;
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG2_SYNC);
            li += smem.rowwise_li_buf[idx_in_warpgroup];

            if (idx_in_warpgroup < H_Q/2) {
                // Calculate output_scale and save
                int head_idx = cta_idx*(H_Q/2) + idx_in_warpgroup;
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + head_idx);
                float output_scale;
                if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                    output_scale = __fdividef(1.0f, li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -mi)));
                } else {
                    output_scale = __fdividef(1.0f, li);
                }
                smem.rowwise_li_buf[idx_in_warpgroup] = li == 0.0f ? 0.0f : output_scale;
                smem.bar_li_full.arrive();

                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if constexpr (IS_PREFILL) {
                    int global_index = args.s_q_idx*params.h_q + head_idx;
                    params.max_logits[global_index] = real_mi*CUDART_LN2_F;
                    params.lse[global_index] = cur_lse;
                } else {
                    if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                        params.lse[args.batch_idx*params.stride_lse_b + args.s_q_idx*params.stride_lse_s_q + head_idx] = cur_lse;
                    } else {
                        float cur_lse_2base = log2f(li) + mi;
                        params.lse_accum[args.n_split_idx*params.stride_lse_accum_split + args.s_q_idx*params.stride_lse_accum_s_q + head_idx] = cur_lse_2base;
                    }
                }
            }
        });
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

// We have two launchers with different kernel names to distinguish prefill and decode

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
sparse_attn_fwd_for_small_topk_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<FwdMode FWD_MODE, int D_QK, ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>
void KernelTemplate<FWD_MODE, D_QK, MODEL_TYPE, EXTRA_MODEL_TYPE>::run(const ArgT& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == H_Q);  // To save some calculation
    KU_ASSERT(params.d_qk == D_QK);
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.model_type == MODEL_TYPE && params.extra_model_type == EXTRA_MODEL_TYPE);
    }

    static_assert(D_Q == 512);
    CUtensorMap tensor_map_q;
    if constexpr (IS_DECODE) {
        if (params.b > 1) {
            KU_ASSERT(params.stride_q_b % params.stride_q_s_q == 0, "In decode mode for V4 sparse fp8 decoding on sm100f, q.stride(0) (on the batch dimension) must be divisible by q.stride(1) (on the sequence dimension).");
            tensor_map_q = ku::make_tensor_map(
                {64ul, H_Q, 2ul, (D_Q/64ul)/2ul, (unsigned long)params.b * (params.stride_q_b / params.stride_q_s_q)},
                ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 64, params.stride_q_s_q}, sizeof(bf16)),
                {64, H_Q/2, 2, (D_Q/64)/2, 1},
                params.q,
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            );
        } else {
            tensor_map_q = ku::make_tensor_map(
                {64ul, H_Q, 2ul, (D_Q/64ul)/2ul, (unsigned long)params.s_q},
                ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 64, params.s_q == 1 ? 0 : params.stride_q_s_q}, sizeof(bf16)),
                {64, H_Q/2, 2, (D_Q/64)/2, 1},
                params.q,
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            );
        }
    } else {
        tensor_map_q = ku::make_tensor_map(
            {64ul, H_Q, 2ul, (D_Q/64ul)/2ul, (unsigned long)params.s_q},
            ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 64, params.stride_q_s_q}, sizeof(bf16)),
            {64, H_Q/2, 2, (D_Q/64)/2, 1},
            params.q,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );  // We use this layout to group Q[0:64] and Q[256:256+64] together, for UTCCP for dual gemm
    }

    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_quant_part[2] = {}, tensor_map_extra_kv_quant_part[2] = {};
    CUtensorMap tensor_map_kv_bf16_part = {}, tensor_map_extra_kv_bf16_part = {};
    if constexpr (IS_DECODE) {
        // Two maps per cache, one per CTA of the pair
        auto make_quant_part_maps = [&]<typename F>(F, const char *name, void* k_ptr, int num_blocks, int64_t stride_kv_block, int64_t stride_kv_row, CUtensorMap (&maps)[2]) {
            maps[0] = make_kv_quant_part_tensor_map<F, QUANT_PART_MAP_DIM0<F, 0>, RAW_TOKEN_SMEM_STRIDE<F>>(
                name, k_ptr, num_blocks, stride_kv_block, stride_kv_row, 0);
            maps[1] = make_kv_quant_part_tensor_map<F, QUANT_PART_MAP_DIM0<F, 1>, RAW_TOKEN_SMEM_STRIDE<F>>(
                name, k_ptr, num_blocks, stride_kv_block, stride_kv_row, QUANT_PART_CTA_OFFSET<F>);
        };
        make_quant_part_maps(OrigKVFormat{}, "k_cache", params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row, tensor_map_kv_quant_part);
        if constexpr (D_BF16 > 0) {
            tensor_map_kv_bf16_part = make_kv_bf16_part_tensor_map<OrigKVFormat, D_BF16, 128>(params.kv, params.num_blocks, params.stride_kv_block);
        }
        if (params.extra_topk > 0) {
            make_quant_part_maps(ExtraKVFormat{}, "extra_k_cache", params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row, tensor_map_extra_kv_quant_part);
            if constexpr (ExtraKVFormat::D_BF16 > 0) {
                tensor_map_extra_kv_bf16_part = make_kv_bf16_part_tensor_map<ExtraKVFormat, D_BF16, 128>(params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block);
            }
        }
    } else {
        tensor_map_kv = ku::make_tensor_map(
            {D_QK, (unsigned long)params.s_kv}, 
            {(unsigned long)params.stride_kv_s_kv*sizeof(bf16)},
            {64, 1},
            params.kv,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    CUtensorMap tensor_map_o;
    if constexpr (IS_DECODE) {
        tensor_map_o = ku::make_tensor_map(
            {64, H_Q, D_V/64, (unsigned long)params.s_q, (unsigned long)params.b},
            ku::make_stride_helper<int>({params.stride_o_h_q, 64, params.stride_o_s_q, params.stride_o_b}, sizeof(bf16)),
            {64, H_Q/2, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    } else {
        tensor_map_o = ku::make_tensor_map(
            {64, H_Q, D_V/64, (unsigned long)params.s_q, 1ul},
            ku::make_stride_helper<int>({D_V, 64, H_Q*D_V, H_Q*D_V}, sizeof(bf16)),
            {64, H_Q/2, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    CUtensorMap tensor_map_o_accum = {};
    if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
        tensor_map_o_accum = ku::make_tensor_map(
            {32, H_Q, D_V/32, (unsigned long)params.s_q, (unsigned long)params.num_sm_parts + params.b},
            ku::make_stride_helper<int>({params.stride_o_accum_h_q, 32, params.stride_o_accum_s_q, params.stride_o_accum_split}, sizeof(float)),
            {32, H_Q/2, B_EPI_SPLITKV/32, 1, 1},
            params.o_accum,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    TmaParams tma_params;
    if constexpr (IS_DECODE) {
        tma_params = {
            tensor_map_q,
            tensor_map_o,
            tensor_map_o_accum,
            {tensor_map_kv_quant_part[0], tensor_map_kv_quant_part[1]},
            tensor_map_kv_bf16_part,
            {tensor_map_extra_kv_quant_part[0], tensor_map_extra_kv_quant_part[1]},
            tensor_map_extra_kv_bf16_part
        };
    } else {
        tma_params = {
            tensor_map_q,
            tensor_map_kv,
            tensor_map_o
        };
    }
    
    using KT = KernelTemplate;
    void (*kernel)(__grid_constant__ const typename KT::ArgT, __grid_constant__ const typename KT::TmaParams);
    if constexpr (IS_PREFILL) {
        kernel = &sparse_attn_fwd_for_small_topk_kernel<KT>;
    } else {
        kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KT>;
    }
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape;
    if constexpr (IS_DECODE) {
        grid_shape = dim3(2*params.s_q, FWD_MODE == FwdMode::DecodeWithSplitKV ? params.num_sm_parts : params.b, 1);
    } else {
        grid_shape = dim3(2*params.s_q, 1, 1);
    }

    cutlass::ClusterLaunchParams launch_params = {
        grid_shape,
        dim3(NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

template<FwdMode FWD_MODE, int D_QK, ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>
void run_sparse_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params) {
    using Kernel = KernelTemplate<FWD_MODE, D_QK, MODEL_TYPE, EXTRA_MODEL_TYPE>;
    Kernel::run(params);
}

}
