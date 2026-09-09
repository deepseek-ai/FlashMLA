/*
Sparse Attention Forward Pass (Phase 1) — SM100, h_q == 64

Forward attention kernel for h_q == 64 query heads.
Uses UTCMMA and TMEM. Single-CTA design with B_TOPK=64. Prefill mode only.

Template parameters:
  FWD_MODE — Forward mode (Prefill)
  D_QK     — Head dimension for QK (512 or 576)

Grid: [s_q, 1, 1]

I/O: See SparseAttnFwdParams in kernels/params.h
*/
#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "kernels/params.h"
#include "kernels/utils.h"
#include "kernels/sm100/helpers.h"
#include "kernels/sm100/common_subroutine.h"
#include "config.h"

namespace sm100::prefill::sparse_fwd::head64 {

using namespace cute;

/*
Pipeline Overview:

| Copy |    MMA    |   Scale & Exp   |

KV0
KV1
KV2
        P0 = QK0^T
                    S0 = exp(P0)
                    scale(O) w.r.t P0
        P1 = QK1^T
                    S1 = exp(P1)
        O += S0V0
KV3                 scale(O) w.r.t P1
        P2 = QK2^T
                    S2 = exp(P2)
        O += S1V1
KV4                 scale(O) w.r.t P2
        P3 = QK3^T
                    S3 = exp(P3)
        O += S2V2
KV5                 scale(O) w.r.t P3

...

        O += S(n-3)V(n-3)
                    scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
                   scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)
*/

using FwdMode = SparseAttnFwdMode;

template<FwdMode FWD_MODE, int D_QK>
template<typename TmaParam>
__device__ void
KernelTemplate<FWD_MODE, D_QK>::sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParam &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid shape: [s_q, 1, 1]

    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        if constexpr (HAVE_ROPE) {
            cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        }
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));

        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_utccp_nope.init(1);
        if constexpr (HAVE_ROPE) {
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_utccp_rope.init(1);
        }
        plan.bar_clc_full.init(1);
        plan.bar_clc_empty.init(NUM_WORKER_THREADS);
        fence_barrier_init();
    } else if (warp_idx == 1 && elect_one_sync()) {
        // Initialize other barriers
        CUTE_UNROLL
        for (int i = 0; i < NUM_BUFS; ++i) {
            plan.bar_qk_nope_done[i].init(1);
            plan.bar_sv_done[i].init(1);
            plan.bar_kv_nope_ready[i][0].init(1);
            plan.bar_kv_nope_ready[i][1].init(1);
            plan.bar_k_valid_ready[i].init(B_TOPK/8);
            plan.bar_k_valid_free[i].init(128);
        }
        plan.bar_p_free.init(128);
        plan.bar_so_ready.init(128);
        if constexpr (HAVE_ROPE) {
            plan.bar_qk_rope_done.init(1);
            plan.bar_kv_rope_ready.init(64);
        }
        plan.bar_o_write_back_done.init(128);
        plan.bar_o_write_back_done_waited.init(4);
        fence_barrier_init();
    } else if (warp_idx == 2) {
        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    struct OuterloopArgs {
        bool outer_loop_phase;
        int s_q_idx;
        int num_k_blocks;
        int topk_length;
    };

    auto issue_q_rope_tma = [&](int s_q_idx) {
        if constexpr (HAVE_ROPE) {
            Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
            Tensor sQ_rope = make_tensor(make_smem_ptr(plan.qk_rope_slot), SmemLayoutQRoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
        }
    };

    auto issue_q_rope_utccp = [&](bool outer_loop_phase) {
        if constexpr (HAVE_ROPE) {
            plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*(D_Q-D_V)*sizeof(bf16));
            plan.bar_prologue_q_rope.wait(outer_loop_phase);
            ku::tcgen05_after_thread_sync();

            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.qk_rope_slot),
                    tile_to_shape(
                        UMMA::Layout_K_SW64_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<32>>{}
                    )
                )
            );

            // Copy the RoPE tile: (2*B_H) rows * 32 cols (64B) (in UTCCP's view), or B_H rows * 64 cols (in our view)
            // A subtile is (2*B_H) rows * 16 cols (256b, 32B) (in UTCCP's view), or B_H rows * 16 cols * 2 (in our view)
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 2; ++subtile_idx) {
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_rope_desc + (subtile_idx*32) / 16,
                    tmem_cols::Q_RoPE + subtile_idx*8
                );
            }
            ku::umma_arrive_noelect(plan.bar_prologue_utccp_rope);
        }
    };

    auto issue_q_nope_tma = [&](int s_q_idx, int qko_slot_idx) {
        Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
        Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.qko_slots[qko_slot_idx].data()), SmemLayoutQNoPE{});
        ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);
    };

    auto issue_q_nope_utccp = [&](bool outer_loop_phase, int qko_slot_idx) {
        plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
        plan.bar_prologue_q_nope.wait(outer_loop_phase);
        ku::tcgen05_after_thread_sync();
        UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
            make_tensor(
                make_smem_ptr(plan.u.qko_slots[qko_slot_idx].data()),
                tile_to_shape(
                    UMMA::Layout_K_SW128_Atom<bf16>{},
                    Shape<Int<B_H*2>, Int<64>>{}    // TODO Explain this layout and dual gemm
                )
            )
        );
        
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
            // A tile is (2*B_H) rows * 64 cols (128B) (in UTCCP's view), or B_H rows * 128 cols (in our view)
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or B_H rows * 16 cols * 2 (in our view)
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_nope_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16,   // Remember that 4 LSBs are not included
                    tmem_cols::Q + tile_idx*32 + subtile_idx*8
                );
            }
        }
        ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);
    };

    auto run_outer_loop = [&](auto loop_body) {
        int outer_loop_phase = false;
        ku::CLCResult next_job = {true, (int)blockIdx.x, 0, 0};
        CUTE_NO_UNROLL
        while (next_job.is_valid) {
            int s_q_idx = next_job.x;
            int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
            int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 2);  // num_k_blocks always >= 2 to simplify synchronizations across outer loop boundries
            OuterloopArgs args = {
                (bool)outer_loop_phase,
                s_q_idx,
                num_k_blocks,
                topk_length
            };
            loop_body(args);

            plan.bar_clc_full.wait(outer_loop_phase);
            next_job = ku::get_clc_query_response<true>(plan.clc_response_obj);
            outer_loop_phase ^= 1;

            plan.bar_clc_empty.arrive();
        }
    };

    RingBufferState rs;

    if (warpgroup_idx == 0) {
        // Scale & Exp warps
        bf16* sS_base = plan.s + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        run_outer_loop([&](const OuterloopArgs &args) {
            // The following three numbers are 
            // - mi: max_logits used to scale Pi (i.e. O := exp2(Pi*scale - mi) @ V)
            // - li: sumexp, i.e. li := sum(exp(Pi*scale - mi))
            // - real_mi: real max logits, i.e. real_mi := max(Pi*scale)
            // where Pi is the i-th row of P, P := QK^T
            // mi and real_mi are always consistent within the two threads that
            // controls one row (i.e. thread 0+64, 1+65, 2+66, ...) after every update
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            cute::tma_store_wait<0>();
            plan.bar_o_write_back_done.arrive();
    
            CUTE_NO_UNROLL
            for (int k = 0; k < args.num_k_blocks; ++k) {
                // Wait for P
                NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
                auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                plan.bar_qk_nope_done[buf_idx].wait(bar_phase);
                plan.bar_k_valid_ready[buf_idx].wait(bar_phase);    // Put the barrier wait here for more code reordering space
                ku::tcgen05_after_thread_sync();
                
                // Load P
                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false  // Prefill keeps P in registers (no store_back_p)
                >(
                    tmem_cols::P,
                    plan.is_k_valid[buf_idx],
                    warp_idx, lane_idx, 
                    [&]() {plan.bar_p_free.arrive();},
                    plan.p_exchange_buf,
                    p
                );
                plan.bar_k_valid_free[buf_idx].arrive();
                
                // Get rowwise max of Pi
                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;
    
                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
                // By this point:
                // - cur_pi_max, real_mi, and mi is identical within each row (i.e. thread 0+64, 1+65, ...)
                // - should_scale_o is identical among every warp, and is identical among threads that controls the same row (i.e. among threads 0~31+64~95; and is identical among threads 32~63+96~127)
    
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
                li = fma(li, scale_for_old, cur_sum);
    
                // Wait for last SV gemm, write S
                if (k > 0) {
                    auto [last_sv_buf_idx, last_sv_bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
                    plan.bar_sv_done[last_sv_buf_idx].wait(last_sv_bar_phase);
                }
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                    *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
                }
    
                // Scale O
                if (k > 0 && should_scale_o) {
                    // plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);   // NOTE We have waited for last SV gemm before
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_V/(128/B_H), 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }
                
                fence_view_async_shared();
                plan.bar_so_ready.arrive();

                rs.update();
            }

            plan.bar_o_write_back_done_waited.wait(args.outer_loop_phase);
    
            // Epilogue
            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }
            
            // Exchange li
            plan.rowwise_li_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_li_buf[idx_in_warpgroup^64];
    
            // Store mi and li
            if (idx_in_warpgroup < B_H) {
                bool is_padding_row = idx_in_warpgroup >= params.h_q;
                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if (!is_padding_row) {
                    int global_index = args.s_q_idx*params.h_q + idx_in_warpgroup;
                    params.max_logits[global_index] = real_mi*CUDART_LN2_F;
                    params.lse[global_index] = cur_lse;
                }
            }
            
            auto [o_slot_idx, __] = rs.offset_by(+2).get<NUM_BUFS>();
    
            // Store O
            float attn_sink = (params.attn_sink == nullptr || idx_in_warpgroup%B_H >= params.h_q)
                ? -CUDART_INF_F : __ldg(params.attn_sink + (idx_in_warpgroup%B_H))*CUDART_L2E_F;
            float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));
            Tensor sO = make_tensor(make_smem_ptr(plan.u.qko_slots[o_slot_idx].data()), SmemLayoutO{});
            Tensor tma_gO = flat_divide(
                tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, args.s_q_idx),
                Shape<Int<B_H>, Int<B_EPI>>{}
            )(_, _, _0{}, _);
            Tensor sO_divided = flat_divide(
                sO,
                Shape<Int<B_H>, Int<B_EPI>>{}
            )(_, _, _0{}, _);
            auto thr_tma = tma_params.tma_O.get_slice(_0{});
    
            float2 o[B_EPI/2];
            bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
            if (!have_valid_indices) {
                // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/2; ++i)
                    o[i].x = o[i].y = 0.0f;
                output_scale = 1.0f;
            }
    
            float2 output_scale_float2 = make_float2(output_scale, output_scale);
    
            bf16* sO_addrs[8];
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/8; ++i) {
                sO_addrs[i] = &sO(idx_in_warpgroup%B_H, i*8);
            }
    
            // Wait for the last GEMM
            {
                auto [last_sv_buf_idx, last_sv_bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
                plan.bar_sv_done[last_sv_buf_idx].wait(last_sv_bar_phase);
                ku::tcgen05_after_thread_sync();
            }
    
            static constexpr int NUM_EPI_SB = D_V/B_EPI_SB;
            static constexpr int NUM_TMA_PARTS = 2;
            CUTE_UNROLL
            for (int c = 0; c < NUM_EPI_SB; ++c) {
                // Each tile: B_H x B_EPI_SB
                CUTE_UNROLL
                for (int k = 0; k < B_EPI_SB/B_EPI/NUM_TMA_PARTS; ++k) {
                    // Load O from tO
                    if (have_valid_indices) {
                        ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*(B_EPI_SB/NUM_TMA_PARTS) + k*B_EPI, o);
                        cutlass::arch::fence_view_async_tmem_load();
                    }
                    // NOTE. We neither signal any barrier after tmem_O is free, nor do we wait for any barrier in the UTCMMA warp, since the first O gemm in the next round depends on S, which depends on this warpgroup

                    // Convert and store
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI/8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        CUTE_UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                        }
                        bf16* o_smem_ptr = sO_addrs[i] + (c*B_EPI_SB + (idx_in_warpgroup/B_H)*(B_EPI_SB/(128/B_H)) + k*B_EPI)*B_H;
                        ku::st_shared(o_smem_ptr, *(__int128_t*)(o_bf16));
                    }
    
                    // Sync
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    
                    // Store into global memory
                    if (warp_idx < NUM_TMA_PARTS && elect_one_sync()) {
                        int epi_chunk_idx = c*(B_EPI_SB/B_EPI) + ((B_EPI_SB/B_EPI)/NUM_TMA_PARTS)*warp_idx + k;
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                            thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                        );
                    }
                }
            }
            cute::tma_store_arrive();
        });

        if (warp_idx == 3) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
        run_outer_loop([&](const OuterloopArgs &args) {
            if (elect_one_sync()) {
                int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q; // [topk]
                CUTE_NO_UNROLL
                for (int k = 0; k < args.num_k_blocks; ++k) {
                    int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                    int max_indices = -1, min_indices = params.s_kv;
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                        max_indices = max(max_indices, int4_max(indices[local_row]));
                        min_indices = min(min_indices, int4_min(indices[local_row]));
                    }
                    bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                    bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;    // Don't skip TMA for the first NUM_BUFS turns to swipe out invalid values in kv buffer
                    
                    if (k == 1) {
                        // Since q_nope coincidences with k["buffer idx of the 1st block"]
                        plan.bar_prologue_utccp_nope.wait(args.outer_loop_phase);
                    } else if (k == 2) {
                        // Since o_buf coincidences with k["buffer idx of the 2nd block"]
                        plan.bar_o_write_back_done.wait(args.outer_loop_phase);
                    }
                    
                    // Copy NoPE
                    auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                    plan.bar_sv_done[buf_idx].wait(bar_phase^1);
                    bf16* sK_nope_base = plan.u.qko_slots[buf_idx].data() + warp_idx*4*64;
    
                    auto load_kv_nope_part = [&](int part_idx) {
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                            CUTE_UNROLL
                            for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                                ku::tma_gather4(
                                    &(tma_params.tensor_map_kv_nope),
                                    plan.bar_kv_nope_ready[buf_idx][part_idx],
                                    sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                    local_col*64,
                                    indices[local_row],
                                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                );
                            }
                        }
                    };
    
                    if (!should_skip_tma) {
                        load_kv_nope_part(0);
                        load_kv_nope_part(1);
                    } else {
                        // NOTE See head128/phase1.cuh for this TMA skipping technique
                        CUTE_UNROLL
                        for (int part_idx = 0; part_idx < 2; ++part_idx)
                            plan.bar_kv_nope_ready[buf_idx][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(bf16));
                    }

                    rs.update();
                }
                if (args.num_k_blocks <= 2) {
                    plan.bar_o_write_back_done.wait(args.outer_loop_phase);
                }
                plan.bar_o_write_back_done_waited.arrive();
            }
            __syncwarp();
        });
    } else {
        // MMA warp
        if (warp_idx == 8 && elect_one_sync()) {
            // Allocate tmem tensors
            TiledMMA tiled_mma_P = TiledMMA_P{};
            TiledMMA tiled_mma_O = TiledMMA_O{};
            // NOTE These tXXX tensors are only for a forged layout (so that CuTe is able to generate correct address in cute::gemm)
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK*2>>{});
            Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<QK_MRGEMM_K_NOPE/2>>{})
            );
            Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<QK_MRGEMM_K_NOPE/2>>{})
            );
            Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<HAVE_ROPE ? QK_MRGEMM_K_ROPE : 64 / (128/B_H)>>{})
            );
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
            tP.data().get() = tmem_cols::P;
            tQ_nope_part0.data().get() = tmem_cols::Q;
            tQ_nope_part1.data().get() = tmem_cols::Q + 64;
            tQ_rope.data().get() = tmem_cols::Q_RoPE;
            tO.data().get() = tmem_cols::O;
            
            run_outer_loop([&](const OuterloopArgs &args) {
                // Copy Q into k["buffer idx of the 1st block"+1]
                // NOTE. As we reach here, we must have already issued the last O gemm of the last round, which means that the penultimate O gemm must be finished (since the last O gemm |-> bar_so_ready (the last S is ready) |-> the penultimate O gemm must be finished). So the corresponding K buffer must be free.
                // For the RoPE part, as we reach here, the last S from the last round must be ready, which means that the corresponding K RoPE buffer must be free.
                int q_slot_idx = rs.offset_by(+1).get<NUM_BUFS>().first;
                issue_q_rope_tma(args.s_q_idx);
                issue_q_nope_tma(args.s_q_idx, q_slot_idx);
                issue_q_rope_utccp(args.outer_loop_phase);
                issue_q_nope_utccp(args.outer_loop_phase, q_slot_idx);
                // NOTE Here we don't need to wait for bar_prologue_utccp_rope, since the copy-in of the first RoPE relies on bar_prologue_utccp_rope

                CUTE_NO_UNROLL
                for (int k = 0; k < args.num_k_blocks+1; ++k) {
                    if (k < args.num_k_blocks) {
                        // Pi = QKi^T
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.qko_slots[buf_idx].data()), SmemLayoutKNoPE_TiledMMA{});
                        Tensor sK_rope = make_tensor(make_smem_ptr(plan.qk_rope_slot), SmemLayoutKRoPE_TiledMMA{});
    
                        auto [__, binary_bar_phase] = rs.get<1>();
                        plan.bar_p_free.wait(binary_bar_phase^1);
                        ku::tcgen05_after_thread_sync();
                        
                        // Wait for K (RoPE)
                        // P = Q(rope) @ K(rope)^T
                        if constexpr (HAVE_ROPE) {
                            plan.bar_kv_rope_ready.wait(binary_bar_phase);
                            ku::tcgen05_after_thread_sync();
                            ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);
                            ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                        }
    
                        // Wait for K (NoPE)
                        if (k == 0) {
                            plan.bar_prologue_utccp_nope.wait(args.outer_loop_phase);
                        }
                        Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                        CUTE_UNROLL
                        for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
                            plan.bar_kv_nope_ready[buf_idx][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
                            plan.bar_kv_nope_ready[buf_idx][kv_nope_part_idx].wait(bar_phase);
                            ku::tcgen05_after_thread_sync();
    
                            // P += Q(nope) @ K(nope)^T
                            bool clear_accum = (!HAVE_ROPE) && kv_nope_part_idx == 0;
                            ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0, sK_nope_divided(_, _, kv_nope_part_idx), tP, clear_accum);
                        }
                        ku::umma_arrive_noelect(plan.bar_qk_nope_done[buf_idx]);
                    }
                    if (k > 0) {
                        // O += S(i-1)V(i-1)
                        auto [buf_idx, bar_phase] = rs.offset_by(-1).get<NUM_BUFS>();
    
                        Tensor sS = make_tensor(make_smem_ptr(plan.s), SmemLayoutS{});
                        Tensor sV = make_tensor(make_smem_ptr(plan.u.qko_slots[buf_idx].data()), SmemLayoutV{});
    
                        // Wait for S(i-1) and O to be scaled
                        auto [__, binary_bar_phase] = rs.offset_by(-1).get<1>();
                        plan.bar_so_ready.wait(binary_bar_phase);
                        ku::tcgen05_after_thread_sync();
    
                        // O += sS @ sV
                        ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
                        ku::umma_arrive_noelect(plan.bar_sv_done[buf_idx]);
                    }

                    rs.update();
                }
                rs = rs.offset_by(-1);
            });
        } else if (warp_idx == 9) {
            // KV valid loading + CLC producer warp
            if (lane_idx < B_TOPK/8) {
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q; // [topk]
                    CUTE_NO_UNROLL
                    for (int k = 0; k < args.num_k_blocks; ++k) {
                        char k_validness_mask = load_indices_and_generate_mask(
                            lane_idx,
                            gIndices + k*B_TOPK,
                            params.s_kv,
                            k*B_TOPK,
                            args.topk_length
                        );
    
                        auto [buf_idx, bar_phase] = rs.get<NUM_BUFS>();
                        plan.bar_k_valid_free[buf_idx].wait(bar_phase^1);
                        plan.is_k_valid[buf_idx][lane_idx] = k_validness_mask;
                        plan.bar_k_valid_ready[buf_idx].arrive();

                        rs.update();
                    }
                });
            } else if (lane_idx == B_TOPK/8) {
                run_outer_loop([&](const OuterloopArgs &args) {
                    plan.bar_clc_empty.wait(args.outer_loop_phase^1);
                    ku::issue_clc_query(plan.bar_clc_full, plan.clc_response_obj);
                    plan.bar_clc_full.arrive_and_expect_tx(sizeof(plan.clc_response_obj));
                });
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            // RoPE loading warp
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = B_H/GROUP_SIZE, ROWS_PER_THREAD = B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.qk_rope_slot), SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q; // [topk]
                    CUTE_NO_UNROLL
                    for (int k = 0; k < args.num_k_blocks; ++k) {
                        auto [_, binary_bar_phase] = rs.get<1>();
                        int indices[ROWS_PER_THREAD];
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                            indices[local_row] = __ldg(gIndices + k*B_TOPK + group_idx + local_row*NUM_GROUPS);
                        plan.bar_qk_rope_done.wait(binary_bar_phase^1);
                        if (k == 0) {
                            plan.bar_prologue_utccp_rope.wait(args.outer_loop_phase); // Wait for Q RoPE's UTCCP so that qk_rope_slot is empty
                        }
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                            int index = indices[local_row];
                            ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                                params.kv + (int64_t)index*params.stride_kv_s_kv + 512 + idx_in_group*8,
                                sK_rope_base + local_row*NUM_GROUPS*32,
                                index >= 0 && index < params.s_kv
                            );  // NOTE Using cp.async instead of TMA is faster here
                            // NOTE Here we only consider the range of `index` instead of also checking against topk_length, as it's noted that under this scenario (i.e. there exists a valid index among indices[topk_length: ] that points to a token who has NaN inside)
                        }
                        cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                        rs.update();
                    }
                });
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<FwdMode FWD_MODE, int D_QK>
void KernelTemplate<FWD_MODE, D_QK>::run(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.topk >= 128);  // To simplify synchronizations between outer loop boundries
    KU_ASSERT(params.h_q == 64);
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);

    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V == 0 ? 64 : D_Q-D_V, params.s_q);    // If 
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        SmemLayoutOTiles<1>{}
    );

    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_O, tma_O,
        tensor_map_kv_nope
    };
    auto kernel = &sparse_attn_fwd_kernel<KernelTemplate<FWD_MODE, D_QK>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<FwdMode FWD_MODE, int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate<FWD_MODE, D_QK>::run(params);
}

}

