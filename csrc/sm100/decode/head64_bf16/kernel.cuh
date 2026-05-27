#include "kernel.h"

#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kerutils/kerutils.cuh"

#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::decode::head64_bf16 {

template<ModelType MODEL_TYPE>
template<typename TmaParam>
__device__ void
KernelTemplate<MODEL_TYPE>
::flash_fwd_splitkv_mla_bf16_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params) {
#if defined(KERUTILS_ENABLE_SM100A)
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_SW128.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q_sw64);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_nope);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_rope);
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_rope_ready[i].init(1);
                plan.bar_nope_ready[i].init(1);  // BF16: TMA loads directly into dequant.nope with swizzle
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_valid_coord_ready[i].init(32);
                plan.bar_valid_coord_free[i].init(128+1+1);  // wg0 + nope_producer + rope_producer
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    struct MainLoopArgs {
        int batch_idx, start_block_idx, end_block_idx;
        bool is_no_split; int n_split_idx;
        bool bar_phase_batch_rel;
        int topk_length, extra_topk_length, num_orig_kv_blocks;
        bool is_last_batch;
    };

    auto run_main_loop = [&](auto f) {
        DecodingSchedMeta sched_meta;
        KU_LDG_256(
            params.tile_scheduler_metadata_ptr + partition_idx,
            &sched_meta,
            ".nc",
            "no_allocate",
            "evict_normal",
            "256B"
        );

        if (sched_meta.begin_req_idx >= params.b) {
            return;
        }
        
        bool bar_phase_batch_rel = 0;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx, bar_phase_batch_rel ^= 1) {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);
            int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
            int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
            bool is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
            int n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);

            MainLoopArgs args = {
                batch_idx, start_block_idx, end_block_idx,
                !is_split, n_split_idx,
                bar_phase_batch_rel,
                topk_length, extra_topk_length,
                orig_topk_padded / B_TOPK,
                batch_idx == sched_meta.end_req_idx
            };

            f(args);
            NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync).arrive_and_wait_unaligned();
        }
    };

    struct RingState {
        int buf_idx = 0;
        bool bar_phase = 0;
        int index_buf_idx = 0;
        bool index_bar_phase = 0;
        CUTE_DEVICE void update() {
            bar_phase ^= (buf_idx == NUM_BUFS-1);
            buf_idx = (buf_idx+1) % NUM_BUFS;
            index_bar_phase ^= (index_buf_idx == NUM_INDEX_BUFS-1);
            index_buf_idx = (index_buf_idx+1) % NUM_INDEX_BUFS;
        }
    };
    RingState rs;

    if (warpgroup_idx == 0) {
        // Scale & Exp warpgroup
        cutlass::arch::warpgroup_reg_alloc<224>();

        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_buf.data()), SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI/8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup%64, (idx_in_warpgroup/64)*128 + i*8);

        const float2 scale = float2 {params.sm_scale_div_log2, params.sm_scale_div_log2};
        bf16* sS_base = plan.s_p.s.data() + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);

        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg((float*)params.attn_sink + (idx_in_warpgroup%64)) * CUDART_L2E_F;
        
        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();

            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();

                // Load P
                float p[B_TOPK/2], p_peer[B_TOPK/2];
                if (warp_idx < 2) {
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P, p);
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P+32, p_peer);
                } else {
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P, p_peer);
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P+32, p);
                }
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                // Reduce within shared mem
                {
                    CUTE_UNROLL
                    for (int i = 0; i < (B_TOPK/2)/4; ++i)
                        plan.s_p.p_exchange_buf[warp_idx^2][i*32 + lane_idx] = *(float4*)(p_peer + i*4);
                    NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
                    CUTE_UNROLL
                    for (int i = 0; i < (B_TOPK/2)/4; ++i) {
                        float2 t[2];
                        *(float4*)t = plan.s_p.p_exchange_buf[warp_idx][i*32 + lane_idx];
                        float2* cur_p = (float2*)(p + i*4);
                        cur_p[0] = ku::float2_add(cur_p[0], t[0]);
                        cur_p[1] = ku::float2_add(cur_p[1], t[1]);
                    }
                }

                // Mask
                uint32_t valid_mask = *((uint32_t*)plan.is_token_valid[rs.index_buf_idx] + (idx_in_warpgroup>=64?1:0));
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK/2; i += 1) {
                    if (!(valid_mask>>i&1))
                        p[i] = -CUDART_INF_F;
                }
                
                // Get rowwise max of Pi
                float cur_pi_max = -CUDART_INF_F;
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2); i += 1) {
                    cur_pi_max = max(cur_pi_max, p[i]);
                }
                cur_pi_max *= params.sm_scale_div_log2;

                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                // Calc scale factor, and scale li
                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                // Calculate S
                __nv_bfloat162 s[(B_TOPK/2)/2];
                float2 neg_new_max = float2 {-new_max, -new_max};
                float2 cur_sum = float2 {0.0f, 0.0f};
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                    float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale, neg_new_max);
                    d.x = exp2f(d.x);
                    d.y = exp2f(d.y);
                    cur_sum = ku::float2_add(cur_sum, d);
                    s[i] = __float22bfloat162_rn(d);
                }
                li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

                // Write S
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK/2/8; i += 1) {
                    *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
                }

                // Scale O
                if (block_idx != args.start_block_idx && should_scale_o) {
                    float2 scale_for_old_float2 = float2 {scale_for_old, scale_for_old}; 
                    ku::tcgen05_after_thread_sync();

                    static constexpr int CHUNK_SIZE = 64;
                    float2 o[CHUNK_SIZE/2];
                    CUTE_UNROLL
                    for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
                        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx*CHUNK_SIZE, o);
                        cutlass::arch::fence_view_async_tmem_load();
                        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                            o[i] = ku::float2_mul(o[i], scale_for_old_float2);
                        }
                        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx*CHUNK_SIZE, o);
                        cutlass::arch::fence_view_async_tmem_store();
                    }
                    ku::tcgen05_before_thread_sync();
                }
                
                fence_view_async_shared();
                plan.bar_so_ready[rs.buf_idx].arrive();

                if (block_idx != args.end_block_idx-1) {
                    rs.update();
                }
            }

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Exchange li
            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_max_buf[idx_in_warpgroup^64];

            // Store li
            if (idx_in_warpgroup < B_H) {
                if (args.is_no_split) {
                    float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                    cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                    float* gSoftmaxLse = (float*)params.lse + args.batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + idx_in_warpgroup;
                    *gSoftmaxLse = cur_lse;
                } else {
                    float cur_lse = log2f(li) + mi;
                    float* gSoftmaxLseAccum = (float*)params.lse_accum + args.n_split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + idx_in_warpgroup;
                    *gSoftmaxLseAccum = cur_lse;
                }
            }
        
            plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
            rs.update();
            ku::tcgen05_after_thread_sync();

            if (args.is_last_batch) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (args.is_no_split) {
                Tensor tma_gO = flat_divide(
                    tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, args.batch_idx),
                    Shape<Int<B_H>, Int<64>>{}
                )(_, _, _0{}, _);
                auto thr_tma = tma_params.tma_O.get_slice(_0{});
                Tensor tma_sO = flat_divide(
                    sO,
                    Shape<Int<B_H>, Int<64>>{}
                )(_, _, _0{}, _);

                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));
                float2 o_scale_float2 = {o_scale, o_scale};
                float2 o[B_EPI/2];
                __nv_bfloat162 o_bf16[B_EPI/2];
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j) {
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[j]);
                    }
                    int col_base = (i*B_EPI>=D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 8; ++j)
                        *(__int128_t*)(sO_bases[j] + col_base*B_H) = *(__int128_t*)(&o_bf16[j*4]);
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    if (warp_idx == 0 && elect_one_sync()) {
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(tma_sO(_, _, col_base/64)),
                            thr_tma.partition_D(tma_gO(_, _, col_base/64))
                        );
                    }
                    if (warp_idx == 1 && elect_one_sync()) {
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(tma_sO(_, _, col_base/64 + (D_V/4)/64)),
                            thr_tma.partition_D(tma_gO(_, _, col_base/64 + (D_V/4)/64))
                        );
                    }
                }
                cute::tma_store_arrive();
            } else {
                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);
                float2 o_scale_float2 = {o_scale, o_scale};
                constexpr int B_EPI = 64;
                float2 o[B_EPI/2];
                Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_accum_buf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j)
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                    int col_base = (idx_in_warpgroup/64)*128 + (i*B_EPI >= D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 4; ++j)
                        *(__int128_t*)&sO(idx_in_warpgroup%64, col_base + j*4) = *(__int128_t*)(&o[j*2]);
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                if (elect_one_sync()) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < B_H/4; ++local_row) {
                        int smem_row = local_row*4 + warp_idx;
                        SM90_BULK_COPY_S2G::copy(
                            &sO(smem_row, _0{}),
                            (float*)params.o_accum + args.n_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + smem_row*params.stride_o_accum_h_q,
                            D_V*sizeof(float)
                        );
                    }
                    cute::tma_store_arrive();
                }
            }
        });

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 4 && elect_one_sync()) {
            // MMA Warp
            run_main_loop([&](const MainLoopArgs &args) {
                if (args.start_block_idx >= args.end_block_idx) {
                    ku::trap();
                }
                // Issue Q (SW128) G->S
                {
                    Tensor gQ = tma_params.tma_Q_SW128.get_tma_tensor(tma_params.shape_Q_SW128)(_, _, s_q_idx, args.batch_idx);
                    Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), SmemLayoutQ_SW128{});
                    ku::launch_tma_copy(
                        tma_params.tma_Q_SW128,
                        gQ,
                        sQ,
                        plan.bar_q_tma,
                        TMA::CacheHintSm90::EVICT_FIRST
                    );
                }
                // Issue Q (SW64) G -> S
                if constexpr (D_Q_SW64 > 0) {
                    cute::SM90_TMA_LOAD_5D::copy(
                        &tma_params.tensor_map_q_sw64,
                        (uint64_t*)&plan.bar_q_tma,
                        (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                        plan.u.qo.q_sw64,
                        0, 0, 0,
                        s_q_idx, args.batch_idx
                    );
                }
                plan.bar_q_tma.arrive_and_expect_tx(B_H*D_Q*sizeof(bf16));
                plan.bar_q_tma.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();
                // Issue Q (SW128) UTCCP
                {
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(plan.u.qo.q.data()),
                            tile_to_shape(
                                UMMA::Layout_K_SW128_Atom<bf16>{},
                                Shape<Int<B_H*2>, Int<64>>{}
                            )
                        )
                    );
                    static_assert(D_Q_SW128%128 == 0);
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q_SW128/128; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 64/16; ++subtile_idx) {
                            SM100_UTCCP_128dp256bit_1cta::copy(
                                sQ_desc + (tile_idx*(B_H*128) + subtile_idx*16) * 2 / 16,
                                tmem_cols::Q + tile_idx*32 + subtile_idx*8
                            );
                        }
                    }
                }
                // Issue Q (SW64) UTCCP
                if constexpr (D_Q_SW64 > 0) {
                    UMMA::SmemDescriptor sQ_SW64_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(plan.u.qo.q_sw64),
                            tile_to_shape(
                                UMMA::Layout_K_SW64_Atom<bf16>{},
                                Shape<Int<B_H*2>, Int<32>>{}
                            )
                        )
                    );
                    static_assert(D_Q_SW64%64 == 0);
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q_SW64/64; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 32/16; ++subtile_idx) {
                            SM100_UTCCP_128dp256bit_1cta::copy(
                                sQ_SW64_desc + (tile_idx*(B_H*64) + subtile_idx*16) * 2 / 16,
                                tmem_cols::Q + (B_H*D_Q_SW128/2/128) + tile_idx*16 + subtile_idx*8
                            );
                        }
                    }
                }
                ku::umma_arrive_noelect(plan.bar_q_utccp);

                // Allocate tmem tensors
                TiledMMA tiled_mma_P = TiledMMA_P{};
                TiledMMA tiled_mma_O = TiledMMA_O{};
                Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
                Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
                tP.data().get() = tmem_cols::P;
                tO.data().get() = tmem_cols::O;

                // Wait for UTCCP
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                // Mainloop
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        // V3.2: RoPE first, then NoPE
                        plan.bar_rope_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();
                        Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_ROPE/2>>{})
                        );
                        tQ_rope.data().get() = tmem_cols::Q_Tail;
                        Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].rope.data()), SmemLayoutKTiles_DualGemm_SW64<2/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);

                        // QK NoPE
                        plan.bar_nope_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();
                        Tensor tQ_nope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_NOPE/2>>{})
                        );
                        tQ_nope.data().get() = tmem_cols::Q;
                        Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()), SmemLayoutKTiles_DualGemm_SW128<512/64/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ_nope, sK_nope, tP, false);
                    } else {
                        // MODEL1: NoPE and RoPE together
                        plan.bar_rope_ready[rs.buf_idx].wait(rs.bar_phase);
                        plan.bar_nope_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();

                        Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_Q/2>>{})
                        );
                        tQ.data().get() = tmem_cols::Q;
                        Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()), SmemLayoutKTiles_DualGemm_SW128<512/64/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx]);

                    // SV
                    plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].nope.data()), SmemLayoutKTilesTransposed_SW128<D_V/64>{});
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == args.start_block_idx);
                    ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);

                    rs.update();
                }
            });
        } else if (warp_idx == 5 && elect_one_sync()) {
            // BF16 NoPE retrieval warp - loads directly into dequant.nope with TMA swizzle
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase^1);
                    int4 cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int row = 0; row < B_TOPK; row += 4) {
                        if (row+4 < B_TOPK)
                            nxt_cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                        CUTE_UNROLL
                        for (int t = 0; t < D_NOPE/64; ++t) {
                            ku::tma_gather4(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_nope : &tma_params.tensor_map_kv_nope,
                                plan.bar_nope_ready[rs.buf_idx],
                                plan.u.kv.dequant[rs.buf_idx].nope.data() + 64*row + t*B_TOPK*64,
                                t*64,
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_nope_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK*D_NOPE*sizeof(bf16));
                    plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
                    rs.update();
                }
            });
        } else if (warp_idx == 6 && elect_one_sync()) {
            // KV RoPE retrieval warp - same as FP8 (RoPE is always BF16)
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    plan.bar_valid_coord_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase^1);
                    } else {
                        plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase^1);
                    }
                    int4 cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int row = 0; row < B_TOPK; row += 4) {
                        if (row+4 < B_TOPK)
                            nxt_cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                        CUTE_UNROLL
                        for (int t = 0; t < D_ROPE/(K_ROPE_SW/2); ++t) {
                            ku::tma_gather4(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_rope : &tma_params.tensor_map_kv_rope,
                                plan.bar_rope_ready[rs.buf_idx],
                                plan.u.kv.dequant[rs.buf_idx].rope.data() + (K_ROPE_SW/2)*row + t*B_TOPK*(K_ROPE_SW/2),
                                t*(K_ROPE_SW/2),
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_rope_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                    plan.bar_valid_coord_free[rs.index_buf_idx].arrive();
                    rs.update();
                }
            });
        } else if (warp_idx == 7) {
            // Indices transformation warp
            // BF16 mode: no scale loading needed
            static_assert(B_TOPK == 64);
            static constexpr int tma_coords_step_per_token = 1;  // BF16: each token is 1 TMA step
            int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE;
            int tma_coords_step_per_extra_block = params.stride_extra_kv_block / TMA_K_STRIDE;
            
            run_main_loop([&](const MainLoopArgs &args) {
                int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*s_q_idx;
                int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*s_q_idx;
                
                struct IsOrigBlock {};
                struct IsExtraBlock {};
                auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                    static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                    int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                    int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                    int abs_pos, my_indices[2];
                    if (!IS_EXTRA_BLOCK) {
                        abs_pos = block_idx*B_TOPK + lane_idx*2;
                        *(int2*)my_indices = __ldg((int2*)(indices + abs_pos));
                    } else {
                        abs_pos = (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2;
                        *(int2*)my_indices = __ldg((int2*)(extra_indices + abs_pos));
                    }
                    plan.bar_valid_coord_free[rs.index_buf_idx].wait(rs.index_bar_phase^1);

                    int tma_coords[2];
                    char valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        int block_idx_local, idx_in_block;
                        block_idx_local = (unsigned int)my_indices[i] / cur_block_size;
                        idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                        bool is_token_valid = my_indices[i] != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                        valid_mask |= is_token_valid << i;
                        tma_coords[i] = is_token_valid ? block_idx_local*cur_tma_coords_step_per_block + idx_in_block*tma_coords_step_per_token : -1;
                    }
                    valid_mask <<= lane_idx%4*2;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                    // No scale loading in BF16 mode
                    *(int2*)(plan.tma_coord[rs.index_buf_idx] + lane_idx*2) = *(int2*)tma_coords;
                    if (lane_idx%4 == 0)
                        plan.is_token_valid[rs.index_buf_idx][lane_idx/4] = valid_mask;
                    
                    plan.bar_valid_coord_ready[rs.index_buf_idx].arrive();
                    rs.update();
                };

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{});
                }

                CUTE_NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{});
                }
            });
        } else {
            run_main_loop([&](const MainLoopArgs &args) {});
        }
    } else {
        // Warpgroup 2: Idle in BF16 mode (NoPE loads directly with TMA swizzle, no dequant needed)
        cutlass::arch::warpgroup_reg_alloc<208>();

        run_main_loop([&](const MainLoopArgs &args) {});
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100 ~ sm119");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_bf16_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::flash_fwd_splitkv_mla_bf16_sparse_kernel_devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE>
void KernelTemplate<MODEL_TYPE>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.extra_topk % B_TOPK == 0, "extra_topk (%d) mod B_TOPK (%d) must be 0", params.extra_topk, B_TOPK);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);

    auto shape_Q_SW128 = make_shape(B_H, D_Q, params.s_q, params.b);
    auto tma_Q_SW128 = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_SW128,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQ_SW128{}
    );

    auto shape_O = make_shape(B_H, D_V, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.stride_o_h_q, _1{}, params.stride_o_s_q, params.stride_o_b)
            )
        ),
        SmemLayoutOBuf_TMA{}
    );

    CUtensorMap tensor_map_q_sw64{};
    if constexpr (D_Q_SW64 > 0) {
        tensor_map_q_sw64 = ku::make_tensor_map(
            {D_Q_SW64, (uint64_t)params.h_q, D_Q_SW64/32, (uint64_t)params.s_q, (uint64_t)params.b},
            ku::make_stride_helper(std::vector<int64_t>{params.stride_q_h_q, (int64_t)32, params.stride_q_s_q, params.stride_q_b}, sizeof(bf16)),
            {32, B_H, D_Q_SW64/32, 1, 1},
            (bf16*)params.q + D_Q_SW128,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
    }

    // BF16 mode: KV data is stored as contiguous BF16
    // NoPE: first D_NOPE bf16 values per token
    // RoPE: next D_ROPE bf16 values per token  
    auto get_nope_rope_tensormap = [&](bool is_extra, void* k_ptr, int num_blocks, int64_t k_batch_stride) -> std::pair<CUtensorMap, CUtensorMap> {
        static_assert(D_NOPE%8 == 0);
        KU_ASSERT((int64_t)k_ptr % 16 == 0, "The base address of %sk_ptr (%p) must be 16B aligned for sparse bf16 attention on sm100f", is_extra?"extra_":"", k_ptr);
        KU_ASSERT(k_batch_stride % TMA_K_STRIDE == 0, "%sk_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", k_batch_stride, TMA_K_STRIDE);
        
        // NoPE tensor map: load D_NOPE bf16 values per token with SW128 swizzle
        // Box: 64 bf16 elements = 128 bytes, matching SWIZZLE_128B
        CUtensorMap tensor_map_kv_nope = ku::make_tensor_map(
            {D_NOPE, (uint64_t)num_blocks * (k_batch_stride/TMA_K_STRIDE)},
            {TMA_K_STRIDE},
            {64, 1},
            k_ptr,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
        // RoPE tensor map: load D_ROPE bf16 values per token (offset by D_NOPE bf16s = D_NOPE*2 bytes)
        CUtensorMap tensor_map_kv_rope = ku::make_tensor_map(
            {D_ROPE, (uint64_t)num_blocks * (k_batch_stride/TMA_K_STRIDE)},
            {TMA_K_STRIDE},
            {K_ROPE_SW/2, 1},
            (uint8_t*)k_ptr + D_NOPE*2,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            K_ROPE_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
        return {tensor_map_kv_nope, tensor_map_kv_rope};
    };

    auto [tensor_map_kv_nope, tensor_map_kv_rope] = get_nope_rope_tensormap(false, params.kv, params.num_blocks, params.stride_kv_block);
    CUtensorMap tensor_map_extra_kv_nope{}, tensor_map_extra_kv_rope{};
    if (params.extra_topk > 0) {
        std::tie(tensor_map_extra_kv_nope, tensor_map_extra_kv_rope) = get_nope_rope_tensormap(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block);
    }

    TmaParams<
        decltype(shape_Q_SW128), decltype(tma_Q_SW128),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_SW128, tma_Q_SW128,
        shape_O, tma_O,
        tensor_map_q_sw64,
        tensor_map_kv_nope,
        tensor_map_kv_rope,
        tensor_map_extra_kv_nope,
        tensor_map_extra_kv_rope
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_bf16_sparse_kernel<KernelTemplate<MODEL_TYPE>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size < 227*1024);
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mla_kernel<<<dim3(params.s_q, params.num_sm_parts, 1), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_bf16_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE>::run(params);
}

}
