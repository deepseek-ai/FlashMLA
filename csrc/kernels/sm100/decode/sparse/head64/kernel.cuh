/*
Sparse MLA Decoding — SM100, h_q == 64

Decoding kernel for h_q == 64 heads. Uses FP8 KV cache with dequantization,
UTCMMA/TMEM architecture, and TMA for data transfers. Supports split-KV partitioning
for large topk values. Config defines per-model-type dimensions (V3.2: d_qk=576,
V4: d_qk=512, RoPE bf16, tile_size=64; V4.1: d_qk=512, RoPE fp8, tile_size=32).

Template parameter: Config (see kernel.h) — the formats of `kv` and `extra_kv` and whether split-KV is enabled

NUM_THREADS=384 (3 warpgroups)

I/O: See SparseAttnDecodeParams in kernels/params.h
*/
#include "kernel.h"

#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kerutils/kerutils.cuh"

#include "kernels/utils.h"
#include "kernels/defines.h"
#include "kernels/sm100/helpers.h"
#include "kernels/sm100/common_subroutine.h"

#include "config.h"

namespace sm100::decode::sparse::head64 {

template<Config CONFIG>
template<typename TmaParam>
__device__ void
KernelTemplate<CONFIG>
::flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params ) {
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
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_quant_part);
        if constexpr (D_BF16 > 0) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_bf16_part);
        }
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            for (int i = 0; i < NUM_BUFS; ++i) {
                if constexpr (D_BF16 > 0)
                    plan.bar_bf16_part_load_ready[i].init(1);
                plan.bar_quant_part_dequant_ready[i].init(128); 
                plan.bar_raw_ready[i].init(1);
                plan.bar_raw_free[i].init(128);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_valid_coord_scale_ready[i].init(32);
                plan.bar_valid_coord_scale_free[i].init(128+128+1+(D_BF16 > 0));
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
        bool bar_phase_batch_rel;    // Bar phase of barriers that are used once per batch
        int topk_length, extra_topk_length, num_orig_kv_blocks;
        bool is_last_batch;
    };

    auto run_main_loop = [&](auto f) {
        // NOTE Putting the following code outside the warpgroup specialization switch results in register spilling.
        // [[maybe_unused]] int begin_req_idx, end_req_idx, sched_begin_block_idx, sched_end_block_idx, begin_n_split_idx, is_first_req_splitted, is_last_req_splitted;
        DecodingSchedMeta sched_meta;
        if constexpr (ENABLE_SPLITKV) {
            KU_LDG_256(
                params.tile_scheduler_metadata_ptr + partition_idx,
                &sched_meta,
                ".nc",
                "no_allocate",
                "evict_normal",
                "256B"
            );
        } else {
            sched_meta.begin_req_idx = sched_meta.end_req_idx = partition_idx;
        }

        if (sched_meta.begin_req_idx >= params.b) {
            return;
        }
        
        bool bar_phase_batch_rel = 0;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx, bar_phase_batch_rel ^= 1) {
            int start_block_idx, end_block_idx;
            bool is_split; int n_split_idx;
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0
            if constexpr (ENABLE_SPLITKV) {
                start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
                is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);
            } else {
                start_block_idx = 0;
                end_block_idx = total_topk_padded / B_TOPK;
                is_split = false;
                n_split_idx = 0;
            }

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
        // The same technique (and highly similar code) as the sm100 sparse prefill head64 kernel
        cutlass::arch::warpgroup_reg_alloc<224>();

        constexpr int B_EPI = 64;   // Must be equal to the size of the swizzle atom
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_buf.data()), SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI/8];   // 64 is the size of the swizzle atom (in number of elements) while 8 is the width of each write
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup%B_H, (idx_in_warpgroup/B_H)*(PV_GEMM_N/(128/B_H)) + i*8);

        const float2 scale = float2 {params.sm_scale_div_log2, params.sm_scale_div_log2};
        bf16* sS_base = plan.s_p.s.data() + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);

        // NOTE Padding rows (idx%B_H >= h_q) must not read
        // attn_sink (OOB); their output is discarded anyway.
        float attn_sink = (params.attn_sink == nullptr || idx_in_warpgroup%B_H >= params.h_q)
            ? -CUDART_INF_F : __ldg((float*)params.attn_sink + (idx_in_warpgroup%B_H)) * CUDART_L2E_F;
        constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;
        
        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();

            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);  // Make sure all intermediate buffers (including p_exchange_buf, rowwise max_buf) are free
                plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);    // Put the barrier wait here for more code reordering space
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();


                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false
                >(
                    tmem_cols::P,
                    plan.is_token_valid[rs.index_buf_idx],
                    warp_idx, lane_idx,
                    [&]() {},
                    plan.s_p.p_exchange_buf,
                    p
                );
                
                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;

                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);    // This also separates "reading p_exchange_buf" and "writing S"
                plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
                // By this point:
                // - cur_pi_max, real_mi, and mi is identical within each row (i.e. thread 0+64, 1+65, ... for HEAD64)
                // - should_scale_o is identical among every warp, and is identical among threads that controls the same row

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
                __nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
                float2 neg_new_max = float2 {-new_max, -new_max};
                float2 cur_sum = float2 {0.0f, 0.0f};
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/2; i += 1) {
                    float2 d = ku::float2_fma(float2{p[i*2], p[i*2+1]}, scale, neg_new_max);
                    d.x = exp2f(d.x);
                    d.y = exp2f(d.y);
                    cur_sum = ku::float2_add(cur_sum, d);
                    s[i] = __float22bfloat162_rn(d);
                }
                li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

                // Write S
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                    *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
                }

                // Scale O
                if (block_idx != args.start_block_idx && should_scale_o) {
                    ku::tcgen05_after_thread_sync();

                    rescale_O<
                        D_V / (128/B_H),
                        64,
                        tmem_cols::O
                    >(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }
                
                fence_view_async_shared();
                plan.bar_so_ready[rs.buf_idx].arrive();

                if (block_idx != args.end_block_idx-1) {
                    rs.update();    // Don't update rs for the last round since we want to wait for the last SV gemm
                }

            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Exchange li
            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_max_buf[idx_in_warpgroup^64];

            // Store li
            if (idx_in_warpgroup < params.h_q) {
                if (args.is_no_split) {
                    float cur_lse = fma(mi, CUDART_LN2_F, logf(li));
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

            static constexpr int O_FOLD = 2;
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
                for (int i = 0; i < (D_V/O_FOLD) / B_EPI; ++i) {
                    // Load
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    // Scale & Convert
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j) {
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[j]);
                    }
                    // Store
                    int col_base = (i >= PV_GEMM_N/O_FOLD/B_EPI ? D_V/2 : 0) + (i*B_EPI%(PV_GEMM_N/O_FOLD));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 8; ++j)
                        *(__int128_t*)(sO_bases[j] + col_base*B_H) = *(__int128_t*)(&o_bf16[j*4]);
                    // Sync
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    // S -> G
                    if (warp_idx < O_FOLD && elect_one_sync()) {
                        int copy_block_idx = warp_idx*(PV_GEMM_N/2/B_EPI) + col_base/B_EPI;
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(tma_sO(_, _, copy_block_idx)),
                            thr_tma.partition_D(tma_gO(_, _, copy_block_idx))
                        );
                    }
                }
                cute::tma_store_arrive();
            } else {
                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);   // Here we leave attn_sink to the combine kernel, otherwise attn_sink will take effect for multiple times
                float2 o_scale_float2 = {o_scale, o_scale};
                constexpr int B_EPI = 64;
                float2 o[B_EPI/2];
                Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_accum_buf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int i = 0; i < (D_V/O_FOLD) / B_EPI; ++i) {
                    // Load
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    // Scale & Convert
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j)
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                    // Store
                    int col_base = (idx_in_warpgroup/B_H)*(PV_GEMM_N/O_FOLD) + (i >= PV_GEMM_N/O_FOLD/B_EPI ? D_V/2 : 0) + (i*B_EPI%(PV_GEMM_N/O_FOLD));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 4; ++j)
                        *(__int128_t*)&sO(idx_in_warpgroup%B_H, col_base + j*4) = *(__int128_t*)(&o[j*2]);
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                if (elect_one_sync()) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < B_H/4; ++local_row) {
                        int smem_row = local_row*4 + warp_idx;
                        if (smem_row < params.h_q) {    // Don't write padding rows (o_accum only has h_q rows)
                            SM90_BULK_COPY_S2G::copy(
                                &sO(smem_row, _0{}),
                                (float*)params.o_accum + args.n_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + smem_row*params.stride_o_accum_h_q,
                                D_V*sizeof(float)
                            );
                        }
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
        const int warp_idx = cutlass::canonical_warp_idx_sync();    // Missing this leads to reg spilling

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
                                Shape<Int<B_H*2>, Int<64>>{}  // *2 to leverage dual GEMM
                            )
                        )
                    );
                    static_assert(D_Q_SW128%128 == 0);
                    #pragma unroll D_Q_SW128/128
                    for (int tile_idx = 0; tile_idx < D_Q_SW128/128; ++tile_idx) {
                        // Each tile: 64 x (64*2) logically, 128 x 64 bf16 on TMEM
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < B_H/16; ++subtile_idx) {
                            // Each subtile: 64 x (16*2) logically, 128 x 16 bf16 (128dp256b) on TMEM
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
                                Shape<Int<B_H*2>, Int<32>>{}  // *2 to leverage dual GEMM
                            )
                        )
                    );
                    static_assert(D_Q_SW64%64 == 0);
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q_SW64/64; ++tile_idx) {
                        // Each tile: 64 x (32*2) logically, 128 x 32 bf16 on TMEM
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 32/16; ++subtile_idx) {
                            // Each subtile: 64 x (16*2) logically, 128 x 16 bf16 (128dp256b) on TMEM
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
                // NOTE These tXXX tensors are only for a forged layout (so that CuTe is able to generate correct address in cute::gemm)
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
                    if constexpr (OrigKVFormat::MODEL_TYPE == ModelType::V32) {
                        // V3.2: RoPE behaves like an extra block with size 64, so we can do RoPE first
                        // QK RoPE
                        plan.bar_bf16_part_load_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();
                        Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_ROPE/2>>{})
                        );
                        tQ_rope.data().get() = tmem_cols::Q_Tail;
                        Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].bf16_part), SmemLayoutKTiles_DualGemm_SW64<2/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);

                        // QK NoPE
                        plan.bar_quant_part_dequant_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();
                        Tensor tQ_nope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_NOPE/2>>{})
                        );
                        tQ_nope.data().get() = tmem_cols::Q;
                        Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].quant_part), SmemLayoutKTiles_DualGemm_SW128<512/64/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ_nope, sK_nope, tP, false);
                    } else {
                        // V4: RoPE is the last 64 dims within the full 512 dim, which couples with the last 64 dim from the NoPE part when performing dual GEMM. i.e.
                        // 
                        // logical view: |0|1|2|3|4|5|6|7| (where 7 is the RoPE part)
                        // dual gemm's view: 
                        // |0|2|4|6|
                        // |1|3|5|7|
                        // 
                        // So we must wait for both the NoPE and the RoPE part, and then perform dual GEMM
                        if constexpr (D_BF16 > 0)
                            plan.bar_bf16_part_load_ready[rs.buf_idx].wait(rs.bar_phase);
                        plan.bar_quant_part_dequant_ready[rs.buf_idx].wait(rs.bar_phase);
                        ku::tcgen05_after_thread_sync();

                        Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                            partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_Q/2>>{})
                        );
                        tQ.data().get() = tmem_cols::Q;
                        Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].quant_part), SmemLayoutKTiles_DualGemm_SW128<512/64/2>{});
                        ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx]);

                    // SV
                    plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.dequant[rs.buf_idx].quant_part), SmemLayoutKTilesTransposed_SW128<D_V/64>{});  // NOTE: For V4, it "expands" to the RoPE part.
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == args.start_block_idx);
                    ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);

                    rs.update();
                }
            });

        } else if (warp_idx == 5 && elect_one_sync()) {
            // Raw (quantized) KV retrieval warp
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                for_each_kv_block<OrigKVFormat, ExtraKVFormat>(args.start_block_idx, args.end_block_idx, args.num_orig_kv_blocks, [&]<typename F>(int block_idx, bool is_extra_block) {
                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_raw_free[rs.buf_idx].wait(rs.bar_phase^1);
                    const CUtensorMap *tensor_map = is_extra_block ? &tma_params.tensor_map_extra_kv_quant_part : &tma_params.tensor_map_kv_quant_part;
                    int4 cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int row = 0; row < B_TOPK; row += 4) {
                        if (row+4 < B_TOPK)
                            nxt_cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                        ku::tma_gather4(
                            tensor_map,
                            plan.bar_raw_ready[rs.buf_idx],
                            plan.u.kv.raw_quant[rs.buf_idx].data() + RAW_TOKEN_SMEM_STRIDE<F>*row,
                            0,
                            cur_indices,
                            (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                        );
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_raw_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK*RAW_TOKEN_SMEM_STRIDE<F>);
                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                    rs.update();
                });
            });
        } else if (warp_idx == 6 && elect_one_sync()) {
            // KV RoPE retrieval warp
            if constexpr (D_BF16 > 0) {
                run_main_loop([&](const MainLoopArgs &args) {
                    plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                    plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                        if constexpr (OrigKVFormat::MODEL_TYPE == ModelType::V32) {
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
                            for (int t = 0; t < D_BF16/(K_ROPE_SW/2); ++t) {
                                ku::tma_gather4(
                                    block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_bf16_part : &tma_params.tensor_map_kv_bf16_part,
                                    plan.bar_bf16_part_load_ready[rs.buf_idx],
                                    plan.u.kv.dequant[rs.buf_idx].bf16_part + (K_ROPE_SW/2)*row + t*B_TOPK*(K_ROPE_SW/2),
                                    t*(K_ROPE_SW/2),
                                    cur_indices,
                                    (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                );
                            }
                            cur_indices = nxt_cur_indices;
                        }
                        plan.bar_bf16_part_load_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK*D_BF16*sizeof(bf16));
                        plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                        rs.update();
                    }
                });
            }
        } else if (warp_idx == 7) {
            // Indices transformation warp
            // Responsible for generating: TMA coordinates, scale factors, and valid masks
            static_assert(B_TOPK == 64);
            static constexpr int tma_coords_step_per_token = 1;   // A token's data is exactly one TMA_K_STRIDE row, for all formats
            int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE; // must < 2G since k_batch_stride < 1T and TMA_K_STRIDE >= 512
            int tma_coords_step_per_extra_block = params.stride_extra_kv_block / ExtraKVFormat::TMA_K_STRIDE;
            uint8_t* k_scales_ptr =
                OrigKVFormat::MODEL_TYPE == ModelType::V32 ?
                (uint8_t*)params.kv + D_NOPE :
                (uint8_t*)params.kv + params.page_block_size*TMA_K_STRIDE;
            uint8_t* extra_k_scales_ptr =
                OrigKVFormat::MODEL_TYPE == ModelType::V32 ?
                (uint8_t*)params.extra_kv + D_NOPE :
                (uint8_t*)params.extra_kv + params.extra_page_block_size*ExtraKVFormat::TMA_K_STRIDE;
            
            run_main_loop([&](const MainLoopArgs &args) {
                int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*s_q_idx;
                int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*s_q_idx;
                
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
                    static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                    using F = std::conditional_t<IS_EXTRA_BLOCK, ExtraKVFormat, OrigKVFormat>;
                    int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                    int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
                    [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
                    uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                    int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                    int abs_pos = IS_EXTRA_BLOCK ? (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2 : block_idx*B_TOPK + lane_idx*2;

                    // Issue the div/mod and the scale LDGs before waiting for the index buffer, so their
                    // latency overlaps with the (usually long) empty-barrier wait.
                    int block_idx_arr[2], idx_in_block_arr[2];
                    alignas(16) uint8_t scales[2][SCALE_SMEM_STRIDE];
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                        int kv_block_idx = (unsigned int)cur_idx / cur_block_size;
                        int idx_in_block = (unsigned int)cur_idx % cur_block_size;
                        block_idx_arr[i] = kv_block_idx;
                        idx_in_block_arr[i] = idx_in_block;
                        bool is_token_valid = cur_idx != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                        if constexpr (F::IS_V32) {
                            int64_t offset = is_token_valid ? kv_block_idx*cur_k_block_stride + idx_in_block*cur_k_row_stride : 0;
                            float4 cur_scale_fp32 = __ldg((float4*)(cur_k_scales_ptr + offset));
                            e8m0 res[4];
                            *(__nv_fp8x2_storage_t*)(res+0) = __nv_cvt_float2_to_e8m0x2(float2{cur_scale_fp32.x, cur_scale_fp32.y}, __NV_NOSAT, cudaRoundZero);
                            *(__nv_fp8x2_storage_t*)(res+2) = __nv_cvt_float2_to_e8m0x2(float2{cur_scale_fp32.z, cur_scale_fp32.w}, __NV_NOSAT, cudaRoundZero);
                            if (!is_token_valid) *(uint32_t*)res = (uint32_t)0;
                            *(uint32_t*)scales[i] = *(uint32_t*)(res);
                        } else {
                            // Each token's row of 1 B scales (padded to whole words, e.g. 7 + 1 B for V4)
                            int64_t offset = kv_block_idx*cur_k_block_stride + idx_in_block*F::NUM_SCALES_EACH_TOKEN;
                            ldg_or_zero<F::NUM_SCALES_EACH_TOKEN>(scales[i], cur_k_scales_ptr + offset, is_token_valid);
                        }
                    }
                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].wait(rs.index_bar_phase^1);

                    int tma_coords[2];
                    char valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < 2; ++i) {
                        int cur_idx = i == 0 ? my_indices.x : my_indices.y;
                        bool is_token_valid = cur_idx != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                        valid_mask |= is_token_valid << i;
                        tma_coords[i] = is_token_valid ? block_idx_arr[i]*cur_tma_coords_step_per_block + idx_in_block_arr[i]*tma_coords_step_per_token : -1; // If the token is invalid because it topk position exceeds topk_length, we must manually fill tma_coords with -1 to avoid copying-in NaN.
                    }
                    valid_mask <<= lane_idx%4*2;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                    if constexpr (SCALE_SMEM_STRIDE == F::NUM_SCALES_EACH_TOKEN) {
                        // The lane's 2 tokens are contiguous in smem: one store
                        copy_bytes<2*SCALE_SMEM_STRIDE>(plan.scales[rs.index_buf_idx][lane_idx*2], scales[0]);
                    } else {
                        // Mixed-format kernel, fp8 tokens: each token uses the first bytes of its 32 B row
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i)
                            copy_bytes<F::NUM_SCALES_EACH_TOKEN>(plan.scales[rs.index_buf_idx][lane_idx*2 + i], scales[i]);
                    }
                    *(int2*)(plan.tma_coord[rs.index_buf_idx] + lane_idx*2) = *(int2*)tma_coords;
                    if (lane_idx%4 == 0)
                        plan.is_token_valid[rs.index_buf_idx][lane_idx/4] = valid_mask;
                    
                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].arrive();
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
        } else {
            run_main_loop([&](const MainLoopArgs &args) {});
        }
    } else {
        // Dequant warpgroup
        cutlass::arch::warpgroup_reg_alloc<208>();

        // Turns the raw (fp8 / fp4) rows of each KV block into the bf16 K tile, see KVBlockDequantizer. The format of a block is
        // that of the cache it comes from, resolved at compile time by for_each_kv_block
        static_assert(NUM_BUFS == 2);
        const Dequantizer<OrigKVFormat> dequant_orig(idx_in_warpgroup);
        const Dequantizer<ExtraKVFormat> dequant_extra(idx_in_warpgroup);   // Identical to dequant_orig unless the extra cache is fp4
        const uint8_t *raw_0 = plan.u.kv.raw_quant[0].data(), *raw_1 = plan.u.kv.raw_quant[1].data();
        const uint32_t dst_0 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[0].quant_part);
        const uint32_t dst_1 = cute::cast_smem_ptr_to_uint(plan.u.kv.dequant[1].quant_part);

        run_main_loop([&](const MainLoopArgs &args) {
            // plan.bar_last_store_done.wait(args.bar_phase_batch_rel); // No need to wait since the raw KV producer must wait
            plan.bar_q_utccp.wait(args.bar_phase_batch_rel);

            for_each_kv_block<OrigKVFormat, ExtraKVFormat>(args.start_block_idx, args.end_block_idx, args.num_orig_kv_blocks, [&]<typename F>(int, bool) {
                plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                plan.bar_raw_ready[rs.buf_idx].wait(rs.bar_phase);
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase^1);
                const auto *dequant = [&] {
                    if constexpr (std::is_same_v<F, OrigKVFormat>) return &dequant_orig; else return &dequant_extra;
                }();
                dequant->run(
                    rs.buf_idx == 0 ? raw_0 : raw_1,
                    plan.scales[rs.index_buf_idx][0],
                    rs.buf_idx == 0 ? dst_0 : dst_1,
                    []{}
                );
                cutlass::arch::fence_view_async_shared();
                plan.bar_quant_part_dequant_ready[rs.buf_idx].arrive();
                plan.bar_raw_free[rs.buf_idx].arrive();
                plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                rs.update();
            });
        });
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100 ~ sm119");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TmaParams tma_params ) {
    Kernel::flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(params, tma_params );
}

template<Config CONFIG>
void KernelTemplate<CONFIG>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.extra_topk % B_TOPK == 0, "extra_topk (%d) mod B_TOPK (%d) must be 0", params.extra_topk, B_TOPK);
    KU_ASSERT(params.h_q == 64);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.model_type == OrigKVFormat::MODEL_TYPE && params.extra_model_type == ExtraKVFormat::MODEL_TYPE);

    // NOTE The head dim of the gmem shape is h_q while the smem box covers
    // B_H rows, so TMA zero-fills rows [h_q, B_H) on load (Q), and drops them on
    // store (O). Same for tensor_map_q_sw64 below.
    auto shape_Q_SW128 = make_shape(params.h_q, D_Q, params.s_q, params.b);
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

    auto shape_O = make_shape(params.h_q, D_V, params.s_q, params.b);
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

    CUtensorMap tensor_map_kv_quant_part = make_kv_quant_part_tensor_map<OrigKVFormat, OrigKVFormat::QUANT_BYTES, RAW_TOKEN_SMEM_STRIDE<OrigKVFormat>>(
        "k_cache", params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row, 0);
    CUtensorMap tensor_map_kv_bf16_part{};
    if constexpr (D_BF16 > 0) {
        tensor_map_kv_bf16_part = make_kv_bf16_part_tensor_map<OrigKVFormat, K_ROPE_SW/2, K_ROPE_SW>(params.kv, params.num_blocks, params.stride_kv_block);
    }
    CUtensorMap tensor_map_extra_kv_quant_part{}, tensor_map_extra_kv_bf16_part{};
    if (params.extra_topk > 0) {
        tensor_map_extra_kv_quant_part = make_kv_quant_part_tensor_map<ExtraKVFormat, ExtraKVFormat::QUANT_BYTES, RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat>>(
            "extra_k_cache", params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row, 0);
        if constexpr (ExtraKVFormat::D_BF16 > 0) {
            tensor_map_extra_kv_bf16_part = make_kv_bf16_part_tensor_map<ExtraKVFormat, K_ROPE_SW/2, K_ROPE_SW>(params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block);
        }
    }

    TmaParams<
        decltype(shape_Q_SW128), decltype(tma_Q_SW128),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q_SW128, tma_Q_SW128,
        shape_O, tma_O,
        tensor_map_q_sw64,
        tensor_map_kv_quant_part,
        tensor_map_kv_bf16_part,
        tensor_map_extra_kv_quant_part,
        tensor_map_extra_kv_bf16_part
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<CONFIG>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 227*1024);
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    
    // NOTE Don't use PDL because of potential compiler bugs!
    mla_kernel<<<dim3(params.s_q, ENABLE_SPLITKV ? params.num_sm_parts : params.b, 1), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params, tma_params );
    KU_CHECK_KERNEL_LAUNCH();

}

template<Config CONFIG>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<CONFIG>::run(params);
}

}
