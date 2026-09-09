#pragma once

#include <cuda_fp8.h>
#include <math_constants.h>   // CUDART_INF_F
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/utils.h"
#include "kernels/sm100/helpers.h"
#include "kernels/sm100/common_subroutine.h"

#include "config.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

static constexpr float MAX_INIT_VAL = -1e30;
static constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4;

#ifdef KERUTILS_ENABLE_SM103A
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = true;
#else
static constexpr bool IS_TMEM_LD_WITH_RED_AVAILABLE = false;
#endif

__device__ __forceinline__
float2 apply_rope(const float2 &x, const float &cur_cos, const float &cur_sin) {
    float2 a = {x.x, x.x};
    float2 b = {cur_cos, cur_sin};
    float2 c = {-x.y * cur_sin, +x.y * cur_cos};
    float2 y = ku::float2_fma(a, b, c);
    return y;
}

// To achieve prefill - decoding alignment while using block_size = 96, decoding must reproduce
// prefill's "natural" blocking of the concatenated indices array ([topk orig slots; extra_topk
// extra slots], tiled by B_TOPK). Since topk (e.g. 128) may not be a multiple of B_TOPK (e.g. 96),
// one KV block may straddle the orig/extra boundary, i.e. the last (partial) block of the orig KV
// and the first tokens of the extra KV are "stitched" into one block. Each KV block is thus
// classified into one of the following three categories:
enum class KVLocation {
    ORIG,           // All slots of this block come from `kv`/`indices`
    ORIG_AND_EXTRA, // This block straddles the boundary: slots < num_orig_slots come from `kv`/`indices`, the rest from `extra_kv`/`extra_indices`
    EXTRA           // All slots of this block come from `extra_kv`/`extra_indices`
};

template<Config CONFIG>
__device__ __forceinline__
void Kernel<CONFIG>::devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params) {
#ifdef KERUTILS_ENABLE_SM100A
    const uint32_t cta_idx = IS_2CTA ? blockIdx.x % 2 : 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t idx_in_warpgroup = threadIdx.x % 128;
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    }

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        if constexpr (IS_DECODE) {
            if constexpr (D_BF16 > 0) {
                cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_bf16_part);
            }
        } else {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        // Init barriers
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
            // bar_kv_slot_full:
            //   Prefill: 1 arrive (arrive_and_expect_tx from the MMA warp) + TMA transactions of the whole KV block
            //   Decode: 128 arrives (the dequant warpgroup) from each CTA + 1 arrive (arrive_and_expect_tx from the MMA warp) from CTA0 + TMA transactions of the bf16 (RoPE) part
            smem.bar_kv_slot_full[i].init(IS_DECODE ? 128*CLUSTER_SIZE + 1 : 1);  // bar_kv_full: Every CTA -> CTA0
            smem.bar_kv_slot_empty[i].init(1); // bar_kv_empty: CTA0 -> Every CTA
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
            smem.bar_indices_full[i].init(32);  // CTA-local
            smem.bar_indices_empty[i].init(IS_DECODE ? 256 : 128);    // CTA-local
        }
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
            smem.bar_tP_full[i].init(1);       // CTA0 -> Every CTA
            if constexpr (NEED_TP_EMPTY_BAR) {
                smem.bar_tP_empty[i].init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
            }
        }
        if constexpr (IS_2CTA && ENABLE_Q_NORM) {
            smem.bar_q_sqr_sum_full.init(128); // CTA-local
        }
        smem.bar_clc_full.init(1);      // CTA0 -> Every CTA
        smem.bar_clc_empty.init(cta_idx == 1 ? 1 : NUM_WORKING_THREADS);   // Every CTA -> CTA0
        smem.bar_tQ_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_tQ_empty.init(1+128);      // CTA0 -> Every CTA (arrive by MMA thread), as well as CTA-local (arrived by exp warpgroup)
        smem.bar_tO_full.init(1);       // CTA0 -> Every CTA
        smem.bar_tO_empty.init(128*CLUSTER_SIZE);   // Every CTA -> CTA0
        smem.bar_SO_full.init(128*CLUSTER_SIZE);    // Every CTA -> CTA0
        smem.bar_SO_empty.init(1);      // CTA0 -> Every CTA
        smem.bar_li_mi_full.init(128);  // CTA-local
        smem.bar_li_mi_empty.init(128); // CTA-local
        if constexpr (IS_DECODE) {
            smem.bar_raw_kv_full.init(1);   // CTA-local
        }
        fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate TMEM
        AllocatorT().allocate(512, smem.tmem_start_addr.data());
        AllocatorT().release_allocation_lock();
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
    }

    if constexpr (IS_2CTA) {
        ku::barrier_cluster_arrive_relaxed();
        ku::barrier_cluster_wait_acquire();
    } else {
        __syncthreads();
    }

    struct OuterloopArgs {
        bool is_valid;
        uint32_t s_q_idx;
        uint32_t job_idx_mod_2;
        uint32_t topk_length;
        uint32_t num_kv_blocks;
        // Decoding only:
        uint32_t extra_topk_length;     // Number of valid extra-topk entries of the current request
        uint32_t num_orig_slots;        // Number of slots occupied by the orig KV in the unified slot space, i.e. slots < num_orig_slots come from `kv`/`indices` (will be set to -1 if we don't have so many valid indices) while the others come from `extra_kv`/`extra_indices`. 0xFFFFFFFF when there is no extra KV (so that every slot belongs to the orig KV)
    };

    auto _make_outer_loop_args = [&](uint32_t job_idx_mod_2, uint32_t cta_x_idx) -> OuterloopArgs {
        uint32_t s_q_idx = cta_x_idx / CLUSTER_SIZE;
        if constexpr (IS_DECODE) {
            uint32_t topk_length = params.topk_length ? (uint32_t)__ldg(params.topk_length + s_q_idx) : (uint32_t)params.topk;
            uint32_t extra_topk_length = params.extra_topk_length ? (uint32_t)__ldg(params.extra_topk_length + s_q_idx) : (uint32_t)params.extra_topk;
            bool have_extra_kv = params.extra_topk > 0;
            uint32_t num_orig_slots, num_kv_blocks;
            if (have_extra_kv) {
                num_orig_slots = (uint32_t)params.topk;
                num_kv_blocks = ku::ceil_div(num_orig_slots + extra_topk_length, (uint32_t)B_TOPK); // When extra_kv is present, always round topk up to the full cycle
            } else {
                num_orig_slots = 0xFFFFFFFFu;
                num_kv_blocks = ku::ceil_div(topk_length, (uint32_t)B_TOPK);
            }
            num_kv_blocks = std::max(num_kv_blocks, 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks,
                extra_topk_length,
                num_orig_slots
            };
        } else {
            uint32_t topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
            uint32_t num_kv_blocks = std::max(ku::ceil_div(topk_length, (uint32_t)B_TOPK), 1u);
            return {
                true,
                s_q_idx,
                job_idx_mod_2,
                topk_length,
                num_kv_blocks
            };
        }
    };

    // A handy function (decode only) to run along all KV blocks in the unified slot space.
    // Should be provided with a template function, which will be invoked as callable<LOC>(kv_block_idx),
    // where LOC (a KVLocation) tells where the tokens of the current block come from. Only the (at most
    // one) block straddling the orig/extra boundary is invoked with ORIG_AND_EXTRA, so that ORIG-only
    // and EXTRA-only blocks stay on branch-free fast paths
    auto run_along_kv_blocks = [&](const OuterloopArgs &cur_args, auto callable) {
        uint32_t num_full_orig_blocks = std::min(cur_args.num_kv_blocks, cur_args.num_orig_slots / B_TOPK);
        bool has_mixed_block = num_full_orig_blocks < cur_args.num_kv_blocks && cur_args.num_orig_slots % B_TOPK != 0;
        CUTE_NO_UNROLL
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_full_orig_blocks; ++kv_block_idx) {
            callable.template operator()<KVLocation::ORIG>(kv_block_idx);
        }
        if (has_mixed_block) {
            callable.template operator()<KVLocation::ORIG_AND_EXTRA>(num_full_orig_blocks);
        }
        CUTE_NO_UNROLL
        for (uint32_t kv_block_idx = num_full_orig_blocks + has_mixed_block; kv_block_idx < cur_args.num_kv_blocks; ++kv_block_idx) {
            callable.template operator()<KVLocation::EXTRA>(kv_block_idx);
        }
    };

    auto get_first_job = [&]() -> OuterloopArgs {
        return _make_outer_loop_args(0, blockIdx.x);
    };
    auto get_next_job = [&](const OuterloopArgs &cur_args) -> OuterloopArgs {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx = ku::get_clc_query_response<true>(smem.clc_response_obj);
        arrive_on_cta0_barrier(smem.bar_clc_empty);

        if (!next_cta0_idx.is_valid) {
            return OuterloopArgs {false};
        } else {
            return _make_outer_loop_args(
                cur_args.job_idx_mod_2^1,
                next_cta0_idx.x
            );
        }
    };

    if (warpgroup_idx == 0) {
        /*
        Q fetching & Epilogue warpgroup

        The timeline of this warpgroup is as follows:
        Q0 Q1 O0 Q2 O1 Q3 O2 Q4 O3 ... Qn O(n-1) On

        Where
        - Qi means loading the Q of the i-th request, computing the sum of squares of each head on the fly,
          performing RoPE transformation, then writing Q to TMEM
        - Oi means reading the O of the i-th request from TMEM, performing RoPE transformation,
          quantizing to FP8, and writing back to global memory

        About cached_cos and cached_sin:
        - D_ROPE is always 64, so the i-th lane caches the i-th cos and sin dimension of the
          corresponding position
        - We always cache the cos and sin of the i-th request and the (i-1)-th request, so cached_cos
          and cached_sin have two slots
        - Every time a new Q is loaded, cached_cos/sin[1] (the first slot) is assigned to the zeroth slot
          (and "sin" is negated to prepare for the conjugate RoPE of O),
          and the cos/sin of the new Q is saved to the 1st slot
        - This way, O RoPE only needs to read from the 0th slot (except for the last O)
        */
        cutlass::arch::warpgroup_reg_alloc<184>();

        #pragma nv_diag_suppress 549    // Uninitialized variable. The following two variables are indeed initialized, but the compiler cannot prove it
        float cached_cos[2], cached_sin[2];
        auto shift_cached_cos_and_cached_sin = [&]() {
            cached_cos[0] = cached_cos[1];
            cached_sin[0] = -cached_sin[1];
        };

        auto load_q_and_save_to_tmem = [&](const OuterloopArgs &cur_job) {
            static constexpr uint32_t NUM_CACHED_BF16_PER_THREAD = H_Q_PER_CTA * D_QK / 128;
            static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16; // LDG 256
            bf16 cached_q[NUM_CACHED_BF16_PER_THREAD];
            float q_sqr_sum = 0.0f;

            uint32_t cur_q_position = __ldg(params.token_positions + cur_job.s_q_idx);
            cached_cos[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + lane_idx);
            cached_sin[1] = __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + D_ROPE / 2 + lane_idx);

            bf16 *q_token_base = params.q + (uint64_t)cur_job.s_q_idx * params.stride_q_s_q;

            static constexpr uint32_t TILE_SIZE = 64;
            CUTE_UNROLL
            for (uint32_t local_tile_idx = 3; local_tile_idx != 0xFFFFFFFF; --local_tile_idx) {
                uint32_t tile_idx =
                    CLUSTER_SIZE == 1 ?
                    local_tile_idx * 2 + (warp_idx / (H_Q_PER_CTA / 32)) :   // Don't use idx_in_warpgroup / H_Q_PER_CTA to hint the compiler that warps does not diverge here
                    (warp_idx / (H_Q_PER_CTA / 32)) * 4 + local_tile_idx;
                CUTE_UNROLL
                for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                    static_assert(NUM_MRGEMM_RAILS == 2);
                    uint32_t h_q_idx = cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA;
                    uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;
                    KU_LDG_256(
                        q_token_base + h_q_idx * NUM_BF16_PER_LOAD + d_q_idx * H_Q,
                        cached_q + local_tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD,
                        ".nc", "no_allocate", "evict_first", "256B"
                    );
                }
                // Perform RoPE
                if (local_tile_idx == 3 && tile_idx == D_VO / TILE_SIZE - 1) {
                    float2 cur_q_sqr_sum = {0.0f, 0.0f};
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                        float2 x = __bfloat1622float2(*(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j));
                        if constexpr (ENABLE_Q_NORM) {
                            cur_q_sqr_sum = ku::float2_fma(x, x, cur_q_sqr_sum);
                        }
                        float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[1], j/2);
                        float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[1], j/2);
                        float2 y = apply_rope(x, cur_cos, cur_sin);
                        *(nv_bfloat162*)(cached_q+local_tile_idx*TILE_SIZE+j) = nv_bfloat162{__float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)};
                    }
                    q_sqr_sum += cur_q_sqr_sum.x + cur_q_sqr_sum.y;
                } else {
                    if constexpr (ENABLE_Q_NORM) {
                        // Accumulate \sum q_i^2 for RMS norm
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < TILE_SIZE; ++i)
                            asm volatile ("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(q_sqr_sum) : "h"(*(uint16_t*)(cached_q+local_tile_idx*TILE_SIZE+i)));
                    }
                }
            }

            if constexpr (ENABLE_Q_NORM) {
                smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] = q_sqr_sum;
            }

            smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2^1);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t NUM_CACHED_UINT32 = NUM_CACHED_BF16_PER_THREAD/2;
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q, cached_q);
            ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32/2>(tmem_cols::Q+NUM_CACHED_UINT32/2, cached_q+NUM_CACHED_UINT32/2*2);   // We split the tmem_st into two parts, otherwise NVCC complains about insufficient registers. I suspect this is because PTXAS ignores the warpgroup_reg_alloc<168> above and uses 128 as the available register count per thread (with 512 total threads, each thread initially has only 128 registers)
            cutlass::arch::fence_view_async_tmem_store();

            ku::tcgen05_before_thread_sync();
            arrive_on_cta0_barrier(smem.bar_tQ_full);
            if constexpr (IS_2CTA && ENABLE_Q_NORM) {
                smem.bar_q_sqr_sum_full.arrive();
            }
        }; 
        auto store_o = [&](const OuterloopArgs &cur_job, const bool &is_last_job) { 
            smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
            float li = 0.0f;
            float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];
            if constexpr (FOLD_FACTOR == 2) {
                li = smem.rowwise_li_buf[idx_in_warpgroup] + smem.rowwise_li_buf[idx_in_warpgroup^64];
            } else {
                static_assert(FOLD_FACTOR == 4);
                li = __fadd_rn(
                    __fadd_rn(smem.rowwise_li_buf[idx_in_warpgroup], smem.rowwise_li_buf[idx_in_warpgroup^64]),
                    __fadd_rn(smem.rowwise_li_buf[idx_in_warpgroup^32], smem.rowwise_li_buf[idx_in_warpgroup^96])
                );
            }
            smem.bar_li_mi_empty.arrive();

            if (idx_in_warpgroup < H_Q_PER_CTA) {
                uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                params.lse[global_index] = cur_lse;
            }

            float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx * H_Q_PER_CTA + idx_in_warpgroup % H_Q_PER_CTA) * CUDART_L2E_F;
            float output_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

            smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
            ku::tcgen05_after_thread_sync();
            
            static constexpr uint32_t MMA_ATOM_N = 256;
            static constexpr uint32_t NUM_MMA_ATOMS = D_VO / MMA_ATOM_N;
            static constexpr uint32_t NUM_O_TMEM_COLS_PER_ATOM = MMA_ATOM_N / FOLD_FACTOR;
            static constexpr uint32_t EPILOGUE_TILE_SIZE = O_QUANT_TILE_SIZE;
            static constexpr uint32_t NUM_EPILOGUE_TILES_PER_ATOM = NUM_O_TMEM_COLS_PER_ATOM / EPILOGUE_TILE_SIZE;
            static_assert(NUM_O_TMEM_COLS_PER_ATOM % EPILOGUE_TILE_SIZE == 0);  // TODO When FOLD_FACTOR is 4 and MODEL_TYPE is V4 (so NUM_O_TMEM_COLS_PER_ATOM is 128), this isn't hold
            fp8_e4m3 output_fp8[NUM_MMA_ATOMS][NUM_O_TMEM_COLS_PER_ATOM];
            uint8_t output_sf[NUM_MMA_ATOMS][NUM_O_TMEM_COLS_PER_ATOM / O_QUANT_TILE_SIZE];

            CUTE_UNROLL
            for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; mma_atom_idx += 1) {
                CUTE_UNROLL
                for (uint32_t epilogue_tile_idx_in_atom = 0; epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM; ++epilogue_tile_idx_in_atom) {
                    // Fetch output from TMEM
                    uint32_t tmem_col_base = tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                    float output[EPILOGUE_TILE_SIZE];
                    float reduce_result_by_tmem_ld;
                    if constexpr (IS_TMEM_LD_WITH_RED_AVAILABLE) {
                        ku::tmem_ld_red_32dp32bNx<EPILOGUE_TILE_SIZE, true, true, true>(tmem_col_base, output, reduce_result_by_tmem_ld);
                    } else {
                        ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(tmem_col_base, output);
                    }
                    cutlass::arch::fence_view_async_tmem_load();

                    // Notify tO's emptyness
                    if (mma_atom_idx+1 == NUM_MMA_ATOMS && epilogue_tile_idx_in_atom+1 == NUM_EPILOGUE_TILES_PER_ATOM) {
                        ku::tcgen05_before_thread_sync();
                        if (!is_last_job) {
                            // Don't arrive on the barrier if this job is the last job, to avoid "cluster target block not present"
                            arrive_on_cta0_barrier(smem.bar_tO_empty);
                        }
                    }

                    // RoPE (conjugate)
                    bool should_perform_rope;
                    {
                        static_assert(FOLD_FACTOR == 2);
                        static_assert(D_ROPE % EPILOGUE_TILE_SIZE == 0);
                        should_perform_rope = 
                            mma_atom_idx + 1 == NUM_MMA_ATOMS && 
                            epilogue_tile_idx_in_atom >= NUM_EPILOGUE_TILES_PER_ATOM - D_ROPE/EPILOGUE_TILE_SIZE && 
                            warp_idx >= 2;
                        if (should_perform_rope) {
                            CUTE_UNROLL
                            for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                                float2 x = *(float2*)(output + j);
                                uint32_t src_lane = j/2 + (epilogue_tile_idx_in_atom + 1 == NUM_EPILOGUE_TILES_PER_ATOM ? EPILOGUE_TILE_SIZE / 2 : 0);
                                float cur_cos = __shfl_sync(0xFFFFFFFF, cached_cos[0], src_lane);
                                float cur_sin = __shfl_sync(0xFFFFFFFF, cached_sin[0], src_lane);
                                float2 y = apply_rope(x, cur_cos, cur_sin);
                                *(float2*)(output + j) = y;
                            }
                        }
                    }

                    // Cast to FP8, and save to global memory
                    float output_abs_max;
                    if (!IS_TMEM_LD_WITH_RED_AVAILABLE || should_perform_rope) {
                        output_abs_max = get_max<EPILOGUE_TILE_SIZE, true>(output) * output_scale;
                    } else {
                        output_abs_max = reduce_result_by_tmem_ld * output_scale;
                    }
                    output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);
                    float sf = output_abs_max / 448.0f;
                    uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t*>(&sf);
                    uint32_t exp_sf = (int32_t)((sf_as_uint32-1) >> 23) + (1 - 127);
                    uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
                    float sf_inv = *reinterpret_cast<float*>(&sf_inv_as_uint32);
                    float cur_multiplier = output_scale * sf_inv;
                    float2 cur_multiplier_float2 = float2(cur_multiplier, cur_multiplier);

                    CUTE_UNROLL
                    for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                        float2 x = *(float2*)(output + j);
                        x = ku::float2_mul(x, cur_multiplier_float2);
                        *(__nv_fp8x2_storage_t*)(output_fp8[mma_atom_idx] + epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE + j) = __nv_cvt_float2_to_fp8x2(
                            x,
                            __NV_SATFINITE,
                            __nv_fp8_interpretation_t::__NV_E4M3
                        );  // NOTE. Here we don't use cvt.f8x4type.f32 since it only has .rs mode, which affects accuracy
                    }
                    output_sf[mma_atom_idx][epilogue_tile_idx_in_atom] = exp_sf + 127;
                }
            }

            uint32_t head_idx = cta_idx*H_Q_PER_CTA + idx_in_warpgroup%H_Q_PER_CTA;
            uint32_t wv_group_idx = head_idx / WV_GROUP_SIZE;
            uint32_t head_idx_in_wv_group = head_idx % WV_GROUP_SIZE;
            CUTE_UNROLL
            for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS; mma_atom_idx += 1) {
                // Store SF
                // Since the weight is per-32 scaled and deep_gemm.einsum requires A and B to have the same scale granularity, the output sf is always stored in a per-32 scaled format, although it will be actually (numerically) per-128 scaled when num_per_channels is 128
                static constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;
                // Layout of O in Tensor Memory:
                // For HEAD_DIM_QK = 64 (head64):
                //   - Atom 0 computes O[:, 0:256]; Atom 1 computes O[:, 256:512]
                //   - Mapping to TMEM:
                //       O[0:128]   -> TMEM[0:64,   0:128]
                //       O[128:256] -> TMEM[64:128,  0:128]
                //       O[256:384] -> TMEM[0:64,   128:256]
                //       O[384:512] -> TMEM[64:128, 128:256]
                //   - Visually (label the four 128-wide O chunks as 0..3):
                //        +---+---+
                //        | 0 | 2 |
                //        +---+---+
                //        | 1 | 3 |
                //        +---+---+
                // For HEAD_DIM_QK = 128 (head128):
                //   - CTA0 holds V[:, 0:256]; CTA1 holds V[:, 256:512]
                //   - Atom 0 computes O[:, 0:128] and O[:, 256:384]
                //     Atom 1 computes O[:, 128:256] and O[:, 384:512]
                //   - Mapping to TMEM:
                //       O[0:128]   -> TMEM[0:64,   0:128]
                //       O[128:256] -> TMEM[0:64,   128:256]
                //       O[256:384] -> TMEM[64:128, 0:128]
                //       O[384:512] -> TMEM[64:128, 128:256]
                //   - Visually (label the four 128-wide O chunks as 0..3):
                //        +---+---+
                //        | 0 | 1 |
                //        +---+---+
                //        | 2 | 3 |
                //        +---+---+
                uint32_t head_dim_idx_base = 
                    CLUSTER_SIZE == 1 ?
                    mma_atom_idx * MMA_ATOM_N + (warp_idx/(H_Q_PER_CTA/32)) * (MMA_ATOM_N/FOLD_FACTOR) :
                    mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM + (warp_idx/(H_Q_PER_CTA/32)) * MMA_ATOM_N;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM / OUTPUT_SAVE_AS_SCALE_GRAN; ++i) {
                    uint32_t sf_block_idx = head_idx_in_wv_group + (head_dim_idx_base/OUTPUT_SAVE_AS_SCALE_GRAN+i) * WV_GROUP_SIZE;
                    *((uint8_t*)(params.out_sf + cur_job.s_q_idx + wv_group_idx*params.stride_out_sf_wv_group + (sf_block_idx/4)*params.stride_out_sf_head_dim) + sf_block_idx%4) = output_sf[mma_atom_idx][i];  // TODO Optimize
                }
                // Store output
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM; i += 32) {
                    uint32_t head_dim_idx = head_dim_idx_base + i;
                    KU_STG_256(
                        params.out_fp8 + (uint64_t)cur_job.s_q_idx*(H_Q*D_VO) + wv_group_idx*(WV_GROUP_SIZE*D_VO) + head_idx_in_wv_group*32 + head_dim_idx*WV_GROUP_SIZE,
                        output_fp8[mma_atom_idx] + i,
                        "no_allocate",
                        "evict_first"
                    );
                }
            }
        };

        OuterloopArgs cur_job = get_first_job();
        load_q_and_save_to_tmem(cur_job);
        do {
            OuterloopArgs next_job = get_next_job(cur_job);
            shift_cached_cos_and_cached_sin();
            if (next_job.is_valid) {
                load_q_and_save_to_tmem(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    } else if (warpgroup_idx == 3) {
        // Scale & Exp warpgroup
        cutlass::arch::warpgroup_reg_alloc<128>();

        OuterloopArgs cur_job = get_first_job();
        uint32_t local_warp_idx = warp_idx - 12;
        static_assert(FOLD_FACTOR == 2);
        bf16* sS_base = smem.s + (local_warp_idx >= 2 ? H_Q_PER_CTA * (B_TOPK/2) : 0) + (idx_in_warpgroup%H_Q_PER_CTA) * 8;
        RingBufferState rs;
        do {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            static constexpr uint32_t NUM_ELEMS_PER_THREAD = B_TOPK * H_Q_PER_CTA / 128;
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            float score_multiplier; // qk_scale * rms_norm's denominator (if q norm is enabled)
            if constexpr (ENABLE_Q_NORM) {
                if constexpr (IS_2CTA) {
                    smem.bar_q_sqr_sum_full.wait(cur_job.job_idx_mod_2);
                } else {
                    smem.bar_tQ_full.wait(cur_job.job_idx_mod_2);
                }

                if constexpr (H_Q_PER_CTA == 64) {
                    score_multiplier = smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup] + smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64];
                } else {
                    static_assert(H_Q_PER_CTA == 32);
                    score_multiplier = __fadd_rn(
                        __fadd_rn(smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup], smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^64]),
                        __fadd_rn(smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^32], smem.q_sqr_sum_buf[cur_job.job_idx_mod_2][idx_in_warpgroup^96])
                    );
                }
                score_multiplier = params.sm_scale_div_log2 * rsqrtf(score_multiplier / D_QK + params.rms_norm_eps);    // rsqrt is translated to `MUFU.RSQ`
            } else {
                score_multiplier = params.sm_scale_div_log2;
            }

            smem.bar_tQ_empty.arrive(); // Must arrive on the empty barrier here, to prevent smem.bar_tQ_full being phase-skipped

            CUTE_NO_UNROLL
            for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();
                smem.bar_tP_full[p_buf_idx].wait(p_bar_phase);
                smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG3_WARP02_SYNC,
                    barrier_ids::WG3_WARP13_SYNC,
                    false
                >(
                    tmem_cols::get_p(p_buf_idx),
                    (char*)&smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {
                        if constexpr (NEED_TP_EMPTY_BAR) {
                            arrive_on_cta0_barrier(smem.bar_tP_empty[p_buf_idx]);
                        }
                    },
                    smem.p_exchange_buf,
                    p
                );

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= score_multiplier;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(64, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx&1));
                smem.bar_indices_empty[indices_buf_idx].arrive();   // Put it here to give the compiler more room for SASS code reordering
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

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
                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>((nv_bfloat162*)s, p, score_multiplier, new_max);
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SO_empty.wait(rs.get<1>().second^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    ku::st_shared(sS_base + i*8*H_Q_PER_CTA, *(__int128_t*)(s + i*8));
                }

                // Rescale O
                if (kv_block_idx > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_VO / FOLD_FACTOR, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                ku::tcgen05_before_thread_sync();
                arrive_on_cta0_barrier(smem.bar_SO_full);
                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // No valid TopK indices
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2^1);
            static_assert(H_Q_PER_CTA % 32 == 0);
            if (local_warp_idx < H_Q_PER_CTA / 32) {
                if constexpr (!IS_DECODE) {
                    uint32_t global_index = cur_job.s_q_idx * H_Q + cta_idx * H_Q_PER_CTA + idx_in_warpgroup;
                    params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                }
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<72>();
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // MMA warp (CTA0 only)
            auto tiled_mma_qk = TiledMMA_QK{};
            auto tiled_mma_sv = TiledMMA_SV{};
            Tensor tQ = tiled_mma_qk.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<D_QK / NUM_MRGEMM_RAILS>>{})
            );
            Tensor tP = partition_fragment_C(tiled_mma_qk, Shape<Int<H_Q_PER_CTA>, Int<B_TOPK * NUM_MRGEMM_RAILS>>{});
            Tensor sS = make_tensor(
                make_smem_ptr(smem.s),
                ku::make_umma_canonical_k_major_layout<H_Q_PER_CTA, B_TOPK, 0>()
            );
            Tensor tO = partition_fragment_C(tiled_mma_sv, Shape<Int<H_Q_PER_CTA>, Int<D_VO>>{});
            tQ.data().get() = tmem_cols::Q;
            tO.data().get() = tmem_cols::O;
            // tP.data() will be assigned in the loop since it has double buffers

            RingBufferState rs_qk, rs_sv;
            auto run_qk_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                if (kv_block_idx == 0) {
                    smem.bar_tQ_full.wait(job.job_idx_mod_2);
                }
                auto [kv_slot_idx, kv_bar_phase] = rs_qk.get<NUM_KV_SLOTS>();
                Tensor sK = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_k_major_layout<B_TOPK / CLUSTER_SIZE * NUM_MRGEMM_RAILS, D_QK / NUM_MRGEMM_RAILS, 128>()
                );
                // Expected TMA transaction bytes on bar_kv_slot_full:
                //   Prefill: the whole KV block; Decode: only the bf16 (RoPE) part (the fp8 part is dequantized by WG1, which arrives on the same barrier)
                if constexpr (IS_DECODE && D_BF16 == 0) {
                    // No bf16 part, so no TMA transaction is expected (expect-tx count must not be 0)
                    smem.bar_kv_slot_full[kv_slot_idx].arrive();
                } else {
                    smem.bar_kv_slot_full[kv_slot_idx].arrive_and_expect_tx(IS_DECODE ? B_TOPK*D_BF16*sizeof(bf16) : B_TOPK*D_QK*sizeof(bf16));
                }
                smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
                auto [p_buf_idx, p_bar_phase] = rs_qk.get<NUM_P_BUFS>();
                if constexpr (NEED_TP_EMPTY_BAR) {
                    smem.bar_tP_empty[p_buf_idx].wait(p_bar_phase ^ 1);   
                }
                tP.data().get() = tmem_cols::get_p(p_buf_idx);

                ku::tcgen05_after_thread_sync();
                ku::utcmma_ts(tiled_mma_qk, tQ, sK, tP, true);
                umma_arrive_on_every_cta(smem.bar_tP_full[p_buf_idx]);

                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tQ_empty);
                }
                rs_qk.update();
            };

            auto run_sv_gemm = [&](const OuterloopArgs &job, uint32_t kv_block_idx) {
                if (kv_block_idx == 0) {
                    smem.bar_tO_empty.wait(job.job_idx_mod_2^1);
                }
                auto [kv_slot_idx, _] = rs_sv.get<NUM_KV_SLOTS>();
                smem.bar_SO_full.wait(rs_sv.get<1>().second);
                Tensor sV = make_tensor(
                    make_smem_ptr(smem.kv_slots[kv_slot_idx]),
                    ku::make_umma_canonical_mn_major_layout<D_VO / CLUSTER_SIZE, B_TOPK, 128>()
                );
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_sv, sS, sV, tO, kv_block_idx == 0);
                umma_arrive_on_every_cta(smem.bar_kv_slot_empty[kv_slot_idx]);
                umma_arrive_on_every_cta(smem.bar_SO_empty);
                if (kv_block_idx == job.num_kv_blocks-1) {
                    umma_arrive_on_every_cta(smem.bar_tO_full);
                }
                rs_sv.update();
            };

            OuterloopArgs cur_job = get_first_job();
            run_qk_gemm(cur_job, 0);
            do {
                CUTE_NO_UNROLL
                for (uint32_t kv_block_idx = 1; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                    run_qk_gemm(cur_job, kv_block_idx);
                    run_sv_gemm(cur_job, kv_block_idx-1);
                }

                OuterloopArgs next_job = get_next_job(cur_job);
                if (next_job.is_valid) {
                    run_qk_gemm(next_job, 0);
                }
                run_sv_gemm(cur_job, cur_job.num_kv_blocks-1);
                
                cur_job = next_job;
            } while (cur_job.is_valid);
        } else if (warp_idx == 9 && elect_one_sync()) {
            // CLC warp
            bool phase = 0;
            while (true) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase^1);
                    ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                }
                smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                
                smem.bar_clc_full.wait(phase&1);
                ku::CLCResult clc_result = ku::get_clc_query_response<true>(smem.clc_response_obj);
                arrive_on_cta0_barrier(smem.bar_clc_empty);
                if (!clc_result.is_valid)
                    break;

                phase ^= 1;
            }
            if constexpr (IS_2CTA) {
                if (cta_idx == 0) {
                    smem.bar_clc_empty.wait(phase); // Wait for all threads' arrival on `bar_clc_empty`, which means that there will be no further operations on distributed shared memory (including barrier arrive and MMA), avoiding the "cluster target block not present" error
                    smem.bar_clc_empty.arrive(1u);  // Transfer the signal above to CTA1
                } else {
                    smem.bar_clc_empty.wait(0);
                }
            }
        } else if (warp_idx == 10) {
            // Indices generator
            // (also generates TMA coords & scales for dequant warps)
            OuterloopArgs cur_job = get_first_job();
            RingBufferState rs;
            static constexpr uint32_t NUM_INDICES_PER_THREAD = B_TOPK / 32;
            static_assert(B_TOPK % 32 == 0);

            do {
                if constexpr (!IS_DECODE) {
                    auto body = [&]<bool CHECK_TOPK_SUBSCRIPT>() {
                        CUTE_NO_UNROLL
                        for (uint32_t kv_block_idx = 0; kv_block_idx < cur_job.num_kv_blocks; ++kv_block_idx) {
                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                            smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                            CUTE_UNROLL
                            for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                                uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                                int cur_index;
                                if constexpr (CHECK_TOPK_SUBSCRIPT) {
                                    // Predicate the load on `pos < topk_length` to prevent IMA
                                    cur_index = pos < cur_job.topk_length ? __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos) : -1;
                                } else {
                                    // topk_length % B_TOPK == 0, so every pos is within the row
                                    cur_index = __ldg(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos);
                                }
                                bool is_index_valid = (uint32_t)cur_index < (uint32_t)params.s_kv;  // Don't need to check `index >= 0`, since if `index < 0` holds, `(uint32_t)index` must lies in 2147483648 ~ 4294967295, which is definitely greater than `params.s_kv`
                                uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                                if (lane_idx == 0) {
                                    *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                                }
                            }

                            smem.bar_indices_full[indices_buf_idx].arrive();
                            rs.update();
                        }
                    };
                    if (cur_job.topk_length % B_TOPK == 0 && cur_job.topk_length != 0)
                        body.template operator()<false>();
                    else
                        body.template operator()<true>();
                } else {
                    int *indices_base = params.indices + (int64_t)cur_job.s_q_idx * params.stride_indices_s_q;
                    int *extra_indices_base = params.extra_indices + (int64_t)cur_job.s_q_idx * params.stride_extra_indices_s_q;
                    run_along_kv_blocks(cur_job, [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase^1);

                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
                            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
                            bool in_extra;
                            if constexpr (LOC == KVLocation::ORIG) {
                                in_extra = false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                in_extra = true;
                            } else {
                                in_extra = pos >= cur_job.num_orig_slots;
                            }
                            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
                            uint32_t valid_len = in_extra ? cur_job.extra_topk_length : cur_job.topk_length;
                            // Predicate the load on `local_pos < valid_len`: the last KV block may run
                            // beyond the end of the indices row (e.g. topk=128 with B_TOPK=96), so an
                            // unconditional load could read OOB of `indices` / `extra_indices`. Slots
                            // beyond `valid_len` are masked out anyway, so skip the load for them
                            int cur_index = local_pos < valid_len ? __ldg((in_extra ? extra_indices_base : indices_base) + local_pos) : -1;
                            bool is_index_valid = cur_index >= 0;
                            
                            // Share coord/scale generation instead of repeating it in all dequant warps.
                            int64_t src_block_stride = in_extra ? params.stride_extra_kv_block : params.stride_kv_block;
                            const auto &fast_divmod = in_extra
                                ? aux_params.fast_divmod_extra_page_block_size
                                : aux_params.fast_divmod_page_block_size;
                            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
                            int idx_in_block;
                            int block_idx = fast_divmod.divmod(idx_in_block, (int)token_idx);
                            uint32_t row = i * 32 + lane_idx;
                            // The extra KV cache may have another format (HAS_FP4_KV); ORIG / EXTRA blocks resolve it at compile time
                            uint32_t tma_k_stride = in_extra ? ExtraKVFormat::TMA_K_STRIDE : OrigKVFormat::TMA_K_STRIDE;
                            uint32_t num_scales_each_token = in_extra ? ExtraKVFormat::NUM_SCALES_EACH_TOKEN : OrigKVFormat::NUM_SCALES_EACH_TOKEN;
                            smem.decode_tma_coords[indices_buf_idx][row] = is_index_valid
                                ? (src_block_stride / tma_k_stride) * block_idx + idx_in_block
                                : -1;

                            uint32_t page_block_size = in_extra ? params.extra_page_block_size : params.page_block_size;
                            uint8_t *kv_base = (uint8_t*)(in_extra ? params.extra_kv : params.kv);
                            uint8_t *scale_ptr = kv_base + page_block_size * tma_k_stride +
                                (int64_t)block_idx * src_block_stride +
                                idx_in_block * num_scales_each_token +
                                cta_idx * (num_scales_each_token / 2);
                            bool is_fp4_token = HAS_FP4_KV && in_extra;   // The dequant warps load the scales of fp4 tokens themselves
                            if (!is_fp4_token) {
                                uint8_t *scale_dst = smem.decode_scales[indices_buf_idx] +
                                    row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                                if constexpr (NUM_SCALES_EACH_TOKEN_PER_CTA == 4) {
                                    uint32_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b32 %0, [%1];"
                                        : "=r"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(uint32_t*)scale_dst = is_index_valid ? scales : 0;
                                } else if constexpr (NUM_SCALES_EACH_TOKEN_PER_CTA == 8) {
                                    uint64_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b64 %0, [%1];"
                                        : "=l"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(uint64_t*)scale_dst = is_index_valid ? scales : 0;
                                } else {
                                    static_assert(NUM_SCALES_EACH_TOKEN_PER_CTA == 16);
                                    __int128_t scales;
                                    asm volatile (
                                        "ld.global.nc.L1::no_allocate.b128 %0, [%1];"
                                        : "=q"(scales)
                                        : "l"((uint64_t)scale_ptr)
                                    );
                                    *(__int128_t*)scale_dst = is_index_valid ? scales : 0;
                                }
                            } else {
                                smem.decode_scale_ptrs[indices_buf_idx][row] = is_index_valid ? scale_ptr : nullptr;
                            }
                            uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
                            if (lane_idx == 0) {
                                *((uint32_t*)smem.is_k_valid[indices_buf_idx] + i) = mask;
                            }
                        }

                        smem.bar_indices_full[indices_buf_idx].arrive();
                        rs.update();
                    });
                }
                cur_job = get_next_job(cur_job);
            } while (cur_job.is_valid);
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<128>();
        if constexpr (IS_DECODE) {
            if constexpr (HAS_FP4_KV) {
                // KV producer for an fp8 `kv` plus an fp4 `extra_kv`: gathers the quantized rows of every selected KV token
                // into the beginning of the KV slot via TMA gather4, dequantizes them in registers, and stores the bf16 result
                // into the KV slot inplace (SW128 K-major layout). fp8 tokens (from `kv`) and fp4
                // tokens (from `extra_kv`) share the code below and differ only in the tensor map, the number of elements per
                // 16 B of raw data (CHUNK_ELEMS), loading of scales, and the conversion instructions. A KV block
                // straddling the orig/extra boundary (KVLocation::ORIG_AND_EXTRA) mixes both kinds of tokens; the format
                // is resolved per 8 rows, so `run()` asserts topk % 8 == 0.
                //
                // The unit of work is a chunk: 16 B of raw data (one LDS.128) = CHUNK_ELEMS elements = CHUNK_ELEMS * 2 B of the bf16
                // 128 B swizzle-atom row of the KV slot (CHUNK_ELEMS / 8 STS.128). Lane-to-token mapping: 4 lanes per row, each
                // owning a quarter of the row's chunks, and 8 consecutive rows (one "token" of the thread) per 8 consecutive lanes,
                // NUM_TOKENS_PER_THREAD tokens per thread. Unlike the fp8-only path, the 4 lanes of one row are spread over the 4
                // wavefronts of an LDS/STS.128 (8 consecutive lanes each), so that one wavefront covers 8 consecutive rows working
                // on the same chunk position (with the fp8-only mapping, two of the 4 lanes of a row would own fp4 chunks of the
                // same parity and their STS.128 would conflict). With C = chunks per lane, and rows relative to the token:
                //
                //   lane          0  1  2  3 |  4  5  6  7 |  8 .. 11 | 12 .. 15 | 16 .. 19 | 20 .. 23 | 24 .. 27 | 28 .. 31
                //   row           0  1  2  3 |  4  5  6  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7 |  0 ..  3 |  4 ..  7
                //   chunks         [0, C)    | rotated by 4|  [C, 2C) | rotated  | [2C, 3C) | rotated  | [3C, 4C) | rotated
                //
                // "rotated by 4": the lanes of rows 4..7 process the same chunks as the lanes of rows 0..3 but in an order shifted
                // by 4 chunks (= 64 B), see get_chunk_base. This makes every wavefront bank-conflict-free:
                //  - Raw LDS.128: the rows of a 4-row gather4 group are RAW_TOKEN_SMEM_STRIDE bytes apart with RAW_TOKEN_SMEM_STRIDE / 16 odd,
                //    so they hit 4 consecutive 16 B bank groups; the groups themselves start 128 B aligned, so the second group
                //    of the wavefront reads chunks rotated by 4 to hit the other 4 bank groups.
                //  - STS.128 into the SW128 K-major layout: a 16 B part of chunk c lands in the 16 B bank group
                //    ((c % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK + j) ^ (row % 8) of its swizzle-atom row. The 8 rows of a
                //    wavefront have 8 distinct row % 8 and, the rotation being a multiple of CHUNKS_PER_ATOM_ROW, the same
                //    c % CHUNKS_PER_ATOM_ROW.
                //
                // Dequant pipeline per KV block:
                //  1. Read the scales of the fp8 tokens from smem, issue gather4 for the raw rows (completing on bar_raw_kv_full),
                //     then load the scales of the fp4 tokens from global memory (their latency overlaps with the gather).
                //  2. Wait for TMA, then read the raw data via LDS.128. Synchronize afterward to prevent the subsequent
                //     write-back from overwriting the read data.
                //  3. Per chunk: fp8: 8x F2FP (e4m3x2 x ue8m0 -> bf16x2); fp4: 16x F2FP (e2m1x2 -> bf16x2) + 16x HMUL2.BF16
                //     (x the bf16 of the e4m3 scale, exact: the product has at most 2 + 4 significant bits). Then the STS.128s.
                static_assert(!OrigKVFormat::IS_FP4 && ExtraKVFormat::IS_FP4 && ExtraKVFormat::D_BF16 == 0);
                uint32_t local_warp_idx = warp_idx - 4;
                static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_LANES_PER_ROW = 4;
                static constexpr uint32_t NUM_ROWS_PER_WAVEFRONT = 32 / NUM_LANES_PER_ROW;   // Rows covered by 8 consecutive lanes (one LDS/STS.128 wavefront)
                static constexpr uint32_t NUM_TOKENS_PER_THREAD = B_TOPK / (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT);
                static_assert(B_TOPK % (NUM_DEQUANT_WARPS * NUM_ROWS_PER_WAVEFRONT) == 0);
                static constexpr uint32_t NUM_ROWS_PER_WARP = NUM_TOKENS_PER_THREAD * NUM_ROWS_PER_WAVEFRONT;
                static constexpr uint32_t MAX_CHUNKS_PER_LANE = OrigKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;   // fp8 rows have the most: 8 / 4
                const uint32_t row_in_wavefront = lane_idx % NUM_ROWS_PER_WAVEFRONT, idx_in_row = lane_idx / NUM_ROWS_PER_WAVEFRONT;
                const uint32_t chunk_rotation = row_in_wavefront / 4 * 4;
                auto get_row_idx = [&](uint32_t token_idx) {
                    return local_warp_idx * NUM_ROWS_PER_WARP + token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
                };
                // The chunks this lane processes are numbered g = 0.. within the lane; chunk g of the row is chunk_base(g / 4) + g % 4,
                // so that all addresses are a per-lane base plus an immediate. The lanes of rows 4..7 rotate the order by 4 chunks:
                // as a rotation of the chunk indices of the row when a lane owns at most 4 chunks, as a swap of its two halves
                // (g ^ 4) when it owns 8
                auto get_chunk_base = [&]<typename F>(uint32_t half) {
                    constexpr uint32_t NUM_CHUNKS_PER_LANE = F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
                    static_assert(NUM_CHUNKS_PER_LANE == 2 || NUM_CHUNKS_PER_LANE == 4 || NUM_CHUNKS_PER_LANE == 8);
                    if constexpr (NUM_CHUNKS_PER_LANE <= 4) {
                        return (idx_in_row * NUM_CHUNKS_PER_LANE + chunk_rotation) % F::NUM_CHUNKS_PER_ROW;
                    } else {
                        return idx_in_row * NUM_CHUNKS_PER_LANE + (half * 4 ^ chunk_rotation);
                    }
                };
                auto get_raw_row_offset = [&](uint32_t row) {
                    return row / 4 * RAW_KV_GROUP_BYTES;   // + (row % 4) * F::RAW_TOKEN_SMEM_STRIDE, which depends on the format
                };

                // STS.128 offsets of the 8 (swizzled) 16 B parts of the first swizzle-atom row of this lane's first token. The other
                // atom rows / tokens are whole swizzle-atom columns / multiples of 8 rows further, i.e. plain offsets
                static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
                static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = NUM_ROWS_PER_WAVEFRONT * 128;
                uint32_t sts_offsets[8];
                {
                    Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK / CLUSTER_SIZE, 128>());
                    CUTE_UNROLL
                    for (uint32_t i = 0; i < 8; ++i) {
                        sts_offsets[i] = (uint32_t)((&sKV(get_row_idx(0), i * 8) - smem.kv_slots[0]) * sizeof(bf16));
                    }
                }

                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                do {
                    // Processes one KV block whose first `num_orig_rows` rows come from `kv` (fp8) and the others from `extra_kv` (fp4),
                    // the first NUM_FP8_TOKENS tokens of this thread being the fp8 ones. NUM_FP8_TOKENS is a template parameter so that
                    // the format of every token, and with it the code operating on the token's registers, is fixed at compile time
                    auto process_block = [&]<uint32_t NUM_FP8_TOKENS>(uint32_t num_orig_rows) {
                        // Runs `callable.template operator()<F>(token_idx)` for every token of this thread, F being its format
                        auto for_each_token = [&](auto callable) {
                            cute::for_each(cute::make_int_sequence<NUM_TOKENS_PER_THREAD>{}, [&](auto token_idx) {
                                if constexpr (token_idx < NUM_FP8_TOKENS) {
                                    callable.template operator()<OrigKVFormat>(token_idx);
                                } else {
                                    callable.template operator()<ExtraKVFormat>(token_idx);
                                }
                            });
                        };

                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                        // The scales of this lane's chunks, per token. fp8 tokens: one ue8m0 per QUANT_TILE_SIZE elements = per 2 chunks,
                        // read from smem here. fp4 tokens: two e4m3 per chunk, loaded from global memory after the gather below
                        static constexpr uint32_t FP8_SCALE_BYTES_PER_LANE = OrigKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW / 2;   // 4 / 2
                        static constexpr uint32_t FP4_SCALE_BYTES_PER_LANE = ExtraKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW * 2;   // 8 / 4
                        static_assert(FP8_SCALE_BYTES_PER_LANE <= FP4_SCALE_BYTES_PER_LANE && FP4_SCALE_BYTES_PER_LANE % 4 == 0);
                        uint32_t cached_scales[NUM_TOKENS_PER_THREAD][FP4_SCALE_BYTES_PER_LANE / 4];
                        const uint8_t *scale_ptrs[NUM_TOKENS_PER_THREAD];
                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            uint32_t row = get_row_idx(token_idx);
                            if constexpr (F::IS_FP4) {
                                scale_ptrs[token_idx] = smem.decode_scale_ptrs[indices_buf_idx][row];
                            } else {
                                // Arranged so that chunk g uses byte g / 2 of the loaded word: a lane owning 8 chunks swaps their two halves
                                // (see get_chunk_base), i.e. the two halves of its 4 scale bytes; a lane owning 4 chunks rotates the chunk
                                // indices, so its 2 scale bytes are read from the rotated position
                                const uint8_t *scale_src = smem.decode_scales[indices_buf_idx] + row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                                if constexpr (FP8_SCALE_BYTES_PER_LANE == 4) {
                                    uint32_t scales = *(uint32_t*)(scale_src + idx_in_row * 4);
                                    cached_scales[token_idx][0] = chunk_rotation ? __byte_perm(scales, scales, 0x1032) : scales;
                                } else {
                                    static_assert(FP8_SCALE_BYTES_PER_LANE == 2);
                                    cached_scales[token_idx][0] = *(uint16_t*)(scale_src + get_chunk_base.template operator()<F>(0) / 2);
                                }
                            }
                        });

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        uint8_t *slot_base = (uint8_t*)smem.kv_slots[kv_slot_idx];

                        cute::for_each(cute::make_int_sequence<NUM_ROWS_PER_WARP / 4>{}, [&](auto i) {
                            // Each tma_gather4 covers 4 consecutive rows of one token, which share one tensor map. Only the
                            // elected lane needs the coordinates, and the 4 rows are consecutive in `decode_tma_coords`, so it
                            // reads them with one LDS.128
                            uint32_t row_start = local_warp_idx * NUM_ROWS_PER_WARP + i * 4;
                            constexpr bool is_fp4 = i / (NUM_ROWS_PER_WAVEFRONT / 4) >= NUM_FP8_TOKENS;
                            auto tensor_map = is_fp4 ?
                                (cta_idx == 0 ? &tma_params.tensor_map_extra_kv_fp4_part_cta0 : &tma_params.tensor_map_extra_kv_fp4_part_cta1) :
                                (cta_idx == 0 ? &tma_params.tensor_map_kv_fp8_part_cta0 : &tma_params.tensor_map_kv_fp8_part_cta1);
                            if (elect_one_sync()) {
                                int4 coords = *(int4*)(smem.decode_tma_coords[indices_buf_idx] + row_start);
                                ku::tma_gather4(
                                    tensor_map,
                                    smem.bar_raw_kv_full,
                                    slot_base + get_raw_row_offset(row_start),
                                    0,
                                    coords,
                                    (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                );
                            }
                        });
                        smem.bar_indices_empty[indices_buf_idx].arrive();

                        // Invalid fp4 tokens (nullptr) are zero-filled by TMA and get scale 0, so they dequantize to 0. The "memory"
                        // clobber keeps the loads behind the gather4s above, otherwise ptxas hoists them and the gather4s wait for them
                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            if constexpr (F::IS_FP4) {
                                uint64_t scale_addr = (uint64_t)(scale_ptrs[token_idx] + get_chunk_base.template operator()<F>(0) * 2);
                                if constexpr (FP4_SCALE_BYTES_PER_LANE == 8) {
                                    uint64_t scales = 0;
                                    if (scale_ptrs[token_idx] != nullptr) {
                                        asm volatile (
                                            "ld.global.nc.L1::no_allocate.b64 %0, [%1];"
                                            : "=l"(scales)
                                            : "l"(scale_addr)
                                            : "memory"
                                        );
                                    }
                                    *(uint64_t*)cached_scales[token_idx] = scales;
                                } else {
                                    static_assert(FP4_SCALE_BYTES_PER_LANE == 4);
                                    uint32_t scales = 0;
                                    if (scale_ptrs[token_idx] != nullptr) {
                                        asm volatile (
                                            "ld.global.nc.L1::no_allocate.b32 %0, [%1];"
                                            : "=r"(scales)
                                            : "l"(scale_addr)
                                            : "memory"
                                        );
                                    }
                                    cached_scales[token_idx][0] = scales;
                                }
                            }
                        });

                        if (idx_in_warpgroup == 0) {
                            smem.bar_raw_kv_full.arrive_and_expect_tx(num_orig_rows * OrigKVFormat::RAW_TOKEN_SMEM_STRIDE + (B_TOPK - num_orig_rows) * ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE);
                        }
                        smem.bar_raw_kv_full.wait(rs.get<1>().second);

                        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4];   // fp4 tokens (half as many chunks) use the first half
                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            uint32_t row = get_row_idx(token_idx);
                            uint8_t *row_base = slot_base + get_raw_row_offset(row) + row % 4 * F::RAW_TOKEN_SMEM_STRIDE;
                            CUTE_UNROLL
                            for (uint32_t g = 0; g < F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW; ++g) {
                                *(__int128_t*)(cached_input[token_idx][g]) = ku::ld_shared(
                                    row_base + get_chunk_base.template operator()<F>(g / 4) * 16 + g % 4 * 16
                                );
                            }
                        });
                        NamedBarrier::arrive_and_wait(128, 7);  // Make sure everyone has finished reading

                        for_each_token([&]<typename F>(uint32_t token_idx) {
                            static constexpr uint32_t C_LANE = F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
                            static constexpr uint32_t NUM_STS_PER_CHUNK = F::CHUNK_ELEMS * sizeof(bf16) / 16;
                            ku::nvbf16x2 fp4_scales[2];   // fp4: the two scales of each of the current pair of chunks, as bf16
                            CUTE_UNROLL
                            for (uint32_t g = 0; g < C_LANE; ++g) {
                                ku::nvbf16x2 data_bf16x2[F::CHUNK_ELEMS / 2];
                                if constexpr (F::IS_FP4) {
                                    if (g % 2 == 0) {
                                        fp8x4_to_bf16x2x2(cached_scales[token_idx][g / 2], fp4_scales);
                                    }
                                    CUTE_UNROLL
                                    for (uint32_t j = 0; j < 4; ++j) {
                                        fp4x8_to_bf16x2x4(cached_input[token_idx][g][j], data_bf16x2 + j * 4);
                                    }
                                    ku::nvbf16x2 scale_lo = __low2bfloat162(fp4_scales[g % 2]), scale_hi = __high2bfloat162(fp4_scales[g % 2]);
                                    CUTE_UNROLL
                                    for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                                        data_bf16x2[k] = __hmul2(data_bf16x2[k], k < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi);
                                    }
                                } else {
                                    __nv_fp8_e8m0 scale = ((__nv_fp8_e8m0*)cached_scales[token_idx])[g * F::CHUNK_ELEMS / F::QUANT_TILE_SIZE];
                                    CUTE_UNROLL
                                    for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                                        data_bf16x2[k] = fp8x2_to_bf16x2_with_scale(((ku::nve4m3x2*)cached_input[token_idx][g])[k], scale);
                                    }
                                }
                                // The chunk covers parts [(g % CHUNKS_PER_ATOM_ROW) * NUM_STS_PER_CHUNK, +NUM_STS_PER_CHUNK) of the atom row
                                // chunk_base / CHUNKS_PER_ATOM_ROW + g % 4 / CHUNKS_PER_ATOM_ROW (exact: chunk_base is a multiple of
                                // CHUNKS_PER_ATOM_ROW), so the swizzled offset is selected at compile time
                                static constexpr uint32_t CHUNKS_PER_ATOM_ROW = 128 / (F::CHUNK_ELEMS * 2);
                                uint32_t atom_col = get_chunk_base.template operator()<F>(g / 4) / CHUNKS_PER_ATOM_ROW + g % 4 / CHUNKS_PER_ATOM_ROW;
                                CUTE_UNROLL
                                for (uint32_t j = 0; j < NUM_STS_PER_CHUNK; ++j) {
                                    ku::st_shared(
                                        slot_base + sts_offsets[g % CHUNKS_PER_ATOM_ROW * NUM_STS_PER_CHUNK + j] + atom_col * STS_ATOM_COL_STRIDE_BYTES + token_idx * STS_TOKEN_STRIDE_BYTES,
                                        *(__int128_t*)(data_bf16x2 + j * 4)
                                    );
                                }
                            }
                        });

                        fence_view_async_shared();
                        arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
                        rs.update();
                    };

                    run_along_kv_blocks(cur_job, [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        if constexpr (LOC == KVLocation::ORIG) {
                            process_block.template operator()<NUM_TOKENS_PER_THREAD>(B_TOPK);
                        } else if constexpr (LOC == KVLocation::EXTRA) {
                            process_block.template operator()<0>(0);
                        } else {
                            // The block straddles the orig/extra boundary. The number of fp8 tokens of this thread is warp-uniform since
                            // topk % 8 == 0 (asserted by `run()`); dispatch to the matching instantiation
                            uint32_t num_orig_rows = cur_job.num_orig_slots - kv_block_idx * B_TOPK;
                            uint32_t num_fp8_tokens = (uint32_t)std::clamp((int)num_orig_rows - (int)(local_warp_idx * NUM_ROWS_PER_WARP), 0, (int)NUM_ROWS_PER_WARP) / NUM_ROWS_PER_WAVEFRONT;
                            [&]<uint32_t... Ks>(std::integer_sequence<uint32_t, Ks...>) {
                                ((num_fp8_tokens == Ks ? process_block.template operator()<Ks>(num_orig_rows) : void()), ...);
                            }(std::make_integer_sequence<uint32_t, NUM_TOKENS_PER_THREAD + 1>{});
                        }
                    });

                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            } else {
                // KV producer: loads the fp8 part of every selected KV token directly from global
                // memory, dequantizes it in registers, and stores the bf16 result into the KV slot (SW128
                // K-major layout, the first D_FP8 columns). Also load the bf16 (RoPE) part via TMA gather4.

                // Maximize shared memory throughput by using LDS.128 and STS.128 with 8 tokens processed per warp.
                // Lane-to-token mapping for the 8 tokens being processed:
                //
                //  0  1  2  3
                //  4  5  6  7
                //  8  9 10 11
                // 12 13 14 15
                // 16 17 18 19
                // 20 21 22 23
                // 24 25 26 27
                // 28 29 30 31
                //
                // Dequant pipeline:
                //  1. Load raw FP8 KV from global memory into the target shared-memory KV buffer via TMA gather4.
                //  2. While TMA is in flight, load scale factors via plain global loads.
                //  3. Wait for TMA, then read raw FP8 KV via LDS.128. Synchronize afterward to prevent the
                //     subsequent write-back from overwriting the read data.
                //  4. Dequantize in registers, then write back bf16 result via STS.128.
                //
                // This pipeline is bank-conflict-free:
                //  - On TMA gather4 load: if the per-row FP8 count (D_FP8_CTA0/1) is a multiple of 128,
                //    the box size is padded by 64B so that lanes i..i+8 see no bank conflicts during LDS.128.
                //  - On STS.128 write-back: swizzling avoids bank conflicts.
                using fp8_e8m0 = __nv_fp8_e8m0;
                uint32_t local_warp_idx = warp_idx - 4;
                static constexpr uint32_t NUM_DEQUANT_WARPS = 4, NUM_ROWS_PER_WARP = 8;
                static constexpr uint32_t GROUP_SIZE = 4;
                static constexpr uint32_t NUM_COLS_PER_GROUP = D_VO / CLUSTER_SIZE / (GROUP_SIZE*16);
                static_assert((D_VO/CLUSTER_SIZE) % (GROUP_SIZE*16) == 0);
                uint32_t group_idx = lane_idx / GROUP_SIZE, idx_in_group = lane_idx % GROUP_SIZE;
                static constexpr uint32_t NUM_CHUNKS_PER_WARP = B_TOPK / (NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP);
                static_assert(B_TOPK % (NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP) == 0);
                static constexpr uint32_t NUM_TOKENS_PER_THREAD = NUM_CHUNKS_PER_WARP;

                auto get_row_idx = [&](uint32_t local_row_idx) {
                    return local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + group_idx;
                };
                auto [sts_base_offset_0, sts_base_offset_1] = [&] {
                    Tensor sKV = make_tensor(make_smem_ptr(smem.kv_slots[0]), ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK, 128>());
                    return std::pair<uint32_t, uint32_t> {
                        (uint32_t)(&sKV(local_warp_idx*NUM_ROWS_PER_WARP+group_idx, idx_in_group*16 + 0) - smem.kv_slots[0]),
                        (uint32_t)(&sKV(local_warp_idx*NUM_ROWS_PER_WARP+group_idx, idx_in_group*16 + 8) - smem.kv_slots[0])
                    };
                }();

                uint32_t d_fp8_this_cta = cta_idx == 0 ? D_FP8_CTA0 : D_FP8_CTA1;
                uint32_t d_fp8_this_cta_padded = d_fp8_this_cta + ((cta_idx==0?IS_CTA0_RAW_KV_PADDED:IS_CTA1_RAW_KV_PADDED)?64:0);

                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                do {
                    auto loop_body = [&]<KVLocation LOC>(uint32_t kv_block_idx) {
                        // Tells whether the unified slot `pos` comes from the extra KV. For ORIG / EXTRA
                        // blocks this is a compile-time constant, so all the source selections below fold
                        // into branch-free code; only the (at most one) ORIG_AND_EXTRA block resolves the
                        // source at runtime, per row
                        auto is_pos_in_extra = [&](uint32_t pos) -> bool {
                            if constexpr (LOC == KVLocation::ORIG) {
                                return false;
                            } else if constexpr (LOC == KVLocation::EXTRA) {
                                return true;
                            } else {
                                return pos >= cur_job.num_orig_slots;
                            }
                        };

                        static constexpr uint32_t NUM_SCALED_EACH_TOKEN_LOCAL = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
                        fp8_e8m0 cached_scales[NUM_TOKENS_PER_THREAD][NUM_SCALED_EACH_TOKEN_LOCAL];
                        int cached_tma_coord[NUM_TOKENS_PER_THREAD];
                        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
                        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            uint32_t row = get_row_idx(local_row_idx);
                            cached_tma_coord[local_row_idx] = smem.decode_tma_coords[indices_buf_idx][row];
                            uint8_t *scale_src = smem.decode_scales[indices_buf_idx] +
                                row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                            if constexpr (NUM_SCALED_EACH_TOKEN_LOCAL == 4) {
                                *(uint32_t*)cached_scales[local_row_idx] = *(uint32_t*)scale_src;
                            } else if constexpr (NUM_SCALED_EACH_TOKEN_LOCAL == 8) {
                                *(uint64_t*)cached_scales[local_row_idx] = *(uint64_t*)scale_src;
                            } else {
                                static_assert(NUM_SCALED_EACH_TOKEN_LOCAL == 16);
                                *(__int128_t*)cached_scales[local_row_idx] = *(__int128_t*)scale_src;
                            }
                        }
                        smem.bar_indices_empty[indices_buf_idx].arrive();

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                    
                        int4 collected_tma_coords[NUM_TOKENS_PER_THREAD][2];
                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            CUTE_UNROLL
                            for (uint32_t i = 0; i < 2; ++i) {
                                // Each tma_gather4 covers 4 consecutive rows, which must share one tensor
                                // map. `run()` asserts topk % 4 == 0 when the extra KV is present, so a
                                // 4-row group never straddles the orig/extra boundary
                                uint32_t row_start = local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + i*4;
                                bool group_in_extra = is_pos_in_extra(kv_block_idx*B_TOPK + row_start);
                                auto tensor_map = group_in_extra ? 
                                    (cta_idx == 0 ? &tma_params.tensor_map_extra_kv_fp8_part_cta0 : &tma_params.tensor_map_extra_kv_fp8_part_cta1) :
                                    (cta_idx == 0 ? &tma_params.tensor_map_kv_fp8_part_cta0 : &tma_params.tensor_map_kv_fp8_part_cta1);
                                int4 coords;
                                coords.x = i == 0 ? cached_tma_coord[local_row_idx] : __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16); // Since the thread being elected is always lane 0 on SM100
                                coords.y = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+4);
                                coords.z = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+8);
                                coords.w = __shfl_sync(0xFFFFFFFF, cached_tma_coord[local_row_idx], i*16+12);
                                collected_tma_coords[local_row_idx][i] = coords;
                                if (elect_one_sync()) {
                                    auto smem_ptr = (fp8_e4m3*)smem.kv_slots[kv_slot_idx] + row_start * d_fp8_this_cta_padded;
                                    ku::tma_gather4(
                                        tensor_map,
                                        smem.bar_raw_kv_full,
                                        smem_ptr,
                                        0,
                                        coords,
                                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                    );
                                }
                            }
                        }

                        if (D_BF16 > 0 && cta_idx+1 == CLUSTER_SIZE && elect_one_sync()) {
                            CUTE_UNROLL
                            for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                                CUTE_UNROLL
                                for (uint32_t i = 0; i < 2; ++i) {
                                    uint32_t row_start = local_row_idx*NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP + local_warp_idx*NUM_ROWS_PER_WARP + i*4;
                                    // Like the fp8 part above, a 4-row gather4 group never straddles the orig/extra boundary
                                    auto tensor_map = is_pos_in_extra(kv_block_idx*B_TOPK + row_start) ? &tma_params.tensor_map_extra_kv_bf16_part : &tma_params.tensor_map_kv_bf16_part;
                                    auto smem_ptr = smem.kv_slots[kv_slot_idx] + (IS_2CTA ? (D_FP8-D_VO/2)/64 : D_FP8/64) * B_TOPK * 64 + row_start * 64;
                                    if constexpr (IS_2CTA) {
                                        ku::tma_gather4_cta_group_2<true>(
                                            tensor_map,
                                            smem.bar_kv_slot_full[kv_slot_idx],
                                            smem_ptr,
                                            0,
                                            collected_tma_coords[local_row_idx][i],
                                            (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                        );
                                    } else {
                                        ku::tma_gather4(
                                            tensor_map,
                                            smem.bar_kv_slot_full[kv_slot_idx],
                                            smem_ptr,
                                            0,
                                            collected_tma_coords[local_row_idx][i],
                                            (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                                        );
                                    }
                                }
                            }
                        }

                        if (idx_in_warpgroup == 0) {
                            smem.bar_raw_kv_full.arrive_and_expect_tx(B_TOPK*d_fp8_this_cta_padded*sizeof(fp8_e4m3));
                        }
                        smem.bar_raw_kv_full.wait(rs.get<1>().second);

                        fp8_e4m3 cached_input[NUM_TOKENS_PER_THREAD][NUM_COLS_PER_GROUP][16];
                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            uint32_t row = get_row_idx(local_row_idx);
                            for (uint32_t local_col_idx = 0; local_col_idx < NUM_COLS_PER_GROUP; ++local_col_idx) {
                                if (MODEL_TYPE == ModelType::V4 && cta_idx+1 == CLUSTER_SIZE && local_col_idx+1 == NUM_COLS_PER_GROUP) {
                                    // Skip the last K/V block for V4
                                    continue;
                                }
                                *(__int128_t*)(cached_input[local_row_idx][local_col_idx]) = ku::ld_shared(
                                    (fp8_e4m3*)smem.kv_slots[kv_slot_idx] + 
                                    row*d_fp8_this_cta_padded + 
                                    local_col_idx*GROUP_SIZE*16 + 
                                    idx_in_group*16
                                );
                            }
                        }
                        NamedBarrier::arrive_and_wait(128, 7);  // Make sure everyone has finished reading

                        CUTE_UNROLL
                        for (uint32_t local_row_idx = 0; local_row_idx < NUM_TOKENS_PER_THREAD; ++local_row_idx) {
                            CUTE_UNROLL
                            for (uint32_t local_col_idx = 0; local_col_idx < NUM_COLS_PER_GROUP; ++local_col_idx) {
                                if (MODEL_TYPE == ModelType::V4 && cta_idx+1 == CLUSTER_SIZE && local_col_idx+1 == NUM_COLS_PER_GROUP) {
                                    // Skip the last K/V block for V4
                                    continue;
                                }
                                ku::nve4m3x2 data_fp8x2[8];
                                ku::nvbf16x2 data_bf16x2[8];
                                *(__int128_t*)data_fp8x2 = *(__int128_t*)(cached_input[local_row_idx][local_col_idx]);
                                static_assert(KV_QUANT_TILE_SIZE == 64 || KV_QUANT_TILE_SIZE == 32);
                                uint32_t scale_idx = KV_QUANT_TILE_SIZE == 64 ? local_col_idx : local_col_idx*2 + (idx_in_group >= GROUP_SIZE/2);
                                CUTE_UNROLL
                                for (uint32_t j = 0; j < 8; ++j) {
                                    data_bf16x2[j] = fp8x2_to_bf16x2_with_scale(data_fp8x2[j], cached_scales[local_row_idx][scale_idx]);
                                }
                                ku::st_shared(
                                    smem.kv_slots[kv_slot_idx] + sts_base_offset_0 + local_row_idx*(NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP*64) + local_col_idx*(B_TOPK*GROUP_SIZE*16),
                                    *(__int128_t*)(data_bf16x2 + 0)
                                );
                                ku::st_shared(
                                    smem.kv_slots[kv_slot_idx] + sts_base_offset_1 + local_row_idx*(NUM_DEQUANT_WARPS*NUM_ROWS_PER_WARP*64) + local_col_idx*(B_TOPK*GROUP_SIZE*16),
                                    *(__int128_t*)(data_bf16x2 + 4)
                                );
                            }
                        }

                        fence_view_async_shared();
                        arrive_on_cta0_barrier(smem.bar_kv_slot_full[kv_slot_idx]);
                        rs.update();
                    };

                    run_along_kv_blocks(cur_job, loop_body);

                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        } else {
            // KV Producer (prefill): gathers the whole bf16 KV block via TMA gather4
            if (elect_one_sync()) {
                OuterloopArgs cur_job = get_first_job();
                RingBufferState rs;
                uint32_t local_warp_idx = warp_idx - 4;
                do {
                    for (uint32_t i = 0; i < cur_job.num_kv_blocks; ++i) {
                        static_assert(B_TOPK % (4*4) == 0);
                        static constexpr uint32_t NUM_ROW_PER_WARP = B_TOPK / 4;
                        int4 topk_idxs[NUM_ROW_PER_WARP / 4];
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            uint32_t pos = i * B_TOPK + row;
                            // Predicate the load on `pos < topk_length` to avoid reading OOB of the
                            // indices row in the last (partial) KV block. Chunks with no slot below
                            // topk_length are masked out by the indices generator warp anyway, so feed
                            // them invalid indices (-1), which tma_gather4 bounds-checks into zero-fill
                            topk_idxs[local_row] = pos < cur_job.topk_length
                                ? __ldg((int4*)(params.indices + cur_job.s_q_idx * params.stride_indices_s_q + pos))
                                : int4{-1, -1, -1, -1};
                        }

                        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
                        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase^1);
                        CUTE_UNROLL
                        for (uint32_t local_row = 0; local_row < NUM_ROW_PER_WARP / 4; local_row += 1) {
                            uint32_t row = local_row * 4 * 4 + local_warp_idx * 4;
                            CUTE_UNROLL
                            for (uint32_t tile_idx = 0; tile_idx < (CLUSTER_SIZE == 2 ? (D_QK/2/64) : (D_QK/64)); ++tile_idx) {
                                /*
                                For cases where CLUSTER_SIZE == 1, each CTA reads the full K/V
                                For cases where CLUSTER_SIZE == 2, CTA0 reads KV[:, :D_QK/2] and CTA1 reads KV[:, D_QK/2:], since dual GeM< is used
                                */
                                if constexpr (CLUSTER_SIZE == 1) {
                                    ku::tma_gather4(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64,
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                } else {
                                    ku::tma_gather4_cta_group_2<true>(
                                        &tma_params.tensor_map_kv,
                                        smem.bar_kv_slot_full[kv_slot_idx],
                                        smem.kv_slots[kv_slot_idx] + tile_idx * B_TOPK * 64 + row * 64,
                                        tile_idx * 64 + cta_idx * (D_QK/2),
                                        topk_idxs[local_row],
                                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                                    );
                                }
                            }
                        }
                        rs.update();
                    }
                    cur_job = get_next_job(cur_job);
                } while (cur_job.is_valid);
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}


template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
fwd_kernel(__grid_constant__ const typename Kernel::Params params, __grid_constant__ const typename Kernel::TMAParams tma_params, __grid_constant__ const typename Kernel::AuxParams aux_params) {
    Kernel::devfunc(params, tma_params, aux_params);
}


template<Config CONFIG>
void Kernel<CONFIG>::run(const Params &params) {
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_VO);
    KU_ASSERT(params.stride_indices_s_q*sizeof(int) % 32 == 0, "indices.stride(0) must be 32B aligned, got %d elements (%d bytes)", params.stride_indices_s_q, (int)(params.stride_indices_s_q*sizeof(int)));

    KU_ASSERT(params.enable_q_norm == ENABLE_Q_NORM);
    KU_ASSERT(params.is_rope_neox_style == false && params.rope_dim == 64);
    KU_ASSERT(params.wv_group_size == WV_GROUP_SIZE);
    KU_ASSERT(params.num_per_channels == O_QUANT_TILE_SIZE);
    KU_ASSERT(params.use_tma_aligned_col_major_sf == true && params.round_sf == true && params.use_packed_ue8m0 == true);

    TMAParams tma_params = {};
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.b == 1, "Only batch size 1 is supported for the fused decoding kernel");
        KU_ASSERT(params.model_type == MODEL_TYPE && params.extra_model_type == EXTRA_MODEL_TYPE);
        if (params.extra_topk > 0) {
            // A KV block may straddle the orig/extra boundary (see KVLocation::ORIG_AND_EXTRA). Since
            // one TMA gather4 covers 4 consecutive rows sharing one tensor map, the boundary (i.e.
            // topk) must be aligned to 4 rows; the common dequant path resolves the format per 8 rows
            KU_ASSERT(params.topk % (HAS_FP4_KV ? 8 : 4) == 0, "topk (%d) must be a multiple of %d when the extra KV cache is used", params.topk, HAS_FP4_KV ? 8 : 4);
        }
        auto make_fp4_kv_tensor_map = [](bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::pair<CUtensorMap, CUtensorMap> {
            using F = ExtraKVFormat;
            KU_ASSERT((int64_t)kv_ptr % 16 == 0, "The base address of %skv (%p) must be 16B aligned", is_extra?"extra_":"", kv_ptr);
            KU_ASSERT(row_stride_bytes == (int)F::BYTES_PER_TOKEN, "%skv_cache.stride(-2) (%d) must be %d, i.e. each page block in the KV cache must be contiguous", is_extra?"extra_":"", row_stride_bytes, (int)F::BYTES_PER_TOKEN);
            KU_ASSERT(block_stride_bytes % F::TMA_K_STRIDE == 0, "%skv_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", block_stride_bytes, (int)F::TMA_K_STRIDE);
            KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE) <= INT32_MAX, "%skv: too many rows for the int32 TMA coordinates", is_extra?"extra_":"");
            // One 2D view per CTA: rows of the CTA's D_FP4 / CLUSTER_SIZE dims, TMA_K_STRIDE bytes apart. The box is 16 B
            // wider than the row (the extra bytes are out of bounds and zero-filled by TMA), see KVFormat::RAW_TOKEN_SMEM_STRIDE
            static_assert(F::RAW_TOKEN_DATA_BYTES % 4 == 0);
            auto make_tensor_map_for_cta = [&](uint32_t cta_idx) {
                return ku::make_tensor_map(
                    {(uint64_t)F::RAW_TOKEN_DATA_BYTES/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / F::TMA_K_STRIDE)},
                    {(uint64_t)F::TMA_K_STRIDE},
                    {F::RAW_TOKEN_SMEM_STRIDE/4, 1},    // Use UINT32 as dtype and divide the box size by 4, like the fp8 part
                    (uint8_t*)kv_ptr + cta_idx * F::RAW_TOKEN_DATA_BYTES,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                );
            };
            CUtensorMap fp4_part_tensor_map_cta1 = {};
            if constexpr (IS_2CTA) {
                fp4_part_tensor_map_cta1 = make_tensor_map_for_cta(1);
            }
            return {make_tensor_map_for_cta(0), fp4_part_tensor_map_cta1};
        };
        auto make_kv_tensor_map = [](bool is_extra, void *kv_ptr, int num_blocks, int64_t block_stride_bytes, int row_stride_bytes) -> std::tuple<CUtensorMap, CUtensorMap, CUtensorMap> {
            KU_ASSERT((int64_t)kv_ptr % 16 == 0, "The base address of %skv (%p) must be 16B aligned", is_extra?"extra_":"", kv_ptr);
            KU_ASSERT(row_stride_bytes == (int)KV_CACHE_BYTES_PER_TOKEN, "%skv_cache.stride(-2) (%d) must be %d, i.e. each page block in the KV cache must be contiguous", is_extra?"extra_":"", row_stride_bytes, (int)KV_CACHE_BYTES_PER_TOKEN);
            KU_ASSERT(block_stride_bytes % TMA_K_STRIDE == 0, "%skv_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", block_stride_bytes, (int)TMA_K_STRIDE);
            KU_ASSERT((uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE) <= INT32_MAX, "%skv: too many rows for the int32 TMA coordinates", is_extra?"extra_":"");
            static_assert(D_FP8_CTA0%4 == 0);
            auto fp8_part_tensor_map_cta0 = ku::make_tensor_map(
                {(uint64_t)D_FP8_CTA0/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                {(uint64_t)TMA_K_STRIDE},
                {RAW_FP8_TOKEN_SMEM_STRIDE_CTA0/4, 1},    // Use UINT32 as dtype and divide the box size by 4, to satisfy the requirement that TMA's box size must <= 256. Add 64 bytes to prevent bank conflict when loading from SMEM
                (uint8_t*)kv_ptr,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            );
            CUtensorMap fp8_part_tensor_map_cta1 = {};
            if constexpr (IS_2CTA) {
                static_assert(D_FP8_CTA1%4 == 0);
                fp8_part_tensor_map_cta1 = ku::make_tensor_map(
                    {(uint64_t)D_FP8_CTA1/4, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                    {(uint64_t)TMA_K_STRIDE},
                    {RAW_FP8_TOKEN_SMEM_STRIDE_CTA1/4, 1},
                    (uint8_t*)kv_ptr + D_FP8_CTA0,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                );
            }
            CUtensorMap bf16_part_tensor_map = {};
            if constexpr (D_BF16 > 0) {
                bf16_part_tensor_map = ku::make_tensor_map(
                    {(uint64_t)D_BF16, (uint64_t)num_blocks * (uint64_t)(block_stride_bytes / TMA_K_STRIDE)},
                    {(uint64_t)TMA_K_STRIDE},
                    {D_BF16, 1},
                    (uint8_t*)kv_ptr + D_FP8,
                    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                );
            }
            return {fp8_part_tensor_map_cta0, fp8_part_tensor_map_cta1, bf16_part_tensor_map};
        };
        {
            auto [fp8_part_cta0, fp8_part_cta1, bf16_part] = make_kv_tensor_map(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
            tma_params.tensor_map_kv_fp8_part_cta0 = fp8_part_cta0;
            tma_params.tensor_map_kv_fp8_part_cta1 = fp8_part_cta1;
            tma_params.tensor_map_kv_bf16_part = bf16_part;
        }
        if (params.extra_topk > 0) {
            if constexpr (HAS_FP4_KV) {
                auto [fp4_part_cta0, fp4_part_cta1] = make_fp4_kv_tensor_map(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
                tma_params.tensor_map_extra_kv_fp4_part_cta0 = fp4_part_cta0;
                tma_params.tensor_map_extra_kv_fp4_part_cta1 = fp4_part_cta1;
            } else {
                auto [fp8_part_cta0, fp8_part_cta1, bf16_part] = make_kv_tensor_map(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
                tma_params.tensor_map_extra_kv_fp8_part_cta0 = fp8_part_cta0;
                tma_params.tensor_map_extra_kv_fp8_part_cta1 = fp8_part_cta1;
                tma_params.tensor_map_extra_kv_bf16_part = bf16_part;
            }
        }
    } else {
        tma_params.tensor_map_kv = ku::make_tensor_map(
            {(uint64_t)D_QK, (uint64_t)params.s_kv},
            {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
            {64, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    auto aux_params = AuxParams {
    };  
    if constexpr (IS_DECODE) {
        aux_params.fast_divmod_page_block_size = cutlass::FastDivmod(params.page_block_size);
        aux_params.fast_divmod_extra_page_block_size = cutlass::FastDivmod(params.extra_kv != nullptr ? params.extra_page_block_size : 1);
    }

    auto kernel = &fwd_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(params.s_q * CLUSTER_SIZE, 1, 1),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params, aux_params
    ));
}


template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params) {
    using KernelType = Kernel<CONFIG>;
    KernelType::run(params);
}


}
