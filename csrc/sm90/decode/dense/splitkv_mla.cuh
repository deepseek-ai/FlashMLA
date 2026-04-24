#include <cutlass/cutlass.h>

#include "utils.h"

#include "params.h"
#include "config.h"
#include "traits.h"

using namespace cute;
using cutlass::arch::NamedBarrier;

namespace sm90 {

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    // In the layout of fragment A and fragment C during WGMMA, data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
    // You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

// Launch TMA copy for a range of KV tile
// A tile has a shape of PAGE_BLOCK_SIZE (64) x 64
template<
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TMA_K_OneTile,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void launch_kv_tiles_copy_tma(
    Tensor<Engine0, Layout0> const &gKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K), swizzled
    TMA_K_OneTile &tma_K,
    TMABarrier* barriers_K,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        auto thr_tma = tma_K.get_slice(_0{});
        Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        Tensor cur_sKV = thr_tma.partition_D(sKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        cute::copy(tma_K.with(reinterpret_cast<typename TMABarrier::ValueType &>(barriers_K[START_HEAD_DIM_TILE_IDX]), 0, cute::TMA::CacheHintSm90::EVICT_FIRST), cur_gKV, cur_sKV);
        if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
            launch_kv_tiles_copy_tma<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, sKV, tma_K, barriers_K, idx_in_warpgroup);
        }
    }
}

// Prefetch some KV tiles
// Currently this is not used because it leads to performance degradation
template<
    int START_HEAD_DIM_TILE_IDX,
    int END_HEAD_DIM_TILE_IDX,
    typename TMA_K_OneTile,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void prefetch_kv_tiles(
    Tensor<Engine0, Layout0> const &gKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    TMA_K_OneTile &tma_K,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        auto thr_tma = tma_K.get_slice(_0{});
        Tensor cur_gKV = thr_tma.partition_S(gKV)(_, _0{}, Int<START_HEAD_DIM_TILE_IDX>{});
        cute::prefetch(tma_K, cur_gKV);
        if constexpr (START_HEAD_DIM_TILE_IDX+1 < END_HEAD_DIM_TILE_IDX) {
            prefetch_kv_tiles<START_HEAD_DIM_TILE_IDX+1, END_HEAD_DIM_TILE_IDX>(gKV, tma_K, idx_in_warpgroup);
        }
    }
}

// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}


// Wait for one KV-tile to be ready, and then calculate P += Q K^T for one Q-tile (BLOCK_SIZE_Mx64) and one KV-tile (PAGE_BLOCK_SIZEx64)
// The Q-tile should be in shared memory
template<
    typename TiledMMA,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void qkt_gemm_one_tile_sQ(
    TiledMMA &tiled_mma,
    Tensor<Engine0, Layout0> const &thr_mma_sQ_tile,	// (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sKV_tile,	// (MMA, 1, 4)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    TMABarrier* barrier,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        barrier->arrive_and_expect_tx(64*64*2);
    }
    barrier->wait(cur_phase ? 1 : 0);

    warpgroup_fence_operand(rP);
    warpgroup_arrive();
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _0{}), thr_mma_sKV_tile(_, _, _0{}), rP);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _1{}), thr_mma_sKV_tile(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _2{}), thr_mma_sKV_tile(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_sQ_tile(_, _, _3{}), thr_mma_sKV_tile(_, _, _3{}), rP);
    warpgroup_commit_batch();
    warpgroup_fence_operand(rP);
}

template<
    typename TiledMMA,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void  qkt_gemm_one_tile_rQ(
    TiledMMA &tiled_mma,
    Tensor<Engine0, Layout0> const &thr_mma_rQ_tile,	// (MMA, 1, 4)
    Tensor<Engine1, Layout1> const &thr_mma_sKV_tile,	// (MMA, 1, 4)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    TMABarrier* barrier,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    if (idx_in_warpgroup == 0) {
        barrier->arrive_and_expect_tx(64*64*2);
    }
    barrier->wait(cur_phase ? 1 : 0);

    warpgroup_fence_operand(const_cast<Tensor<Engine0, Layout0> &>(thr_mma_rQ_tile));
    warpgroup_fence_operand(rP);
    warpgroup_arrive();
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _0{}), thr_mma_sKV_tile(_, _, _0{}), rP);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _1{}), thr_mma_sKV_tile(_, _, _1{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _2{}), thr_mma_sKV_tile(_, _, _2{}), rP);
    cute::gemm(tiled_mma, thr_mma_rQ_tile(_, _, _3{}), thr_mma_sKV_tile(_, _, _3{}), rP);
    warpgroup_commit_batch();
    warpgroup_fence_operand(rP);
    warpgroup_fence_operand(const_cast<Tensor<Engine0, Layout0> &>(thr_mma_rQ_tile));
}

// Pipelined TMA wait and Q K^T gemm
// In order to overlap memory copy (G->S copy for K) and computation, we divide both Q and K into tiles of shape (BLOCK_SIZE_M, 64), and (PAGE_BLOCK_SIZE, 64) respectively, and then do the computation as follows:
// - Wait for the 0-th tile to be ready using `barrier.wait()`
// - Compute Q K^T for the 0-th tile
// - Wait for the 1-st tile to be ready
// - Compute Q K^T for the 1-st tile
// ...
// This gives latter tiles more time to be ready, and thus can overlap the memory copy and computation
template<
    typename T, // Traits
    int PHASE_IDX,	// See comments in the code
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3
> 
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm(
    Tensor<Engine0, Layout0> &sQ,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (PAGE_BLOCK_SIZE, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    Tensor<Engine3, Layout3> &rQ8,	// The 8-th tile of Q. We store it separately to leave some room for storing sP1
    TMABarrier* barriers,
    bool &cur_phase,
    int idx_in_warpgroup
) {
    Tensor sQ_tiled = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, _64>{})(_, _, _0{}, _);	// (BLOCK_SIZE_M, 64, 9)
    Tensor sKV_tiled = flat_divide(sKV, Shape<Int<T::PAGE_BLOCK_SIZE>, _64>{})(_, _, _0{}, _);	// (PAGE_BLOCK_SIZE, 64, 9)
    TiledMMA tiled_mma_sQ = (typename T::TiledMMA_QK_sQ){};
    ThrMMA thr_mma_sQ = tiled_mma_sQ.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sQ_tiled = thr_mma_sQ.partition_fragment_A(sQ_tiled);	// (MMA, 1, 4, 9)
    Tensor thr_mma_sKV_tiled = thr_mma_sQ.partition_fragment_B(sKV_tiled);	// (MMA, 1, 4, 9)
    TiledMMA tiled_mma_rQ = (typename T::TiledMMA_QK_rQ){};

    #define QKT_GEMM_ONE_TILE(TILE_IDX) \
        if constexpr(TILE_IDX != 8) { \
            qkt_gemm_one_tile_sQ(tiled_mma_sQ, thr_mma_sQ_tiled(_, _, _, Int<TILE_IDX>{}), thr_mma_sKV_tiled(_, _, _, Int<TILE_IDX>{}), rP, barriers + TILE_IDX, cur_phase, idx_in_warpgroup); \
        } else { \
            qkt_gemm_one_tile_rQ(tiled_mma_rQ, rQ8, thr_mma_sKV_tiled(_, _, _, Int<TILE_IDX>{}), rP, barriers + TILE_IDX, cur_phase, idx_in_warpgroup); \
        }

    if constexpr (PHASE_IDX == 0) {
        // In PHASE-0, warpgroup 0 calculates Q K^T for the first 4 tiles
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::Zero;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
    } else if constexpr (PHASE_IDX == 1) {
        // In PHASE-1, warpgroup 1 calculates Q K^T for all the 9 tiles
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::Zero;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);
        QKT_GEMM_ONE_TILE(0);
        QKT_GEMM_ONE_TILE(1);
        QKT_GEMM_ONE_TILE(2);
        QKT_GEMM_ONE_TILE(3);
        cur_phase ^= 1;
    } else {
        // In PHASE-2, warpgroup 0 calculates Q K^T for the last 5 tiles
        static_assert(PHASE_IDX == 2);
        tiled_mma_sQ.accumulate_ = GMMA::ScaleOut::One;
        tiled_mma_rQ.accumulate_ = GMMA::ScaleOut::One;
        QKT_GEMM_ONE_TILE(4);
        QKT_GEMM_ONE_TILE(5);
        QKT_GEMM_ONE_TILE(6);
        QKT_GEMM_ONE_TILE(7);
        QKT_GEMM_ONE_TILE(8);
        cur_phase ^= 1;
    }
}


template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_qkt_gemm_no_pipeline(
    Tensor<Engine0, Layout0> &sQ,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine1, Layout1> &sKV,	// (BLOCK_SIZE_M, HEAD_DIM_K)
    Tensor<Engine2, Layout2> &rP,	// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_QK_sQ){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sQ = thr_mma.partition_fragment_A(sQ);	// (MMA, 1, 576/16=36)
    Tensor thr_mma_sKV = thr_mma.partition_fragment_B(sKV);	// (MMA, 1, 576/16=36)
    gemm<true, -1>(tiled_mma, thr_mma_sQ, thr_mma_sKV, rP);
}


// Compute O += PV, where P resides in register
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_localP(
    Tensor<Engine0, Layout0> &rP,	// ((2, 2, 8), 1, 1), fragment A layout
    Tensor<Engine1, Layout1> &sKV_half,	// (HEAD_DIM_V/2, PAGE_BLOCK_SIZE)
    Tensor<Engine2, Layout2> &rO,	// ((2, 2, 32), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_PV_LocalP){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor rP_retiled = make_tensor(rP.data(), Layout<
        Shape<Shape<_2, _2, _2>, _1, _4>,
        Stride<Stride<_1, _2, _4>, _0, _8>
    >{});
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half);	// (MMA, 1, 64/16=4)
    gemm<false, -1>(tiled_mma, rP_retiled, thr_mma_sKV_half, rO);
}


// Compute O += PV, where P resides in shared memory
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
> 
__forceinline__ __device__ void warpgroup_cooperative_pv_gemm_remoteP(
    Tensor<Engine0, Layout0> &sP,
    Tensor<Engine1, Layout1> &sKV_half,	// (HEAD_DIM_V/2, PAGE_BLOCK_SIZE)
    Tensor<Engine2, Layout2> &rO,	// ((2, 2, 32), 1, 1)
    int idx_in_warpgroup
) {
    TiledMMA tiled_mma = (typename T::TiledMMA_PV_RemoteP){};
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor thr_mma_sP = thr_mma.partition_fragment_A(sP);
    Tensor thr_mma_sKV_half = thr_mma.partition_fragment_B(sKV_half);	// (MMA, 1, 64/16=4)
    gemm<false, -1>(tiled_mma, thr_mma_sP, thr_mma_sKV_half, rO);
}


template<
    typename T,
    bool DO_OOB_FILLING,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4
>
__forceinline__ __device__ void wg0_bunch_0(
    Tensor<Engine0, Layout0> &rPb,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &rP0,	// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rO0,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine4, Layout4> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup
) {
    // This piece of code is tightly coupled [Accumulate's layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png)
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            if constexpr (DO_OOB_FILLING) {
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP0(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP0(i) : MAX_INIT_VAL;
                rP0(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP0(i+1) : MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP0(i), rP0(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        
        // Update sM and sL
        cur_max *= scale_softmax_log2;
        float new_max = max(sM(row_idx), cur_max);
        float scale_for_old = exp2f(sM(row_idx) - new_max);
        __syncwarp();   // Make sure all reads have finished before updating sM
        if (idx_in_warpgroup%4 == 0) {
            sScale0(row_idx) = scale_for_old;
            sM(row_idx) = new_max;
        }
        
        // Scale-O
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_for_old;
            rO0(i+1) *= scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rP0(i) = exp2f(rP0(i)*scale_softmax_log2 - new_max);
            rP0(i+1) = exp2f(rP0(i+1)*scale_softmax_log2 - new_max);
            rPb(i) = (typename T::InputT)rP0(i);
            rPb(i+1) = (typename T::InputT)rP0(i+1);
            cur_sum += rP0(i) + rP0(i+1);
        }
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
}


template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5
>
__forceinline__ __device__ void wg1_bunch_0(
    Tensor<Engine0, Layout0> &rP1b,	// ((2, 2, 8), 1, 1)
    Tensor<Engine1, Layout1> &sScale1,	// (BLOCK_SIZE_M)
    Tensor<Engine2, Layout2> &rO1,	// ((2, 2, 32), 1, 1)
    Tensor<Engine3, Layout3> &sM,	// (BLOCK_SIZE_M)
    float rL[2],
    int rRightBorderForQSeq[2],
    Tensor<Engine4, Layout4> const &sScale0,	// (BLOCK_SIZE_M)
    Tensor<Engine5, Layout5> &rP1,	// ((2, 2, 8), 1, 1)
    float scale_softmax_log2,
    int start_token_idx,
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);

        // Mask, and get row-wise max
        float cur_max = MAX_INIT_VAL;
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
            if constexpr (IS_BLK1_LAST || IS_BLK2_LAST) {
                // Need to apply the mask when either this block is the last one, or
                // the next block is the last one (because of the causal mask)
                int token_idx = start_token_idx + (i/4)*8 + idx_in_warpgroup%4*2;
                rP1(i) = token_idx < rRightBorderForQSeq[local_row_idx] ? rP1(i) : MAX_INIT_VAL;
                rP1(i+1) = token_idx+1 < rRightBorderForQSeq[local_row_idx] ? rP1(i+1) : MAX_INIT_VAL;
            } else if constexpr (IS_BLK0_LAST) {
                rP1(i) = rP1(i+1) = MAX_INIT_VAL;
            }
            cur_max = max(cur_max, max(rP1(i), rP1(i+1)));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
        cur_max *= scale_softmax_log2;

        float old_max = sM(row_idx);
        float new_max = max(old_max, cur_max);
        float scale_for_old = exp2f(old_max - new_max);
        __syncwarp();
        if (idx_in_warpgroup%4 == 0) {
            sM(row_idx) = new_max;
            sScale1(row_idx) = scale_for_old;
        }

        // Scale, exp, and get row-wise expsum
        float cur_sum = 0;
        if constexpr (!IS_BLK0_LAST) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = local_row_idx ? 2 : 0; i < size(rP1); i += 4) {
                rP1(i) = exp2f(rP1(i)*scale_softmax_log2 - new_max);
                rP1(i+1) = exp2f(rP1(i+1)*scale_softmax_log2 - new_max);
                rP1b(i) = (typename T::InputT)rP1(i);
                rP1b(i+1) = (typename T::InputT)rP1(i+1);
                cur_sum += rP1(i) + rP1(i+1);
            }
        }

        // Scale O
        float cur_scale_for_o1 = scale_for_old * sScale0(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO1); i += 4) {
            rO1(i) *= cur_scale_for_o1;
            rO1(i+1) *= cur_scale_for_o1;
        }

        // Update rL
        rL[local_row_idx] = rL[local_row_idx]*cur_scale_for_o1 + cur_sum;
    }
}


// Save rPb (64x64, bfloat16/half) to sP using the stmatrix instruction
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void save_rPb_to_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> &sP,
    int idx_in_warpgroup
) {
    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, typename T::InputT>{},
        (typename T::TiledMMA_QK_sQ){}
    );
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}


// Retrieve rPb (64x64, bfloat16/half) from sP using the ldmatrix instruction
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void retrieve_rP_from_sP(
    Tensor<Engine0, Layout0> &rPb,
    Tensor<Engine1, Layout1> const &sP,
    int idx_in_warpgroup
) {
    TiledCopy s2r_copy = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, typename T::InputT>{},
        (typename T::TiledMMA_PV_LocalP){}
    );
    ThrCopy thr_copy = s2r_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_sP = thr_copy.partition_S(sP);
    Tensor thr_copy_rPb = thr_copy.retile_D(rPb);
    cute::copy(s2r_copy, thr_copy_sP, thr_copy_rPb);
}


// Rescale rP0 and save the result to rPb
template<
    typename T,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void wg0_scale_rP0(
    Tensor<Engine0, Layout0> const &sScale1,	// (BLOCK_M)
    Tensor<Engine1, Layout1> const &rP0,		// ((2, 2, 8), 1, 1)
    Tensor<Engine2, Layout2> &rPb,		// ((2, 2, 8), 1, 1)
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rP0); i += 4) {
            rPb(i) = (typename T::InputT)(rP0(i)*scale_factor);
            rPb(i+1) = (typename T::InputT)(rP0(i+1)*scale_factor);
        }
    }
}


// Rescale rO0 according to sScale1
template<
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void wg0_rescale_rO0(
    Tensor<Engine0, Layout0> &rO0,
    Tensor<Engine1, Layout1> &sScale1,
    float rL[2],
    int idx_in_warpgroup
) {
    CUTLASS_PRAGMA_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
        float scale_factor = sScale1(row_idx);
        CUTLASS_PRAGMA_UNROLL
        for (int i = local_row_idx ? 2 : 0; i < size(rO0); i += 4) {
            rO0(i) *= scale_factor;
            rO0(i+1) *= scale_factor;
        }
        rL[local_row_idx] *= scale_factor;
    }
}


// Fill out-of-bound V with 0.0
// We must fill it since it may contain NaN, which may propagate to the final result
template<
    typename T,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ void fill_oob_V(
    Tensor<Engine0, Layout0> &sV,	// tile_to_shape(GMMA::Layout_MN_SW128_Atom<InputT>{}, Shape<Int<HALF_HEAD_DIM>, Int<T::PAGE_BLOCK_SIZE>>{}, LayoutRight{} );
    int valid_window_size,
    int idx_in_warpgroup
) {
    Tensor sV_int64 = make_tensor(
        make_smem_ptr((int64_t*)(sV.data().get().get())),
        tile_to_shape(
            GMMA::Layout_MN_SW128_Atom<cute::int64_t>{},
            Shape<Int<256/(64/16)>, Int<T::PAGE_BLOCK_SIZE>>{},
            LayoutRight{}
        )
    );
    valid_window_size = max(valid_window_size, 0);
    int head_dim_size = size<0>(sV_int64);	// 128%head_dim_size == 0 should holds
    for (int token_idx = valid_window_size + (idx_in_warpgroup/head_dim_size); token_idx < size<1>(sV); token_idx += (128/head_dim_size)) {
        sV_int64(idx_in_warpgroup%head_dim_size, token_idx) = 0;
    }
}


// Store O / OAccum
template<
    typename T,
    bool IS_NO_SPLIT,
    typename TMAParams,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void store_o(
    Tensor<Engine0, Layout0> &rO,	// ((2, 2, 32), 1, 1)
    Tensor<Engine1, Layout1> &gOorAccum,	// (BLOCK_SIZE_M, HEAD_DIM_V)
    float rL[2],
    char* sO_addr,
    TMAParams &tma_params,
    int batch_idx,
    int k_head_idx,
    int m_block_idx,
    int num_valid_seq_q,
    int warpgroup_idx,
    int idx_in_warpgroup
) {
    using InputT = typename T::InputT;
    if constexpr (IS_NO_SPLIT) {
        // Should convert the output to bfloat16 / float16, and save it to O
        Tensor sOutputBuf = make_tensor(make_smem_ptr((InputT*)sO_addr), tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}
        ));

        Tensor rOb = make_tensor_like<InputT>(rO);
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < size(rO); ++idx) {
            rOb(idx) = (InputT)(rO(idx) / rL[idx%4 >= 2]);
        }

        Tensor sMyOutputBuf = local_tile(sOutputBuf, Shape<_64, _256>{}, make_coord(_0{}, warpgroup_idx));
        TiledCopy r2s_tiled_copy = make_tiled_copy_C(
            Copy_Atom<SM90_U32x4_STSM_N, InputT>{},
            (typename T::TiledMMA_PV_LocalP){}
        );
        ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);
        Tensor r2s_thr_copy_rOb = r2s_thr_copy.retile_S(rOb);
        Tensor r2s_thr_copy_sMyOutputBuf = r2s_thr_copy.partition_D(sMyOutputBuf);
        cute::copy(r2s_tiled_copy, r2s_thr_copy_rOb, r2s_thr_copy_sMyOutputBuf);
        cutlass::arch::fence_view_async_shared();
        
        __syncthreads();

        if (threadIdx.x == 0) {
            Tensor tma_gO = tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, k_head_idx, batch_idx);	// (seqlen_q, HEAD_DIM)
            auto thr_tma = tma_params.tma_O.get_slice(_0{});
            Tensor my_tma_gO = flat_divide(tma_gO, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{})(_, _, m_block_idx, _0{});
            cute::copy(
                tma_params.tma_O,
                thr_tma.partition_S(sOutputBuf),
                thr_tma.partition_D(my_tma_gO)
            );
            cute::tma_store_arrive();
        }
    } else {
        // Should save the result to OAccum
        Tensor sOutputBuf = make_tensor(make_smem_ptr((float*)sO_addr), Layout<
            Shape<_64, _512>,
            Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
        >{});
    
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < size(rO); idx += 2) {
            int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (idx%4 >= 2 ? 8 : 0);
            int col = warpgroup_idx*256 + (idx_in_warpgroup%4)*2 + idx/4*8;
            *(float2*)((float*)sO_addr + sOutputBuf.layout()(row, col)) = float2 {
                rO(idx) / rL[idx%4 >= 2],
                rO(idx+1) / rL[idx%4 >= 2],
            };
        }
        cutlass::arch::fence_view_async_shared();
        
        __syncthreads();
        
        int row = threadIdx.x;
        if (row < num_valid_seq_q) {
            SM90_BULK_COPY_S2G::copy(&sOutputBuf(row, _0{}), &gOorAccum(row, _0{}), T::HEAD_DIM_V*sizeof(float));
            cute::tma_store_arrive();
        }
    }
}

template<
    typename T,
    typename TmaParams, typename Tensor0
>
__forceinline__ __device__ void launch_q_copy(
    TmaParams const &tma_params,
    int batch_idx,
    int m_block_idx,
    int k_head_idx,
    Tensor0 &sQ,
    TMABarrier* barrier_Q
) {
    if (threadIdx.x == 0) {
        Tensor tma_gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, k_head_idx, batch_idx);	// (seqlen_q, HEAD_DIM)
        auto thr_tma = tma_params.tma_Q.get_slice(_0{});
        Tensor my_tma_gQ = flat_divide(tma_gQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_K>>{})(_, _, m_block_idx, _0{});
        cute::copy(
            tma_params.tma_Q.with(reinterpret_cast<typename TMABarrier::ValueType &>(*barrier_Q), 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
            thr_tma.partition_S(my_tma_gQ),
            thr_tma.partition_D(sQ)
        );
        barrier_Q->arrive_and_expect_tx(64*576*2);
    }
}

template<
    typename T,
    bool IS_R,
    typename Engine0, typename Layout0
>
__forceinline__ __device__ auto get_half_V(
    Tensor<Engine0, Layout0> &sK
) {
    Tensor sV = make_tensor(sK.data(), (typename T::SmemLayoutV){});
    return flat_divide(sV, Shape<Int<T::HEAD_DIM_V/2>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, Int<(int)IS_R>{}, _0{});
}

template<
    typename T,
    bool IS_BLK0_LAST,	// "BLK0" means block_idx+0, "BLK1" means block_idx+1, ...
    bool IS_BLK1_LAST,
    typename TMAParams,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11
>
__forceinline__ __device__ void wg0_subroutine(
    Tensor<Engine0, Layout0> &tma_gK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> &sK0,
    Tensor<Engine3, Layout3> &sK1,
    Tensor<Engine4, Layout4> &sP0,
    Tensor<Engine5, Layout5> &sP1,
    Tensor<Engine6, Layout6> &sM,
    Tensor<Engine7, Layout7> &sScale0,
    Tensor<Engine8, Layout8> &sScale1,
    Tensor<Engine9, Layout9> &rQ8,
    Tensor<Engine10, Layout10> &rP0,
    Tensor<Engine11, Layout11> &rO0,
    float rL[2],
    int rRightBorderForQSeq[2],
    TMABarrier barriers_K0[9],
    TMABarrier barriers_K1[9],
    bool &cur_phase_K0,
    const TMAParams &tma_params,
    const DenseAttnDecodeParams &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup
) {
    int start_token_idx = block_idx * T::PAGE_BLOCK_SIZE;
    #define GET_BLOCK_INDEX(block_idx) ((block_idx) >= end_block_idx ? 0 : __ldg(block_table_ptr + (block_idx)))
    int nxt_block0_index = GET_BLOCK_INDEX(block_idx+2);
    int nxt_block1_index = GET_BLOCK_INDEX(block_idx+3);

    Tensor sV0L = get_half_V<T, 0>(sK0);
    Tensor sV1L = get_half_V<T, 0>(sK1);

    Tensor rPb = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    // Calc P0 = softmax(P0)
    wg0_bunch_0<T, IS_BLK0_LAST||IS_BLK1_LAST>(rPb, rP0, rO0, sScale0, sM, rL, rRightBorderForQSeq, params.scale_softmax_log2, start_token_idx, idx_in_warpgroup);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale0Ready);

    // Issue rO0 += rPb @ sV0L
    if constexpr (IS_BLK0_LAST) {
        fill_oob_V<T>(sV0L, seqlen_k-start_token_idx, idx_in_warpgroup);
        cutlass::arch::fence_view_async_shared();
    }
    warpgroup_cooperative_pv_gemm_localP<T>(rPb, sV0L, rO0, idx_in_warpgroup);

    // Wait for rO0, launch TMA for the next V0L
    cute::warpgroup_wait<0>();
    
    // Wait for warpgroup 1, rescale P0, notify warpgroup 1
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale1Ready);
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        // Put it here seems to be faster, don't know why
        launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, nxt_block0_index), sK0, tma_params.tma_K, barriers_K0, idx_in_warpgroup);
    }
    wg0_scale_rP0<T>(sScale1, rP0, rPb, idx_in_warpgroup);
    save_rPb_to_sP<T>(rPb, sP0, idx_in_warpgroup);
    cutlass::arch::fence_view_async_shared();
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sP0Ready);
    
    // Wait for warpgroup 1, rescale O0, issue rO0 += rPb @ sV1L
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (IS_BLK1_LAST) {
            fill_oob_V<T>(sV1L, seqlen_k-start_token_idx-T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
            cutlass::arch::fence_view_async_shared();
        }
        NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
        wg0_rescale_rO0(rO0, sScale1, rL, idx_in_warpgroup);
        warpgroup_cooperative_pv_gemm_remoteP<T>(sP1, sV1L, rO0, idx_in_warpgroup);
    }
    
    // Issue P0 = Q @ K0^T
    // Since TMAs for these 4 tiles are launched right after rO0 += rPb @ sV0L finishes, they should have already finished. Therefore, we issue the first 4 tiles to fill the pipeline.
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 0>(sQ, sK0, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup);
    }

    // Wait for rO0 += rPb @ sV1L, launch TMA
    if (!IS_BLK0_LAST && !IS_BLK1_LAST && __builtin_expect(block_idx+3 < end_block_idx, true)) {
        cute::warpgroup_wait<4>();
        launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, nxt_block1_index), sK1, tma_params.tma_K, barriers_K1, idx_in_warpgroup);
    }
    
    // Issue P0 = Q @ K0^T
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        warpgroup_cooperative_qkt_gemm<T, 2>(sQ, sK0, rP0, rQ8, barriers_K0, cur_phase_K0, idx_in_warpgroup);
    }
    
    // Wait for P0 = Q @ K0^T
    cute::warpgroup_wait<0>();
}


template<
    typename T,
    bool IS_BLK0_LAST,
    bool IS_BLK1_LAST,
    bool IS_BLK2_LAST,
    typename TMAParams,
    typename Engine0, typename Layout0,
    typename Engine1, typename Layout1,
    typename Engine2, typename Layout2,
    typename Engine3, typename Layout3,
    typename Engine4, typename Layout4,
    typename Engine5, typename Layout5,
    typename Engine6, typename Layout6,
    typename Engine7, typename Layout7,
    typename Engine8, typename Layout8,
    typename Engine9, typename Layout9,
    typename Engine10, typename Layout10,
    typename Engine11, typename Layout11
>
__forceinline__ __device__ void wg1_subroutine(
    Tensor<Engine0, Layout0> &tma_gK,
    Tensor<Engine1, Layout1> &sQ,
    Tensor<Engine2, Layout2> &sK0,
    Tensor<Engine3, Layout3> &sK1,
    Tensor<Engine4, Layout4> &sP0,
    Tensor<Engine5, Layout5> &sP1,
    Tensor<Engine6, Layout6> &sM,
    Tensor<Engine7, Layout7> &sScale0,
    Tensor<Engine8, Layout8> &sScale1,
    Tensor<Engine9, Layout9> &rQ8,
    Tensor<Engine10, Layout10> &rP1,
    Tensor<Engine11, Layout11> &rO1,
    float rL[2],
    int rRightBorderForQSeq[2],
    TMABarrier barriers_K0[9],
    TMABarrier barriers_K1[9],
    bool &cur_phase_K1,
    const TMAParams &tma_params,
    const DenseAttnDecodeParams &params,
    int* block_table_ptr,
    int seqlen_k,
    int block_idx,
    int end_block_idx,
    int idx_in_warpgroup
) {
    int start_token_idx = block_idx * T::PAGE_BLOCK_SIZE;
    int nxt_block0_index = GET_BLOCK_INDEX(block_idx+2);
    int nxt_block1_index = GET_BLOCK_INDEX(block_idx+3);

    Tensor rP1b = make_tensor<T::InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
    
    Tensor sV0R = get_half_V<T, 1>(sK0);
    Tensor sV1R = get_half_V<T, 1>(sK1);

    // Wait for rP1 and warpgroup 0, run bunch 1, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sScale0Ready);
    wg1_bunch_0<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>(rP1b, sScale1, rO1, sM, rL, rRightBorderForQSeq, sScale0, rP1, params.scale_softmax_log2, start_token_idx+T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
    NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::sScale1Ready);

    // Save rPb to sP, and issue rO1 += rP1b @ sV1R
    // We do this after notifying warpgroup 1, since both "saving rPb to sP" and "issuing" WGMMA are high-latency operations
    if constexpr (!IS_BLK0_LAST) {
        save_rPb_to_sP<T>(rP1b, sP1, idx_in_warpgroup);
    }
    if constexpr (!IS_BLK0_LAST) {
        if constexpr (IS_BLK1_LAST) {
            fill_oob_V<T>(sV1R, seqlen_k-start_token_idx-T::PAGE_BLOCK_SIZE, idx_in_warpgroup);
            cutlass::arch::fence_view_async_shared();
        }
        warpgroup_cooperative_pv_gemm_localP<T>(rP1b, sV1R, rO1, idx_in_warpgroup);
        if constexpr (!IS_BLK1_LAST) {
            // We use this proxy for making sP1 visible to the async proxy
            // We skip it if IS_BLK1_LAST, since in that case we have already put a fence
            cutlass::arch::fence_view_async_shared();
        }
    }
    
    // Wait for sP0, issue rO1 += sP0 @ sV0R, notify warpgroup 0
    NamedBarrier::arrive_and_wait(T::NUM_THREADS, NamedBarriers::sP0Ready);
    if constexpr (IS_BLK0_LAST) {
        fill_oob_V<T>(sV0R, seqlen_k-start_token_idx, idx_in_warpgroup);
        cutlass::arch::fence_view_async_shared();
    }
    warpgroup_cooperative_pv_gemm_remoteP<T>(sP0, sV0R, rO1, idx_in_warpgroup);
    if constexpr (!IS_BLK0_LAST) {
        NamedBarrier::arrive(T::NUM_THREADS, NamedBarriers::rO1sP0sV0RIssued);
    }
    
    // Wait for rO1 += rP1b @ sV1R, launch TMA for the next V1R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST) {
        cute::warpgroup_wait<1>();
        launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, nxt_block1_index), sK1, tma_params.tma_K, barriers_K1, idx_in_warpgroup);
    }
    
    // Wait for rO1 += sP0 @ sV0R, launch TMA for the next V0R
    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST) {
        cute::warpgroup_wait<0>();
        launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, nxt_block0_index), sK0, tma_params.tma_K, barriers_K0, idx_in_warpgroup);
    }

    if constexpr (!IS_BLK0_LAST && !IS_BLK1_LAST && !IS_BLK2_LAST) {
        // Issue rP1 = sQ @ sK1, wait
        warpgroup_cooperative_qkt_gemm<T, 1>(sQ, sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup);
    }
    
    // We put the `cute::warpgroup_wait<0>()` out of the `if` statement above, otherwise
    // nvcc cannot correctly analyse the loop, and will think that we are using accumulator
    // registers during the WGMMA pipeline, which results in `WARPGROUP.ARRIVE` and `WARPGROUP.DEPBAR.LE` being inserted in SASS and WGMMA instructions being serialized.
    // This is also the reason why we put QK^T here, instead of the first operation in the loop
    cute::warpgroup_wait<0>();
}

// A helper function for determining the length of the causal mask for one q token
__forceinline__ __device__ int get_mask_len(const DenseAttnDecodeParams &params, int m_block_idx, int local_seq_q_idx) {
    int global_seq_q_idx = m_block_idx*Config::BLOCK_SIZE_M + local_seq_q_idx;
    if (global_seq_q_idx < params.q_seq_per_hk) {
        int s_q_idx = global_seq_q_idx / params.q_head_per_hk;
        return params.s_q - s_q_idx - 1;
    } else {
        // Out-of-bound request, regard as no masks
        return 0;
    }
}

template<typename T, typename TmaParams>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const DenseAttnDecodeParams params, __grid_constant__ const TmaParams tma_params) {
    // grid shape: [
    // 	num_m_blocks (=ceil_div(seqlen_q_ori*(num_q_heads//num_kv_heads))),
    // 	num_kv_heads,
    // 	num_sm_parts
    // ]
    // An "sm part" is responsible for all the BLOCK_SIZE_M q_heads in the m_block (as specified by m_block_idx), under one kv head (as specified by k_head_idx), of a segment (as specified by [start_block_idx, end_block_idx]) of one request (as specified by batch_idx).
    // If is_no_split is True, then this request is exclusively assigned to this sm_part, so we shall write the result directly into params.o_ptr and params.softmax_lse_ptr. Otherwise, write to oaccum_ptr and softmax_lseaccum_ptr, with the corresponding split idx being (n_split_idx + num_splits_ptr[batch_idx])
    // For the complete schedule of the kernel, please read our deep-dive write-up (link can be found in the README.md file).

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = threadIdx.x / 128;
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_sQ.data()), (typename T::SmemLayoutQ){});
    Tensor sK0 = make_tensor(make_smem_ptr(plan.smem_sK0.data()), (typename T::SmemLayoutK){});
    Tensor sK1 = make_tensor(make_smem_ptr(plan.smem_sK1.data()), (typename T::SmemLayoutK){});
    Tensor sP0 = make_tensor(make_smem_ptr(plan.smem_sP0.data()), (typename T::SmemLayoutP0){});
    Tensor sP1 = flat_divide(sQ, Shape<Int<T::BLOCK_SIZE_M>, Int<T::PAGE_BLOCK_SIZE>>{})(_, _, _0{}, _8{}); // Overlap with sQ's 8-th tile
    Tensor sM = make_tensor(make_smem_ptr(plan.smem_sM.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sL_reduction_wksp = make_tensor(make_smem_ptr(plan.sL_reduction_wksp.data()), make_shape(Int<2*T::BLOCK_SIZE_M>{}));
    Tensor sScale0 = make_tensor(make_smem_ptr(plan.smem_sScale0.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    Tensor sScale1 = make_tensor(make_smem_ptr(plan.smem_sScale1.data()), make_shape(Int<T::BLOCK_SIZE_M>{}));
    char* sO_addr = (char*)plan.smem_sK0.data();	// Overlap with sK0 and sK1
    
    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
    }

    // Define TMA stuffs
    Tensor tma_gK = tma_params.tma_K.get_tma_tensor(tma_params.shape_K)(_, _, k_head_idx, _);
    TMABarrier* barriers_K0 = plan.barriers_K0;
    TMABarrier* barriers_K1 = plan.barriers_K1;
    TMABarrier* barrier_Q = &(plan.barrier_Q);

    // Initialize TMA barriers
    if (threadIdx.x == 0) {
        barrier_Q->init(1);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 9; ++i) {
            barriers_K0[i].init(1);
            barriers_K1[i].init(1);
        }
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    bool cur_phase_Q = 0, cur_phase_K0 = 0, cur_phase_K1 = 0;

    
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    // Copy the first Q
    launch_q_copy<T>(tma_params, sched_meta.begin_req_idx, m_block_idx, k_head_idx, sQ, barrier_Q);

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : cute::ceil_div(seqlen_k, kBlockN);
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        
        int rRightBorderForQSeq[2];
        if (params.is_causal) {
            // The causal mask looks like:
            // XXXX
            // XXXX
            // ...
            // XXXX
            //  XXX
            //  XXX
            //  ...
            //  XXX
            //   XX
            //   XX
            //  ...
            //   XX
            // Firstly, there is a common_mask_len, which is the minimum length of causal masks among all tokens. Since the length of the causal mask decreases monotonically, the common_mask_len is the length of the causal mask for the last token. We consider the common_mask_len as a "reduction in the length of the k-sequence.", and adjust end_block_idx based on it, to save some calculation.
            // Besides, a token may have some extra masks other than the common mask. We use rRightBorderForQSeq to denote it, which means the right border of the k-sequence for the particular q token. In this way, (seqlen_k-common_mask_len) - rRightBorderForQSeq < 64 holds, which means that we only need to apply the causal mask to the last two KV blocks
            // NOTE This may lead to start_block_idx >= end_block_idx which needs some special handling
            int common_mask_len = get_mask_len(params, m_block_idx, T::BLOCK_SIZE_M-1);
            int last_block_in_seq = cute::ceil_div(seqlen_k-common_mask_len, kBlockN);
            end_block_idx = batch_idx == sched_meta.end_req_idx ? min(sched_meta.end_block_idx, last_block_in_seq) : last_block_in_seq;

            CUTLASS_PRAGMA_UNROLL
            for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                int row_idx = get_AorC_row_idx(local_row_idx, idx_in_warpgroup);
                rRightBorderForQSeq[local_row_idx] = min(seqlen_k-get_mask_len(params, m_block_idx, row_idx), end_block_idx*T::PAGE_BLOCK_SIZE);
            }
        } else {
            rRightBorderForQSeq[0] = rRightBorderForQSeq[1] = seqlen_k;
        }

        // Define global tensors
        using InputT = typename T::InputT;
        InputT* o_ptr = (InputT*)params.o_ptr + batch_idx*params.o_batch_stride + m_block_idx*T::BLOCK_SIZE_M*params.o_row_stride + k_head_idx*params.o_head_stride;	// (BLOCK_SIZE_M, HEAD_DIM_V) : (params.o_row_stride, 1)
        float* softmax_lse_ptr = (float*)params.softmax_lse_ptr + (batch_idx*params.h_k + k_head_idx)*params.q_seq_per_hk + m_block_idx*T::BLOCK_SIZE_M;	// (BLOCK_SIZE_M) : (1)
        int* block_table_ptr = params.block_table + batch_idx*params.block_table_batch_stride;	// (/) : (1)
        
        Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
            Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{},
            make_stride(params.o_row_stride, _1{})
        ));
        Tensor gSoftmaxLse = make_tensor(make_gmem_ptr(softmax_lse_ptr), Layout<
            Shape<Int<T::BLOCK_SIZE_M>>,
            Stride<_1>
        >{});

        // Copy K0 and K1
        launch_kv_tiles_copy_tma<0, 9>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx)), sK0, tma_params.tma_K, barriers_K0, threadIdx.x);
        if (start_block_idx+1 < end_block_idx) {
            launch_kv_tiles_copy_tma<4, 9>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx+1)), sK1, tma_params.tma_K, barriers_K1, threadIdx.x);
            launch_kv_tiles_copy_tma<0, 4>(tma_gK(_, _, __ldg(block_table_ptr + start_block_idx+1)), sK1, tma_params.tma_K, barriers_K1, threadIdx.x);
        }

        Tensor rO = partition_fragment_C((typename T::TiledMMA_PV_LocalP){}, Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V / 2>>{});	// ((2, 2, 32), 1, 1)
        float rL[2];
        rL[0] = rL[1] = 0.0f;
        
        // Clear buffers
        cute::fill(rO, 0.);
        if (threadIdx.x < size(sM)) {
            sM[threadIdx.x] = MAX_INIT_VAL_SM;
        }

        // Wait for Q
        barrier_Q->wait(cur_phase_Q);
        cur_phase_Q ^= 1;

        Tensor rQ8 = make_tensor<InputT>(Shape<Shape<_2, _2, _2>, _1, _4>{});
        retrieve_rP_from_sP<T>(rQ8, local_tile(sQ, Shape<_64, _64>{}, Coord<_0, _8>{}), idx_in_warpgroup);

        if (warpgroup_idx == 0) {
            // Warpgroup 0
            Tensor rP0 = make_tensor<float>((typename T::rP0Layout){});
            
            // NOTE We don't use the pipelined version of Q K^T here since it leads
            // to a slow-down (or even register spilling, thanks to the great NVCC)
            // Wait for K0
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 9; ++i) {
                if (idx_in_warpgroup == 0)
                    barriers_K0[i].arrive_and_expect_tx(64*64*2);
                barriers_K0[i].wait(cur_phase_K0);
            }
            cur_phase_K0 ^= 1;
            
            // Issue P0 = Q @ K0^T, wait
            if (start_block_idx-16777216 < end_block_idx) {     // NOTE We use this `if` to prevent register spilling
                warpgroup_cooperative_qkt_gemm_no_pipeline<T>(sQ, sK0, rP0, idx_in_warpgroup);
            }
            // We add a barrier here, making sure that previous writes to sM are visible to warpgroup 0
            NamedBarrier::arrive_and_wait(128, NamedBarriers::sMInitialized);
            cute::warpgroup_wait<0>();

            #define LAUNCH_WG0_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST) \
                wg0_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST>( \
                    tma_gK, sQ, sK0, sK1, sP0, sP1, sM, sScale0, sScale1, \
                    rQ8, rP0, rO, rL, rRightBorderForQSeq, \
                    barriers_K0, barriers_K1, cur_phase_K0, \
                    tma_params, params, \
                    block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup \
                );

            int block_idx = start_block_idx;
            #pragma unroll 1
            for (; block_idx < end_block_idx-2; block_idx += 2) {
                LAUNCH_WG0_SUBROUTINE(false, false);
            }

            if (block_idx+1 < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(false, true);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG0_SUBROUTINE(true, false);
            }

        } else {
            // Warpgroup 1
            Tensor rP1 = make_tensor<float>((typename T::rP0Layout){});
            
            if (start_block_idx+1 < end_block_idx) {
                // Issue rP1 = sQ @ sK1, wait
                warpgroup_cooperative_qkt_gemm<T, 1>(sQ, sK1, rP1, rQ8, barriers_K1, cur_phase_K1, idx_in_warpgroup);
                cute::warpgroup_wait<0>();
            }

            #define LAUNCH_WG1_SUBROUTINE(IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST) \
                wg1_subroutine<T, IS_BLK0_LAST, IS_BLK1_LAST, IS_BLK2_LAST>( \
                    tma_gK, sQ, sK0, sK1, sP0, sP1, sM, sScale0, sScale1, \
                    rQ8, rP1, rO, rL, rRightBorderForQSeq, \
                    barriers_K0, barriers_K1, cur_phase_K1, \
                    tma_params, params, \
                    block_table_ptr, seqlen_k, block_idx, end_block_idx, idx_in_warpgroup \
                );

            int block_idx = start_block_idx;
            #pragma unroll 1
            for (; block_idx < end_block_idx-3; block_idx += 2) {
                LAUNCH_WG1_SUBROUTINE(false, false, false);
            }

            if (block_idx+2 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, false, true);
                block_idx += 2;
                LAUNCH_WG1_SUBROUTINE(true, false, false);
            } else if (block_idx+1 < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(false, true, false);
            } else if (block_idx < end_block_idx) {
                LAUNCH_WG1_SUBROUTINE(true, false, false);
            }
        }

        // Reduce rL across threads within the same warp
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        // Reduce rL across warpgroups
        int my_row = get_AorC_row_idx(0, idx_in_warpgroup);
        if (idx_in_warpgroup%4 == 0) {
            sL_reduction_wksp[my_row + warpgroup_idx*64] = rL[0];
            sL_reduction_wksp[my_row + 8 + warpgroup_idx*64] = rL[1];
        }
        __syncthreads();
        if (warpgroup_idx == 0) {
            rL[0] += sL_reduction_wksp[my_row + 64];
            rL[1] += sL_reduction_wksp[my_row + 8 + 64];
        } else {
            if (idx_in_warpgroup%4 == 0) {
                sL_reduction_wksp[my_row] += rL[0];
                sL_reduction_wksp[my_row + 8] += rL[1];
            }
            __syncwarp();
            rL[0] = sL_reduction_wksp[my_row];
            rL[1] = sL_reduction_wksp[my_row+8];
        }

        // Prune out when rL is 0.0f or NaN
        // rL may be 0.0f if there are large values (~10^12) in QK^T, which leads
        // to exp2f(P(i)*scale-max) = 0.0f or +inf due to FMA error.
        // When this happens, we set rL to 1.0f. This aligns with the old version
        // of the MLA kernel.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i)
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];

        // Copy Q for the next batch
        if (batch_idx+1 <= sched_meta.end_req_idx) {
            launch_q_copy<T>(tma_params, batch_idx+1, m_block_idx, k_head_idx, sQ, barrier_Q);
        } else {
            // Allow the next kernel (the combine kernel) to launch
            // The next kernel MUST be the combine kernel
            cudaTriggerProgrammaticLaunchCompletion();
        }

        int num_valid_seq_q = min(params.q_seq_per_hk - m_block_idx*T::BLOCK_SIZE_M, T::BLOCK_SIZE_M);
        if (is_no_split) {
            store_o<T, true>(rO, gO, rL, sO_addr, tma_params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                gSoftmaxLse(i) = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + sM(i) / (float)M_LOG2E;
            }

            cute::tma_store_wait<0>();
        } else {
            // Don't use __ldg because of PDL and instruction reordering
            int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;
            float* oaccum_ptr = (float*)params.oaccum_ptr + ((split_idx*params.h_k + k_head_idx)*params.q_seq_per_hk + m_block_idx*T::BLOCK_SIZE_M)*T::HEAD_DIM_V;	// (BLOCK_SIZE_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
            float* softmax_lseaccum_ptr = (float*)params.softmax_lseaccum_ptr + (split_idx*params.h_k + k_head_idx)*params.q_seq_per_hk + m_block_idx*T::BLOCK_SIZE_M;	// (BLOCK_SIZE_M) : (1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>,
                Stride<Int<T::HEAD_DIM_V>, _1>
            >{});
            Tensor gSoftmaxLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr), Layout<
                Shape<Int<T::BLOCK_SIZE_M>>,
                Stride<_1>
            >{});
            store_o<T, false>(rO, gOAccum, rL, sO_addr, tma_params, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

            int i = threadIdx.x;
            if (i < num_valid_seq_q) {
                float cur_L = sL_reduction_wksp[i];
                gSoftmaxLseAccum(i) = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY : log2f(cur_L) + sM(i);
            }

            cute::tma_store_wait<0>();
        }

        if (batch_idx != sched_meta.end_req_idx)
            __syncthreads();
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}


template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    FLASH_ASSERT(params.d == Config::HEAD_DIM_K);
    FLASH_ASSERT(params.d_v == Config::HEAD_DIM_V);

    using T = Traits<InputT>;
    auto shape_Q = make_shape(params.q_seq_per_hk, params.d, params.h_k, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((InputT*)params.q_ptr),
            make_layout(
                shape_Q,
                make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride)
            )
        ),
        tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_K>>{}
        )
    );
    auto shape_K = make_shape(Int<T::PAGE_BLOCK_SIZE>{}, Int<T::HEAD_DIM_K>{}, params.h_k, params.num_blocks);
    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((InputT*)params.k_ptr),
            make_layout(
                shape_K,
                make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride)
            )
        ),
        tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Layout<
                Shape<Int<T::PAGE_BLOCK_SIZE>, Int<64>>,
                Stride<Int<T::HEAD_DIM_K>, _1>
            >{}
        )
    );
    auto shape_O = make_shape(params.q_seq_per_hk, params.d_v, params.h_k, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((InputT*)params.o_ptr),
            make_layout(
                shape_O,
                make_stride(params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride)
            )
        ),
        tile_to_shape(
            GMMA::Layout_K_SW128_Atom<InputT>{},
            Shape<Int<T::BLOCK_SIZE_M>, Int<T::HEAD_DIM_V>>{}
        )
    );
    TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_K), decltype(tma_K), decltype(shape_O), decltype(tma_O)> tma_params = {
        shape_Q, tma_Q,
        shape_K, tma_K,
        shape_O, tma_O
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_kernel<T, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    const int num_m_block = cute::ceil_div(params.q_seq_per_hk, T::BLOCK_SIZE_M);
    cudaLaunchAttribute mla_kernel_attributes[1];
    mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t mla_kernel_config = {
        dim3(num_m_block, params.h_k, params.num_sm_parts),
        dim3(T::NUM_THREADS, 1, 1),
        smem_size,
        params.stream,
        mla_kernel_attributes,
        1
    };
    cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
