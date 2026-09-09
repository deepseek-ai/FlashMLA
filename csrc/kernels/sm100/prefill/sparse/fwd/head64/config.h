#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/defines.h"
#include "kernels/params.h"

namespace sm100::prefill::sparse_fwd::head64 {

using namespace cute;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
struct KernelTemplate {

static constexpr uint32_t B_H = 64;   // Head block size. This kernel only supports h_q == B_H

template<
    typename Shape_Q_NoPE, typename TMA_Q_NoPE,
    typename Shape_Q_RoPE, typename TMA_Q_RoPE,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_NoPE shape_Q_nope; TMA_Q_NoPE tma_Q_nope;
    Shape_Q_RoPE shape_Q_rope; TMA_Q_RoPE tma_Q_rope;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv_nope;
};

static_assert(D_QK == 576 || D_QK == 512);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan
static constexpr int D_NOPE = D_V;
static constexpr int D_ROPE = D_QK - D_V;
static constexpr bool HAVE_ROPE = D_ROPE > 0;

static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 3;
static constexpr int NUM_THREADS = 128 + 128 + 128;
static constexpr int NUM_WORKER_THREADS = 128 + 128 + 1 + B_TOPK/8+1 + (HAVE_ROPE?64:0);

static constexpr int B_EPI = 64;
static constexpr int B_EPI_SB = 256;    // "SB" means SuperBlock
static_assert(D_V % B_EPI_SB == 0);
static_assert(B_EPI_SB % ((128/B_H)*B_EPI) == 0);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 400: Q
    // 400 ~ 464: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_RoPE = 256 + 128;
    static constexpr int P = 400;
};

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_Q-D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // TODO Explain why SW64

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_EPI*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<D_V/B_EPI>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKTiles<8>;
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // Re-view K-NoPE as B_TOPK*2 x D_V/2 for dual gemm

using SmemLayoutKRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<64/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<B_H>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        static_assert(B_H <= B_TOPK);
        array_aligned<bf16, B_TOPK*D_V> qko_slots[NUM_BUFS];
    } u;
    bf16 qk_rope_slot[B_TOPK*(D_Q-D_V)];
    float p_exchange_buf[4][32 * (B_TOPK/(128/B_H))];
    bf16 s[B_H*B_TOPK];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q_nope, bar_prologue_q_rope, bar_prologue_utccp_nope, bar_prologue_utccp_rope;
    transac_bar_t bar_qk_nope_done[NUM_BUFS], bar_qk_rope_done;    // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_nope_ready[NUM_BUFS][2], bar_kv_rope_ready;
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_o_write_back_done;
    transac_bar_t bar_o_write_back_done_waited; // Whether the barrier above has been waited for. Used to prevent double arrive
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    transac_bar_t bar_clc_full, bar_clc_empty;
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
    ku::CLCResponseObj clc_response_obj;
};

static constexpr int QK_MRGEMM_N = (128/B_H)*B_TOPK;
static constexpr int QK_MRGEMM_K_NOPE = D_V / (128/B_H);
static constexpr int QK_MRGEMM_K_ROPE = D_ROPE / (128/B_H);

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, QK_MRGEMM_N, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
};

template<typename TmaParam>
static __device__ void
sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParam &tma_params);

static void run(const SparseAttnFwdParams& params);

};

}
