#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines.h"
#include "params.h"

namespace sm100::prefill::sparse_fwd::head64 {

using namespace cute;

struct KernelTemplate {

static constexpr int MMA_M = 64;    // Head block size of the MMA; equals h_q (64)

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;
};

struct float2x2 {
    float2 lo, hi;
};

static constexpr int D_QK = 512;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 3;
static constexpr int NUM_THREADS = 128 + 128 + 128;
static constexpr int NUM_WORKER_THREADS = 128 + 128 + 1 + B_TOPK/8+1;

static constexpr int B_EPI = 64;
static constexpr int B_EPI_SB = 256;    // "SB" means SuperBlock
static_assert(D_V % B_EPI_SB == 0);
static_assert(B_EPI_SB % ((128/MMA_M)*B_EPI) == 0);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 384: Q
    // 400 ~ 464: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int P = 400;
};

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<B_EPI*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<D_V/B_EPI>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutK = SmemLayoutKTiles<8>;
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutK{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutK_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // Re-view K as B_TOPK*2 x D_V/2 for dual gemm

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<MMA_M>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        static_assert(MMA_M <= B_TOPK);
        array_aligned<bf16, B_TOPK*D_V> qko_slots[NUM_BUFS];
    } u;
    float p_exchange_buf[4][32 * (B_TOPK/(128/MMA_M))];
    bf16 s[MMA_M*B_TOPK];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q, bar_prologue_utccp;
    transac_bar_t bar_qk_done[NUM_BUFS];    // Pi = QKi^T done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_ready[NUM_BUFS][2];
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

static constexpr int QK_MRGEMM_N = (128/64)*B_TOPK;
static constexpr int QK_MRGEMM_K = D_V / (128/64);

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, QK_MRGEMM_N, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, 256, UMMA::Major::K, UMMA::Major::MN>{}
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
