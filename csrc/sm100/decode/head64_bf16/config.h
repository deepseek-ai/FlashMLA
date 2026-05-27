#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::head64_bf16 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using namespace cute;

enum NamedBarriers : uint32_t {
    main_loop_sync = 0,
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    everyone_sync = 4
};

template<ModelType MODEL_TYPE>
struct KernelTemplate {

static constexpr int D_Q = MODEL_TYPE == ModelType::V32 ? 576 : 512;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_NOPE = MODEL_TYPE == ModelType::V32 ? 512 : 448;
static constexpr int D_ROPE = 64;
static constexpr bool V_HAVE_ROPE = MODEL_TYPE == ModelType::V32 ? false : true;
// BF16 mode: each token is simply d_qk bf16 values
// Layout: [D_NOPE bf16 values][D_ROPE bf16 values] = D_Q bf16 values total
// Stride per token in bytes: D_Q * 2
static constexpr int TMA_K_STRIDE = D_Q * 2;  // bytes per token for TMA coordinate calculation
static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(V_HAVE_ROPE ? (D_NOPE + D_ROPE == D_V) : (D_NOPE == D_V));

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;
// BF16 mode: no dequant warpgroup needed, but we keep 3 warpgroups for NoPE producer
static constexpr int NUM_THREADS = 128*3;
static constexpr float MAX_INIT_VAL = -1e30f;

static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = MODEL_TYPE == ModelType::V32 ? 64 : 0;
static_assert(D_Q_SW128 + D_Q_SW64 == D_Q);
static constexpr int K_ROPE_SW = MODEL_TYPE == ModelType::V32 ? 64 : 128;

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_sw64;
    CUtensorMap tensor_map_kv_nope;  // BF16 NoPE data
    CUtensorMap tensor_map_kv_rope;  // BF16 RoPE data
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_rope;
};

// Tensor memory columns
struct tmem_cols {
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_Tail = 256 + B_H*D_NOPE/2/128;
    static constexpr int P = 400;
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<NUM_TILES*64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q_SW128/64>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{}
));

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{},
    Layout<
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW64 = decltype(composition(
    SmemLayoutKTiles_SW64<NUM_TILES>{},
    Layout<
        Shape<Int<32*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

// BF16 mode: TMA loads NoPE directly with swizzle into dequant.nope (no raw_nope needed)
struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            bf16 q_sw64[B_H*D_Q_SW64];
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                array_aligned<bf16, B_H*D_NOPE> nope;  // SW128 swizzled, TMA loads directly here
                array_aligned<bf16, B_H*D_ROPE> rope;  // SW64/SW128 swizzled, TMA loads directly here
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H*D_Q));
            // No raw_nope buffer needed: TMA swizzle loads BF16 NoPE directly into dequant.nope
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    // No scales in BF16 mode
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[NUM_BUFS];
    transac_bar_t bar_nope_ready[NUM_BUFS];
    transac_bar_t bar_valid_coord_ready[NUM_INDEX_BUFS], bar_valid_coord_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_bf16_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}
