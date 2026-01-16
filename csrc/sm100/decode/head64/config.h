#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::head64 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e8m0 = __nv_fp8_e8m0;
using e4m3 = cutlass::float_e4m3_t;
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
static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 128 : 64;
static constexpr bool V_HAVE_ROPE = MODEL_TYPE == ModelType::V32 ? false : true;
static constexpr int NUM_SCALES_EACH_TOKEN = MODEL_TYPE == ModelType::V32 ? 4 : 8;    // Padding is included
static constexpr int TMA_K_STRIDE = MODEL_TYPE == ModelType::V32 ? D_NOPE+2*D_ROPE+4*(D_NOPE/QUANT_TILE_SIZE) : D_NOPE+2*D_ROPE;   // Stride of K's tensormap. This stride must 1) be a factor of the actual stride between tokens 2) large enough to cover the entire KV cache. Since TMA copy's coordinate can only be 32bit signed integers, this number must >= 128, perferrably >= 256. So we set this to 656 for V32 and 576 for MODEL1. Extra padding may be necessary for KV blocks.
static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(V_HAVE_ROPE ? (D_NOPE + D_ROPE == D_V) : (D_NOPE == D_V));

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) & is_token_valid & scales
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 1/32 utcmma + 1/32 raw KV producer + 1/32 rope producer + 32 index+scale+valid_mask producer + 128 dequant
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = MODEL_TYPE == ModelType::V32 ? 64 : 0;
static_assert(D_Q_SW128 + D_Q_SW64 == D_Q);
static constexpr int K_ROPE_SW = MODEL_TYPE == ModelType::V32 ? 64 : 128; // RoPE part stored in SW64 (for V32) or SW128 (for MODEL1), in bytes

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_sw64;  // Invalid if D_Q_SW64 == 0
    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_rope;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + 64*D_Q/256: Q
    // 400 ~ 464: P
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
)); // A TMA tile

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
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

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            bf16 q_sw64[B_H*D_Q_SW64];  // NOTE D_Q_SW64 may be 0 but array_aligned<bf16, 0> will have a size of 16, so we use array here. The former tensor (`q`) promises its alignment.
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                array_aligned<bf16, B_H*D_NOPE> nope; // NoPE part, dequantized
                array_aligned<bf16, B_H*D_ROPE> rope; // RoPE part, dequantized. SW64 in v32 mode, SW128 in MODEL1 mode
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H*D_Q)); // So that Q does not covers raw_nope
            array_aligned<e4m3, B_H*D_NOPE> raw_nope[NUM_BUFS];  // Raw (quantized) NoPE part
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[NUM_BUFS];
    transac_bar_t bar_nope_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}