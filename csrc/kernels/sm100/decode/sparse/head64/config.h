#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "kernels/defines.h"
#include "kernels/kv_cache_format.h"
#include "kernels/sm100/dequant_utils.cuh"


namespace sm100::decode::sparse::head64 {

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

template<Config CONFIG>
struct KernelTemplate {

static constexpr uint32_t B_H = 64;   // Head block size. This kernel only supports h_q == B_H
static constexpr bool ENABLE_SPLITKV = CONFIG.ENABLE_SPLITKV;

using OrigKVFormat = KVCacheFormat<CONFIG.MODEL_TYPE>;         // Format of `kv`
using ExtraKVFormat = KVCacheFormat<CONFIG.EXTRA_MODEL_TYPE>;  // Format of `extra_kv`
static_assert(is_valid_kv_format_pair(CONFIG.MODEL_TYPE, CONFIG.EXTRA_MODEL_TYPE));

static constexpr int D_Q = OrigKVFormat::D_QK;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_NOPE = OrigKVFormat::D_NOPE;
static constexpr int D_ROPE = OrigKVFormat::D_ROPE;
static constexpr int D_FP8 = OrigKVFormat::D_FP8;     // K dimensions stored as fp8 and needing dequant
static constexpr int D_BF16 = OrigKVFormat::D_BF16;   // K dimensions stored as bf16 and not needing dequant
static constexpr int QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr bool V_HAVE_ROPE = OrigKVFormat::MODEL_TYPE == ModelType::V32 ? false : true;
static constexpr int NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;    // Padding is included
static constexpr int TMA_K_STRIDE = OrigKVFormat::TMA_K_STRIDE;   // Stride of K's tensormap. This stride must 1) be a factor of the actual stride between tokens 2) large enough to cover the entire KV cache. Since TMA copy's coordinate can only be 32bit signed integers, this number must >= 128, perferrably >= 256. So we set this to 656 for V32, 576 for V4 and 512 for V41. Extra padding may be necessary for KV blocks.
static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(D_FP8 + D_BF16 == D_Q);
static_assert(V_HAVE_ROPE ? (D_NOPE + D_ROPE == D_V) : (D_NOPE == D_V));

static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) & is_token_valid & scales

// Both caches are dequantized into the same bf16 tile (B_TOPK x D_FP8, plus the bf16 part) by the same warpgroup, see KVBlockDequantizer
static_assert(ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == D_FP8 && ExtraKVFormat::D_BF16 == D_BF16);
// Bytes between two raw (quantized) rows in shared memory, i.e. the box of their tensor map. fp4 rows are padded by 32 B so that the 4 rows of a
// gather4 group start in 4 different quarters of the 32 banks (288 / 4 = 72 = 8 mod 32) and the LDS.32 of the dequantizer has no
// bank conflict; fp8 rows have no padding
template<typename F> static constexpr int RAW_TOKEN_SMEM_STRIDE = F::IS_FP4 ? F::QUANT_BYTES + 32 : F::D_FP8;
// One tma_gather4 writes its 4 rows RAW_TOKEN_SMEM_STRIDE apart, and its destination must be 128 B aligned
static_assert(4 * RAW_TOKEN_SMEM_STRIDE<OrigKVFormat> % 128 == 0 && 4 * RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat> % 128 == 0);
// One row of 1 B scales per token. In a kernel with an fp4 extra_kv the rows are 32 B and the fp8 tokens use their first 16 B
static constexpr int SCALE_SMEM_STRIDE = std::max(OrigKVFormat::NUM_SCALES_EACH_TOKEN, ExtraKVFormat::NUM_SCALES_EACH_TOKEN);
template<typename F> using Dequantizer = KVBlockDequantizer<F, D_FP8, B_TOPK, RAW_TOKEN_SMEM_STRIDE<F>, SCALE_SMEM_STRIDE>;
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 1/32 utcmma + 1/32 raw KV producer + 1/32 rope producer + 32 index+scale+valid_mask producer + 128 dequant
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = OrigKVFormat::MODEL_TYPE == ModelType::V32 ? 64 : 0;
static_assert(D_Q_SW128 + D_Q_SW64 == D_Q);
static constexpr int K_ROPE_SW = OrigKVFormat::MODEL_TYPE == ModelType::V41 ? 0 : (OrigKVFormat::MODEL_TYPE == ModelType::V32 ? 64 : 128); // RoPE part stored in SW64 (for V32) or SW128 (for V4), in bytes. 0 for V41 (no separate bf16 RoPE, loaded as fp8 in D_FP8)

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_sw64;  // Invalid if D_Q_SW64 == 0
    CUtensorMap tensor_map_kv_quant_part;        // The quantized (fp8) part of `kv`, one raw row per token
    CUtensorMap tensor_map_kv_bf16_part;         // The bf16 (RoPE) part of `kv`. Invalid if D_BF16 == 0
    CUtensorMap tensor_map_extra_kv_quant_part;  // Same for `extra_kv` (fp8 or fp4). Invalid if extra_topk == 0
    CUtensorMap tensor_map_extra_kv_bf16_part;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + B_H*D_Q/256: Q
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
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<64*NUM_TILES>>{},
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
    Shape<Int<B_TOPK>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<32*NUM_TILES>>{},
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
                alignas(1024) bf16 quant_part[B_TOPK*D_FP8];   // Quantized-origin (fp8 / fp4) part, dequantized to bf16
                alignas(1024) bf16 bf16_part[B_TOPK*D_BF16];   // bf16-origin part, swizzled as K_ROPE_SW
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H*D_Q)); // So that Q does not cover raw_quant
            array_aligned<uint8_t, B_TOPK*D_FP8, 128> raw_quant[NUM_BUFS];  // Raw (quantized) rows of a KV block, RAW_TOKEN_SMEM_STRIDE<F> apart. For V41, includes both NoPE and RoPE. 128 B aligned for gather4
            static_assert(B_TOPK * RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat> <= B_TOPK * D_FP8);
        } kv;
    } u;
    union {
        float p_exchange_buf[4][32 * (B_TOPK/(128/B_H))];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) uint8_t scales[NUM_INDEX_BUFS][B_TOPK][SCALE_SMEM_STRIDE];   // ue8m0 (fp8) or e4m3 (fp4), see KVBlockDequantizer
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_bf16_part_load_ready[NUM_BUFS];
    transac_bar_t bar_quant_part_dequant_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

static constexpr int PV_GEMM_N = 256;
using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, PV_GEMM_N, UMMA::Major::K, UMMA::Major::MN>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params );

static void run(const SparseAttnDecodeParams &params);

};

}