#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines.h"
#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/dequant_utils.cuh"
#include "cuda_kernels/sm100/kv_cache_utils.cuh"

namespace sm100::decode::sparse::head64 {

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

template<Config CONFIG>
struct KernelTemplate {

static constexpr int MMA_M = 64;    // Head block size of the MMA; equals h_q (64)
static constexpr bool ENABLE_SPLITKV = CONFIG.ENABLE_SPLITKV;

using OrigKVFormat = KVCacheFormat<CONFIG.MODEL_TYPE>;         // Format of `kv`
using ExtraKVFormat = KVCacheFormat<CONFIG.EXTRA_MODEL_TYPE>;  // Format of `extra_kv`
static_assert(is_valid_kv_format_pair(CONFIG.MODEL_TYPE, CONFIG.EXTRA_MODEL_TYPE));

static constexpr int D_Q = OrigKVFormat::D_QK;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_FP8 = OrigKVFormat::D_FP8;     // K dimensions stored as fp8 and needing dequant
static constexpr int QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr int NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;    // Padding is included
static constexpr int TMA_K_STRIDE = OrigKVFormat::BYTES_PER_TOKEN;   // Interleaved raw data and scales form one TMA row

static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) and is_token_valid

// Both caches are dequantized into the same bf16 tile (B_TOPK x D_FP8) by the same warpgroup, see KVBlockDequantizer
static_assert(ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == D_FP8);
// Interleaved raw data and scales share a row. FP8 uses 576 B (512 raw + 16 scales + 48 padding):
// adjacent rows start 64 B apart modulo 128 B, avoiding bank conflicts for the dequantizer's LDS.64.
// FP4 uses 288 B (256 raw + 32 scales), which also preserves its conflict-free LDS.32 mapping.
template<typename F> static constexpr int RAW_TOKEN_SMEM_STRIDE = F::IS_FP4 ? F::BYTES_PER_TOKEN : F::QUANT_BYTES + 64;
static_assert(RAW_TOKEN_SMEM_STRIDE<OrigKVFormat> >= OrigKVFormat::BYTES_PER_TOKEN && RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat> >= ExtraKVFormat::BYTES_PER_TOKEN);
// Each gather4 destination must be 128 B aligned.
static_assert(4 * RAW_TOKEN_SMEM_STRIDE<OrigKVFormat> % 128 == 0 && 4 * RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat> % 128 == 0);
static constexpr int RAW_BLOCK_SMEM_BYTES = B_TOPK * std::max(RAW_TOKEN_SMEM_STRIDE<OrigKVFormat>, RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat>);
template<typename F> using Dequantizer = KVBlockDequantizer<F, D_FP8, B_TOPK, RAW_TOKEN_SMEM_STRIDE<F>, RAW_TOKEN_SMEM_STRIDE<F>>;
static constexpr int NUM_THREADS = 128*3;  // 128 exp (wg0) + 1/32 utcmma + 1/32 raw KV producer + 32 index+valid_mask producer (wg1, its warp 6 is idle) + 128 dequant (wg2)
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv_quant_part;        // Interleaved raw data and scales of `kv`, one row per token
    CUtensorMap tensor_map_extra_kv_quant_part;  // Same for `extra_kv` (fp8 or fp4). Invalid if extra_topk == 0
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + 64*D_Q/256: Q. Here we use 64*D_Q/256 instead of MMA_M since we always use dual-rail gemm
    // 400 ~ 464: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int P = 400;
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<NUM_TILES*64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q/64>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<64>>{}
)); // A TMA tile

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<MMA_M>, Int<D_V>>,
    Stride<Int<516>, _1>	// We use stride = 516 here to avoid bank conflict
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<MMA_M>, Int<B_TOPK>>{},
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

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                alignas(1024) bf16 quant_part[B_TOPK*D_FP8];   // Quantized-origin (fp8 / fp4) part, dequantized to bf16
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (MMA_M*D_Q)); // So that Q does not cover raw_quant
            array_aligned<uint8_t, RAW_BLOCK_SMEM_BYTES, 128> raw_quant[NUM_BUFS];  // Interleaved raw data and scales, RAW_TOKEN_SMEM_STRIDE<F> bytes per row
        } kv;
    } u;
    union {
        float p_exchange_buf[4][32 * (B_TOPK/(128/MMA_M))];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_quant_part_dequant_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_ready[NUM_INDEX_BUFS], bar_valid_coord_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

static constexpr int PV_GEMM_N = 256;
using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, PV_GEMM_N, UMMA::Major::K, UMMA::Major::MN>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}
