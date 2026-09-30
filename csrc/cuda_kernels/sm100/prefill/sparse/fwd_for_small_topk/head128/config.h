#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cutlass/float8.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines.h"
#include "params.h"
#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/dequant_utils.cuh"
#include "cuda_kernels/sm100/kv_cache_utils.cuh"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

using namespace cute;

template<SparseAttnFwdMode FWD_MODE, ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>
struct KernelTemplate {

using ArgT = SparseFwdArgT<FWD_MODE>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool IS_PREFILL = !IS_DECODE;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp8_e8m0 = __nv_fp8_e8m0;

static constexpr int D_QK = 512;

struct TmaParamsForPrefill {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_o;
};

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv_quant_part[2];        // One map per CTA: each CTA of the pair gathers its half of the token
    CUtensorMap tensor_map_extra_kv_quant_part[2];  // Only available if extra_kv is enabled
};

using TmaParams = std::conditional_t<
    IS_DECODE,
    TmaParamsForDecode,
    TmaParamsForPrefill
>;

using OrigKVFormat = KVCacheFormat<MODEL_TYPE>;         // Format of `kv`, decode only
using ExtraKVFormat = KVCacheFormat<EXTRA_MODEL_TYPE>;  // Format of the paged `extra_kv`, decode only
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static_assert(OrigKVFormat::D_QK == D_QK);
static_assert(ExtraKVFormat::D_FP8 + ExtraKVFormat::D_FP4 == OrigKVFormat::D_FP8);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

static constexpr int H_Q = 128;    // For 2 CTAs
static constexpr int B_TOPK = 64; // For 2 CTAs
static constexpr int NUM_THREADS = 128*4;
static constexpr int NUM_WORKER_THREADS = IS_PREFILL ? (128 + 4 + (B_TOPK/8) + 1 + 128)*2 + 1 : (128 + 128 + 1 + 32 + 1 + 32 + 128)*2 - 1;

// For non-decode mode, we have 4 (half-)KV buffers
// For decode mode, we have 3 (half-)KV buffers with two raw KV buffers
static constexpr int NUM_K_BUFS = IS_DECODE ? 3 : 4;
static constexpr int NUM_RAW_K_BUFS = IS_DECODE ? 2 : 0;
static constexpr int NUM_INDEX_BUFS = IS_DECODE ? 4 : 4;

static constexpr int QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr int NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;
// Decode: CTA0 gathers raw and stages scales separately; CTA1 gathers raw + trailing scales.
template<typename F, int CTA> using KVPart = KVCachePart<F, 2, CTA>;
template<typename F> static constexpr int RAW_TOKEN_SMEM_STRIDE = F::QUANT_BYTES/2 + 32;
// One tma_gather4 writes its 4 rows RAW_TOKEN_SMEM_STRIDE apart, and its destination must be 128 B aligned
static_assert(4 * RAW_TOKEN_SMEM_STRIDE<OrigKVFormat> % 128 == 0 && 4 * RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat> % 128 == 0);
static constexpr int SCALE_STAGE_ROW_BYTES = std::max(KVPart<OrigKVFormat, 0>::NUM_SCALES, KVPart<ExtraKVFormat, 0>::NUM_SCALES);
// The scale stage follows the entire TMA tile, including its zero-filled padding.
static constexpr int SCALE_STAGE_OFFSET = B_TOPK * std::max(RAW_TOKEN_SMEM_STRIDE<OrigKVFormat>, RAW_TOKEN_SMEM_STRIDE<ExtraKVFormat>);
static constexpr int RAW_BUFFER_BYTES = SCALE_STAGE_OFFSET + B_TOPK * SCALE_STAGE_ROW_BYTES;
template<typename F> using DequantizerT = KVBlockDequantizer<F, D_K/2, B_TOPK, RAW_TOKEN_SMEM_STRIDE<F>, RAW_TOKEN_SMEM_STRIDE<F>>;

static constexpr int B_EPI = 64;                // Epilogue block size for normal case (i.e. prefill or non-splitkv decoding)
static constexpr int B_EPI_SPLITKV = 32;        // Epilogue block size for splitkv decoding
static constexpr int NUM_EPI_SPLITKV_BUFS = 4;  // The number of epilogue buffers for splitkv decoding
static_assert((H_Q/2)*D_Q*sizeof(bf16) >= NUM_EPI_SPLITKV_BUFS*(H_Q/2)*(B_EPI_SPLITKV*2)*sizeof(float));

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: Output accumulator
    // 256 ~ 384: Q
    // 384 ~ 448: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int P = 384;
};

struct SharedMemoryPlan {
    array_aligned<bf16, B_TOPK*(D_K/2)> K[NUM_K_BUFS];
    array_aligned<bf16, (H_Q/2)*B_TOPK> S;
    array_aligned<bf16, (H_Q/2)*D_Q> Q; // Will be output for epilogue
    array_aligned<uint8_t, RAW_BUFFER_BYTES, 128> K_raw[NUM_RAW_K_BUFS];
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    
    transac_bar_t bar_sQ_full, bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_tOut_full, bar_tOut_empty;
    transac_bar_t bar_KV_full[NUM_K_BUFS], bar_KV_empty[NUM_K_BUFS];
    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done, bar_SV_done;
    transac_bar_t bar_S_O_full;
    transac_bar_t bar_li_full, bar_li_empty;

    // The following barriers are prefill-only
    transac_bar_t bar_clc_full, bar_clc_empty;

    // The following barriers are decode-only
    transac_bar_t bar_raw_KV_full[NUM_RAW_K_BUFS], bar_raw_KV_empty[NUM_RAW_K_BUFS];
    transac_bar_t bar_valid_coord_full[NUM_INDEX_BUFS], bar_valid_coord_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, H_Q, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
));

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG2_SYNC = 1;
    static constexpr int WG2_WARP02_SYNC = 2;
    static constexpr int WG2_WARP13_SYNC = 3;
};

static __device__ void
sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params);

static void run(const ArgT& params);

};

}
