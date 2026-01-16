#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cutlass/float8.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::fwd_for_small_topk::head128 {

using namespace cute;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
struct KernelTemplate {

using ArgT = SparseFwdArgT<FWD_MODE>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool IS_PREFILL = !IS_DECODE;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp8_e8m0 = __nv_fp8_e8m0;

struct TmaParamsForPrefill {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_o;
};

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
    CUtensorMap tensor_map_extra_kv_nope;   // Only available if extra_kv is enabled
    CUtensorMap tensor_map_extra_kv_rope;
};

using TmaParams = std::conditional_t<
    IS_DECODE,
    TmaParamsForDecode,
    TmaParamsForPrefill
>;

static_assert(D_QK == 512);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

static constexpr int H_Q = 128;    // For 2 CTAs
static constexpr int B_TOPK = 64; // For 2 CTAs
static constexpr int NUM_THREADS = 128*4;
static constexpr int NUM_WORKER_THREADS = IS_PREFILL ? (128 + 4 + (B_TOPK/8) + 1 + 128)*2 + 1 : (128 + 128 + 1 + 32 + 2 + 128)*2;

// For non-decode mode, we have 4 (half-)KV buffers
// For decode mode, we have 3 (half-)KV buffers with two raw KV buffers
static constexpr int NUM_K_BUFS = IS_DECODE ? 3 : 4;
static constexpr int NUM_RAW_K_BUFS = IS_DECODE ? 2 : 0;
static constexpr int NUM_INDEX_BUFS = IS_DECODE ? 4 : 4;

static constexpr int D_NOPE = 448;
static constexpr int D_ROPE = 64;
static constexpr int TMA_K_STRIDE_FOR_DECODING = D_NOPE + 2*D_ROPE;
static constexpr int NUM_SCALES_EACH_TOKEN = 8; // 7 scales + 1 padding

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
    array_aligned<bf16, (H_Q/2)*D_Q> Q; // Will be output for epilogue
    array_aligned<bf16, B_TOPK*(D_K/2)> K[NUM_K_BUFS];
    array_aligned<fp8_e4m3, B_TOPK*(D_K/2)> K_raw[NUM_RAW_K_BUFS];
    array_aligned<bf16, (H_Q/2)*B_TOPK> S;
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) fp8_e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN/2];
    
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
    transac_bar_t bar_valid_coord_scales_full[NUM_INDEX_BUFS], bar_valid_coord_scales_empty[NUM_INDEX_BUFS];

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
