/*
Fused "Q Norm + Q RoPE + Core Attention + O RoPE + O FP8 Cast" kernel for DeepSeek V4 / V4.1
(d_qk = d_v = 512, h_kv = 1, token-level sparse (exact top-k) attention) on SM100f.

Instead of launching separate kernels for RMSNorm, RoPE, attention, and output quantization, this
kernel fuses the whole "q_b_proj output -> wv_proj input" segment of the MLA block:
1. Q Norm (V4 only, MODEL_TYPE == ModelType::V4): computes the per-head RMSNorm denominator
   rsqrt(sum(q^2)/d_qk + eps) on the fly while loading Q, and folds it into the softmax scale.
   V4.1 (ModelType::V41) skips this step
2. Q RoPE: applies non-neox-style RoPE (rope_dim = 64) to the last D_ROPE dims of each Q head,
   using `token_positions` and `cos_sin_cache`
3. Core attention: token-level sparse attention over the (at most) `topk` KV tokens selected by
   `indices`. Invalid indices (< 0 or >= s_kv) and positions beyond `topk_length` are masked out.
   Supports an optional per-head `attn_sink` (affects output but not lse / max_logits)
4. O RoPE: applies the conjugate RoPE to the last D_ROPE dims of each output head
5. O FP8 Cast: quantizes the output to fp8_e4m3 with per-32-element ue8m0 scale factors
   (round_sf + packed ue8m0, TMA-aligned col-major sf layout)

Template parameters:
- FWD_MODE:   SparseAttnFwdMode::Prefill or SparseAttnFwdMode::Decode
- MODEL_TYPE: ModelType::V4 (Q norm enabled, V4 KV cache layout) or ModelType::V41
- EXTRA_MODEL_TYPE: the format of the extra KV cache (decode only), MODEL_TYPE or ModelType::V41_FP4
              (V4.1 with fp4 KV cache, e4m3 per-16 scales) with MODEL_TYPE == V41
- H_Q:        number of Q heads, 64 or 128

I/O (see `csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp` and the Python docstrings in
`flash_mla/fused_norm_rope_attn_rope_cast.py` for the full parameter list):
- q: [s_q, h_q, d_qk], bf16, WITHOUT RoPE applied, in the PERMUTED layout produced by the
  `permute_q_b_proj` kernel (16-element d-chunks interleaved across heads). Each token's
  h_q*d_qk elements must be contiguous
- Prefill: kv [s_kv, 1, d_qk] (bf16, non-paged) + indices [s_q, 1, topk]
  Decode:  paged quantized KV cache(s) (V4 / V4.1 / V4.1 fp4 format, see below) + indices_in_kvcache [s_q, topk],
  plus an optional secondary ("extra") KV cache with its own indices / topk_length
- out_fp8: [s_q, n_wv_group, wv_group_size * d_v], fp8_e4m3, in the permuted layout expected by
  the `permute_wv_proj`-transformed weights; out_sf: packed ue8m0 scale factors (always per-32)
- lse / max_logits (prefill only for max_logits): [s_q, h_q], fp32

Execution structure:
- Persistent kernel scheduled via CLC (Cluster Launch Control); the grid is
  (s_q * CLUSTER_SIZE, 1, 1) and each cluster processes one query token per job
- CLUSTER_SIZE = ceil(H_Q / 64): 1 CTA for h64, a 2-CTA cluster (dual-CTA UMMA, CTA0 owns
  V[:, 0:256] and CTA1 owns V[:, 256:512]) for h128. Each CTA has 512 threads (4 warpgroups):
  - WG0: Q fetching (q_sqr_sum + Q RoPE + store to TMEM) & O epilogue (TMEM load, O RoPE,
    FP8 quant, store to gmem). Timeline: Q0 Q1 O0 Q2 O1 ... Qn O(n-1) On
  - WG1: KV producer. Prefill: gathers the bf16 KV block via TMA gather4. Decode: loads the fp8 / fp4
    part from the paged cache and dequantizes it into the smem KV slots in registers
  - WG2: MMA warp (warp 8, issues UMMAs on CTA0 only), CLC warp (warp 9), indices / validity
    mask generator (warp 10), and (decode only) TMA warp for the bf16 KV part (warp 11)
  - WG3: Scale & Exp: reduces P, maintains the online softmax state (mi / li), produces S
- KV tokens are processed in blocks of B_TOPK (96 for 2-CTA, 64 for 1-CTA) with NUM_KV_SLOTS-deep
  software pipelining

Multi-rail GeMM is always used:
- For cases where CLUSTER_SIZE is 1, we FOLD Q by FOLD_FACTOR and do a "batched gemm" in batch size = FOLD_FACTOR, and reduce P on shared memory
- For cases where CLUSTER_SIZE is 2, we also fold Q by FOLD_FACTOR and perform batched GeMM, and reduce P on shared memory

Decoding mode (FWD_MODE == SparseAttnFwdMode::Decode):
- Batch size must be 1; the grid covers the s_q query tokens
- The KV cache is a paged FP8 cache with the same format as `sm100::decode::sparse::head64` (V4 layout).
  The fp8 (D_FP8) part is loaded from global memory and dequantized in-place into the KV slots by
  warpgroup 1 (no intermediate raw-fp8 smem buffer), while the bf16 (D_BF16, RoPE) part is loaded via
  TMA gather4 by warp 11. An optional extra (secondary) KV cache is supported
- EXTRA_MODEL_TYPE == ModelType::V41_FP4 selects an fp4 extra KV cache: every token is 512 e2m1 + 32 e4m3 scales,
  each page block stores [page_block_size x 256 B data rows] + [page_block_size x 32 B scale rows]. Warpgroup 1 then
  dequantizes both caches with a common code path (see there) instead of the fp8-only one
- Split-KV is not supported (the fused RoPE + FP8-quant epilogue cannot be combined across splits)
*/

#pragma once

#include <cutlass/fast_math.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "kernels/defines.h"
#include "kernels/params.h"
#include "kernels/kv_cache_format.h"

#include "kernel.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using namespace cute;

template<Config CONFIG>
struct Kernel {

static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;

using Params = ParamT<FWD_MODE>;

static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(H_Q == 64 || H_Q == 128);

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

// Model parameters
static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t O_QUANT_TILE_SIZE = 32;
static constexpr uint32_t D_ROPE = 64;
static constexpr uint32_t D_NOPE = D_QK - D_ROPE;
static constexpr uint32_t WV_GROUP_SIZE = 8;
static constexpr bool ENABLE_Q_NORM = CONFIG.ENABLE_Q_NORM;

// Cluster shape selection
static constexpr uint32_t CLUSTER_SIZE = ku::ceil_div((uint32_t)H_Q, 64u);
static constexpr uint32_t IS_2CTA = CLUSTER_SIZE == 2;

// Tiling Shape Selection
static constexpr uint32_t B_TOPK = CLUSTER_SIZE == 2 ? 96 : 64;
static constexpr uint32_t H_Q_PER_CTA = 64;

// Paged quantized KV cache format for decoding, plus the constants of this kernel's common dequant path. The members are re-exported as uint32_t
template<ModelType MT>
struct KVFormat {
    using Base = KVCacheFormat<MT>;
    static constexpr bool IS_FP4 = Base::IS_FP4;
    static constexpr uint32_t D_FP4 = Base::D_FP4;
    static constexpr uint32_t D_FP8 = Base::D_FP8;
    static constexpr uint32_t D_BF16 = Base::D_BF16;
    static constexpr uint32_t QUANT_TILE_SIZE = Base::QUANT_TILE_SIZE;
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN = Base::NUM_SCALES_EACH_TOKEN;
    static constexpr uint32_t TMA_K_STRIDE = Base::TMA_K_STRIDE;
    static constexpr uint32_t BYTES_PER_TOKEN = Base::BYTES_PER_TOKEN;
    // The common dequant path of WG1 only (HAS_FP4_KV). A raw row is this CTA's part of the quantized data of a token; it is
    // gathered with a box 16 B wider than its data (zero-filled by TMA) so that RAW_TOKEN_SMEM_STRIDE / 16 is odd, which keeps the
    // LDS.128 of WG1 free of bank conflicts. A chunk is 16 B of raw data, by one LDS.128
    static constexpr uint32_t RAW_TOKEN_DATA_BYTES = Base::QUANT_BYTES / CLUSTER_SIZE;
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = RAW_TOKEN_DATA_BYTES + 16;   // 272 / 144 (fp4), 528 / 272 (V41 fp8)
    static constexpr uint32_t NUM_CHUNKS_PER_ROW = RAW_TOKEN_DATA_BYTES / 16;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;
};
using OrigKVFormat = KVFormat<MODEL_TYPE>;      // Format of `kv`
using ExtraKVFormat = KVFormat<EXTRA_MODEL_TYPE>;   // Format of `extra_kv`
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static constexpr bool HAS_FP4_KV = ExtraKVFormat::IS_FP4;   // Selects the common (fp8 + fp4) dequant path of WG1 instead of the fp8-only one
// Shorthands for the format of `kv`, which is also the format of `extra_kv` unless HAS_FP4_KV
static constexpr uint32_t D_FP8 = OrigKVFormat::D_FP8;
static constexpr uint32_t D_BF16 = OrigKVFormat::D_BF16;
static constexpr uint32_t KV_QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
static constexpr uint32_t TMA_K_STRIDE = OrigKVFormat::TMA_K_STRIDE;
static constexpr uint32_t KV_CACHE_BYTES_PER_TOKEN = OrigKVFormat::BYTES_PER_TOKEN;
// The common dequant path gathers the raw rows of a KV block into the beginning of the KV slot (one 4-row group per gather4) and
// dequantizes them in place. A group is padded to 128 B since the destination of a gather4 must be 128 B aligned
static constexpr uint32_t RAW_KV_GROUP_BYTES = ku::ceil_div(4 * std::max(OrigKVFormat::RAW_TOKEN_SMEM_STRIDE, ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE), 128u) * 128;
static_assert(!HAS_FP4_KV || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= B_TOPK * D_QK / CLUSTER_SIZE * sizeof(bf16));

static constexpr uint32_t NUM_THREADS = 512;
static constexpr uint32_t NUM_WORKING_THREADS = 
    CLUSTER_SIZE == 1 ? (
        IS_DECODE ?
        128 + 128 + (1+1+32) + 128 :   // WG0 + WG3 + (MMA + CLC + indices) + WG1 (dequant)
        128 + 128 + (1+1+32) + 4                            // WG0 + WG3 + (MMA + CLC + indices) + WG1 (KV producer, 1 elected thread per warp)
    ) : (
        IS_DECODE ? 
        128*2 + 128*2 + (1+2+32*2) + 128*2 :
        128*2 + 128*2 + (1+2+32*2) + 4*2
    );

static constexpr uint32_t FOLD_FACTOR = 128 / H_Q_PER_CTA;
static constexpr uint32_t NUM_MRGEMM_RAILS = 2; // The number of "rails" (batch size) during multi-rail GeMM. Currently must be 2
static constexpr uint32_t NUM_P_ELEMS_PER_THREAD = H_Q_PER_CTA * B_TOPK / 128;

static constexpr uint32_t NUM_KV_SLOTS = 3;
static constexpr uint32_t NUM_INDICES_BUFS = 4;
static constexpr uint32_t NUM_P_BUFS = CLUSTER_SIZE == 2 ? 1 : 2;
static constexpr uint32_t NEED_TP_EMPTY_BAR = NUM_P_BUFS == 1;  // Don't need to wait for P's emptiness as long as P has >= 2 buffers, since "we are issuing P[i]" <-- "O[i-2] has been issued" <-- "S[i-2] is ready" <-- "P[i-2] is free"

struct tmem_cols {
    static constexpr uint32_t O = 0;
    static constexpr uint32_t Q = O + D_VO / FOLD_FACTOR;
    static constexpr uint32_t P_0 = Q + D_QK / NUM_MRGEMM_RAILS / 2;  // /2 since 2 bf16 is packed in 1 uint32
    static constexpr uint32_t P_1 = P_0 + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR;

    static constexpr uint32_t get_p(const uint32_t &p_buf_idx) {
        if constexpr (NUM_P_BUFS == 1) {
            return P_0;
        } else if constexpr (NUM_P_BUFS == 2) {
            return p_buf_idx ? P_1 : P_0;
        } else {
            static_assert(NUM_P_BUFS == 1 || NUM_P_BUFS == 2);
        }
    }
    static_assert(get_p(NUM_P_BUFS-1) + B_TOPK*NUM_MRGEMM_RAILS/FOLD_FACTOR <= 512);
};

using MMAAtom_QK = cute::conditional_t<
    CLUSTER_SIZE == 2,
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>,
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>
>;
using TiledMMA_QK = decltype(make_tiled_mma(MMAAtom_QK{}));
using TiledMMA_SV = cute::conditional_t<
    CLUSTER_SIZE == 2,
    decltype(make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
    )),
    decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, H_Q, 256, UMMA::Major::K, UMMA::Major::MN>{}))
>;

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK / CLUSTER_SIZE];    // Cluster size = 1: the whole KV; cluster size = 2: half KV
    CUTE_ALIGNAS(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    CUTE_ALIGNAS(1024) float p_exchange_buf[4][32*NUM_P_ELEMS_PER_THREAD];
    CUTE_ALIGNAS(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][ku::find_next_power_of_2(B_TOPK/8)];
    // Decode: WG10 produces metadata once for the four dequant warps. 16 B aligned so that the fp4 path can read
    // the coordinates of 4 consecutive rows with one LDS.128
    CUTE_ALIGNAS(16) int decode_tma_coords[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK];
    uint8_t decode_scales[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK * NUM_SCALES_EACH_TOKEN_PER_CTA];   // Scales of the fp8 tokens
    // fp4 tokens: 4 buffers of 32 B scales per token do not fit into shared memory, so the dequant warps load the scales themselves
    // and only get their global addresses here (nullptr for an invalid token)
    const uint8_t *decode_scale_ptrs[(IS_DECODE && HAS_FP4_KV) ? NUM_INDICES_BUFS : 0][B_TOPK];
    float q_sqr_sum_buf[ENABLE_Q_NORM ? 2 : 0][128];    // We have 2 q_sqr_sum_buf to save some barriers, as Q[i+2] starts to fetch -> O[i] have finished -> Q[i]'s q_sqr_sum_buf is useless
    float rowwise_max_buf[128];
    float rowwise_mi_buf[H_Q_PER_CTA];
    float rowwise_li_buf[128];    // 128: warpgroup size

    transac_bar_t bar_kv_slot_full[NUM_KV_SLOTS], bar_kv_slot_empty[NUM_KV_SLOTS];
    transac_bar_t bar_indices_full[NUM_INDICES_BUFS], bar_indices_empty[NUM_INDICES_BUFS];
    transac_bar_t bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_q_sqr_sum_full;   // Only used for 2-CTA
    transac_bar_t bar_tO_empty, bar_tO_full;
    transac_bar_t bar_tP_full[NUM_P_BUFS], bar_tP_empty[NEED_TP_EMPTY_BAR ? NUM_P_BUFS : 0];
    transac_bar_t bar_SO_full, bar_SO_empty;
    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_li_mi_full, bar_li_mi_empty;
    transac_bar_t bar_raw_kv_full;

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};
static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

struct TMAParams {
    // Prefill only
    CUtensorMap tensor_map_kv;                  // the whole (bf16, non-paged) KV cache
    // Decode only
    CUtensorMap tensor_map_kv_fp8_part_cta0;
    CUtensorMap tensor_map_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta1;
    CUtensorMap tensor_map_kv_bf16_part;        // the bf16 (RoPE) part of the paged KV cache
    CUtensorMap tensor_map_extra_kv_bf16_part;  // the bf16 (RoPE) part of the extra paged KV cache. Invalid if extra_topk == 0
    CUtensorMap tensor_map_extra_kv_fp4_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta1;
};
static constexpr uint32_t D_FP8_CTA0 = CLUSTER_SIZE == 1 ? D_FP8 : D_VO/2;
static constexpr uint32_t D_FP8_CTA1 = D_FP8 - D_FP8_CTA0;
static constexpr bool IS_CTA0_RAW_KV_PADDED = D_FP8_CTA0 % 128 == 0;
static constexpr bool IS_CTA1_RAW_KV_PADDED = D_FP8_CTA1 % 128 == 0;
// Bytes of one raw fp8 row in shared memory, i.e. the box of the fp8 tensor maps: the fp8-only dequant path pads a row by 64 B
// when needed, the common path (HAS_FP4_KV) by 16 B (KVFormat::RAW_TOKEN_SMEM_STRIDE)
static constexpr uint32_t RAW_FP8_TOKEN_SMEM_STRIDE_CTA0 = HAS_FP4_KV ? OrigKVFormat::RAW_TOKEN_SMEM_STRIDE : D_FP8_CTA0 + (IS_CTA0_RAW_KV_PADDED ? 64 : 0);
static constexpr uint32_t RAW_FP8_TOKEN_SMEM_STRIDE_CTA1 = HAS_FP4_KV ? OrigKVFormat::RAW_TOKEN_SMEM_STRIDE : D_FP8_CTA1 + (IS_CTA1_RAW_KV_PADDED ? 64 : 0);

using AllocatorT = std::conditional_t<IS_2CTA, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

struct AuxParams {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};

// Some helper functions for buffer arrival
static __device__ __forceinline__ void umma_arrive_on_every_cta(transac_bar_t &bar) {
    // Perform UMMA arrive, possibly with multicast, to every CTA
    if constexpr (IS_2CTA) {
        ku::umma_arrive_multicast_2x1SM_noelect(bar, 1|2);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void umma_arrive_on_cta0(transac_bar_t &bar) {
    // Perform UMMA arrive on CTA0
    if constexpr (IS_2CTA) {
        ku::umma_arrive_2x1SM_noelect(bar);
    } else {
        ku::umma_arrive_noelect(bar);
    }
}
static __device__ __forceinline__ void arrive_on_cta0_barrier(transac_bar_t &bar) {
    if constexpr (IS_2CTA) {
        bar.arrive(0u);
    } else {
        bar.arrive();
    }
}

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG3_SYNC = 1;
    static constexpr int WG3_WARP02_SYNC = 2;
    static constexpr int WG3_WARP13_SYNC = 3;
};

static __device__ __forceinline__ void
devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params);

static void run(const Params &params);

};

}
