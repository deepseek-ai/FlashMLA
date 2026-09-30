/*
Fused "Q Norm + Q RoPE + Core Attention + O RoPE + O FP8 Cast" kernel for DeepSeek V4.1
(d_qk = d_v = 512, h_kv = 1, token-level sparse attention) on SM100f.

Instead of launching separate kernels for RMSNorm, RoPE, attention, and output quantization, this
kernel fuses the whole "q_b_proj output -> wv_proj input" segment of the MLA block:
1. Q Norm (ENABLE_Q_NORM, the API's `enable_q_norm`): computes the per-head RMSNorm denominator
   rsqrt(sum(q^2)/d_qk + eps) on the fly while loading Q, and folds it into the softmax scale
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
- MODEL_TYPE: the format of the `kv` cache: ModelType::V41 (V4.1 fp8 layout)
- EXTRA_MODEL_TYPE: the format of the extra KV cache (decode only): ModelType::V41, or ModelType::V41_FP4
              (V4.1 with an fp4 extra KV cache, e4m3 per-16 scales); prefill passes V41
- H_Q:        number of Q heads: 64 (1 CTA) or 128 (2-CTA cluster)
- ENABLE_Q_NORM: the API's `enable_q_norm` flag

I/O (see `csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp` and the Python docstrings in
`flash_mla/fused_norm_rope_attn_rope_cast.py` for the full parameter list):
- q: [s_q, h_q, d_qk], bf16, WITHOUT RoPE applied, in the PERMUTED layout produced by the
  `permute_q_b_proj` kernel (16-element d-chunks interleaved across heads). Each token's
  h_q*d_qk elements must be contiguous
- Prefill: kv [s_kv, 1, d_qk] (bf16, non-paged) + indices [s_q, 1, topk]
  Decode:  paged quantized KV cache(s) (V4.1 fp8 / V4.1 fp4 format, see below) + indices_in_kvcache [s_q, topk],
  plus an optional secondary ("extra") KV cache with its own indices / topk_length
- out_fp8: [s_q, n_wv_group, wv_group_size * d_v], fp8_e4m3, in the permuted layout expected by
  the `permute_wv_proj`-transformed weights; out_sf: packed ue8m0 scale factors (always per-32)
- lse / max_logits (prefill only for max_logits): [s_q, h_q], fp32

Execution structure:
- Persistent kernel scheduled via CLC (Cluster Launch Control); the grid is
  (s_q * CLUSTER_SIZE, 1, 1) and each cluster processes one query token per job
- CLUSTER_SIZE = H_Q / 64: 1 CTA for H_Q = 64, a 2-CTA cluster (dual-CTA UMMA, CTA0 owns V[:, 0:256] and
  CTA1 owns V[:, 256:512]) for H_Q = 128. Each CTA has 512 threads (4 warpgroups):
  - WG0: Q fetching (q_sqr_sum + Q RoPE + store to TMEM) & O epilogue (TMEM load, O RoPE,
    FP8 quant, store to gmem). Timeline: Q0 Q1 O0 Q2 O1 ... Qn O(n-1) On
  - WG1: KV producer. Prefill: gathers the bf16 KV block via TMA gather4. Decode: loads the fp8 / fp4
    part from the paged cache and dequantizes it into the smem KV slots in registers
  - WG2: MMA warp (warp 8, issues UMMAs on CTA0 only), CLC warp (warp 9), indices / validity
    mask generator (warp 10), and warp 11: the decode KV producer's gather front (idle in prefill)
  - WG3: Scale & Exp: reduces P, maintains the online softmax state (mi / li), produces S
- KV tokens are processed in blocks of B_TOPK (96 for 2-CTA, 64 for 1-CTA) with NUM_KV_SLOTS-deep
  software pipelining

Multi-rail GeMM is always used:
- For cases where CLUSTER_SIZE is 1, we FOLD Q by FOLD_FACTOR and do a "batched gemm" in batch size = FOLD_FACTOR, and reduce P on shared memory
- For cases where CLUSTER_SIZE is 2, we also fold Q by FOLD_FACTOR and perform batched GeMM, and reduce P on shared memory

Decoding mode (FWD_MODE == SparseAttnFwdMode::Decode):
- Batch size must be 1; the grid covers the s_q query tokens
- Each KV cache token stores its raw data followed immediately by its scales: 512 + 16 B for fp8,
  or 256 + 32 B for fp4. Warp 11 gathers raw data and scales together with TMA gather4, except that
  CTA0 of h128 gathers only its raw half and uses cp.async for its scale subset. Both transports complete
  on the same per-block barrier. WG1 dequantizes in place into the slot's bf16 layout (DequantRows)
- EXTRA_MODEL_TYPE == ModelType::V41_FP4 selects an fp4 extra KV cache: every token is 512 e2m1 + 32 e4m3 scales,
  each page block stores [page_block_size, 288 B] interleaved token rows. A KV block may then
  mix fp8 tokens of `kv` and fp4 tokens of `extra_kv`; the format is resolved per 8 rows
- Split-KV is not supported (the fused RoPE + FP8-quant epilogue cannot be combined across splits)
*/

#pragma once

#include <cutlass/fast_math.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/defines.h"
#include "params.h"
#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/kv_cache_utils.cuh"

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
// Each CTA computes 64 rows of Q / P / O, and H_Q is 64 (1 CTA) or 128 (2 CTAs), so every row of Q / P / O is a head. A .ws UMMA
// instruction takes about 12 + N/4 cycles for M = 64 (the tensor datapath is 64 rows wide), so M decides the TMEM / shared-memory
// footprint and the softmax work per thread
static constexpr uint32_t H_Q_PER_CTA = 64;   // rows of Q / P / O per CTA
static constexpr uint32_t MMA_M = CLUSTER_SIZE * H_Q_PER_CTA;   // M of the UMMA atom: the 2-CTA atom spans both CTAs' rows

template<ModelType MT>
struct KVFormat : KVCacheFormat<MT> {
    using Base = KVCacheFormat<MT>;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;   // The number of elements that one LDS.128 can load
};

// An odd number of 16 B chunks per row, together with the rotation in DequantRows,
// keeps LDS.128 conflict-free. Both CTAs share this row stride and one dequant body.
template<ModelType MT, uint32_t CTA>
struct KVFormatCta : KVCachePart<KVFormat<MT>, CLUSTER_SIZE, CTA> {
    using F = KVFormat<MT>;
    using Part = KVCachePart<F, CLUSTER_SIZE, CTA>;
    static constexpr uint32_t NUM_QUANT_CHUNKS_PER_ROW = Part::QUANT_BYTES / 16;
    static constexpr uint32_t NUM_QUANT_CHUNKS_PER_LANE = NUM_QUANT_CHUNKS_PER_ROW / 4;
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = (ku::ceil_div(Part::QUANT_BYTES + F::NUM_SCALES_EACH_TOKEN, 16) | 1) * 16;
    static_assert(NUM_QUANT_CHUNKS_PER_ROW % 4 == 0);
};
using OrigKVFormat = KVFormat<MODEL_TYPE>;          // Format of `kv`
using ExtraKVFormat = KVFormat<EXTRA_MODEL_TYPE>;   // Format of `extra_kv`
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static constexpr bool HAS_FP4_KV = ExtraKVFormat::IS_FP4;   // The extra KV cache is fp4
// Shorthands for the format of `kv`, which is also the format of `extra_kv` unless HAS_FP4_KV
static constexpr uint32_t D_FP8 = OrigKVFormat::D_FP8;
static constexpr uint32_t KV_QUANT_TILE_SIZE = OrigKVFormat::QUANT_TILE_SIZE;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN = OrigKVFormat::NUM_SCALES_EACH_TOKEN;
static constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = NUM_SCALES_EACH_TOKEN / CLUSTER_SIZE;
static constexpr uint32_t KV_CACHE_BYTES_PER_TOKEN = OrigKVFormat::BYTES_PER_TOKEN;
// Decode gathers the raw rows of a KV block into the beginning of the KV slot (one 4-row group per gather4) and dequantizes
// them in place. A group is padded to 128 B: the destination of a gather4 must be 128 B aligned (a 64 B aligned one raises a
// misaligned address error; the PTX ISA's 16 B does not hold for this instruction). Both CTAs of a cluster use the same group
// stride. Both caches hold the same dims, so a CTA's rows of both are placed the same way
static constexpr uint32_t RAW_KV_GROUP_BYTES = ku::ceil_div(4 * std::max({
    KVFormatCta<MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE, KVFormatCta<MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE,
    KVFormatCta<EXTRA_MODEL_TYPE, 0>::RAW_TOKEN_SMEM_STRIDE, KVFormatCta<EXTRA_MODEL_TYPE, CLUSTER_SIZE - 1>::RAW_TOKEN_SMEM_STRIDE}), 128u) * 128;
static_assert(!IS_DECODE || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= B_TOPK * D_QK / CLUSTER_SIZE * sizeof(bf16));   // the raw rows fit in the slot
// CTA0 stages its scale subset in the unused tail of the KV slot. A mixed fp8/fp4 block uses
// a common 16 B row; a pure fp8 block needs 8 B per token. All raw and scale bytes are read before
// the dequant warpgroup's named barrier allows bf16 write-back to overwrite the slot.
static constexpr uint32_t SCALE_STAGE_ROW_BYTES = HAS_FP4_KV ? 16 : NUM_SCALES_EACH_TOKEN_PER_CTA;
static constexpr uint32_t SCALE_STAGE_OFFSET = B_TOPK * (D_QK / CLUSTER_SIZE * sizeof(bf16) - SCALE_STAGE_ROW_BYTES);
static_assert(!IS_DECODE || !IS_2CTA || B_TOPK / 4 * RAW_KV_GROUP_BYTES <= SCALE_STAGE_OFFSET);

// Decode's per-block code nearly fills the instruction cache (h128 q-norm kernel: hit rate 82.7% with the per-job
// code unrolled, 89.6% with it compact), so the code that runs once per job (the O epilogue and the q-norm sum of
// squares) is kept small there, which adds some latency to each job: loops rolled, sums read back from TMEM.
// Prefill's code fits in the cache, and its per-job code is on the critical path between jobs, so it takes the
// unrolled forms
static constexpr bool COMPACT_PER_JOB_CODE = IS_DECODE;

// Registers per thread of each warpgroup.
// Decode's WG1 dequantizes a KV block in registers and needs 128. The prefill KV producer issues TMA gathers and uses
// about 50, so the 1-CTA prefill kernels give WG0 256 for the software-pipelined O epilogue (EPILOGUE_TILES_AHEAD)
static constexpr bool WIDE_WG0 = !IS_DECODE && CLUSTER_SIZE == 1;
static constexpr uint32_t WG0_REGS = WIDE_WG0 ? 256 : 184;
static constexpr uint32_t WG1_REGS = WIDE_WG0 ? 56 : 128;
static constexpr uint32_t WG2_REGS = 72;
static constexpr uint32_t WG3_REGS = 128;
static_assert(WG0_REGS + WG1_REGS + WG2_REGS + WG3_REGS == 4 * 128);
// setmaxnreg takes a multiple of 8 in [24, 256]
static_assert(WG0_REGS % 8 == 0 && WG1_REGS % 8 == 0 && WG2_REGS % 8 == 0 && WG3_REGS % 8 == 0);
static_assert(24 <= WG1_REGS && 24 <= WG2_REGS && WG0_REGS <= 256 && WG3_REGS <= 256);
// Q load: a thread's 16 LDG.256 are issued one 16 KB tile at a time, each tile's addresses depending on the data of
// the previous tile (load_q_and_save_to_tmem). 1-CTA kernels only: on the 2-CTA kernels the burst overlaps no
// shared-memory instruction of the critical path, so the grouping has nothing to hide and its serial load latencies
// only add to the job
static constexpr bool GROUP_Q_LOADS = CLUSTER_SIZE == 1;

static constexpr uint32_t NUM_THREADS = 512;
// Every thread that calls get_next_job arrives on bar_clc_empty once per CLC round, and this count
// is that barrier's expected arrival count. In decode warp 11 runs the job loop too (its front
// crosses job boundaries), so its 32 threads are counted
static constexpr uint32_t NUM_WORKING_THREADS =
    CLUSTER_SIZE == 1 ? (
        IS_DECODE ?
        128 + 128 + (1+1+32) + 128 + 32 :   // WG0 + WG3 + (MMA + CLC + indices) + WG1 (dequant) + warp 11 (gather front)
        128 + 128 + (1+1+32) + 4                            // WG0 + WG3 + (MMA + CLC + indices) + WG1 (KV producer, 1 elected thread per warp)
    ) : (
        IS_DECODE ?
        128*2 + 128*2 + (1+2+32*2) + 128*2 + 32*2 :   // ... + both CTAs' warp 11
        128*2 + 128*2 + (1+2+32*2) + 4*2
    );

// The 128 TMEM lanes form FOLD_FACTOR groups of H_Q_PER_CTA lanes. A .ws UMMA instruction reads the A operand of lane group g from
// that group's lanes and writes the g-th N-slice of D to them, so the QK GEMM runs NUM_MRGEMM_RAILS "rails": B is the KV block
// viewed as NUM_MRGEMM_RAILS x B_TOPK rows of D_QK / NUM_MRGEMM_RAILS dims (rail j = 64-dim atom columns j, j+2, j+4, j+6), and lane
// group g holds head rows x the K-view of rail g. The rails' partial P are summed through shared memory by the softmax warpgroup
static constexpr uint32_t FOLD_FACTOR = 128 / H_Q_PER_CTA;
static constexpr uint32_t NUM_MRGEMM_RAILS = 2; // The number of "rails" (batch size) during multi-rail GeMM. Currently must be 2
static_assert(FOLD_FACTOR % NUM_MRGEMM_RAILS == 0);
static constexpr uint32_t NUM_P_ELEMS_PER_THREAD = H_Q_PER_CTA * B_TOPK / 128;
// O epilogue tiles per job: (D_VO / FOLD_FACTOR) columns of O per thread in tiles of O_QUANT_TILE_SIZE
static constexpr uint32_t NUM_EPILOGUE_TILES = D_VO / FOLD_FACTOR / O_QUANT_TILE_SIZE;
// O epilogue: the TMEM load of a tile is issued this many tiles before the tile is converted; each tile loaded ahead
// holds 32 fp32 registers, so 6 tiles need the 256 of WIDE_WG0. 0 selects the per-tile loop: a tile's
// load, conversion and stores per iteration. The 2-CTA prefill kernels use 0. Their first SV of a job waits on the softmax's
// first S, and S is published later than tO_empty, so an earlier tO_empty does not start the SV earlier, while the stores the
// pipeline concentrates into a shorter time delay that softmax block
static constexpr uint32_t EPILOGUE_TILES_AHEAD = WIDE_WG0 ? std::min(6u, NUM_EPILOGUE_TILES) : 0;

static constexpr uint32_t NUM_KV_SLOTS = 3;
static constexpr uint32_t NUM_INDICES_BUFS = 4;
// Raw-gather barriers: an alternating pair, so two gather batches can be issued but not completed at the same
// time: the front of block k+1 waits for the dequant warps to be past block k-1, the previous user of
// bar_raw_kv_full[(k+1)%2], not past block k. With one barrier the front of block k+1 could be issued only
// after block k's rows had arrived, so warp 11's front time added to the transport time of every block
static constexpr uint32_t NUM_RAW_BARS = 2;
static constexpr uint32_t NUM_P_BUFS = CLUSTER_SIZE == 2 ? 1 : 2;
static constexpr uint32_t NEED_TP_EMPTY_BAR = NUM_P_BUFS == 1;  // Don't need to wait for P's emptiness as long as P has >= 2 buffers, since "we are issuing P[i]" <-- "O[i-2] has been issued" <-- "S[i-2] is ready" <-- "P[i-2] is free"

struct tmem_cols {
    // O: D_VO / FOLD_FACTOR = 256 columns; Q: the K-view of one rail per lane group, D_QK / NUM_MRGEMM_RAILS bf16 = 128 columns;
    // P: B_TOPK x NUM_MRGEMM_RAILS fp32 per lane group = 64 columns each
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
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>,
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, MMA_M, B_TOPK * NUM_MRGEMM_RAILS, UMMA::Major::K, UMMA::Major::K>
>;
using TiledMMA_QK = decltype(make_tiled_mma(MMAAtom_QK{}));
using TiledMMA_SV = cute::conditional_t<
    CLUSTER_SIZE == 2,
    decltype(make_tiled_mma(
        SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
    )),
    decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, MMA_M, 256, UMMA::Major::K, UMMA::Major::MN>{}))
>;

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK / CLUSTER_SIZE];    // Cluster size = 1: the whole KV; cluster size = 2: half KV
    CUTE_ALIGNAS(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    CUTE_ALIGNAS(1024) float p_exchange_buf[4][32*NUM_P_ELEMS_PER_THREAD];
    CUTE_ALIGNAS(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][ku::find_next_power_of_2(B_TOPK/8)];
    // Decode: WG10 produces metadata once for the four dequant warps. 16 B aligned so that the fp4 path can read
    // the coordinates of 4 consecutive rows with one LDS.128
    CUTE_ALIGNAS(16) int decode_tma_coords[IS_DECODE ? NUM_INDICES_BUFS : 0][B_TOPK];
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
    // Completion of a block's raw gathers; block k uses index k % NUM_RAW_BARS (see NUM_RAW_BARS)
    transac_bar_t bar_raw_kv_full[IS_DECODE ? NUM_RAW_BARS : 0];
    // Dequant has passed the raw+scale completion wait; warp 11 may reuse this barrier index.
    transac_bar_t bar_wg1_past_raw[IS_DECODE ? NUM_RAW_BARS : 0];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};
static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

struct TMAParams {
    // Prefill only
    CUtensorMap tensor_map_kv;                  // the whole (bf16, non-paged) KV cache
    // Decode only: the raw rows of each CTA's part of a token (KVFormatCta), per paged KV cache
    CUtensorMap tensor_map_kv_fp8_part_cta0;
    CUtensorMap tensor_map_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp8_part_cta1;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta0;
    CUtensorMap tensor_map_extra_kv_fp4_part_cta1;
};

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
    static constexpr int WG1_DEQUANT_SYNC = 7;  // The dequant warpgroup, between reading a block's raw rows and writing its bf16 rows back
};

static __device__ __forceinline__ void
devfunc(const Params &params, const TMAParams &tma_params, const AuxParams &aux_params);

static void run(const Params &params);

};

}
