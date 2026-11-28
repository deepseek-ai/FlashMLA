#pragma once

#include "kernel.h"
#include "cuda_kernels/kv_cache_format.h"


#include "kerutils/kerutils_for_ascend_npu.h"

#include "fast_div_mod.h"

static constexpr uint32_t NUM_UINT8_IN_VEC = 256 / sizeof(uint8_t);
static constexpr uint32_t NUM_BF16_IN_VEC = 256 / sizeof(bf16);
static constexpr uint32_t NUM_FLOAT_IN_VEC = 256 / sizeof(float);
static constexpr uint32_t NUM_UINT32_IN_VEC = 256 / sizeof(uint32_t);
static constexpr uint32_t NUM_UINT64_IN_VEC = 256 / sizeof(uint64_t);

namespace ascend::prefill::sparse_fwd {

__aicore__ static constexpr uint32_t constexpr_max(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

__aicore__ static constexpr uint32_t constexpr_min(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

template<Config CONFIG>
struct Kernel {
    
static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;
static constexpr uint32_t NUM_HEAD_SPLITS = ku::ceil_div(H_Q, 64u);    // Number of AI cores that share one q token
static constexpr uint32_t H_Q_PER_AI_CORE = H_Q / NUM_HEAD_SPLITS;   // Number of query heads handled by one AI core
static_assert(H_Q % NUM_HEAD_SPLITS == 0);
static_assert(H_Q_PER_AI_CORE % 8 == 0);
static constexpr uint32_t MMA_M = ku::ceil(H_Q_PER_AI_CORE, 32u);   // Pad MMA_M to multiples of 32 because 1) MMA_M = 32 is already memory-bound (L2 bw bound) 2) MMA_M = 32 corresponds to MMA_M_PER_V = 16 which is the height of one L0C fractal. MMA_M smaller than 32 will cause strange problems when copying from FixPipe to UB
static constexpr uint32_t INDEX_BLOCK_SIZE = CONFIG.INDEX_BLOCK_SIZE;
static constexpr bool HAVE_ATTN_SINK = CONFIG.HAVE_ATTN_SINK;
static constexpr uint32_t O_QUANT_TILE_SIZE = 32;
static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(INDEX_BLOCK_SIZE % NUM_UINT32_IN_VEC == 0);
static constexpr bool HAVE_Q_O_ROPE_AND_O_CAST = CONFIG.HAVE_Q_O_ROPE_AND_O_CAST;
static constexpr bool HAVE_Q_NORM = CONFIG.HAVE_Q_NORM;
static_assert(!(HAVE_Q_NORM && !HAVE_Q_O_ROPE_AND_O_CAST));

using Params = ParamT<CONFIG.FWD_MODE, HAVE_Q_O_ROPE_AND_O_CAST>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

static constexpr bool HAVE_FP4 = IS_DECODE && (MODEL_TYPE == ModelType::V41_FP4 || EXTRA_MODEL_TYPE == ModelType::V41_FP4);
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static_assert(!IS_DECODE || ((MODEL_TYPE == ModelType::V41 || MODEL_TYPE == ModelType::V41_FP4)
    && (EXTRA_MODEL_TYPE == ModelType::V41 || EXTRA_MODEL_TYPE == ModelType::V41_FP4)));

// Model parameters
static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t D_ROPE = 64;
static constexpr uint32_t D_NOPE = D_QK - D_ROPE;
static constexpr uint32_t RESCALE_THRESHOLD = 6;   // O rescale+accumulation won't be triggered unless (cur_max - max_for_scale) * sm_scale > RESCALE_THRESHOLD
static constexpr uint32_t SCALE_GRAN = 32;
static constexpr uint32_t NUM_SCALES_PER_TOKEN = D_QK / SCALE_GRAN;
using FP8Format = KVCacheFormat<ModelType::V41>;
using FP4Format = KVCacheFormat<ModelType::V41_FP4>;
// GM records stay compact; MTE2 destination rows must be aligned to 32 bytes.
static constexpr uint32_t FP8_UB_TOKEN_BYTES = ku::ceil((uint32_t)FP8Format::BYTES_PER_TOKEN, 32u);
static constexpr uint32_t FP4_UB_TOKEN_BYTES = ku::ceil((uint32_t)FP4Format::BYTES_PER_TOKEN, 32u);

// Tile size selection
static constexpr uint32_t NUM_VEC_CORE_PER_AI_CORE = 2;
static constexpr uint32_t MMA_M_PER_V = MMA_M / NUM_VEC_CORE_PER_AI_CORE;
static constexpr uint32_t B_TOPK = 64;
static constexpr uint32_t B_TOPK_PER_V = B_TOPK / NUM_VEC_CORE_PER_AI_CORE;
static_assert(INDEX_BLOCK_SIZE % B_TOPK == 0);

static constexpr uint32_t OUTPUT_FRAG_D_VO = 128;
static constexpr uint32_t NUM_OUTPUT_FRAG_BUFS = 1;
static constexpr uint32_t NUM_OUTPUT_FRAGS = D_VO / OUTPUT_FRAG_D_VO;
static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);

static constexpr uint32_t L12L0_TILE_SIZE = 128;
static constexpr uint32_t NUM_L0A_BUFS = 4;

static constexpr uint32_t NUM_L1_Q_BUFS = 2;
static constexpr uint32_t NUM_L1_KV_BUFS = 4;
static constexpr uint32_t NUM_UB_KV_BUFS = !IS_DECODE ? 2 : 3;
// For prefill, NUM_UB_KV_BUFS = 2 measures consistently ~1% faster than 3. We suspect that more UB
// KV buffers create more bursts on the memory system and increase the tail latency, but we are not
// sure: the memory system is chaotic, so the 1% is hard to attribute to a single cause.
static constexpr uint32_t NUM_INDEX_BUFS = 4;    // Keep indices live across overlapping queries until softmax consumes them
static constexpr uint32_t NUM_SM_SCALE_DIV_RMS_BUFS = 4;    // We have a total pipeline depth of 8, and each q token has at least 2 blocks to process (controlled by the `max` in `_get_outer_loop_args`), so 8/2=4 bufs for sm_scale_div_rms is enough
static constexpr uint32_t NUM_ROPE_COS_SIN_BUFS = 4;    // The same reason as above
static constexpr uint32_t NUM_ROW_DENORM_BUFS = 2;  // 2 is enough since it's produced in `ACTIVATE(3, run_get_final_lse_and_denorm);` and consumed in o copy out stage 1 or 2 (`ACTIVATE(1, run_o_copy_out)` or `ACTIVATE(0, run_o_copy_out)`), and `ACTIVATE(0, run_o_copy_out)` runs before `ACTIVATE(3, run_get_final_lse_and_denorm);` in the systolic array loop.

// Q fragment on UB, used when Q RoPE / Q norm is enabled
static constexpr uint32_t Q_FRAG_D_QK = 128;
static constexpr uint32_t NUM_UB_Q_FRAG_BUFS = 2;
static constexpr uint32_t NUM_Q_FRAGS = D_QK / Q_FRAG_D_QK;
static_assert(D_QK % Q_FRAG_D_QK == 0);

static constexpr uint32_t FRACTAL_H = 16;   // In number of rows
static constexpr uint32_t FRACTAL_W = 16;   // In number of elements

static constexpr uint32_t GRID_SHAPE = 32;
static_assert(GRID_SHAPE % NUM_HEAD_SPLITS == 0);
// Number of q tokens by which the outer loop advances on one AI core. With a head split the GRID_SHAPE
// cores are paired up (core 2i and core 2i+1 handle the same q tokens), so each core only walks
// GRID_SHAPE / NUM_HEAD_SPLITS q tokens per step
static constexpr uint32_t Q_TOKENS_PER_JOB_STEP = GRID_SHAPE / NUM_HEAD_SPLITS;

using gathered_kv_t = std::conditional_t<IS_DECODE, float8_e4m3_t, bf16>;
struct UnifiedBufferMemoryPlan {
    float output_accum[NUM_OUTPUT_FRAGS][MMA_M_PER_V * OUTPUT_FRAG_D_VO];
    float cur_output_frag[NUM_OUTPUT_FRAG_BUFS][MMA_M_PER_V][OUTPUT_FRAG_D_VO];

    gathered_kv_t gathered_kv[NUM_UB_KV_BUFS][B_TOPK_PER_V][IS_DECODE ? FP8_UB_TOKEN_BYTES : D_VO];
    bf16 gathered_kv_in_nz[(B_TOPK_PER_V+1)*D_VO];    // Add extra padding to avoid bank conflict during nd->nz

    float p[MMA_M_PER_V*B_TOPK];

    float attn_sink[HAVE_ATTN_SINK ? MMA_M_PER_V : 0];

    // The following two row_max are NOT scaled (NOT multiplied by sm_scale)
    float row_max[2][constexpr_max(MMA_M_PER_V, NUM_FLOAT_IN_VEC)];    // row0 row1 row2 ...
    float row_max_for_o[2][constexpr_max(MMA_M_PER_V, NUM_FLOAT_IN_VEC)];
    
    // One vector of [8 rows][8 column pairs] partial sums per 8 rows: row_sum[:, 0:8] are the pair sums that form row0,
    // row_sum[:, 8:16] are the pair sums for row1, and so on.
    float row_sum[2][(MMA_M_PER_V/8)*NUM_FLOAT_IN_VEC];
    float row_denorm[NUM_ROW_DENORM_BUFS][4*NUM_FLOAT_IN_VEC];  // For cases where `HAVE_Q_O_ROPE_AND_O_CAST` is true, each number is repeated for 4 times. Otherwise, no repeat. Use 4*NUM_FLOAT_IN_VEC instead of `constexpr_max` because we always write 4 `vector_float` into `row_denorm` regarding `MMA_M_PER_V`

    float row_rescale_factor[2][2][NUM_FLOAT_IN_VEC];   // row0 row1 row2 ..., not defined for KV block0 in each q token

    int32_t indices[INDEX_BLOCK_SIZE];

    // Indices after transformation
    // We pack every two KV tokens together and use one MTE2 copy (who has two rows) to copy them in together
    // The SIMD VF (process_kv_index_vf) is responsible for generating base pointers and strides for the MTE2 copy
    // Re-calculated as long as kv_block_idx % INDEX_BLOCK_SIZE == 0
    // All indices are being viewed and processed as uint32, so that negative invalid indices will be converted to numbers > INT_MAX and we only need to check whether a index is >= params.s_kv to see whether it's valid
    // All gathers for the previous address group are issued before process_kv_index_vf
    // can replace it, so one address buffer is sufficient.
    // Softmax consumes smaller_indices/larger_indices later and needs multiple index buffers.
    int64_t global_kv_ptr_offset[INDEX_BLOCK_SIZE / 2];
    int64_t src_stride_in_gather2[INDEX_BLOCK_SIZE / 2];
    uint32_t smaller_indices[NUM_INDEX_BUFS][INDEX_BLOCK_SIZE / 2];
    uint32_t larger_indices[NUM_INDEX_BUFS][INDEX_BLOCK_SIZE / 2];
    static_assert((INDEX_BLOCK_SIZE/2) % 8 == 0);
    
    // RoPE cos&sin buffer: (D_ROPE/2) x cos, followed by (D_ROPE/2) x sin
    // The reading may OOB for `D_ROPE/2` floats, so this array should never be put at the end of UB
    alignas(32) float rope_cos_sin_buf[HAVE_Q_O_ROPE_AND_O_CAST ? NUM_ROPE_COS_SIN_BUFS : 0][D_ROPE];

    // Originally, we'd like to prevent NaN appearing in `gathered_kv` by:
    //  - Filling `gathered_kv` by valid values at the beginning of the kernel
    //  - When a token is invalid, skip its gather
    // This keeps values in `gathered_kv` always valid.
    // However, packed FP4 bytes can be NaN encodings when interpreted as FP8. So we separate fp4 and fp8's raw slots to avoid this.
    uint8_t gathered_kv_fp4[HAVE_FP4 ? NUM_UB_KV_BUFS : 0][B_TOPK_PER_V][FP4_UB_TOKEN_BYTES];

    // An scratch for dequant VF.
    uint8_t vf_scratchpad[
        constexpr_max(
            (IS_DECODE ? B_TOPK_PER_V : 0) * NUM_SCALES_PER_TOKEN * sizeof(bf16),   // For FP8 KV dequant
            (IS_DECODE && HAVE_FP4 ? B_TOPK_PER_V : 0) * FP4Format::NUM_SCALES_EACH_TOKEN * sizeof(bf16)    // For FP4 KV dequant
        )
    ];

    // Scratchpad for `get_final_output_vf`
    alignas(32) float o_block_max[HAVE_Q_O_ROPE_AND_O_CAST ? constexpr_max(MMA_M_PER_V * (D_VO / SCALE_GRAN) * 4, NUM_FLOAT_IN_VEC) : 0];

    alignas(32) uint8_t o_scale_factors[HAVE_Q_O_ROPE_AND_O_CAST ? constexpr_max(MMA_M_PER_V * (D_VO / SCALE_GRAN), NUM_UINT8_IN_VEC) : 0];

    alignas(32) bf16 q_frag[HAVE_Q_O_ROPE_AND_O_CAST ? NUM_UB_Q_FRAG_BUFS : 0][MMA_M_PER_V * Q_FRAG_D_QK]; // In NZ layout

    alignas(32) float q_norm_accum_wksp[HAVE_Q_O_ROPE_AND_O_CAST ? (MMA_M_PER_V/4) * NUM_FLOAT_IN_VEC : 0];
    alignas(32) float sm_scale_div_rms[HAVE_Q_O_ROPE_AND_O_CAST ? NUM_SM_SCALE_DIV_RMS_BUFS : 0][constexpr_max(MMA_M_PER_V, NUM_FLOAT_IN_VEC)]; // One value per head

    alignas(64) float row_max_max_delta[2][2];  // Must be put at the end to guarantee alignment of the previous arrays. alignas(64) makes it cover a cache line, which `asc_ub_dcci_single` requires
};
static_assert(sizeof(UnifiedBufferMemoryPlan) <= 256 * 1024);   // We consider UB's usable capacity to be 256 KB rather than 248 KB, because our VFs never spill registers, which is guaranteed by `--cce-disable-vf-stack-reserved-ubuf` in `setup.py`

struct SSBufferMemoryPlan {
    uint32_t is_rescale_triggered[8][8][NUM_VEC_CORE_PER_AI_CORE];    // Allocate 8 slots for job_idx and 8 slots for kv_block_idx so that there will be no data race
};

static constexpr uint32_t NUM_Q_TOKENS_PER_SF_TRANSPOSE = IS_DECODE ? 16 : (H_Q > 64 ? 16 : 32);
static constexpr uint32_t NUM_SF_TRANSPOSE_BUFS = 4;
struct UnifiedBufferMemoryPlanForSFTranspose {
    uint8_t output_accum_placeholder[sizeof(UnifiedBufferMemoryPlan::output_accum)];    // To prevent it being overwritten
    uint8_t buf[NUM_SF_TRANSPOSE_BUFS][NUM_Q_TOKENS_PER_SF_TRANSPOSE * H_Q * NUM_SCALES_PER_TOKEN];
};
static_assert(sizeof(UnifiedBufferMemoryPlanForSFTranspose) <= 256 * 1024);

struct CrossCoreFlags {
    // C -> V
    static constexpr uint32_t UBUF_P_FULL = 0;
    static constexpr uint32_t L1_S_EMPTY = 1;
    static constexpr uint32_t SS_BUFFER_FULL_C2V = 2;
    static constexpr uint32_t UBUF_O_FRAG_FULL = 4;     // [NUM_OUTPUT_FRAG_BUFS]
    static constexpr uint32_t UBUF_O_ACCUM_FULL = UBUF_O_FRAG_FULL + NUM_OUTPUT_FRAG_BUFS;  // [NUM_OUTPUT_FRAGS]
    static constexpr uint32_t L1_KV_BUF_EMPTY = UBUF_O_ACCUM_FULL + NUM_OUTPUT_FRAGS;       // [NUM_L1_KV_BUFS]
    static constexpr uint32_t UBUF_Q_FRAG_FULL = L1_KV_BUF_EMPTY + NUM_L1_KV_BUFS;          // [NUM_UB_Q_FRAG_BUFS]
    static_assert(UBUF_Q_FRAG_FULL + NUM_UB_Q_FRAG_BUFS <= 16);
    // V -> C
    static constexpr uint32_t UBUF_P_EMPTY = 0;
    static constexpr uint32_t L1_S_FULL = 1;
    static constexpr uint32_t SS_BUFFER_FULL_V2C = 2;
    static constexpr uint32_t UBUF_O_FRAG_EMPTY = 4;    // [NUM_OUTPUT_FRAG_BUFS]
    static constexpr uint32_t UBUF_O_ACCUM_EMPTY = UBUF_O_FRAG_EMPTY + NUM_OUTPUT_FRAG_BUFS;// [NUM_OUTPUT_FRAGS]
    static constexpr uint32_t L1_KV_BUF_FULL = UBUF_O_ACCUM_EMPTY + NUM_OUTPUT_FRAGS;       // [NUM_L1_KV_BUFS]
    static constexpr uint32_t UBUF_Q_FRAG_EMPTY = L1_KV_BUF_FULL + NUM_L1_KV_BUFS;          // [NUM_UB_Q_FRAG_BUFS]
    static constexpr uint32_t L1_Q_ROPE_FRAG_FULL = UBUF_Q_FRAG_EMPTY + NUM_UB_Q_FRAG_BUFS;
    static_assert(L1_Q_ROPE_FRAG_FULL < 16);
};

struct VectorCoreFlags {
    // V -> MTE3
    static constexpr uint32_t S_FULL = 0;
    static constexpr uint32_t Q_FRAG_BUF_FULL = 1;  // [NUM_UB_Q_FRAG_BUFS]
    // MTE3 -> V
    static constexpr uint32_t Q_FRAG_BUF_EMPTY = 0; // [NUM_UB_Q_FRAG_BUFS]
    // V -> MTE2
    static constexpr uint32_t ROPE_COS_SIN_BUF_EMPTY = 0;    // [NUM_ROPE_COS_SIN_BUFS]
    // MTE2 -> V
    static constexpr uint32_t ATTN_SINK_FULL = 0;
    static constexpr uint32_t ROPE_COS_SIN_BUF_FULL = 1;    // [NUM_ROPE_COS_SIN_BUFS]
    // V -> S
    static constexpr uint32_t ROW_MAX_MAX_DELTA_FULL = 0;
    // MTE3 -> MTE2
    static constexpr uint32_t ATTN_SCORE_FULL = 0;
};

struct VectorCoreBufIDs {
    static constexpr uint32_t RAW_INDICES = 0;
    static constexpr uint32_t PROCESSED_INDICES = 1;
    static constexpr uint32_t FINAL_LOGITS_LSE = 2; // [2]
    static constexpr uint32_t O_ACCUM = 4;  // [D_VO / OUTPUT_FRAG_D_VO] = [4]
    static constexpr uint32_t GATHERED_KV = 24; // [NUM_UB_KV_BUFS]
    static constexpr uint32_t KV_IN_NZ = 28;
};

struct CubeCoreFlags {
    // MTE2 -> MTE1
    static constexpr uint32_t L1_Q_FULL = 0;    // [2]
    // MTE1 -> MTE2
    static constexpr uint32_t L1_Q_EMPTY = 0;   // [2]
    // M -> FIX
    static constexpr uint32_t L0C_O_FULL = 0;
    // FIX -> M
    static constexpr uint32_t L0C_O_EMPTY = 0;
};

struct CubeCoreBufIDs {
    static constexpr uint32_t L0A_BUF = 0;
    static constexpr uint32_t L0B_BUF = L0A_BUF + NUM_L0A_BUFS;
};

struct InterCoreFlags {
    static constexpr uint32_t SYNC_BEFORE_SF_TRANSPOSE = 0;
};

template<ModelType CACHE_MODEL_TYPE = MODEL_TYPE>
static __simd_vf__ void process_kv_index_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t index_buf_idx, uint32_t s_kv, uint32_t stride_kv_block, int valid_len, uint32_t page_block_size, FastDivMod page_block_size_fast_div_mod);
static __simd_vf__ void clear_kv_buf_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);
static __simd_vf__ void kv_nd2nz_for_prefill_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);
template<ModelType CACHE_MODEL_TYPE = MODEL_TYPE>
static __simd_vf__ void kv_dequant_and_nd2nz_for_decode_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);

static __simd_vf__ void q_norm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, uint32_t q_frag_idx, float sm_scale, float rms_norm_eps);
static __simd_vf__ void q_rope_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t rope_cos_sin_buf_idx);

template<bool IS_BLOCK0> static __simd_vf__ void softmax_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, float sm_scale, uint32_t job_idx, uint32_t kv_block_idx, uint32_t index_buf_idx, uint32_t ub_indices_arr_offset, uint32_t s_kv, uint32_t subblock_idx);
template<bool PERFORM_ADD> static __simd_vf__ void add_and_scale_o_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, float sm_scale, uint32_t job_idx, uint32_t kv_block_idx);
static __simd_vf__ void get_final_lse_and_denorm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, float sm_scale);

template<bool HAVE_PREVIOUS_OUTPUT_ACCUM> static __simd_vf__ void get_final_output_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx);
template<bool HAVE_PREVIOUS_OUTPUT_ACCUM> static __simd_vf__ void get_final_output_vf_with_cast_and_rope_stage0(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx);
template<bool HAVE_PREVIOUS_OUTPUT_ACCUM> static __simd_vf__ void get_final_output_vf_with_cast_and_rope_stage1(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, uint32_t rope_cos_sin_buf_idx);
static __simd_vf__ void get_final_output_vf_with_cast_and_rope_stage2(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx);

struct AuxParams {
    FastDivMod page_block_size_fast_div_mod;
    FastDivMod extra_page_block_size_fast_div_mod;
};

static __aicore__ void sparse_attn_fwd_kernel_devfunc(const Params params, const AuxParams aux_params);

static void run(const Params &params);

};

}
