#pragma once

#include "kernel.h"
#include "cuda_kernels/kv_cache_format.h"

#include "kerutils/kerutils_for_ascend_npu.h"

#include "fast_div_mod.h"

static constexpr uint32_t NUM_BF16_IN_VEC = 256 / sizeof(bf16);
static constexpr uint32_t NUM_FLOAT_IN_VEC = 256 / sizeof(float);
static constexpr uint32_t NUM_UINT32_IN_VEC = 256 / sizeof(uint32_t);
static constexpr uint32_t NUM_UINT64_IN_VEC = 256 / sizeof(uint64_t);

namespace ascend::prefill::sparse_fwd {

__aicore__ static constexpr uint32_t constexpr_max(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

template<Config CONFIG>
struct Kernel {
    
static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;
static constexpr uint32_t MMA_M = ku::ceil(H_Q, 32u);   // Pad MMA_M to multiples of 32 because 1) MMA_M = 32 is already memory-bound (L2 bw bound) 2) MMA_M = 32 corresponds to MMA_M_PER_V = 16 which is the height of one L0C fractal. MMA_M smaller than 32 will cause strange problems when copying from FixPipe to UB
static constexpr uint32_t INDEX_BLOCK_SIZE = CONFIG.INDEX_BLOCK_SIZE;
static constexpr bool HAVE_ATTN_SINK = CONFIG.HAVE_ATTN_SINK;
static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(INDEX_BLOCK_SIZE % NUM_UINT32_IN_VEC == 0);
using Params = ParamT<CONFIG.FWD_MODE>;
static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool HAVE_FP4 = IS_DECODE && (MODEL_TYPE == ModelType::V41_FP4 || EXTRA_MODEL_TYPE == ModelType::V41_FP4);
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));
static_assert(!IS_DECODE || ((MODEL_TYPE == ModelType::V41 || MODEL_TYPE == ModelType::V41_FP4)
    && (EXTRA_MODEL_TYPE == ModelType::V41 || EXTRA_MODEL_TYPE == ModelType::V41_FP4)));

// Model parameters
static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t RESCALE_THRESHOLD = 6;   // O rescale+accumulation won't be triggered unless (cur_max - max_for_scale) * sm_scale <= RESCALE_THRESHOLD
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
static constexpr uint32_t NUM_OUTPUT_FRAG_BUFS = 2;
static_assert(D_VO % OUTPUT_FRAG_D_VO == 0);

static constexpr uint32_t L12L0_TILE_SIZE = 128;
static constexpr uint32_t NUM_L0A_BUFS = 4;

static constexpr uint32_t NUM_L1_Q_BUFS = 2;
static constexpr uint32_t NUM_L1_KV_BUFS = 4;
static constexpr uint32_t NUM_UB_KV_BUFS = 3;
static constexpr uint32_t NUM_INDEX_BUFS = 4;    // Since `ACTIVATE(1, run_softmax)` is followed by `ACTIVATE(5, run_process_index)`, at least 4 index bufs are necessary

static constexpr uint32_t FRACTAL_H = 16;   // In number of rows
static constexpr uint32_t FRACTAL_W = 16;   // In number of elements

static constexpr uint32_t GRID_SHAPE = 32;

using gathered_kv_t = std::conditional_t<IS_DECODE, float8_e4m3_t, bf16>;
struct UnifiedBufferMemoryPlan {
    gathered_kv_t gathered_kv[NUM_UB_KV_BUFS][B_TOPK_PER_V][IS_DECODE ? FP8_UB_TOKEN_BYTES : D_VO];
    bf16 fp8_scales_bf16[IS_DECODE ? B_TOPK_PER_V : 0][NUM_SCALES_PER_TOKEN];    // VF scratch, serialized by KV_IN_NZ

    bf16 gathered_kv_in_nz[(B_TOPK_PER_V+1)*D_VO];    // Add extra padding to avoid bank conflict during nd->nz

    float output_accum[MMA_M_PER_V][D_VO];
    float cur_output_frag[NUM_OUTPUT_FRAG_BUFS][MMA_M_PER_V][OUTPUT_FRAG_D_VO];

    float p[MMA_M_PER_V*B_TOPK];

    float attn_sink[HAVE_ATTN_SINK ? MMA_M_PER_V : 0];

    // The following two row_max are NOT scaled (NOT multiplied by sm_scale)
    float row_max[2][constexpr_max(MMA_M_PER_V*2, NUM_FLOAT_IN_VEC)];    // row0 row0 row1 row1 row2 row2 ...
    float row_max_for_o[2][constexpr_max(MMA_M_PER_V*2, NUM_FLOAT_IN_VEC)];
    
    float row_sum[2][constexpr_max(MMA_M_PER_V*2, NUM_FLOAT_IN_VEC)];    // row0 row0 row1 row1 row2 row2 ...
    float row_denorm[2][constexpr_max(MMA_M_PER_V, NUM_FLOAT_IN_VEC)];

    // The following variables are not defined for block0
    float row_scales[2][2][NUM_FLOAT_IN_VEC];   // row0 row0 row1 row1 row2 row2 ...
    // `row_max_max_delta` is also not defined for block0, but it must be put at the end to guarantee alignment of the previous arrays

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

    // Packed FP4 bytes can be NaN encodings when interpreted as FP8. Separate raw slots
    // preserve the finite-FP8 invariant when a masked gather skips a reused buffer.
    uint8_t gathered_kv_fp4[HAVE_FP4 ? NUM_UB_KV_BUFS : 0][B_TOPK_PER_V][FP4_UB_TOKEN_BYTES];
    // Only the VF uses this scratch; KV_IN_NZ serializes consecutive dequant VFs.
    bf16 fp4_scales_bf16[HAVE_FP4 ? B_TOPK_PER_V : 0][FP4Format::NUM_SCALES_EACH_TOKEN];

    float row_max_max_delta[2][2];  // Must be put at the end to guarantee alignment of the previous arrays
};
static_assert(sizeof(UnifiedBufferMemoryPlan) <= 256 * 1024);

struct SSBufferMemoryPlan {
    uint32_t is_rescale_triggered[8][8][NUM_VEC_CORE_PER_AI_CORE];    // Allocate 8 slots for job_idx and 8 slots for kv_block_idx so that there will be no data race
};

struct CrossCoreFlags {
    // C -> V
    static constexpr uint32_t UBUF_P_FULL = 0;
    static constexpr uint32_t L1_S_EMPTY = 1;
    static constexpr uint32_t SS_BUFFER_FULL_C2V = 2;
    static constexpr uint32_t UBUF_O_FRAG_FULL = 4;     // [NUM_OUTPUT_FRAG_BUFS]
    static constexpr uint32_t L1_KV_BUF_EMPTY = UBUF_O_FRAG_FULL + NUM_OUTPUT_FRAG_BUFS; // [NUM_L1_KV_BUFS]
    // V -> C
    static constexpr uint32_t UBUF_P_EMPTY = 0;
    static constexpr uint32_t L1_S_FULL = 1;
    static constexpr uint32_t SS_BUFFER_FULL_V2C = 2;
    static constexpr uint32_t UBUF_O_FRAG_EMPTY = 4;    // [NUM_OUTPUT_FRAG_BUFS]
    static constexpr uint32_t L1_KV_BUF_FULL = UBUF_O_FRAG_EMPTY + NUM_OUTPUT_FRAG_BUFS;  // [NUM_L1_KV_BUFS]
};

struct VectorCoreFlags {
    // V -> MTE3
    static constexpr uint32_t S_FULL = 0;
    // MTE3 -> V
    // V -> MTE2
    static constexpr uint32_t ATTN_SINK_FULL = 2;
    // MTE2 -> V
    // V -> S
    static constexpr uint32_t ROW_MAX_MAX_DELTA_FULL = 0;
};

struct VectorCoreBufIDs {
    static constexpr uint32_t RAW_INDICES = 0;
    static constexpr uint32_t PROCESSED_INDICES = 1;
    static constexpr uint32_t FINAL_LOGITS_LSE = 2; // [2]
    static constexpr uint32_t O_ACCUM = 4;  // [D_VO / OUTPUT_FRAG_D_VO] = [4]
    static constexpr uint32_t GATHERED_KV = 8; // [NUM_UB_KV_BUFS]
    static constexpr uint32_t KV_IN_NZ = GATHERED_KV + NUM_UB_KV_BUFS;
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

template<ModelType CACHE_MODEL_TYPE = MODEL_TYPE>
static __simd_vf__ void process_kv_index_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t index_buf_idx, uint32_t s_kv, uint32_t stride_kv_block, int valid_len, uint32_t page_block_size, FastDivMod page_block_size_fast_div_mod);
static __simd_vf__ void clear_kv_buf_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);
static __simd_vf__ void kv_nd2nz_for_prefill_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);
template<ModelType CACHE_MODEL_TYPE = MODEL_TYPE>
static __simd_vf__ void kv_dequant_and_nd2nz_for_decode_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t kv_buf_idx);
template<bool IS_BLOCK0> static __simd_vf__ void softmax_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, float sm_scale, float rescale_threshold_div_sm_scale, uint32_t job_idx, uint32_t kv_block_idx, uint32_t index_buf_idx, uint32_t ub_indices_arr_offset, uint32_t s_kv, uint32_t subblock_idx);
template<bool PERFORM_SCALE, bool PERFORM_ADD> static __simd_vf__ void add_and_scale_o_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx, uint32_t kv_block_idx);
static __simd_vf__ void get_final_lse_and_denorm_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t job_idx, float sm_scale);  // TODO Move this parameter to Config after using deep_jit
template<bool ADD_OUTPUT_ACCUM> static __simd_vf__ void get_final_output_vf(__ubuf__ UnifiedBufferMemoryPlan& ub_buf, uint32_t output_frag_idx, uint32_t cur_o_frag_buf_idx, uint32_t job_idx, uint32_t kv_block_idx, bool is_warmup_mode);

struct AuxParams {
    float rescale_threshold_div_sm_scale;   // = RESCALE_THRESHOLD / sm_scale
    FastDivMod page_block_size_fast_div_mod;
    FastDivMod extra_page_block_size_fast_div_mod;
    uint32_t kv_cache_hint_for_decoding;
};

static __aicore__ void sparse_attn_fwd_kernel_devfunc(const Params params, const AuxParams aux_params);

static void run(const Params &params);

};

}
