#pragma once

#include <cutlass/float8.h>

#include "kernels/params.h"

namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

// Local alias: `kernels/defines.h` calls this type `fp8`; this header needs the explicit name
// (see `csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/config.h` for the same pattern)
using fp8_e4m3 = cutlass::float_e4m3_t;

// Compile-time configuration for the fused_norm_rope_attn_rope_cast core attention kernel.
// Notes:
//   - For Prefill mode, MODEL_TYPE is always V4 since V4 and V41 has no difference.
//   - For Decode mode, MODEL_TYPE (the format of `kv`) can be V4 or V41, and EXTRA_MODEL_TYPE (the format of `extra_kv`)
//     can be MODEL_TYPE or, for MODEL_TYPE == V41, V41_FP4 (fp4 KV cache).
struct Config {
    SparseAttnFwdMode FWD_MODE;
    ModelType MODEL_TYPE;          // V4 only for prefill; V4 or V41 for decode
    ModelType EXTRA_MODEL_TYPE;    // Decode only, the format of the extra KV cache. Equals MODEL_TYPE for prefill
    uint32_t H_Q;
    bool ENABLE_Q_NORM;
};

// Parameters for the fused Q-b-norm + Q RoPE + Core Attention Forward (prefill/decoding) + O RoPE + O Cast kernel
template<typename Base>
struct ParamsTemplate : Base {
    // Q Norm
    bool enable_q_norm;
    float rms_norm_eps;

    // Q/O RoPE
    uint32_t* __restrict__ token_positions;   // [s_q]
    bool is_rope_neox_style;    // Must be false
    uint32_t rope_dim;          // Must be 64
    float* __restrict__ cos_sin_cache;  // [*, rope_dim], must be contiguous, from vllm.RotaryEmbedding.cos_sin_cache

    // O Cast
    uint32_t n_wv_group;
    uint32_t wv_group_size;
    uint32_t num_per_channels;          // Must be 128 (for v4) or 32 (for v4.1)
    bool use_tma_aligned_col_major_sf;  // Must be true
    bool round_sf;  // Must be true
    bool use_packed_ue8m0;  // Must be true

    // Output
    fp8_e4m3* __restrict__ out_fp8; // [s_q, n_wv_group, wv_group_size * d_v]
    uint32_t* __restrict__ out_sf;   // [s_q, n_wv_group, (wv_group_size*d_v) / 32 / 4], contiguous on the first (s_q) dim
    uint32_t stride_out_sf_wv_group, stride_out_sf_head_dim;
};

template<SparseAttnFwdMode FWD_MODE>
using ParamT = std::conditional_t<is_decode_v<FWD_MODE>, ParamsTemplate<SparseAttnDecodeParams>, ParamsTemplate<SparseAttnFwdParams>>;

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}
