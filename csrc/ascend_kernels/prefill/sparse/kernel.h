#pragma once

#include "params.h"


namespace ascend::prefill::sparse_fwd {

struct Config {
    SparseAttnFwdMode FWD_MODE;
    uint32_t H_Q;
    uint32_t INDEX_BLOCK_SIZE;
    bool HAVE_ATTN_SINK;
    ModelType MODEL_TYPE;  // Primary decoding cache; prefill uses BF16 tensors.
    ModelType EXTRA_MODEL_TYPE;
    bool HAVE_Q_O_ROPE_AND_O_CAST;  // Whether or not to run Q RoPE, O RoPE (conj), and O cast-to-fp8
    bool HAVE_Q_NORM;   // Whether or not to run RMSNorm on Q
};

template<SparseAttnFwdMode FWD_MODE, bool HAVE_Q_O_ROPE_AND_O_CAST>
using ParamT = std::conditional_t<HAVE_Q_O_ROPE_AND_O_CAST,
    std::conditional_t<
        is_decode_v<FWD_MODE>,
        ParamsTemplateForFusedNormRoPEAttnRoPECast<SparseAttnDecodeParams>,
        ParamsTemplateForFusedNormRoPEAttnRoPECast<SparseAttnFwdParams>
    >,
    std::conditional_t<
        is_decode_v<FWD_MODE>,
        SparseAttnDecodeParams,
        SparseAttnFwdParams
    >
>;

template<Config CONFIG>
void run_sparse_fwd_kernel(const ParamT<CONFIG.FWD_MODE, CONFIG.HAVE_Q_O_ROPE_AND_O_CAST>& params);

}
