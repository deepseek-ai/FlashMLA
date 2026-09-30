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
};

template<SparseAttnFwdMode FWD_MODE>
using ParamT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;

template<Config CONFIG>
void run_sparse_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}
