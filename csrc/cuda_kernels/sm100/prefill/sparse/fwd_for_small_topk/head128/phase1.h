#pragma once

#include "params.h"
#include "cuda_kernels/defines.h"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

template<SparseAttnFwdMode FWD_MODE, ModelType MODEL_TYPE = ModelType::V41, ModelType EXTRA_MODEL_TYPE = MODEL_TYPE>
void run_sparse_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params);

}
