#pragma once

#include "kernels/params.h"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

template<SparseAttnFwdMode FWD_MODE, int D_QK, ModelType MODEL_TYPE = ModelType::V4, ModelType EXTRA_MODEL_TYPE = MODEL_TYPE>
void run_sparse_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params);

}
