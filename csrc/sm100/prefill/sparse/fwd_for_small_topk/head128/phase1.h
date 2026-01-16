#pragma once

#include "params.h"

namespace sm100::fwd_for_small_topk::head128 {

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params);

}
