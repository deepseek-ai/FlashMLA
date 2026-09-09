#pragma once

#include "kernels/params.h"

namespace sm100::prefill::sparse_fwd::head128 {

template<int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
