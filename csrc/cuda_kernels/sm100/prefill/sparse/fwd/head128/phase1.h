#pragma once

#include "params.h"
#include "cuda_kernels/defines.h"

namespace sm100::prefill::sparse_fwd::head128 {

void run_sparse_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
