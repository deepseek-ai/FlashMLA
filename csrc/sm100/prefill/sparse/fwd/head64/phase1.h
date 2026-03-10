#pragma once

#include "params.h"

namespace sm100::fwd::head64 {

template<int D_QK, int WIN = 128, int INDEXER_TOPK = 0>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
