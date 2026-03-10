#pragma once

#include "../../../params.h"

namespace sm90::fwd {

template<int D_QK, bool HAVE_TOPK_LENGTH, int WIN = 128, int INDEXER_TOPK = 0>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
