#pragma once

#include "../../../params.h"

namespace sm90::fwd {

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
