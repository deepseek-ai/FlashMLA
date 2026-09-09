#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::prefill::sparse_fwd::head64 {

template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const SparseAttnFwdParams& params);

}
