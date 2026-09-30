#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::prefill::sparse_fwd::head64 {

void run_sparse_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate::run(params);
}

}
