#include "../phase1.h"
#include "../phase1.cuh"

namespace sm90::fwd {

template void run_fwd_phase1_kernel<576, false>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<576, false, 128, 512>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<576, false, 128, 2048>(const SparseAttnFwdParams& params);

}
