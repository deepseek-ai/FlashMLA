#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd::head64 {

template void run_fwd_phase1_kernel<576>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<576, 512>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<576, 1024>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<576, 2048>(const SparseAttnFwdParams& params);

}
