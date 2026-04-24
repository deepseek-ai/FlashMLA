#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd::head64 {

template void run_fwd_phase1_kernel<512>(const SparseAttnFwdParams& params);

}
