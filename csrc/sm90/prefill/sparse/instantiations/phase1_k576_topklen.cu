#include "../phase1.h"
#include "../phase1.cuh"

namespace sm90::fwd {

template void run_fwd_phase1_kernel<576, true>(const SparseAttnFwdParams& params);

}
