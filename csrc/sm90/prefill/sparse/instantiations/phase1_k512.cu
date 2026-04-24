#include "../phase1.h"
#include "../phase1.cuh"

namespace sm90::fwd {

// NOTE (intlsy): We instantiate run_fwd_phase1_kernel in two .cu files as functions with HAVE_TOPK_LENGTH
// = true / false respectively, to compile them in parallel.
template void run_fwd_phase1_kernel<512, false>(const SparseAttnFwdParams& params);

}
