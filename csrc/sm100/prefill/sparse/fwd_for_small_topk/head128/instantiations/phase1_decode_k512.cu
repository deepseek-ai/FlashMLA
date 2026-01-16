#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd_for_small_topk::head128 {

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(const SparseAttnDecodeParams& params);

}
