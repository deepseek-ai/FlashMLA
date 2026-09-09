#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

template void run_sparse_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Decode, 512, ModelType::V41, ModelType::V41_FP4>(const SparseAttnDecodeParams& params);

}
