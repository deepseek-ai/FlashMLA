#include "../splitkv_mla.cuh"

namespace sm80 {
template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(DenseAttnDecodeParams &params);
}

// Note: cute_traits.h sanity check moved to a separate TU
// (cute_traits_sanity.cu) so that issues with cute compose with
// splitkv_mla.cuh's includes don't block the production raw-PTX kernel.
