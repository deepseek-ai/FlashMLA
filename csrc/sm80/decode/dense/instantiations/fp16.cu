#include "../splitkv_mla.cuh"

#ifndef FLASH_MLA_DISABLE_FP16
namespace sm80 {
template void run_flash_splitkv_mla_kernel<cutlass::half_t>(DenseAttnDecodeParams &params);
}
#endif
