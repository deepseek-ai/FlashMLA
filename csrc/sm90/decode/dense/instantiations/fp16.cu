#include "../splitkv_mla.cuh"
#include "../splitkv_mla.h"

namespace sm90 {

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_splitkv_mla_kernel<cutlass::half_t>(DenseAttnDecodeParams &params);
#endif

}
