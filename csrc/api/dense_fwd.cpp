#include "common.h"

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
#include "cuda_kernels/sm100/prefill/dense/interface.h"
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
TORCH_LIBRARY_IMPL(flash_mla, CUDA, m) {
    m.impl("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
}
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA
